"""
Quebec French Continual Pretraining Pipeline for LLAMA-3B
Adapts to Quebec French using continual pretraining
"""

import os
import json
import torch
import argparse
from argparse import BooleanOptionalAction
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

import numpy as np
from tqdm import tqdm
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
    set_seed
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers.trainer_callback import TrainerCallback
import bitsandbytes as bnb

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    model_name: str = "/home/k_ammade/Projects/CPT_scratch/llama_1b"
    use_lora: bool = True
    use_8bit: bool = False
    use_4bit: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    gradient_checkpointing: bool = True
    fsdp_enable: bool = True
    fsdp_sharding: str = "full_shard"
    fsdp_min_num_params: float = 1e8
    fsdp_wrap_cls: str = "LlamaDecoderLayer"


@dataclass
class DataConfig:
    """Configuration for data processing"""
    train_file: str = "train.txt"
    max_length: int = 2048
    stride: int = 512
    batch_size: int = 8
    preprocessing_num_workers: int = 4
    tokenizer_batch_size: int = 1000


@dataclass
class TrainingConfig:
    """Configuration for training"""
    output_dir: str = "./quebec_llama3.2_1b"
    num_epochs: int = 4
    learning_rate: float = 2e-6
    warmup_ratio: float = 0.3
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 8
    fp16: bool = True
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 2
    seed: int = 42
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None


class QuebecFrenchDataProcessor:
    """Handles data loading and preprocessing for Quebec French text"""
    
    def __init__(self, tokenizer, config: DataConfig):
        self.tokenizer = tokenizer
        self.config = config
        
    def load_text_data(self, file_path: str) -> List[str]:
        """Load text data from file with minimal cleaning for CPT"""
        logger.info(f"Loading data from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # MINIMAL filtering - only remove completely empty lines
        # Preserve whitespace-only lines as they may indicate paragraph breaks
        texts = []
        for line in lines:
            # Only remove lines that are completely empty (not even whitespace)
            if line.strip() or line.isspace():  
                # Keep original formatting - only strip final newline
                texts.append(line.rstrip('\n\r'))
        
        logger.info(f"Loaded {len(texts)} text samples (minimal filtering)")
        
        # Show raw examples to verify preservation
        # logger.info("\nRaw text samples (showing exact formatting):")
        # logger.info("-" * 40)
        # for i, text in enumerate(texts[:3]):
        #     logger.info(f"Sample {i+1} (length: {len(text)} chars): {repr(text[:100])}")
        
        return texts
    
    def create_dataset(self, texts: List[str]) -> Dataset:
        """Create HuggingFace dataset from texts"""
        return Dataset.from_dict({"text": texts})
    
    def group_texts(self, examples):
        """Group texts into fixed-length chunks with proper padding"""
        # Concatenate all texts with EOS tokens
        all_text_parts = []
        for batch_texts in examples['input_ids']:
            all_text_parts.extend(batch_texts)
            all_text_parts.append(self.tokenizer.eos_token_id)

        result = {'input_ids': [], 'attention_mask': [], 'labels': []}
        L = self.config.max_length
        
        # Create fixed-length chunks
        for i in range(0, len(all_text_parts), L):
            chunk = all_text_parts[i:i + L]
            
            # IMPORTANT: Pad the last chunk to max_length if it's shorter
            if len(chunk) < L:
                # Pad with pad_token_id
                padding_length = L - len(chunk)
                chunk = chunk + [self.tokenizer.pad_token_id] * padding_length
                attention_mask = [1] * (L - padding_length) + [0] * padding_length
                # For labels, mask out padding tokens with -100
                labels = all_text_parts[i:i + (L - padding_length)] + [-100] * padding_length
            else:
                attention_mask = [1] * L
                labels = chunk.copy()
            
            # Ensure all are exactly max_length
            assert len(chunk) == L, f"Chunk length {len(chunk)} != {L}"
            assert len(attention_mask) == L, f"Attention mask length {len(attention_mask)} != {L}"
            assert len(labels) == L, f"Labels length {len(labels)} != {L}"
            
            result['input_ids'].append(chunk)
            result['attention_mask'].append(attention_mask)
            result['labels'].append(labels)

        return result
    
    def tokenize_function(self, examples):
        """Tokenize texts with proper padding and truncation"""
        # Tokenize without padding (we'll pad in group_texts)
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=self.config.max_length,
        )
    
    def prepare_dataset(self, train_texts: List[str]) -> Dataset:
        """Prepare dataset using ALL data for training (no validation split)"""
        # Create dataset with all data
        train_dataset = self.create_dataset(train_texts)
        
        # Tokenize dataset
        logger.info("Tokenizing dataset...")
        tokenized_train = train_dataset.map(
            self.tokenize_function,
            batched=True,
            batch_size=self.config.tokenizer_batch_size,
            num_proc=self.config.preprocessing_num_workers,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing train dataset"
        )
        
        # Group texts into chunks with padding
        logger.info("Grouping texts into fixed-length chunks...")
        tokenized_train = tokenized_train.map(
            self.group_texts,
            batched=True,
            batch_size=self.config.tokenizer_batch_size,
            num_proc=self.config.preprocessing_num_workers,
            desc="Grouping train texts"
        )
        
        # Filter out any potential empty examples (shouldn't happen with fixed-length chunks)
        tokenized_train = tokenized_train.filter(lambda x: len(x['input_ids']) > 0)
        
        # Verify all sequences have the same length
        logger.info("Verifying sequence lengths...")
        sample_lengths = [len(tokenized_train[i]['input_ids']) for i in range(min(10, len(tokenized_train)))]
        logger.info(f"Sample sequence lengths: {sample_lengths}")
        if len(set(sample_lengths)) > 1:
            logger.warning(f"WARNING: Variable sequence lengths detected: {set(sample_lengths)}")
        
        return tokenized_train


class ModelSetup:
    """Handles model initialization and configuration"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    def get_quantization_config(self):
        """Get quantization configuration if needed"""
        if self.config.use_4bit:
            from transformers import BitsAndBytesConfig
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
        elif self.config.use_8bit:
            from transformers import BitsAndBytesConfig
            return BitsAndBytesConfig(load_in_8bit=True)
        return None
    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer"""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Decide compute dtype once (bf16 on A100+, else fp16)
        is_ampere = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
        amp_dtype = torch.bfloat16 if is_ampere else torch.float16
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            local_files_only=True,
            trust_remote_code=True
        )
        
        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        tokenizer.padding_side = "right"
        
        # Quantization config (only allowed when NOT using FSDP)
        quantization_config = self.get_quantization_config()

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": amp_dtype,
            "low_cpu_mem_usage": True,
        }

        if quantization_config is not None and not getattr(self.config, "fsdp_enable", False):
            model_kwargs.update({
                "quantization_config": quantization_config,
                "device_map": "auto",
            })

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            local_files_only=True,
            **model_kwargs
        )
        
        # Enable grad checkpointing early
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False

        # Safer attention backend defaults
        try:
            model.config.attn_implementation = "sdpa"
        except Exception:
            pass
        
        # Ensure model is in training mode
        model.train()
        
        # Prepare model for training 
        if self.config.use_4bit or self.config.use_8bit:
            model = prepare_model_for_kbit_training(
                model, 
                use_gradient_checkpointing=self.config.gradient_checkpointing
            )
        
        # Enable gradient checkpointing before LoRA setup
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        # Setup LoRA if specified
        if self.config.use_lora:
            logger.info("Setting up LoRA...")
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                init_lora_weights=True,
                use_rslora=False,
            )
            model = get_peft_model(model, lora_config)
            
            model.train()
            
            # Print trainable parameters for verification
            model.print_trainable_parameters()
            
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Number of trainable parameters: {trainable_params}")
            if trainable_params == 0:
                raise ValueError("No trainable parameters found! Check your LoRA configuration.")
        
        return model, tokenizer


class PerplexityCallback(TrainerCallback):
    """Custom callback to log perplexity during training"""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if "loss" in logs:
                logs["perplexity"] = np.exp(logs["loss"])
            if "eval_loss" in logs:
                logs["eval_perplexity"] = np.exp(logs["eval_loss"])


class FixedLengthDataCollator(DataCollatorForLanguageModeling):
    """Custom data collator that ensures all sequences are the same length"""
    
    def __call__(self, features):
        # Verify all sequences have the same length
        input_ids_lengths = [len(f['input_ids']) for f in features]
        if len(set(input_ids_lengths)) > 1:
            raise ValueError(f"Found variable sequence lengths in batch: {set(input_ids_lengths)}")
        
        # Call parent collator
        batch = super().__call__(features)
        
        # Additional verification
        for key in ['input_ids', 'attention_mask', 'labels']:
            if key in batch:
                if batch[key].dim() != 2:
                    raise ValueError(f"{key} should be 2D tensor, got {batch[key].dim()}D")
        
        return batch


class QuebecFrenchTrainer:
    """Main trainer class for Quebec French adaptation"""
    
    def __init__(
        self,
        model_config: ModelConfig,
        data_config: DataConfig,
        training_config: TrainingConfig
    ):
        self.model_config = model_config
        self.data_config = data_config
        self.training_config = training_config
        
        # Set seed for reproducibility
        set_seed(training_config.seed)

        # Setup model and tokenizer
        model_setup = ModelSetup(model_config)
        self.model, self.tokenizer = model_setup.setup_model_and_tokenizer()
        
        # Setup data processor
        self.data_processor = QuebecFrenchDataProcessor(self.tokenizer, data_config)
    
    def inspect_training_data_single(self, train_dataset: Dataset, num_samples: int = 3):
        """Inspect a few training samples to verify data processing (single dataset version)"""
        logger.info("=" * 80)
        logger.info("TRAINING DATA INSPECTION")
        logger.info("=" * 80)
        
        logger.info(f"Dataset info:")
        logger.info(f"  - Train samples: {len(train_dataset)}")
        logger.info(f"  - Features: {train_dataset.features}")
        
        # Check sequence lengths
        logger.info("\nChecking sequence lengths consistency:")
        lengths = []
        for i in range(min(100, len(train_dataset))):
            sample = train_dataset[i]
            input_len = len(sample['input_ids'])
            attn_len = len(sample['attention_mask'])
            label_len = len(sample['labels'])
            lengths.append((input_len, attn_len, label_len))
            
            if input_len != self.data_config.max_length:
                logger.warning(f"Sample {i}: input_ids length {input_len} != max_length {self.data_config.max_length}")
        
        unique_lengths = set(lengths)
        logger.info(f"Unique length combinations (input, attention, labels): {unique_lengths}")
        
        logger.info("\nSample training examples:")
        logger.info("-" * 50)
        
        for i in range(min(num_samples, len(train_dataset))):
            sample = train_dataset[i]
            
            logger.info(f"\n--- SAMPLE {i+1} ---")
            logger.info(f"Input IDs length: {len(sample['input_ids'])}")
            logger.info(f"Attention mask length: {len(sample['attention_mask'])}")
            logger.info(f"Labels length: {len(sample['labels'])}")
            
            # Check for padding
            pad_count = sum(1 for token_id in sample['input_ids'] if token_id == self.tokenizer.pad_token_id)
            logger.info(f"Padding tokens: {pad_count}")
            
            # Check labels masking
            masked_labels = sum(1 for label in sample['labels'] if label == -100)
            logger.info(f"Masked labels (-100): {masked_labels}")
            
            # Decode the input to see actual text
            decoded_text = self.tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
            logger.info(f"Decoded text preview (first 200 chars):")
            logger.info(f"'{decoded_text[:200]}...'")
            
            # Show first and last few tokens
            logger.info(f"First 10 token IDs: {sample['input_ids'][:10]}")
            logger.info(f"Last 10 token IDs: {sample['input_ids'][-10:]}")
        
        logger.info("=" * 80)
    
    def train(self, inspect_data: bool = True, inspect_samples: int = 3):
        """Run training using only train file"""
        # Load ONLY train data - no validation logic
        train_texts = self.data_processor.load_text_data(self.data_config.train_file)
        logger.info(f"Using train data only: {len(train_texts)} text lines")
        
        # Prepare training dataset
        train_dataset = self.data_processor.prepare_dataset(train_texts)
        logger.info(f"Final training dataset size: {len(train_dataset)} chunks")
        
        # Verify model setup
        logger.info("Verifying model setup for CPT...")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
        
        if trainable_params == 0:
            raise ValueError("No trainable parameters found!")
        
        # Inspect data if requested
        if inspect_data:
            self.inspect_training_data_single(train_dataset, inspect_samples)
        
        # Determine dtype
        use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8  # A100/H100
        fp16_flag = not use_bf16
        
        # FSDP config
        fsdp_list = []
        fsdp_cfg = None
        if getattr(self.model_config, "fsdp_enable", False):
            fsdp_list = [self.model_config.fsdp_sharding, "auto_wrap"]
            fsdp_cfg = {
                "transformer_layer_cls_to_wrap": self.model_config.fsdp_wrap_cls,
                "xla": False,
                "cpu_offload": False,
                "use_orig_params": True,
                "sync_module_states": False,
            }

        training_args = TrainingArguments(
            output_dir=self.training_config.output_dir,
            num_train_epochs=self.training_config.num_epochs,
            per_device_train_batch_size=self.data_config.batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            warmup_ratio=self.training_config.warmup_ratio,
            weight_decay=self.training_config.weight_decay,
            fp16=fp16_flag,
            bf16=use_bf16,
            logging_steps=self.training_config.logging_steps,
            save_steps=self.training_config.save_steps * 2,
            do_eval=False,
            save_strategy="steps",
            save_total_limit=self.training_config.save_total_limit,
            push_to_hub=self.training_config.push_to_hub,
            hub_model_id=self.training_config.hub_model_id,
            report_to="tensorboard",
            logging_dir=f"{self.training_config.output_dir}/logs",
            seed=self.training_config.seed,
            dataloader_num_workers=4,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            optim="adamw_torch",
            max_grad_norm=1.0,
            logging_first_step=True,
            # FSDP knobs
            fsdp=" ".join(fsdp_list) if fsdp_list else None,
            fsdp_config=fsdp_cfg,
            ddp_find_unused_parameters=False,
        )
        
        # Setup custom data collator that ensures fixed lengths
        data_collator = FixedLengthDataCollator(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=None  # We're already padding in group_texts
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            callbacks=[PerplexityCallback()]
        )
        
        # Start training
        logger.info("Starting Quebec French CPT (train-only)...")
        train_result = trainer.train()
        
        # Save model
        logger.info("Saving adapted model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.training_config.output_dir)
        
        # Save metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        return trainer, metrics
    
    def generate_sample(self, prompt: str, max_length: int = 100):
        """Generate a sample to test Quebec French adaptation"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id
            )   
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def save_config(config_dict: Dict, output_dir: str):
    """Save configuration to JSON file"""
    config_path = Path(output_dir) / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"Configuration saved to {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Quebec French CPT for LLaMA")

    # Model arguments
    parser.add_argument("--model_name", type=str, default="/home/k_ammade/Projects/CPT_scratch/llama_1b")
    parser.add_argument("--use_lora", action=BooleanOptionalAction, default=True)
    parser.add_argument("--use_4bit", action=BooleanOptionalAction, default=False)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)

    # FSDP arguments
    parser.add_argument("--fsdp_enable", action=BooleanOptionalAction, default=True)
    parser.add_argument("--fsdp_sharding", type=str, default="full_shard",
                        choices=["full_shard", "shard_grad_op"])
    parser.add_argument("--fsdp_min_num_params", type=float, default=1e8)
    parser.add_argument("--fsdp_wrap_cls", type=str, default="LlamaDecoderLayer")

    # Data arguments
    parser.add_argument("--train_file", type=str, required=True, help="Path to Quebec French training corpus (text file)")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--inspect_data", action=BooleanOptionalAction, default=False)
    parser.add_argument("--inspect_samples", type=int, default=0)

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./quebec_french_llama3.2_1b")
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)

    args = parser.parse_args()
    
    # Create configurations
    model_config = ModelConfig(
        model_name=args.model_name,
        use_lora=args.use_lora,
        use_4bit=args.use_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        fsdp_enable=args.fsdp_enable,
        fsdp_sharding=args.fsdp_sharding,
        fsdp_min_num_params=args.fsdp_min_num_params,
        fsdp_wrap_cls=args.fsdp_wrap_cls,
    )
    
    data_config = DataConfig(
        train_file=args.train_file,
        max_length=args.max_length,
        batch_size=args.batch_size
    )
    
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    # Save configuration
    config_dict = {
        "model": asdict(model_config),
        "data": asdict(data_config),
        "training": asdict(training_config)
    }
    save_config(config_dict, args.output_dir)
    
    # Initialize trainer
    trainer = QuebecFrenchTrainer(model_config, data_config, training_config)
    
    # Run training
    trainer_obj, train_metrics = trainer.train(inspect_data=args.inspect_data, 
                                               inspect_samples=args.inspect_samples)
    
    # Test Quebec French generation
    logger.info("Testing Quebec French adaptation...")
    quebec_prompts = [
        "À matin j'ai",
        "Pis là, tu sais ben que", 
        "C'est ben correct, mais"
    ]
    
    for prompt in quebec_prompts:
        generated = trainer.generate_sample(prompt, max_length=50)
        logger.info(f"Prompt: '{prompt}' → Generated: '{generated}'")
    
    logger.info("CPT completed successfully!")
    logger.info(f"Final training loss: {train_metrics.get('train_loss', 'N/A')}")


if __name__ == "__main__":
    main()