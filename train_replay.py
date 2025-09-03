"""
Quebec French Continual Pretraining Pipeline for LLAMA-3B
Adapts to Quebec French using continual pretraining
"""

import os
import json
import random
import math
import tiktoken
import torch
import argparse
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
    """Configuration for model setup"""
    model_name: str = "meta-llama/Llama-3.2-3B"
    use_lora: bool = True
    use_8bit: bool = False
    use_4bit: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    gradient_checkpointing: bool = True


@dataclass
class DataConfig:
    """Configuration for data processing"""
    train_file: str = "train.txt"
    val_file: Optional[str] = "val.txt"
    replay_file: str = "croissant.jsonl"
    max_length: int = 2048
    stride: int = 512
    batch_size: int = 8
    val_split: float = 0.1
    preprocessing_num_workers: int = 4
    tokenizer_batch_size: int = 1000


@dataclass
class TrainingConfig:
    """Configuration for training"""
    output_dir: str = "./quebec_llama3.2_3b"
    num_epochs: int = 1
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.1
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
        
    def load_text_data(self, file_path: str, text_field: str = "text") -> List[str]:
        """Load text data from file (supports .txt and .jsonl formats)
        
        Args:
            file_path: Path to the input file
            text_field: Field name to extract from JSONL objects (default: "text")
        
        Returns:
            List of text strings
        """
        logger.info(f"Loading data from {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        texts = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_ext == '.jsonl':
                # Handle JSONL format
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        json_obj = json.loads(line)
                        
                        # Extract text from the specified field
                        if text_field in json_obj:
                            text = json_obj[text_field]
                            if isinstance(text, str) and text.strip():
                                texts.append(text.strip())
                        else:
                            # If text_field not found, try common alternatives
                            for field in ["text", "content", "message", "data"]:
                                if field in json_obj:
                                    text = json_obj[field]
                                    if isinstance(text, str) and text.strip():
                                        texts.append(text.strip())
                                    break
                            else:
                                logger.warning(f"Line {line_num}: No text field found in JSON object")
                                
                    except json.JSONDecodeError as e:
                        logger.warning(f"Line {line_num}: Invalid JSON - {e}")
                        continue
            else:
                # Handle regular text format
                lines = f.readlines()
                # Filter empty lines and strip whitespace
                texts = [line.strip() for line in lines if line.strip()]
        
        logger.info(f"Loaded {len(texts)} text samples")
        
        # Show a few raw examples
        logger.info("-" * 40)
        for i, text in enumerate(texts[:3]):
            logger.info(f"Sample {i+1} (length: {len(text)} chars): {text[:100]}{'...' if len(text) > 100 else ''}")
        
        return texts
    
    def create_dataset(self, texts: List[str]) -> Dataset:
        """Create HuggingFace dataset from texts"""
        return Dataset.from_dict({"text": texts})
    
    def group_texts(self, examples):
        """Group texts into chunks of max_length"""
        # Check if the input is already tokenized
        if not isinstance(examples['input_ids'][0], list):
            # If input_ids are not lists, they're already flat - just chunk them
            total_length = len(examples['input_ids'])
            
            # We drop the small remainder, you can customize this part to your needs
            if total_length >= self.config.max_length:
                total_length = (total_length // self.config.max_length) * self.config.max_length
            
            # Split by chunks of max_length
            result = {
                'input_ids': [examples['input_ids'][i : i + self.config.max_length]
                             for i in range(0, total_length, self.config.max_length)],
                'attention_mask': [examples['attention_mask'][i : i + self.config.max_length]
                                  for i in range(0, total_length, self.config.max_length)]
            }
        else:
            # Original logic for nested lists
            # Concatenate all texts
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            
            # Drop the last chunk if it's too small
            if total_length >= self.config.max_length:
                total_length = (total_length // self.config.max_length) * self.config.max_length
            
            # Split by chunks of max_length
            result = {
                k: [t[i : i + self.config.max_length] 
                    for i in range(0, total_length, self.config.max_length)]
                for k, t in concatenated_examples.items()
            }
        
        # Add labels (same as input_ids for CLM)
        result["labels"] = result["input_ids"].copy()
        return result
    
    def tokenize_function(self, examples):
        """Tokenize texts with proper padding and truncation"""
        # Simpler tokenization without return_overflowing_tokens for now
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=self.config.max_length,
        )
        
        
    def prepare_dataset(self, train_texts: List[str], val_texts: Optional[List[str]] = None) -> DatasetDict:
        """Prepare train and validation datasets"""
        # Create datasets
        train_dataset = self.create_dataset(train_texts)
        
        if val_texts:
            val_dataset = self.create_dataset(val_texts)
        else:
            # Split train data if no validation provided
            split = train_dataset.train_test_split(test_size=self.config.val_split, seed=42)
            train_dataset = split["train"]
            val_dataset = split["test"]
        
        # Tokenize datasets
        logger.info("Tokenizing datasets...")
        tokenized_train = train_dataset.map(
            self.tokenize_function,
            batched=True,
            batch_size=self.config.tokenizer_batch_size,
            num_proc=self.config.preprocessing_num_workers,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing train dataset"
        )
        
        tokenized_val = val_dataset.map(
            self.tokenize_function,
            batched=True,
            batch_size=self.config.tokenizer_batch_size,
            num_proc=self.config.preprocessing_num_workers,
            remove_columns=val_dataset.column_names,
            desc="Tokenizing validation dataset"
        )
        
        # Group texts into chunks
        logger.info("Grouping texts into chunks...")
        tokenized_train = tokenized_train.map(
            self.group_texts,
            batched=True,
            batch_size=self.config.tokenizer_batch_size,
            num_proc=self.config.preprocessing_num_workers,
            desc="Grouping train texts"
        )
        
        tokenized_val = tokenized_val.map(
            self.group_texts,
            batched=True,
            batch_size=self.config.tokenizer_batch_size,
            num_proc=self.config.preprocessing_num_workers,
            desc="Grouping validation texts"
        )
        
        # Filter out empty examples
        tokenized_train = tokenized_train.filter(lambda x: len(x['input_ids']) > 0)
        tokenized_val = tokenized_val.filter(lambda x: len(x['input_ids']) > 0)
        
        return DatasetDict({
            "train": tokenized_train,
            "validation": tokenized_val
        })


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
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        # Add pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Load model with quantization if specified
        quantization_config = self.get_quantization_config()
        
        # FIXED: Consistent model loading approach
        model_kwargs = {
            "trust_remote_code": True,
        }
        
        if quantization_config is not None:
            model_kwargs.update({
                "quantization_config": quantization_config,
                "device_map": "auto",
                "torch_dtype": torch.float16,
            })
        else:
            model_kwargs.update({
                "torch_dtype": torch.float16,  # Changed from float32 to float16 for consistency
            })
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        # FIXED: Ensure model is in training mode
        model.train()
        
        # Prepare model for training (FIXED: Do this before LoRA setup)
        if self.config.use_4bit or self.config.use_8bit:
            model = prepare_model_for_kbit_training(
                model, 
                use_gradient_checkpointing=self.config.gradient_checkpointing
            )
        
        # FIXED: Enable gradient checkpointing before LoRA setup
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
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.data.to(torch.float32)
            
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


def count_tokens_in_file(file_path, encoding_name="cl100k_base"):
        """Counts the number of tokens in a text file using tiktoken."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            encoding = tiktoken.get_encoding(encoding_name)
            tokens = encoding.encode(text)
            return len(tokens)
        except FileNotFoundError:
            return f"Error: File '{file_path}' not found."
        except Exception as e:
            return f"An error occurred: {e}"
        
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
    
    def inspect_training_data(self, datasets: DatasetDict, num_samples: int = 3):
        """Inspect a few training samples to verify data processing"""
        logger.info("=" * 80)
        logger.info("TRAINING DATA INSPECTION")
        logger.info("=" * 80)
        
        train_dataset = datasets["train"]
        
        logger.info(f"Dataset info:")
        logger.info(f"  - Train samples: {len(train_dataset)}")
        logger.info(f"  - Validation samples: {len(datasets['validation'])}")
        logger.info(f"  - Features: {train_dataset.features}")
        
        logger.info("\nSample training examples:")
        logger.info("-" * 50)
        
        for i in range(min(num_samples, len(train_dataset))):
            sample = train_dataset[i]
            
            logger.info(f"\n--- SAMPLE {i+1} ---")
            logger.info(f"Input IDs length: {len(sample['input_ids'])}")
            logger.info(f"Attention mask length: {len(sample['attention_mask'])}")
            logger.info(f"Labels length: {len(sample['labels'])}")
            
            # Decode the input to see actual text
            decoded_text = self.tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
            logger.info(f"Decoded text preview (first 200 chars):")
            logger.info(f"'{decoded_text[:200]}...'")
            
            # Show first and last few tokens
            logger.info(f"First 10 token IDs: {sample['input_ids'][:10]}")
            logger.info(f"Last 10 token IDs: {sample['input_ids'][-10:]}")
            
            # Decode first and last few tokens to text
            first_tokens_text = self.tokenizer.decode(sample['input_ids'][:10], skip_special_tokens=True)
            last_tokens_text = self.tokenizer.decode(sample['input_ids'][-10:], skip_special_tokens=True)
            logger.info(f"First 10 tokens as text: '{first_tokens_text}'")
            logger.info(f"Last 10 tokens as text: '{last_tokens_text}'")
            
            # Check for special tokens
            special_token_count = sum(1 for token_id in sample['input_ids'] 
                                    if token_id in [self.tokenizer.eos_token_id, 
                                                   self.tokenizer.bos_token_id, 
                                                   self.tokenizer.pad_token_id])
            logger.info(f"Special tokens count: {special_token_count}")
            
            # Verify labels match input_ids (for causal LM)
            labels_match = sample['input_ids'] == sample['labels']
            logger.info(f"Labels match input_ids: {labels_match}")
        
        # Check vocabulary coverage for Quebec French
        logger.info("\n" + "-" * 50)
        logger.info("QUEBEC FRENCH VOCABULARY ANALYSIS")
        logger.info("-" * 50)
        
        # Sample some Quebec French specific words/phrases if present
        quebec_indicators = ["qu'", "pis", "icitte", "toé", "moé", "tsé", "ben", "pantoute", "à soir"]
        
        sample_text = ""
        for i in range(min(5, len(train_dataset))):
            sample_text += self.tokenizer.decode(train_dataset[i]['input_ids'], skip_special_tokens=True)
        
        found_quebec_terms = []
        for term in quebec_indicators:
            if term.lower() in sample_text.lower():
                found_quebec_terms.append(term)
        
        logger.info(f"Quebec French indicators found: {found_quebec_terms}")
        
        # Token statistics
        all_token_ids = []
        for i in range(min(100, len(train_dataset))):  # Sample first 100 examples
            all_token_ids.extend(train_dataset[i]['input_ids'])
        
        unique_tokens = len(set(all_token_ids))
        vocab_size = len(self.tokenizer)
        logger.info(f"Unique tokens in sample: {unique_tokens:,}")
        logger.info(f"Total vocabulary size: {vocab_size:,}")
        logger.info(f"Vocabulary coverage: {100 * unique_tokens / vocab_size:.2f}%")
        
        logger.info("=" * 80)
    
    def train(self, inspect_data: bool = True, inspect_samples: int = 3):
        """Run the training process"""
        # Load and prepare data
        train_texts = self.data_processor.load_text_data(self.data_config.train_file)
        replay_texts = self.data_processor.load_text_data(self.data_config.replay_file)
        val_texts = None

        file_name = "/home/k_ammade/CPT_scratch/data/83M_data/train.txt"
        
        total_tokens = count_tokens_in_file(file_name)
        print(file_name)
        print(total_tokens)
        target_replay_tokens = math.floor(0.1 * total_tokens) # 10% replay
        
        encoding = tiktoken.get_encoding("cl100k_base")
        
        current_replay_tokens = sum(len(encoding.encode(text)) for text in replay_texts)
        
        enc = tiktoken.get_encoding("cl100k_base")
        def toklen(s: str) -> int:
            return len(enc.encode(s))

        # replay_texts must be a List[str]
        assert isinstance(replay_texts, list) and all(isinstance(t, str) for t in replay_texts), \
            "replay_texts must be List[str]"

        current_replay_tokens = sum(toklen(text) for text in replay_texts)

        # (optional) sample more replay to hit the 10% target
        needed = max(0, target_replay_tokens - current_replay_tokens)
        if needed > 0 and replay_texts:
            shuffled = replay_texts[:]  # don’t mutate original
            random.shuffle(shuffled)
            extra, acc = [], 0
            for t in shuffled:
                if acc >= needed:
                    break
                extra.append(t)
                acc += toklen(t)
            # prepend replay to the current task data
            train_texts = extra + train_texts
        
        if current_replay_tokens < target_replay_tokens:
            needed_tokens = target_replay_tokens - current_replay_tokens
            
            # Shuffle train texts for random sampling
            shuffled_train = replay_texts.copy()
            random.shuffle(shuffled_train)
            
            additional_texts = []
            accumulated_tokens = 0
            
            for text in shuffled_train:
                if accumulated_tokens >= needed_tokens:
                    break
                    
                text_tokens = len(encoding.encode(text))
                additional_texts.append(text)
                accumulated_tokens += text_tokens
            
            # Add the additional texts to replay_texts
            train_texts.extend(additional_texts)
            
            logger.info(f"Added {len(additional_texts)} texts ({accumulated_tokens} tokens) from train to replay")
            logger.info(f"Replay data now contains {len(train_texts)} texts")
        
        if self.data_config.val_file and os.path.exists(self.data_config.val_file):
            val_texts = self.data_processor.load_text_data(self.data_config.val_file)
        
        datasets = self.data_processor.prepare_dataset(train_texts, val_texts)
        
        logger.info(f"Train dataset size: {len(datasets['train'])}")
        logger.info(f"Validation dataset size: {len(datasets['validation'])}")
        
        # ADDED: Inspect training data to see actual inputs
        if inspect_data:
            self.inspect_training_data(datasets, num_samples=inspect_samples)
        
        # FIXED: Enhanced training arguments
        training_args = TrainingArguments(
            output_dir=self.training_config.output_dir,
            num_train_epochs=self.training_config.num_epochs,
            per_device_train_batch_size=self.data_config.batch_size,
            per_device_eval_batch_size=self.data_config.batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            warmup_ratio=self.training_config.warmup_ratio,
            weight_decay=self.training_config.weight_decay,
            fp16=self.training_config.fp16,
            logging_steps=self.training_config.logging_steps,
            save_steps=self.training_config.save_steps,
            eval_steps=self.training_config.eval_steps,
            eval_strategy="steps",
            save_strategy="steps",
            save_total_limit=self.training_config.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
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
        )
        
        # Setup data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Initialize trainer (using standard Trainer - custom trainer removed for simplicity)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            data_collator=data_collator,
            callbacks=[PerplexityCallback()]
        )
        
        # ADDED: Show actual batch from DataLoader
        if inspect_data:
            logger.info("\n" + "=" * 80)
            logger.info("ACTUAL TRAINING BATCH INSPECTION")
            logger.info("=" * 80)
            
            train_dataloader = trainer.get_train_dataloader()
            sample_batch = next(iter(train_dataloader))
            
            logger.info(f"Batch keys: {list(sample_batch.keys())}")
            logger.info(f"Batch size: {sample_batch['input_ids'].shape[0]}")
            logger.info(f"Sequence length: {sample_batch['input_ids'].shape[1]}")
            logger.info(f"Input IDs shape: {sample_batch['input_ids'].shape}")
            logger.info(f"Attention mask shape: {sample_batch['attention_mask'].shape}")
            logger.info(f"Labels shape: {sample_batch['labels'].shape}")
            
            # Show first sample in batch
            first_sample = {k: v[0] for k, v in sample_batch.items()}
            decoded_batch_text = self.tokenizer.decode(first_sample['input_ids'], skip_special_tokens=True)
            logger.info(f"First sample in batch (decoded): '{decoded_batch_text[:200]}...'")
            logger.info("=" * 80)
        
        # FIXED: Final check before training
        logger.info("Verifying model setup before training...")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
        
        if trainable_params == 0:
            raise ValueError("No trainable parameters found! Training cannot proceed.")
        
        # Start training
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # Save final model
        logger.info("Saving final model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.training_config.output_dir)
        
        # Save training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        # Evaluate on validation set
        logger.info("Running final evaluation...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        
        return trainer, metrics, eval_metrics
    
    def generate_sample(self, prompt: str, max_length: int = 100):
        """Generate a sample text for quick testing"""
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
    parser = argparse.ArgumentParser(description="Quebec French Continual Pretraining for LLAMA-3B")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--use_4bit", action="store_true", default=False)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    # Data arguments
    parser.add_argument("--train_file", type=str, default="/home/k_ammade/Projects/QuebecCPT/CPT_scratch/data/train.txt")
    parser.add_argument("--val_file", type=str, default="/home/k_ammade/Projects/QuebecCPT/CPT_scratch/data/val.txt")
    parser.add_argument("--replay_file", type=str, default="/home/k_ammade/Projects/QuebecCPT/CPT_scratch/data/croissant.jsonl")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--inspect_data", action="store_true", default=False)
    parser.add_argument("--inspect_samples", type=int, default=4)
        
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./quebec_french_llama_3.1_8B")
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    
    args = parser.parse_args()
    
    # Create configurations
    model_config = ModelConfig(
        model_name=args.model_name,
        use_lora=args.use_lora,
        use_4bit=args.use_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )
    
    data_config = DataConfig(
        train_file=args.train_file,
        val_file=args.val_file,
        replay_file = args.replay_file,
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
    trainer_obj, train_metrics, eval_metrics = trainer.train(inspect_data=args.inspect_data, 
                                                            inspect_samples=args.inspect_samples)
    
    # Test generation
    logger.info("Testing generation with Quebec French prompt...")
    test_prompt = "Pour certains c'est un symbole"
    generated = trainer.generate_sample(test_prompt, max_length=100)
    logger.info(f"Generated text: {generated}")
    
    logger.info("Training completed successfully!")
    logger.info(f"Final training loss: {train_metrics.get('train_loss', 'N/A')}")
    logger.info(f"Final eval perplexity: {eval_metrics.get('eval_perplexity', 'N/A')}")


if __name__ == "__main__":
    main()
    

# python /home/k_ammade/Projects/QuebecCPT/CPT_scratch/train_replay.py --train_file /home/k_ammade/Projects/QuebecCPT/CPT_scratch/data/23M_data/train.txt --val_file /home/k_ammade/Projects/QuebecCPT/CPT_scratch/data/23M_data/val.txt --replay_file /home/k_ammade/Projects/QuebecCPT/CPT_scratch/data/croissant.jsonl --model_name croissantllm/CroissantLLMChat-v0.1 --output_dir /home/k_ammade/Projects/QuebecCPT/CPT_scratch/quebec_croissant_chat_23M_replay --use_lora --inspect_data --inspect_samples 8