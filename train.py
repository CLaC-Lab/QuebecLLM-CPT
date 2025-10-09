#!/usr/bin/env python
"""
Quebec French Continual Pretraining Pipeline - FIXED VERSION
Updated --> Fixed tensor dimension mismatch issue in data processing
"""

import os
import json
import torch
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import unicodedata

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
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    gradient_checkpointing: bool = True


@dataclass
class DataConfig:
    """Configuration for data processing"""
    train_file: str = "train.txt"
    max_length: int = 1024  # Reduced from 2048 to match error
    stride: int = 512  # 50% overlap for better context preservation
    batch_size: int = 8
    preprocessing_num_workers: int = 4
    tokenizer_batch_size: int = 1000
    min_length: int = 50  # Filter out very short segments
    

@dataclass
class TrainingConfig:
    """Configuration for training"""
    output_dir: str = "./quebec_french_llama3.2_3b_3E"
    num_epochs: int = 3
    learning_rate: float = 1e-5  # Slightly higher for better Quebec French adaptation
    warmup_ratio: float = 0.1    # Reduced warmup for CPT
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 8
    fp16: bool = True
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 50  # More frequent logging
    save_total_limit: int = 3  # Keep more checkpoints for CPT
    seed: int = 42
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None


class QuebecFrenchDataProcessor:
    """Enhanced data processor for Quebec French text - FIXED VERSION"""
    
    def __init__(self, tokenizer, config: DataConfig):
        self.tokenizer = tokenizer
        self.config = config
        
    def load_text_data(self, file_path: str) -> List[str]:
        """Load and lightly clean Quebec French data"""
        logger.info(f"Loading Quebec French data from {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Quebec French specific cleaning while preserving dialect
        texts = []
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Very light normalization - preserve Quebec French characteristics
            # Only fix obvious encoding issues, keep dialect intact
            line = line.replace('\ufeff', '')  # Remove BOM
            line = ' '.join(line.split())      # Normalize whitespace
            
            line = unicodedata.normalize("NFC", line)
            
            # Keep lines that have substance (minimum length)
            if len(line) >= 10:  # Very permissive minimum
                texts.append(line)
        
        logger.info(f"Loaded {len(texts)} Quebec French text segments")
        
        # Analyze Quebec French content
        self._analyze_quebec_content(texts[:100])  # Sample first 100
        
        return texts
    
    def _analyze_quebec_content(self, sample_texts: List[str]):
        """Analyze Quebec French characteristics in the data"""
        quebec_indicators = [
            # Common Quebec French words/expressions
            'pis', 'icitte', 'astheure', 'pantoute', 'toé', 'moé', 'tsé', 'ben',
            'à matin', 'à soir', 'su', 'chu', 'chus', 'criss', 'câlice', 'tabarnak',
            'drett', 'tight', 'cute', 'checker', 'watcher', 'chialer', 'niaiser',
            # Quebec specific contractions/pronunciations
            "j'l'ai", "m'a", "s'a", "t'a", "d'l'", "c'l'", 
            # Quebec institutions/places
            'québec', 'montréal', 'cegep', 'dep', 'université laval',
        ]
        
        found_indicators = set()
        total_text = ' '.join(sample_texts).lower()
        
        for indicator in quebec_indicators:
            if indicator in total_text:
                found_indicators.add(indicator)
        
        logger.info(f"Quebec French indicators found: {sorted(list(found_indicators))}")
        logger.info(f"Quebec content coverage: {len(found_indicators)}/{len(quebec_indicators)} indicators")
        
        # Check for potential issues
        if len(found_indicators) < 5:
            logger.warning("Low Quebec French content detected - verify data source")
    
    def create_dataset(self, texts: List[str]) -> Dataset:
        """Create HuggingFace dataset from texts"""
        return Dataset.from_dict({"text": texts})
    
    def tokenize_function(self, examples):
        """Simple tokenization function for Quebec French"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=self.config.max_length,
        )
    
    def group_texts(self, examples):
        """FIXED: Group texts into chunks of max_length with proper handling"""
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # FIXED: Only process if we have enough tokens for at least one full chunk
        if total_length < self.config.max_length:
            logger.warning(f"Batch too short ({total_length} tokens), skipping to avoid tensor size mismatch")
            # Return empty batch that will be filtered out
            return {k: [] for k in concatenated_examples.keys()}
        
        # Calculate how many complete chunks we can make
        num_chunks = total_length // self.config.max_length
        total_length = num_chunks * self.config.max_length  # Use only complete chunks
        
        # Split by chunks of max_length - FIXED: Only create complete chunks
        result = {
            k: [t[i : i + self.config.max_length] for i in range(0, total_length, self.config.max_length)]
            for k, t in concatenated_examples.items()
        }
        
        # Add labels (copy of input_ids for causal LM)
        result["labels"] = result["input_ids"].copy()
        
        # FIXED: Verify all chunks have correct length
        for key in result:
            for chunk in result[key]:
                if len(chunk) != self.config.max_length:
                    logger.error(f"Invalid chunk length: {len(chunk)}, expected: {self.config.max_length}")
                    raise ValueError(f"Chunk length mismatch: got {len(chunk)}, expected {self.config.max_length}")
        
        return result
    
    def prepare_dataset(self, train_texts: List[str]) -> Dataset:
        """Prepare dataset with improved processing for Quebec French"""
        # Create initial dataset
        train_dataset = self.create_dataset(train_texts)
        logger.info(f"Created dataset with {len(train_dataset)} text segments")
        
        # Step 1: Tokenize texts
        logger.info("Tokenizing Quebec French text...")
        tokenized_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            batch_size=self.config.tokenizer_batch_size,
            num_proc=self.config.preprocessing_num_workers,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing Quebec French text"
        )
        
        # Step 2: Group texts into chunks
        logger.info("Grouping texts into chunks...")
        lm_dataset = tokenized_dataset.map(
            self.group_texts,
            batched=True,
            batch_size=self.config.tokenizer_batch_size,
            num_proc=self.config.preprocessing_num_workers,
            desc="Grouping texts"
        )
        
        # FIXED: Filter out empty examples created by our new logic
        def filter_examples(example):
            return len(example['input_ids']) >= self.config.min_length
        
        lm_dataset = lm_dataset.filter(
            filter_examples,
            desc="Filtering short examples"
        )
        
        logger.info(f"Final dataset size: {len(lm_dataset)} chunks")
        return lm_dataset


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
        """Initialize model and tokenizer with Quebec French considerations"""
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
        
        # Check tokenizer's French capability
        self._check_french_tokenization(tokenizer)
        
        # Load model with quantization if specified
        quantization_config = self.get_quantization_config()
        
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
                "torch_dtype": torch.float16, 
            })
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        # Prepare model for training
        model.train()
        
        if self.config.use_4bit or self.config.use_8bit:
            model = prepare_model_for_kbit_training(
                model, 
                use_gradient_checkpointing=self.config.gradient_checkpointing
            )
        
        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        # Setup LoRA for Quebec French adaptation
        if self.config.use_lora:
            logger.info("Setting up LoRA for Quebec French adaptation...")
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
            
            # Ensure proper training setup
            model.train()
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.data.to(torch.float32)
            
            model.print_trainable_parameters()
            
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if trainable_params == 0:
                raise ValueError("No trainable parameters found!")
        
        return model, tokenizer
    
    def _check_french_tokenization(self, tokenizer):
        """Check how well the tokenizer handles Quebec French"""
        quebec_samples = [
            "À matin j'ai été au dépanneur pis j'ai acheté d'la bière.",
            "Criss que ça me frustre quand chu pogné dans l'traffic!",
            "Ben là, tsé veux dire, c'est ben correct ton affaire.",
            "J'vas aller checker ça à soir, c'est sûr."
        ]
        
        logger.info("Checking tokenizer performance on Quebec French:")
        for sample in quebec_samples:
            tokens = tokenizer.tokenize(sample)
            token_count = len(tokens)
            char_count = len(sample)
            ratio = char_count / token_count if token_count > 0 else 0
            
            logger.info(f"Sample: '{sample[:50]}...'")
            logger.info(f"  Tokens: {token_count}, Chars: {char_count}, Ratio: {ratio:.2f}")
            logger.info(f"  Tokenization: {tokens[:10]}...")  # Show first 10 tokens


class PerplexityCallback(TrainerCallback):
    """Enhanced callback with Quebec French specific metrics"""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if "loss" in logs:
                perplexity = np.exp(logs["loss"])
                logs["perplexity"] = perplexity
                
                # Quebec French adaptation progress indicator
                if perplexity < 3.0:
                    logs["quebec_adaptation"] = "excellent"
                elif perplexity < 5.0:
                    logs["quebec_adaptation"] = "good"
                elif perplexity < 10.0:
                    logs["quebec_adaptation"] = "moderate"
                else:
                    logs["quebec_adaptation"] = "poor"


class QuebecFrenchTrainer:
    """Enhanced trainer for Quebec French CPT"""
    
    def __init__(
        self,
        model_config: ModelConfig,
        data_config: DataConfig,
        training_config: TrainingConfig
    ):
        self.model_config = model_config
        self.data_config = data_config
        self.training_config = training_config
        
        set_seed(training_config.seed)
        
        # Setup model and tokenizer
        model_setup = ModelSetup(model_config)
        self.model, self.tokenizer = model_setup.setup_model_and_tokenizer()
        
        # Setup data processor
        self.data_processor = QuebecFrenchDataProcessor(self.tokenizer, data_config)
    
    def inspect_quebec_data(self, train_dataset: Dataset, num_samples: int = 3):
        """Quebec French specific data inspection"""
        logger.info("=" * 80)
        logger.info("QUEBEC FRENCH DATA INSPECTION")
        logger.info("=" * 80)
        
        logger.info(f"Dataset info:")
        logger.info(f"  - Training chunks: {len(train_dataset)}")
        logger.info(f"  - Features: {train_dataset.features}")
        
        # Analyze sample data for Quebec French characteristics
        quebec_terms_found = set()
        total_tokens = 0
        
        for i in range(min(num_samples, len(train_dataset))):
            sample = train_dataset[i]
            
            logger.info(f"\n--- QUEBEC SAMPLE {i+1} ---")
            logger.info(f"Input length: {len(sample['input_ids'])} tokens")
            
            # FIXED: Verify all samples have consistent length
            if len(sample['input_ids']) != self.data_config.max_length:
                logger.error(f"INCONSISTENT LENGTH DETECTED: {len(sample['input_ids'])}, expected: {self.data_config.max_length}")
                raise ValueError(f"Dataset contains inconsistent sequence lengths")
            
            # Decode to check content
            decoded_text = self.tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
            logger.info(f"Text preview: '{decoded_text[:200]}...'")
            
            # Look for Quebec French indicators
            text_lower = decoded_text.lower()
            quebec_indicators = ['pis', 'icitte', 'toé', 'moé', 'tsé', 'ben', 'chu', 'criss', 'à matin', 'à soir']
            
            for term in quebec_indicators:
                if term in text_lower:
                    quebec_terms_found.add(term)
            
            total_tokens += len(sample['input_ids'])
        
        # Summary statistics
        avg_length = total_tokens / min(num_samples, len(train_dataset))
        logger.info(f"\nQuebec French Analysis:")
        logger.info(f"  - Average chunk length: {avg_length:.1f} tokens")
        logger.info(f"  - Expected chunk length: {self.data_config.max_length} tokens")
        logger.info(f"  - Quebec terms found: {sorted(list(quebec_terms_found))}")
        logger.info(f"  - Quebec content richness: {len(quebec_terms_found)}/10 indicators")
        
        if len(quebec_terms_found) < 3:
            logger.warning("Limited Quebec French content detected in samples")
        
        logger.info("=" * 80)
    
    def train(self, inspect_data: bool = True, inspect_samples: int = 3):
        """Train Quebec French adapted model"""
        # Load Quebec French data
        train_texts = self.data_processor.load_text_data(self.data_config.train_file)
        logger.info(f"Loaded {len(train_texts)} Quebec French text segments")
        
        # Prepare training dataset
        train_dataset = self.data_processor.prepare_dataset(train_texts)
        logger.info(f"Prepared {len(train_dataset)} training chunks")
        
        # FIXED: Enhanced training arguments with better data handling
        training_args = TrainingArguments(
            output_dir=self.training_config.output_dir,
            num_train_epochs=self.training_config.num_epochs,
            per_device_train_batch_size=self.data_config.batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            warmup_ratio=self.training_config.warmup_ratio,
            weight_decay=self.training_config.weight_decay,
            fp16=self.training_config.fp16,
            logging_steps=self.training_config.logging_steps,
            save_steps=self.training_config.save_steps,
            eval_strategy="no",
            save_strategy="steps",
            save_total_limit=self.training_config.save_total_limit,
            push_to_hub=self.training_config.push_to_hub,
            hub_model_id=self.training_config.hub_model_id,
            report_to=["tensorboard"],
            logging_dir=f"{self.training_config.output_dir}/logs",
            seed=self.training_config.seed,
            dataloader_num_workers=2,  # FIXED: Reduced workers to avoid data loading issues
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            optim="adamw_torch",
            max_grad_norm=1.0,
            logging_first_step=True,
            # Quebec French specific optimizations
            lr_scheduler_type="cosine",  # Better for CPT
            save_safetensors=True,
            load_best_model_at_end=False,  # Not applicable for no eval
            # FIXED: Add data loading configurations
            dataloader_drop_last=True,  # Drop incomplete batches
            ignore_data_skip=True,      # Continue on data loading issues
            #fsdp=os.environ.get("HF_FSDP", None),
            #fsdp_transformer_layer_cls_to_wrap=os.environ.get("FSDP_TRANSFORMER_CLS_TO_WRAP", None)
        )
        
        # Data collator with explicit padding configuration
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=None, 
            return_tensors="pt"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            callbacks=[PerplexityCallback()]
        )
        
        # Inspect data if requested
        if inspect_data:
            self.inspect_quebec_data(train_dataset, inspect_samples)
        
        # Training verification
        logger.info("Starting Quebec French CPT...")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Training efficiency: {100 * trainable_params / total_params:.2f}%")
        
        # Start training
        train_result = trainer.train()
        
        # Save Quebec French adapted model
        logger.info("Saving Quebec French adapted model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.training_config.output_dir)
        
        # Save training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        return trainer, metrics
    
    def test_quebec_generation(self, prompts: List[str] = None, max_length: int = 100):
        """Test Quebec French generation capabilities"""
        if prompts is None:
            prompts = [
                "À matin j'ai",
                "Pis là, tu sais ben que", 
                "C'est ben correct, mais",
                "Chu allé au dépanneur pis",
                "Criss que ça",
                "Ben voyons donc",
                "J'vas aller checker"
            ]
        
        logger.info("Testing Quebec French generation:")
        logger.info("-" * 50)
        
        for prompt in prompts:
            generated = self.generate_sample(prompt, max_length)
            logger.info(f"Prompt: '{prompt}'")
            logger.info(f"Generated: '{generated}'")
            logger.info("-" * 30)
    
    def generate_sample(self, prompt: str, max_length: int = 100):
        """Generate Quebec French text sample"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.1  # Reduce repetition
            )   
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    
def save_config(config_dict: Dict, output_dir: str):
    """Save configuration to JSON file"""
    config_path = Path(output_dir) / "quebec_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"Quebec French configuration saved to {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Quebec French CPT for LLaMA")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B")  # FIXED: Use 1B model as in error log
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--use_4bit", action="store_true", default=False)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    # Data arguments
    parser.add_argument("--train_file", type=str, required=False, default="/home/o_vanesb/QuebecLLM-CPT/data/train.txt", 
                       help="Path to Quebec French training corpus")
    parser.add_argument("--max_length", type=int, default=1024)  # FIXED: Match error log
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--inspect_data", action="store_true", default=True)
    parser.add_argument("--inspect_samples", type=int, default=5)
        
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./llama_3b_6E")  # FIXED: Match error log
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    
    # Paralellization arguments
    parser.add_argument("--fsdp", type=str, default=False,
                        help='Enable FSDP sharding, e.g. "full_shard auto_wrap"')
    parser.add_argument("--fsdp_transformer_layer_cls_to_wrap", type=str, default="LlamaDecoderLayer",
                        help="Transformer layer class to wrap for FSDP auto_wrap")
    
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
        "training": asdict(training_config),
        "quebec_french_version": "1.1_FIXED"  # FIXED: Updated version
    }
    save_config(config_dict, args.output_dir)
    
    # Initialize Quebec French trainer
    trainer = QuebecFrenchTrainer(model_config, data_config, training_config)
    
    if args.fsdp:
        os.environ["HF_FSDP"] = args.fsdp
        os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = args.fsdp_transformer_layer_cls_to_wrap

    trainer_obj, train_metrics = trainer.train(
         inspect_data=args.inspect_data, 
         inspect_samples=args.inspect_samples
     )
    
    # Test Quebec French adaptation
    trainer.test_quebec_generation()
    
    logger.info("Quebec French CPT completed successfully!")
    logger.info(f"Final training loss: {train_metrics.get('train_loss', 'N/A')}")
    
    # Save final summary
    summary = {
        "completion_time": datetime.now().isoformat(),
        "final_loss": train_metrics.get('train_loss'),
        "total_steps": train_metrics.get('train_runtime'),
        "quebec_french_adapted": True,
        "fixed_version": "1.1"
    }
    
    summary_path = Path(args.output_dir) / "quebec_training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()