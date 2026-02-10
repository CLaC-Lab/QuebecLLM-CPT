#!/usr/bin/env python
"""
Continual Pretraining Pipeline
"""

import os
import json
import torch
import random
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
    max_length: int = 1024
    stride: int = 512
    batch_size: int = 8
    preprocessing_num_workers: int = 1
    tokenizer_batch_size: int = 1000
    min_length: int = 50
    replay_file: str = "replay.txt"
    replay_percent: float = 0.1

@dataclass
class TrainingConfig:
    """Configuration for training"""
    output_dir: str = "./cpt_model"
    num_epochs: int = 3
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.00
    gradient_accumulation_steps: int = 8
    fp16: bool = True
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 50
    save_total_limit: int = 3
    seed: int = 42
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    fsdp: Optional[str] = None
    fsdp_transformer_layer_cls_to_wrap: Optional[str] = None


class DataProcessor:
    """Data processor for continual pretraining"""

    def __init__(self, tokenizer, config: DataConfig):
        self.tokenizer = tokenizer

        self.config = config

    def load_text_data(self, file_path: str) -> List[str]:
        """Load and lightly clean text data"""
        logger.info(f"Loading data from {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        texts = []
        for line in tqdm(lines, desc=f"Loading text"):
            line = line.strip()

            if not line:
                continue

            # Light normalization
            line = line.replace('\ufeff', '')  # Remove BOM
            line = ' '.join(line.split())      # Normalize whitespace
            line = unicodedata.normalize("NFC", line)

            if len(line) >= 10:
                texts.append(line)

        logger.info(f"Loaded {len(texts)} text segments")
        return texts

    def create_dataset(self, texts: List[str]) -> Dataset:
        """Create HuggingFace dataset from texts"""
        return Dataset.from_dict({"text": texts})

    def tokenize_function(self, examples):
        """
        Tokenize without truncation, and append EOS to each document so that
        cross-document boundaries are not learned as intra-document text.
        """
        tok = self.tokenizer(
            examples["text"],
            #add_special_tokens=False,
            padding=False,
            truncation=False
        )
        #eos_id = self.tokenizer.eos_token_id
        # append EOS to every sample
        #tok["input_ids"]      = [ids + [eos_id] for ids in tok["input_ids"]]
        #tok["attention_mask"] = [am  + [1]      for am  in tok["attention_mask"]]
        return tok


    def group_texts(self, examples):
        """
        Concatenate tokens within the batch and chunk into fixed-length blocks.
        If stride > 0 and < max_length, use a sliding window (overlap).
        """
        block = self.config.max_length
        stride = self.config.stride

        # sanity checks
        if stride is not None and stride >= block:
            raise ValueError(f"stride ({stride}) must be < max_length ({block})")

        # concatenate all sequences in the batch
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_len = len(concatenated["input_ids"])

        if total_len < block:
            # not enough to form a single block from this batch
            return {k: [] for k in concatenated.keys()}

        # choose stepping: non-overlap or sliding window
        if stride and stride > 0:
            step = block - stride
            starts = range(0, total_len - block + 1, step)
            result = {k: [concatenated[k][i:i+block] for i in starts]
                    for k in concatenated.keys()}
        else:
            # non-overlapping blocks
            cut = (total_len // block) * block
            result = {k: [concatenated[k][i:i+block] for i in range(0, cut, block)]
                    for k in concatenated.keys()}

        # causal LM labels = inputs
        result["labels"] = [ids[:] for ids in result["input_ids"]]

        return result


    def prepare_dataset(self, train_texts: List[str]) -> Dataset:
        """Prepare dataset with improved processing"""
        train_dataset = self.create_dataset(train_texts)
        logger.info(f"Created dataset with {len(train_dataset)} text segments")

        # Step 1: Tokenize texts
        logger.info("Tokenizing text...")
        tokenized_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            batch_size=self.config.tokenizer_batch_size,
            num_proc=self.config.preprocessing_num_workers,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing text"
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

        def filter_examples(example):
            return len(example['input_ids']) >= self.config.min_length

        lm_dataset = lm_dataset.filter(
            filter_examples,
            desc="Filtering short examples"
        )

        logger.info(f"Final dataset size: {len(lm_dataset)} chunks")
        return lm_dataset

class ChatProcessor:
    """Data processor for instruction tuning. Input is chat template"""
    def __init__(self, tokenizer, config: DataConfig):
        self.tokenizer = tokenizer

        if tokenizer.chat_template is None:
            print("No default chat template. Setting one")
            with open("./default_chat_template.txt", "r") as f_open:
                chat_template = f_open.read()
            tokenizer.chat_template = chat_template

        self.config = config
        self.tokenizer = tokenizer

    def load_text_data(self, file_path: str) -> List[str]:
        """Load and lightly clean text data"""
        logger.info(f"Loading data from {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        texts = []
        replay_texts = []
        for line in tqdm(lines, desc=f"Loading text"):
            if not line:
                continue
            chat_obj = json.loads(line)
            chat_input = self.tokenizer.apply_chat_template(chat_obj, tokenize=False, add_generation_prompt=True)
            texts.append(chat_input)
        logger.info(f"Loaded {len(lines)} instruction examples")

        logger.info(f"Replay path {self.config.replay_file}")
        replay_file = self.config.replay_file
        if replay_file is not None:
            logger.info(f"Loading replay data from {replay_file}")

            with open(replay_file, 'r', encoding='utf-8') as f:
                replay_lines = f.readlines()

            for line in replay_lines:
                line = line.strip()

                if not line:
                    continue
                line = line.replace('\ufeff', '')  # Remove BOM
                line = ' '.join(line.split())      # Normalize whitespace
                line = unicodedata.normalize("NFC", line)

                if len(line) >= 10:
                    replay_texts.append(line)

            logger.info(f"Loaded {len(replay_texts)} texts")
            replay_texts = replay_texts[:round(len(replay_texts)*self.config.replay_percent)]
            logger.info(f"Using {len(replay_texts)}")

            logger.info(f"Merging with instructions")
            print(type(texts))
            texts = texts + replay_texts
            random.shuffle(texts)

        logger.info(f"Training with {len(texts)} text segments")
        return texts

    def group_texts(self, examples):
        """
        Concatenate tokens within the batch and chunk into fixed-length blocks.
        If stride > 0 and < max_length, use a sliding window (overlap).
        """
        block = self.config.max_length
        stride = self.config.stride

        # sanity checks
        if stride is not None and stride >= block:
            raise ValueError(f"stride ({stride}) must be < max_length ({block})")

        # concatenate all sequences in the batch
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_len = len(concatenated["input_ids"])

        if total_len < block:
            # not enough to form a single block from this batch
            return {k: [] for k in concatenated.keys()}

        # choose stepping: non-overlap or sliding window
        if stride and stride > 0:
            step = block - stride
            starts = range(0, total_len - block + 1, step)
            result = {k: [concatenated[k][i:i+block] for i in starts]
                    for k in concatenated.keys()}
        else:
            # non-overlapping blocks
            cut = (total_len // block) * block
            result = {k: [concatenated[k][i:i+block] for i in range(0, cut, block)]
                    for k in concatenated.keys()}

        # causal LM labels = inputs
        result["labels"] = [ids[:] for ids in result["input_ids"]]

        return result

    def tokenize_function(self, examples):
        """
        Tokenize without truncation, and append EOS to each document so that
        cross-document boundaries are not learned as intra-document text.
        """
        tok = self.tokenizer(
            examples["text"],
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )
        eos_id = self.tokenizer.eos_token_id
        # append EOS to every sample
        tok["input_ids"]      = [ids + [eos_id] for ids in tok["input_ids"]]
        tok["attention_mask"] = [am  + [1]      for am  in tok["attention_mask"]]
        return tok

    def create_dataset(self, texts: List[str]) -> Dataset:
        """Create HuggingFace dataset from texts"""
        return Dataset.from_dict({"text": texts})

    def prepare_dataset(self, train_texts: List[str]) -> Dataset:
        """Prepare dataset with improved processing"""
        train_dataset = self.create_dataset(train_texts)
        logger.info(f"Created dataset with {len(train_dataset)} text segments")

        # Step 1: Tokenize texts
        logger.info("Tokenizing text...")
        tokenized_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            batch_size=self.config.tokenizer_batch_size,
            num_proc=self.config.preprocessing_num_workers,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing text"
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

    def setup_model_and_tokenizer(self, use_fsdp=False):
        """Initialize model and tokenizer"""
        logger.info(f"Loading model: {self.config.model_name}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True, 
            add_eos_token=True, 
            use_fast=True
        )

        # Add pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

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
            # FIXED: For FSDP, don't specify device_map - let FSDP handle it
            model_kwargs.update({
                "torch_dtype": torch.float16,
            })
            if not use_fsdp:
                # Only use device_map if not using FSDP
                model_kwargs["device_map"] = "auto"

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

        # Setup LoRA (must be done before FSDP wrapping)
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

            # Ensure proper training setup
            model.train()
            if not use_fsdp:
                # Only cast to float32 if not using FSDP
                for param in model.parameters():
                    if param.requires_grad:
                        param.data = param.data.to(torch.float32)

            model.print_trainable_parameters()

            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if trainable_params == 0:
                raise ValueError("No trainable parameters found!")

        return model, tokenizer


class PerplexityCallback(TrainerCallback):
    """Callback for tracking perplexity"""
    def __init__(self):
        super().__init__()
        self.log_arr = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if "loss" in logs:
                perplexity = np.exp(logs["loss"])
                logs["perplexity"] = float(perplexity)
                self.log_arr.append(logs)

    def on_train_end(self, args, state, control, **kwargs):
        print(self.log_arr)
        data = {
            "epoch": [item["epoch"] for item in self.log_arr],
            "loss": [item["loss"] for item in self.log_arr],
            "perplexity": [item["perplexity"] for item in self.log_arr],
            "learning_rate": [item["learning_rate"] for item in self.log_arr],
            "grad_norm": [item["grad_norm"] for item in self.log_arr]
        }
        with open(f"{args.output_dir}/logs/train_log.json", "w") as f_open:
            f_open.write(json.dumps(data))


class ContinualPretrainingTrainer:
    """Trainer for continual pretraining"""

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

        # Check if FSDP is enabled
        use_fsdp = training_config.fsdp is not None

        # Setup model and tokenizer
        model_setup = ModelSetup(model_config)
        self.model, self.tokenizer = model_setup.setup_model_and_tokenizer(use_fsdp=use_fsdp)

        # Setup data processor
        self.data_processor = DataProcessor(self.tokenizer, data_config)

    def inspect_data(self, train_dataset: Dataset, num_samples: int = 3):
        """Inspect training data"""
        logger.info("=" * 80)
        logger.info("DATA INSPECTION")
        logger.info("=" * 80)

        logger.info(f"Dataset info:")
        logger.info(f"  - Training chunks: {len(train_dataset)}")
        logger.info(f"  - Features: {train_dataset.features}")

        if len(train_dataset) == 0:
            logger.error("=" * 80)
            logger.error("EMPTY DATASET ERROR")
            logger.error("=" * 80)
            logger.error("No training chunks were created. This usually means:")
            logger.error("  1. Your text segments are too short")
            logger.error("  2. All segments are shorter than max_length")
            logger.error(f"  3. Current max_length: {self.data_config.max_length}")
            logger.error("\nSuggestions:")
            logger.error("  - Add more/longer text to your training file")
            logger.error("  - Reduce max_length parameter")
            logger.error("  - Check that train_file path is correct")
            logger.error("=" * 80)
            raise ValueError("Empty dataset: no training chunks created")

        total_tokens = 0

        for i in range(min(num_samples, len(train_dataset))):
            sample = train_dataset[i]

            logger.info(f"\n--- SAMPLE {i+1} ---")
            logger.info(f"Input length: {len(sample['input_ids'])} tokens")

            # FIXED: Verify all samples have consistent length
            if len(sample['input_ids']) != self.data_config.max_length:
                logger.error(f"INCONSISTENT LENGTH DETECTED: {len(sample['input_ids'])}, expected: {self.data_config.max_length}")
                raise ValueError(f"Dataset contains inconsistent sequence lengths")

            # Decode to check content
            decoded_text = self.tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
            logger.info(f"Text preview: '{decoded_text[:200]}...'")

            total_tokens += len(sample['input_ids'])

        # Summary statistics
        avg_length = total_tokens / min(num_samples, len(train_dataset))
        logger.info(f"\nSummary:")
        logger.info(f"  - Average chunk length: {avg_length:.1f} tokens")
        logger.info(f"  - Expected chunk length: {self.data_config.max_length} tokens")
        logger.info("=" * 80)

    def train(self, inspect_data: bool = True, inspect_samples: int = 3):
        """Train model"""
        # Load data
        train_texts = self.data_processor.load_text_data(self.data_config.train_file)
        logger.info(f"Loaded {len(train_texts)} text segments")

        # Prepare training dataset
        train_dataset = self.data_processor.prepare_dataset(train_texts)
        logger.info(f"Prepared {len(train_dataset)} training chunks")

        # FIXED: Enhanced training arguments with better data handling and FSDP support
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
            dataloader_num_workers=2,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            optim="adamw_torch",
            max_grad_norm=1.0,
            logging_first_step=True,
            lr_scheduler_type="cosine",
            save_safetensors=True,
            load_best_model_at_end=False,
            # FIXED: Add data loading configurations
            dataloader_drop_last=True,
            ignore_data_skip=True,
            # FIXED: FSDP config from environment variable (new method)
            # Don't set fsdp or fsdp_transformer_layer_cls_to_wrap here
            # They should be set via FSDP_CONFIG environment variable
        )

        # Data collator
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
            self.inspect_data(train_dataset, inspect_samples)

        # Training verification
        logger.info("Starting continual pretraining...")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Training efficiency: {100 * trainable_params / total_params:.2f}%")

        # Start training
        train_result = trainer.train()

        # Save model
        logger.info("Saving model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.training_config.output_dir)

        # Save training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        return trainer, metrics

    def generate_sample(self, prompt: str, max_length: int = 100):
        """Generate text sample"""
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
                repetition_penalty=1.1
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class InstructionTuningTrainer:
    """Trainer for instruction tuning"""
    def __init__(
        self,
        model_config: ModelConfig,
        data_config: DataConfig,
        training_config: TrainingConfig
    ):
        self.model_config = model_config
        self.data_config = data_config
        self.training_config = training_config
        print("Instruction Tuning")
        set_seed(training_config.seed)

        # Check if FSDP is enabled
        use_fsdp = training_config.fsdp is not None

        # Setup model and tokenizer
        model_setup = ModelSetup(model_config)
        self.model, self.tokenizer = model_setup.setup_model_and_tokenizer(use_fsdp=use_fsdp)

        # Setup data processor
        self.data_processor = ChatProcessor(self.tokenizer, data_config)

    def inspect_data(self, train_dataset: Dataset, num_samples: int = 3):
        """Inspect training data"""
        logger.info("=" * 80)
        logger.info("DATA INSPECTION")
        logger.info("=" * 80)

        logger.info(f"Dataset info:")
        logger.info(f"  - Training chunks: {len(train_dataset)}")
        logger.info(f"  - Features: {train_dataset.features}")

        if len(train_dataset) == 0:
            logger.error("EMPTY DATASET ERROR")
            raise ValueError("Empty dataset: no trainingcs chunks created")

        total_tokens = 0

        for i in range(min(num_samples, len(train_dataset))):
            sample = train_dataset[i]

            logger.info(f"\n--- SAMPLE {i+1} ---")
            logger.info(f"Input length: {len(sample['input_ids'])} tokens")

            # FIXED: Verify all samples have consistent length
            if len(sample['input_ids']) != self.data_config.max_length:
                logger.error(f"INCONSISTENT LENGTH DETECTED: {len(sample['input_ids'])}, expected: {self.data_config.max_length}")
                raise ValueError(f"Dataset contains inconsistent sequence lengths")

            # Decode to check content
            decoded_text = self.tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
            logger.info(f"Text preview: '{decoded_text[:200]}...'")

            total_tokens += len(sample['input_ids'])

        # Summary statistics
        avg_length = total_tokens / min(num_samples, len(train_dataset))
        logger.info(f"\nSummary:")
        logger.info(f"  - Average chunk length: {avg_length:.1f} tokens")
        logger.info(f"  - Expected chunk length: {self.data_config.max_length} tokens")
        logger.info("=" * 80)

    def train(self, inspect_data: bool = True, inspect_samples: int = 3):
        """Train model"""
        # Load data
        train_texts = self.data_processor.load_text_data(self.data_config.train_file)
        logger.info(f"Loaded {len(train_texts)} instruction segments")

        # Prepare training dataset
        train_dataset = self.data_processor.prepare_dataset(train_texts)
        logger.info(f"Prepared {len(train_dataset)} training chunks")

        # FIXED: Enhanced training arguments with better data handling and FSDP support
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
            dataloader_num_workers=2,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            optim="adamw_torch",
            max_grad_norm=1.0,
            logging_first_step=True,
            lr_scheduler_type="cosine",
            #save_safetensors=True,
            load_best_model_at_end=False,
            # FIXED: Add data loading configurations
            dataloader_drop_last=True,
            ignore_data_skip=True,
            # FIXED: FSDP config from environment variable (new method)
            # Don't set fsdp or fsdp_transformer_layer_cls_to_wrap here
            # They should be set via FSDP_CONFIG environment variable
        )

        # Data collator
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
            self.inspect_data(train_dataset, inspect_samples)

        # Training verification
        logger.info("Starting instruction tuning...")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Training efficiency: {100 * trainable_params / total_params:.2f}%")

        # Start training
        train_result = trainer.train()

        # Save model
        logger.info("Saving model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.training_config.output_dir)

        # Save training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        return trainer, metrics

    def generate_sample(self, prompt: str, max_length: int = 100):
        """Generate text sample"""
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
                repetition_penalty=1.1
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
    parser = argparse.ArgumentParser(description="Continual Pretraining for LLaMA")

    # Model arguments
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--use_4bit", action="store_true", default=False)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)

    # Data arguments
    parser.add_argument("--train_file", type=str, required=True,
                       help="Path to training corpus")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--inspect_data", action="store_true", default=True)
    parser.add_argument("--inspect_samples", type=int, default=5)
    parser.add_argument("--preprocessing_num_workers", type=int, default=1)
    

    # Training arguments
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--stride", type=int, default=128,
                    help="Sliding-window overlap in tokens; 0 means no overlap")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replay_file", type=str, required=False, help="Path to replay data", default=None)
    parser.add_argument("--replay_percent", type=float, required=False, default=0.2)

    # Parallelization arguments
    parser.add_argument("--fsdp", type=str, default=None,
                        help='Enable FSDP sharding, e.g. "full_shard auto_wrap"')
    parser.add_argument("--fsdp_transformer_layer_cls_to_wrap", type=str, default="LlamaDecoderLayer",
                        help="Transformer layer class to wrap for FSDP auto_wrap")

    # CPT arguments
    parser.add_argument("--it", action='store_true')

    args = parser.parse_args()

    name = args.model_name.split("/")[-1] if not args.model_name.endswith("/") else args.model_name[:-1].split("/")[-1]
    replay = True if args.replay_file is not None else False
    output_dir = f"./models/{name}-{args.num_epochs}E"
    if replay:
        output_dir = output_dir + f"-replay-{args.replay_percent}"

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
        batch_size=args.batch_size,
        stride=args.stride,
        preprocessing_num_workers=args.preprocessing_num_workers,
        replay_file=args.replay_file,
        replay_percent=args.replay_percent
    )

    training_config = TrainingConfig(
        output_dir=output_dir,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        fsdp=args.fsdp,
        fsdp_transformer_layer_cls_to_wrap=args.fsdp_transformer_layer_cls_to_wrap
    )

    # Save configuration
    config_dict = {
        "model": asdict(model_config),
        "data": asdict(data_config),
        "training": asdict(training_config),
        "version": "1.1_FIXED"
    }
    save_config(config_dict, output_dir)

    # Initialize trainer
    trainer = ContinualPretrainingTrainer(model_config, data_config, training_config) if args.it is False else InstructionTuningTrainer(model_config, data_config, training_config)

    trainer_obj, train_metrics = trainer.train(
         inspect_data=args.inspect_data,
         inspect_samples=args.inspect_samples
     )

    logger.info("Continual pretraining completed successfully!")
    logger.info(f"Final training loss: {train_metrics.get('train_loss', 'N/A')}")

    # Save final summary
    summary = {
        "completion_time": datetime.now().isoformat(),
        "final_loss": train_metrics.get('train_loss'),
        "total_steps": train_metrics.get('train_runtime'),
        "fixed_version": "1.1"
    }

    summary_path = Path(output_dir) / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
