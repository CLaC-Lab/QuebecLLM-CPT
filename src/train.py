#!/usr/bin/env python

"""
Modified training script for CPT using pre-tokenized data.
Adapted to work with JSONL files containing {"text": ..., "input_ids": ...} format.
"""
import os
import json
from dataclasses import dataclass, field
from typing import Optional, Sequence, Dict
import random
import copy
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
import transformers
import datasets
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    Trainer,
    set_seed,
    AutoModelForCausalLM,
    AutoTokenizer,
)

# Import custom trainers if available
try:
    from train_utils import (
        NoShuffleSeq2SeqTrainer,
        WSDTrainer,
        WSDNoShuffleTrainer,
    )
except ImportError:
    print("Warning: Custom trainer classes not found. Using default Trainer.")
    NoShuffleSeq2SeqTrainer = Trainer
    WSDTrainer = Trainer
    WSDNoShuffleTrainer = Trainer

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")
    flash_attention: Optional[bool] = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    no_shuffle: bool = field(
        default=False, metadata={"help": "Whether to shuffle the training data."}
    )
    preprocess_num_workers: int = field(
        default=32,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    use_pretokenized: bool = field(
        default=True, metadata={"help": "Whether the data is already tokenized."}
    )
    group_texts: bool = field(
        default=True, metadata={"help": "Whether to group texts to max_length."}
    )
    single_file: bool = field(
        default=False,
        metadata={"help": "Whether to load a single file or multiple files."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_wsd: bool = field(default=False)


class SupervisedDataset(Dataset):
    def __init__(self, data):
        super(SupervisedDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.data[idx]


@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer
    model_max_length: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Pad sequences to the same length
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]

        # Find max length in batch
        max_len = min(max(len(ids) for ids in input_ids), self.model_max_length)

        # Pad sequences
        padded_input_ids = []
        padded_labels = []
        attention_mask = []

        for ids, labs in zip(input_ids, labels):
            # Truncate if necessary
            if len(ids) > max_len:
                ids = ids[:max_len]
                labs = labs[:max_len]

            # Calculate padding needed
            padding_length = max_len - len(ids)

            # Pad input_ids
            padded_ids = ids + [self.tokenizer.pad_token_id] * padding_length
            padded_input_ids.append(padded_ids)

            # Pad labels (use -100 for padding positions)
            padded_labs = labs + [-100] * padding_length
            padded_labels.append(padded_labs)

            # Create attention mask
            mask = [1] * len(ids) + [0] * padding_length
            attention_mask.append(mask)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def load_pretokenized_data(data_path, model_max_length, single_file=False):
    """Load pre-tokenized JSONL files."""
    all_data = []

    if single_file:
        files = [data_path]
    else:
        # Get all JSONL files in directory
        files = []
        i = 0
        for root, _, filenames in os.walk(data_path):
            for filename in filenames:
                if filename.endswith('.jsonl') and i < 100:
                    files.append(os.path.join(root, filename))
                    i += 1

    print(f"Found {len(files)} files to load")

    for file_path in files:
        print(f"Loading {file_path}")
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                if 'input_ids' in data:
                    # Ensure we have both input_ids and labels
                    item = {
                        'input_ids': data['input_ids'],
                        'labels': copy.deepcopy(data['input_ids'])
                    }
                    all_data.append(item)

    return all_data


def group_and_chunk_texts(data, model_max_length, tokenizer):
    """Group texts into chunks of model_max_length."""
    grouped_data = []
    current_chunk = []

    for item in data:
        current_chunk.extend(item['input_ids'])

        # Add EOS token if not present
        if current_chunk and current_chunk[-1] != tokenizer.eos_token_id:
            current_chunk.append(tokenizer.eos_token_id)

        # Create chunks of model_max_length
        while len(current_chunk) >= model_max_length:
            chunk = current_chunk[:model_max_length]
            grouped_data.append({
                'input_ids': chunk,
                'labels': copy.deepcopy(chunk)
            })
            current_chunk = current_chunk[model_max_length:]

    # Don't forget the last chunk if it's substantial
    if len(current_chunk) > model_max_length // 4:  # Only keep if > 25% of max length
        grouped_data.append({
            'input_ids': current_chunk,
            'labels': copy.deepcopy(current_chunk)
        })

    return grouped_data


def make_supervised_data_module(tokenizer, data_args, training_args):
    """Create data module for pre-tokenized data."""

    # Load pre-tokenized data
    print("Loading pre-tokenized data...")
    data = load_pretokenized_data(
        data_args.data_path,
        training_args.model_max_length,
        data_args.single_file
    )
    print(f"Loaded {len(data)} sequences")

    # Group texts if requested
    if data_args.group_texts:
        print(f"Grouping texts into chunks of {training_args.model_max_length}")
        data = group_and_chunk_texts(data, training_args.model_max_length, tokenizer)
        print(f"After grouping: {len(data)} sequences")

    # Sample some examples
    if LOCAL_RANK == 0:
        for idx in random.sample(range(min(len(data), 5)), min(len(data), 3)):
            print(f"\nSample {idx}:")
            print(f"Length: {len(data[idx]['input_ids'])}")
            print(f"Text: {tokenizer.decode(data[idx]['input_ids'][:100])}...")
            print("-" * 50)

    # Create dataset
    train_dataset = SupervisedDataset(data)

    # Create data collator
    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer,
        model_max_length=training_args.model_max_length
    )

    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator
    )

def get_peft_tokenizer(model_args, training_args):
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    lora_config = LoraConfig(
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, 
        bias = "none", 

    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        trust_remote_code=True,
    )
    model = get_peft_model(model, lora_config)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_model_tokenizer(model_args, training_args):
    """Load model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        attn_implementation="flash_attention_2" if model_args.flash_attention else None,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {
            "use_reentrant": False
        }

    # Load model and tokenizer
    model, tokenizer = get_peft_tokenizer(model_args, training_args) #get_model_tokenizer()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Create data module
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
    )

    # Model settings
    model.is_parallelizable = True
    model.model_parallel = True

    # Select trainer class
    trainer_class = Trainer
    if data_args.no_shuffle:
        if training_args.use_wsd:
            trainer_class = WSDNoShuffleTrainer
        else:
            trainer_class = NoShuffleSeq2SeqTrainer
    elif training_args.use_wsd:
        trainer_class = WSDTrainer

    # Create trainer
    trainer = trainer_class(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )

    # Disable cache for training
    model.config.use_cache = False

    # Start training
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # Save model and state
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()