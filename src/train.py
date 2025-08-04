# # encoding: utf-8
# # The code is adapted from tatsu-lab/stanford_alpaca. The original code is licensed under the Apache 2.0 License.
# #    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
# #
# #    Licensed under the Apache License, Version 2.0 (the "License");
# #    you may not use this file except in compliance with the License.
# #    You may obtain a copy of the License at
# #
# #        http://www.apache.org/licenses/LICENSE-2.0
# #
# #    Unless required by applicable law or agreed to in writing, software
# #    distributed under the License is distributed on an "AS IS" BASIS,
# #    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# #    See the License for the specific language governing permissions and
# #    limitations under the License.
# """
# Training script for CPT using Hugging Face Transformers and Datasets libraries.
#
# This script supports loading and preprocessing text datasets or loading preprocessed datasets from disk. It allows for customization of training behavior through various command-line arguments.
#
# Key Parameters:
# - `use_wsd` (bool): If set to True, the script uses the WSD optimizer for training, which may impact the training dynamics. Default is False.
# - `no_shuffle` (bool): If set to True, the training data will not be shuffled during training. This can be useful for certain training strategies where data order is important. Default is False.
# - `load_text_dataset` (bool): If set to True, the script will load raw text data and perform preprocessing (tokenization and grouping) before training. After preprocessing, the script will save the processed dataset to disk and exit. If False, it assumes that a preprocessed dataset is provided and loads it directly from disk. Default is False.
# - `single_dataset` (bool): If set to True, the script will load a single dataset from the specified `data_path`. If False, it will load and concatenate multiple datasets found in the `data_path` directory. Default is False.
#
# Usage:
# Run the script with the desired arguments to start training the model. Command-line arguments allow you to specify the model, data paths, and various training configurations.
#
# For more detailed configurations and options, refer to the argument definitions in the script.
# """
# import os
#
# # Set the environment variable to use the cache
# SAVE_PATH = "~/.cache"
# os.environ["TMPDIR"] = os.path.join(SAVE_PATH, "tmp")
# os.environ["HF_DATASETS_CACHE"] = os.path.join(SAVE_PATH, "hf_datasets_cache")
# os.environ["HF_HOME"] = os.path.join(SAVE_PATH, "hf_home")
# os.makedirs(os.environ["TMPDIR"], exist_ok=True)
# os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
# os.makedirs(os.environ["HF_HOME"], exist_ok=True)
# from dataclasses import dataclass, field
# from typing import Optional, Sequence, Dict
# import random
# import copy
# import torch
# import torch.distributed as dist
# from torch.utils.data import Dataset
# import transformers
# import datasets
# from transformers import (
#     Trainer,
#     set_seed,
#     AutoModelForCausalLM,
#     AutoTokenizer,
# )
#
# from train_utils import (
#     NoShuffleSeq2SeqTrainer,
#     WSDTrainer,
#     WSDNoShuffleTrainer,
# )
#
# LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
#
#
# @dataclass
# class ModelArguments:
#     model_name_or_path: Optional[str] = field(default="")
#     flash_attention: Optional[bool] = field(default=False)
#
#
# @dataclass
# class DataArguments:
#     data_path: str = field(
#         default=None, metadata={"help": "Path to the training data."}
#     )
#     no_shuffle: bool = field(
#         default=False, metadata={"help": "Whether to shuffle the training data."}
#     )
#     preprocess_num_workers: int = field(
#         default=32,
#         metadata={"help": "The number of processes to use for the preprocessing."},
#     )
#     load_text_dataset: bool = field(
#         default=False, metadata={"help": "Whether the dataset is text or input ids."}
#     )
#     min_text_length: int = field(
#         default=20, metadata={"help": "Minimum text length to include in the dataset."}
#     )
#     single_dataset: bool = field(
#         default=False,
#         metadata={"help": "Whether to load a single dataset."},
#     )
#
#
# @dataclass
# class TrainingArguments(transformers.TrainingArguments):
#     cache_dir: Optional[str] = field(default=None)
#     model_max_length: int = field(
#         default=2048,
#         metadata={
#             "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
#         },
#     )
#     use_wsd: bool = field(default=False)
#
#
# class SupervisedDataset(Dataset):
#     def __init__(self, train_dataset):
#         super(SupervisedDataset, self).__init__()
#         self.sources = train_dataset
#
#     def __len__(self):
#         return len(self.sources)
#
#     def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
#         ipt_ids = self.sources[idx]["input_ids"]
#         return dict(input_ids=ipt_ids, labels=copy.deepcopy(ipt_ids))
#
#
# @dataclass
# class DataCollatorForSupervisedDataset(object):
#     data_args: DataArguments
#     tokenizer: transformers.PreTrainedTokenizer
#
#     def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
#         return dict(
#             input_ids=torch.tensor([d["input_ids"] for d in instances]),
#             labels=torch.tensor([d["labels"] for d in instances]),
#         )
#
#
# def make_supervised_data_module(tokenizer, data_args, training_args, model_args):
#     def process_example(example):
#         tokenized_list = tokenizer(example["text"], add_special_tokens=False)
#         token_ids = [
#             tokenized + [tokenizer.eos_token_id]
#             for tokenized in tokenized_list["input_ids"]
#         ]
#         return {"input_ids": token_ids}
#
#     def group_texts(examples):
#         processed_data = []
#         tmp = []
#         for example in examples["input_ids"]:
#             if len(example) < data_args.min_text_length:
#                 continue
#             tmp.extend(example)
#             while len(tmp) >= training_args.model_max_length:
#                 processed_data.append(tmp[: training_args.model_max_length])
#                 tmp = tmp[training_args.model_max_length :]
#         return {"input_ids": processed_data}
#
#     # Load and preprocess the text dataset if needed
#     if data_args.load_text_dataset:
#         world_size = int(os.environ.get("WORLD_SIZE", "1"))
#         rank = int(os.environ.get("RANK", "0"))
#         if rank == 0 and LOCAL_RANK == 0:
#             total_paths = [
#                 os.path.join(data_args.data_path, file_name)
#                 for file_name in os.listdir(data_args.data_path)
#                 if file_name.endswith(".jsonl") or file_name.endswith(".json")
#             ]
#             # Process per 60G
#             file_sizes = [os.path.getsize(path) for path in total_paths]
#             file_size_limit = 60 * 1024**3
#             agg_paths = []
#             temp_size = 0
#             temp_paths = []
#             for i, size in enumerate(file_sizes):
#                 temp_size += size
#                 temp_paths.append(total_paths[i])
#                 if temp_size >= file_size_limit:
#                     agg_paths.append(temp_paths)
#                     temp_size = 0
#                     temp_paths = []
#             if temp_paths:
#                 agg_paths.append(temp_paths)
#             for i, paths in enumerate(agg_paths):
#                 data_save_dir = os.path.join(
#                     training_args.output_dir,
#                     data_args.data_path.split("/")[-1] + f"_{i}",
#                 )
#                 print(
#                     f"Start processing data list: {paths} | {len(paths)} files | save to {data_save_dir}"
#                 )
#                 raw_train_dataset = datasets.Dataset.from_json(
#                     path_or_paths=paths,
#                     num_proc=data_args.preprocess_num_workers,
#                 )
#                 print(raw_train_dataset)
#                 print(f"Raw dataset size: {len(raw_train_dataset)}")
#                 print("Tokenizing dataset")
#                 train_dataset = raw_train_dataset.map(
#                     process_example,
#                     batched=True,
#                     num_proc=data_args.preprocess_num_workers,
#                     remove_columns=raw_train_dataset.column_names,
#                     desc="Running tokenizer on train dataset",
#                 )
#                 print("Tokenizing dataset finished")
#                 print(
#                     f"Grouping texts with sequence length {training_args.model_max_length}. Filter texts shorter than {data_args.min_text_length}"
#                 )
#                 train_dataset = train_dataset.map(
#                     group_texts,
#                     batched=True,
#                     num_proc=data_args.preprocess_num_workers,
#                     desc=f"Grouping texts with sequence length {training_args.model_max_length}",
#                 )
#                 if not os.path.exists(training_args.output_dir):
#                     os.mkdir(training_args.output_dir)
#                 print(train_dataset)
#                 train_dataset.save_to_disk(data_save_dir)
#         if world_size > 1:
#             dist.barrier()
#         print(
#             f"Preprocess finished. Please set `load_text_dataset` to False and reload from {data_save_dir} with `single_dataset` set to True | Exit"
#         )
#         exit(0)
#
#     if data_args.single_dataset:
#         train_dataset = datasets.load_from_disk(data_args.data_path)
#         print(train_dataset)
#     else:
#         train_dataset = []
#         for data_name in os.listdir(data_args.data_path):
#             train_dataset.append(
#                 datasets.load_from_disk(os.path.join(data_args.data_path, data_name))
#             )
#             print(f"Dataset {data_name} loaded")
#         print(len(train_dataset))
#         train_dataset = datasets.concatenate_datasets(train_dataset)
#         print(train_dataset)
#
#     print(f"train dataset size: {len(train_dataset)}")
#     if LOCAL_RANK == 0:
#         for index in [0] + list(random.sample(range(len(train_dataset)), 1)):
#             print(f"Sample {index} of the training set: {train_dataset[index]}.")
#             print("---------" * 9)
#             if isinstance(train_dataset[index]["input_ids"][0], list):
#                 print(tokenizer.decode(train_dataset[index]["input_ids"][0]))
#             else:
#                 print(tokenizer.decode(train_dataset[index]["input_ids"]))
#             print("=========" * 9)
#
#     train_dataset = SupervisedDataset(train_dataset=train_dataset)
#     data_collator = DataCollatorForSupervisedDataset(
#         data_args=data_args, tokenizer=tokenizer
#     )
#     return dict(
#         train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
#     )
#
#
# def get_model_tokenizer(model_args, data_args, training_args):
#     model = AutoModelForCausalLM.from_pretrained(
#         model_args.model_name_or_path,
#         attn_implementation="flash_attention_2" if model_args.flash_attention else None,
#         cache_dir=training_args.cache_dir,
#     )
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_args.model_name_or_path,
#         cache_dir=training_args.cache_dir,
#         model_max_length=training_args.model_max_length,
#         padding_side="right",
#     )
#     assert tokenizer.eos_token_id is not None, "Tokenizer must have an EOS token"
#     assert model.get_output_embeddings().weight.data.size(0) == len(
#         tokenizer
#     ), "The vocabulary size of the model and the tokenizer should be the same"
#     return model, tokenizer
#
#
# def train():
#     parser = transformers.HfArgumentParser(
#         (ModelArguments, DataArguments, TrainingArguments)
#     )
#     model_args, data_args, training_args = parser.parse_args_into_dataclasses()
#
#     if training_args.gradient_checkpointing:
#         training_args.gradient_checkpointing_kwargs = {
#             "use_reentrant": False
#         }  # OR gradient_checkpointing_kwargs={'use_reentrant':True}, please refer to https://github.com/huggingface/transformers/issues/26969
#
#     model, tokenizer = get_model_tokenizer(model_args, data_args, training_args)
#
#     set_seed(training_args.seed)
#
#     data_module = make_supervised_data_module(
#         tokenizer=tokenizer,
#         data_args=data_args,
#         training_args=training_args,
#         model_args=model_args,
#     )
#     model.is_parallelizable = True
#     model.model_parallel = True
#     trainer_class = Trainer
#     if data_args.no_shuffle:
#         if training_args.use_wsd:
#             trainer_class = WSDNoShuffleTrainer
#         else:
#             trainer_class = NoShuffleSeq2SeqTrainer
#     elif training_args.use_wsd:
#         trainer_class = WSDTrainer
#     trainer = trainer_class(
#         model=model,
#         tokenizer=tokenizer,
#         args=training_args,
#         **data_module,
#     )
#     model.config.use_cache = False
#     trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
#     trainer.save_state()
#     trainer.save_model(output_dir=training_args.output_dir)
#
#
# if __name__ == "__main__":
#     train()


# encoding: utf-8
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
                if filename.endswith('.jsonl') and i < 4:
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