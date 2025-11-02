# Step 1: Data Preparation 

This script prepares and cleans Quebec French text data for training your language model. It handles both plain text (`.txt`) and JSON Lines (`.jsonl`) formats, performing essential preprocessing tasks to ensure high-quality training data.

---

## Overview

The data preparation script performs the following key operations:

* **Text Normalization**: Normalizes Unicode characters and fixes common encoding issues.
* **Cleaning**: Removes URLs, email addresses, and excessive punctuation.
* **Filtering**: Removes low-quality texts based on length, word count, repetition, and character ratios.
* **Train/Val Split**: Automatically splits your data into training and validation sets.
* **Format Support**: Works with both **`.txt`** (plain text) and **`.jsonl`** (JSON Lines) formats.

### Prerequisites

The script requires at least **Python 3.6+** with **no external dependencies** (uses only the standard library).

---

## Input Formats

The script supports two primary input formats:

### Plain Text Format (`.txt`)

Each line contains one text sample, such as: "Bonjour, comment ça va? C'est une belle journée à Montréal. J'aime le poutine et les queues de castor."

### JSONL Format (`.jsonl`)

Each line is a JSON object with a text field:

```json
{"text": "Bonjour, comment ça va?"}
{"text": "C'est une belle journée à Montréal."}
{"text": "J'aime le poutine et les queues de castor."}
```

## Usage
Basic Usage - Single File
Process a plain text file:

python prepare_data.py --input corpus.txt --output ./data

### Merging Multiple Files

Combine multiple files (you can mix .txt and .jsonl): `python prepare_data.py --input file1.txt file2.jsonl file3.txt --output ./data --merge`

### Adjust text length filters: `python prepare_data.py --input corpus.txt --output ./data --min_length 20 --max_length 5000`

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input` | str (multiple) | Required | Input file(s) - supports `.txt` and `.jsonl` formats |
| `--output` | str | `./data` | Output directory for processed files |
| `--merge` | flag | False | Merge multiple input files into single train/val sets |
| `--val_split` | float | `0.1` | Validation split ratio (0.0-0.5) |
| `--min_length` | int | `10` | Minimum text length in characters |
| `--max_length` | int | `10000` | Maximum text length in characters |
| `--seed` | int | `42` | Random seed for reproducible splits |

---

## Output

The script generates two files in the output directory:

* `train.txt` (or `train_1.txt` for single files) - Training data
* `val.txt` (or `val_1.txt` for single files) - Validation data

## Data Quality Filters

The script applies several quality filters to ensure clean training data:

1. Length Filter: Removes texts shorter than min_length or longer than max_length
2. Word Count: Requires at least 3 words per text
3. Repetition Check: Filters texts with excessive word repetition (vocabulary diversity < 30%)
4. Character Ratio: Requires at least 70% alphabetic characters or spaces


# Step 2: Model Training

## Key Features

- LoRA Support: Parameter-efficient fine-tuning with Low-Rank Adaptation
- Quantization: Optional 4-bit/8-bit quantization for memory efficiency
- Sliding Window: Configurable stride for overlapping context windows
- Gradient Checkpointing: Reduce memory usage during training
- FSDP Support: Fully Sharded Data Parallel for multi-GPU training
- Data Inspection: Built-in data validation and preview
- Perplexity Tracking: Automatic perplexity calculation during training
- TensorBoard Logging: Real-time training metrics visualization

## Prerequisite Installs

`pip install torch transformers datasets peft bitsandbytes accelerate tensorboard`

## Basic Usage

For single GPU training (with LoRA): 

```python
python train.py \
  --train_file ./output/train.txt \
  --output_dir ./cpt_model \
  --model_name meta-llama/Llama-3.2-1B \
  --num_epochs 3 \
  --batch_size 8 \
  --max_length 1024 \
  --learning_rate 1e-5
``` 

To turn on 4-bit quantization and reduce memory usage even further:

```python
python train.py \
  --train_file ./output/train.txt \
  --output_dir ./cpt_model \
  --model_name meta-llama/Llama-3.2-1B \
  --use_4bit \
  --batch_size 16 \
  --gradient_accumulation_steps 4
``` 

### Command Line Arguments

#### Model Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_name` | str | `meta-llama/Llama-3.2-1B` | HuggingFace model identifier |
| `--use_lora` | flag | True | Enable LoRA for parameter-efficient training |
| `--use_4bit` | flag | False | Enable 4-bit quantization (requires bitsandbytes) |
| `--lora_r` | int | `16` | LoRA rank (higher = more parameters) |
| `--lora_alpha` | int | `32` | LoRA alpha scaling parameter |

#### Data Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--train_file` | str | Required | Path to training corpus (from step 1) |
| `--max_length` | int | `1024` | Maximum sequence length in tokens |
| `--stride` | int | `128` | Sliding window overlap (0 = no overlap) |
| `--batch_size` | int | `8` | Per-device training batch size |
| `--inspect_data` | flag | True | Enable data inspection before training |
| `--inspect_samples` | int | `5` | Number of samples to inspect |

#### Training Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output_dir` | str | `./cpt_model` | Directory for model checkpoints and logs |
| `--num_epochs` | int | `3` | Number of training epochs |
| `--learning_rate` | float | `1e-5` | Initial learning rate |
| `--gradient_accumulation_steps` | int | `8` | Steps to accumulate gradients |
| `--seed` | int | `42` | Random seed for reproducibility |

#### Parallelization Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--fsdp` | str | None | FSDP strategy (e.g., "full_shard auto_wrap") |
| `--fsdp_transformer_layer_cls_to_wrap` | str | `LlamaDecoderLayer` | Layer class for FSDP wrapping |

### Data Processing Pipeline

The script processes your training data through several stages:

1. **Loading**: Reads text from `train.txt` (one document per line)
2. **Tokenization**: Converts text to tokens, appends EOS token to each document
3. **Chunking**: Groups tokens into fixed-length sequences
   - **Non-overlapping** (stride=0): `[0:1024], [1024:2048], ...`
   - **Overlapping** (stride=512): `[0:1024], [512:1536], [1024:2048], ...`
4. **Filtering**: Removes sequences shorter than `min_length`
5. **Batching**: Creates training batches with proper padding

### Training Configuration

#### LoRA Configuration (Default)

- **Rank (r)**: 16
- **Alpha**: 32
- **Dropout**: 0.1
- **Target Modules**: `q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

#### Optimizer Settings

- **Optimizer**: AdamW
- **Learning Rate Schedule**: Cosine with warmup
- **Warmup Ratio**: 0.03 (3% of total steps)
- **Weight Decay**: 0.0
- **Max Gradient Norm**: 1.0

#### Memory Optimization

- **Mixed Precision**: FP16 enabled by default
- **Gradient Checkpointing**: Enabled for LoRA training
- **4-bit Quantization**: Uses NF4 with double quantization

### Output Structure

After training, the output directory should contain:
```
cpt_model/
├── checkpoint-500/          # Intermediate checkpoints
├── checkpoint-1000/
├── adapter_config.json      # LoRA configuration
├── adapter_model.safetensors # LoRA weights
├── config.json              # Model configuration
├── training_args.bin        # Training arguments
├── trainer_state.json       # Training state
├── config.json              # Pipeline configuration
├── training_summary.json    # Final metrics
└── logs/                    # TensorBoard logs
    └── events.out.tfevents.*
