# Step 1: Data Preparation for Quebec French Language Model

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