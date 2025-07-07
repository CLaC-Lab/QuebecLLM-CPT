# encoding: utf-8
"""
This script tokenizes text data using a specified tokenizer and saves the tokenized data to a target folder.
It supports multiprocessing to speed up the tokenization process.
It can also decode previously tokenized data back to text.

Functions:
    get_tgt_folder(file_path, model_name):
        Get the target folder path based on the file path and model name.

    tokenize_text(dataset, tgt_folder, idx, text_key, tokenizer_path, include_text):
        Tokenize text data and save it to the target folder.

    decode_tokenized_file(input_path, output_path, tokenizer_path):
        Decode a file containing tokenized data back to text.

    process_file_chunk(args):
        Process a chunk of data from a file.

    read_jsonl_file(file_path, max_lines=None):
        Generator to read JSONL file line by line.

Main:
    The script takes several command-line arguments:
        --mode: Mode of operation ('tokenize' or 'decode')
        --tokenizer_path: Path to the tokenizer.
        --model_name: Name of the model (for tokenize mode).
        --data_path: Path to the data files (for tokenize mode).
        --num_files: Number of files to process (for tokenize mode).
        --text_key: Key to access text data in the dataset (for tokenize mode).
        --num_worker: Number of worker processes for multiprocessing.
        --skip_exist: Whether to skip existing processed files.
        --include_text: Whether to include original text in tokenized output.
        --input_file: Input file for decode mode.
        --output_file: Output file for decode mode.

    The script processes each file in the specified data path, tokenizes the text data, and saves the tokenized data to the target folder.
"""

import argparse
import os
import json
import random
import pathlib
import logging
import multiprocessing as mp
from typing import List, Dict, Generator, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

random.seed(45)
MAX_DATA = int(1e6)


def get_tgt_folder(file_path: str, model_name: str) -> Tuple[str, bool]:
    """Get the target folder path based on the file path and model name."""
    # Create a more flexible path replacement
    base_dir = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)

    # Create target directory structure
    tgt_base = base_dir.replace("/data", f"/{model_name}_data_ids")
    if tgt_base == base_dir:  # If replacement didn't work
        tgt_base = os.path.join(base_dir, f"{model_name}_data_ids")

    file_stem = os.path.splitext(file_name)[0]
    tgt_folder = os.path.join(tgt_base, file_stem, "wo_ppl")

    is_exists = os.path.exists(tgt_folder)
    pathlib.Path(tgt_folder).mkdir(parents=True, exist_ok=True)

    return tgt_folder, is_exists


def tokenize_text(dataset: List[Dict], tgt_path: str, text_key: str, tokenizer_path: str,
                  include_text: bool = False) -> None:
    """Tokenize text data and save it to a specific file."""
    # Initialize tokenizer for this process
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    try:
        with open(tgt_path, 'w') as fout:
            for data in tqdm(dataset, desc=f"Tokenizing {os.path.basename(tgt_path)}"):
                try:
                    if text_key not in data:
                        logger.warning(f"Key '{text_key}' not found in data. Available keys: {list(data.keys())}")
                        continue

                    text = data[text_key]
                    input_ids = tokenizer(text, add_special_tokens=False)["input_ids"]

                    if include_text:
                        new_data = {
                            "text": text,
                            "input_ids": input_ids
                        }
                    else:
                        new_data = {"input_ids": input_ids}

                    fout.write(json.dumps(new_data, ensure_ascii=False) + "\n")
                except Exception as e:
                    logger.error(f"Error tokenizing data: {e}")
                    continue
    except Exception as e:
        logger.error(f"Error writing to file {tgt_path}: {e}")
        raise


def decode_tokenized_file(input_path: str, output_path: str, tokenizer_path: str) -> None:
    """Decode a file containing tokenized data back to text."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in tqdm(fin, desc=f"Decoding {os.path.basename(input_path)}"):
            try:
                data = json.loads(line.strip())
                if 'input_ids' in data:
                    # Decode the tokens back to text
                    decoded_text = tokenizer.decode(data['input_ids'], skip_special_tokens=True)

                    # Create new data with both text and tokens
                    new_data = {
                        'text': decoded_text,
                        'input_ids': data['input_ids']
                    }

                    fout.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                else:
                    # If no input_ids, write as is
                    fout.write(line)
            except Exception as e:
                logger.error(f"Error processing line: {e}")
                continue


def process_file_chunk(args: Tuple[List[Dict], str, str, str, bool]) -> None:
    """Process a chunk of data (wrapper for multiprocessing)."""
    dataset, tgt_path, text_key, tokenizer_path, include_text = args
    tokenize_text(dataset, tgt_path, text_key, tokenizer_path, include_text)


def read_jsonl_file(file_path: str, max_lines: int = None) -> Generator[Dict, None, None]:
    """Generator to read JSONL file line by line."""
    line_count = 0
    with open(file_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            if max_lines and line_count >= max_lines:
                break

            line = line.strip()
            if not line:
                continue

            try:
                yield json.loads(line)
                line_count += 1
            except json.JSONDecodeError as e:
                logger.warning(f"Error decoding JSON on line {line_count + 1}: {e}")
                continue


def read_json_file(file_path: str) -> List[Dict]:
    """Read a regular JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as fin:
            data = json.load(fin)
            if isinstance(data, list):
                return data
            else:
                return [data]
    except json.JSONDecodeError as e:
        logger.error(f"Error reading JSON file {file_path}: {e}")
        return []


def process_file(file_path: str, args: argparse.Namespace, tokenizer_path: str) -> None:
    """Process a single file."""
    tgt_folder, is_exists = get_tgt_folder(file_path, args.model_name)

    if is_exists and args.skip_exist:
        logger.info(f"Skipping existing folder: {tgt_folder}")
        return

    logger.info(f"Processing: {file_path}")
    logger.info(f"Target folder: {tgt_folder}")

    # Determine file type and read accordingly
    if file_path.endswith('.jsonl'):
        process_jsonl_file(file_path, tgt_folder, args, tokenizer_path)
    elif file_path.endswith('.json'):
        dataset = read_json_file(file_path)
        if dataset:
            process_data_batch(dataset, tgt_folder, args, tokenizer_path, batch_num=0)
    else:
        logger.warning(f"Unsupported file format: {file_path}")


def process_jsonl_file(file_path: str, tgt_folder: str, args: argparse.Namespace, tokenizer_path: str) -> None:
    """Process a JSONL file in batches."""
    batch_size = MAX_DATA // args.num_worker  # Divide work among workers
    batch = []
    batch_num = 0

    for data in read_jsonl_file(file_path):
        batch.append(data)

        if len(batch) >= batch_size:
            process_data_batch(batch, tgt_folder, args, tokenizer_path, batch_num)
            batch = []
            batch_num += 1

    # Process remaining data
    if batch:
        process_data_batch(batch, tgt_folder, args, tokenizer_path, batch_num)


def process_data_batch(dataset: List[Dict], tgt_folder: str, args: argparse.Namespace,
                       tokenizer_path: str, batch_num: int) -> None:
    """Process a batch of data using multiprocessing."""
    if not dataset:
        return

    # Validate text_key
    if args.text_key not in dataset[0]:
        logger.error(f"Key '{args.text_key}' not found. Available keys: {list(dataset[0].keys())}")
        raise KeyError(f"Unknown key: {args.text_key}")

    # Shuffle data
    random.shuffle(dataset)

    # Split data among workers
    num_workers = min(args.num_worker, len(dataset))
    if num_workers == 1:
        # Single process mode
        tgt_path = os.path.join(tgt_folder, f"batch_{batch_num}_part_0.jsonl")
        tokenize_text(dataset, tgt_path, args.text_key, tokenizer_path, args.include_text)
    else:
        # Multiprocessing mode
        chunk_size = len(dataset) // num_workers
        chunks = []

        for i in range(num_workers):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < num_workers - 1 else len(dataset)
            chunk = dataset[start_idx:end_idx]

            if chunk:  # Only process non-empty chunks
                tgt_path = os.path.join(tgt_folder, f"batch_{batch_num}_part_{i}.jsonl")
                chunks.append((chunk, tgt_path, args.text_key, tokenizer_path, args.include_text))

        # Process chunks in parallel
        pool = mp.Pool(num_workers)
        try:
            pool.map(process_file_chunk, chunks)
        finally:
            pool.close()
            pool.join()
            pool.terminate()  # Ensure all processes are terminated

        logger.info(f"Completed batch {batch_num} with {num_workers} workers")


def tokenize_mode(args: argparse.Namespace) -> None:
    """Run tokenization mode."""
    # Validate tokenizer path
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        logger.info(f"Successfully loaded tokenizer from {args.tokenizer_path}")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return

    # Process files
    files_processed = 0

    for root, _, files in os.walk(args.data_path):
        # Filter for supported file types
        supported_files = [f for f in files if f.endswith(('.json', '.jsonl'))]
        random.shuffle(supported_files)

        for file_name in supported_files:
            if files_processed >= args.num_files:
                break

            file_path = os.path.join(root, file_name)
            try:
                process_file(file_path, args, args.tokenizer_path)
                files_processed += 1
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                continue

        if files_processed >= args.num_files:
            break

    logger.info(f"Processing complete. Processed {files_processed} files.")


def decode_mode(args: argparse.Namespace) -> None:
    """Run decode mode."""
    if not args.input_file or not args.output_file:
        logger.error("Both --input_file and --output_file are required for decode mode")
        return

    logger.info(f"Loading tokenizer from {args.tokenizer_path}")
    try:
        # Test loading tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return

    # Decode the file
    decode_tokenized_file(args.input_file, args.output_file, args.tokenizer_path)
    logger.info(f"Decoding complete. Output saved to {args.output_file}")


def main():
    # Set multiprocessing start method for macOS compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

    parser = argparse.ArgumentParser(description="Tokenize text data or decode tokenized data")

    # Mode selection
    parser.add_argument("--mode", type=str, choices=['tokenize', 'decode'], default='tokenize',
                        help="Mode of operation: 'tokenize' or 'decode'")

    # Common arguments
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to the tokenizer")
    parser.add_argument("--num_worker", type=int, default=4,
                        help="Number of worker processes for multiprocessing")

    # Tokenize mode arguments
    parser.add_argument("--model_name", type=str,
                        help="Name of the model (required for tokenize mode)")
    parser.add_argument("--data_path", type=str,
                        help="Path to the data files (required for tokenize mode)")
    parser.add_argument("--num_files", type=int,
                        help="Number of files to process (required for tokenize mode)")
    parser.add_argument("--text_key", type=str,
                        help="Key to access text data in the dataset (required for tokenize mode)")
    parser.add_argument("--skip_exist", action='store_true',
                        help="Whether to skip existing processed files")
    parser.add_argument("--include_text", action='store_true',
                        help="Whether to include original text in tokenized output")

    # Decode mode arguments
    parser.add_argument("--input_file", type=str,
                        help="Input file for decode mode")
    parser.add_argument("--output_file", type=str,
                        help="Output file for decode mode")

    args = parser.parse_args()

    # Run appropriate mode
    if args.mode == 'tokenize':
        # Validate required arguments for tokenize mode
        required_args = ['model_name', 'data_path', 'num_files', 'text_key']
        missing_args = [arg for arg in required_args if not getattr(args, arg)]
        if missing_args:
            logger.error(f"Missing required arguments for tokenize mode: {', '.join(missing_args)}")
            return

        tokenize_mode(args)
    else:  # decode mode
        decode_mode(args)


if __name__ == "__main__":
    main()

