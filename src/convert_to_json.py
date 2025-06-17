#!/usr/bin/env python3
# encoding: utf-8
"""
Convert plain text files to JSONL format for tokenization.
Each line becomes a JSON object with a 'text' field.
"""

import json
import argparse
import os


def convert_txt_to_jsonl(input_file, output_file, chunk_size=None):
    """
    Convert a text file to JSONL format.

    Args:
        input_file: Path to input .txt file
        output_file: Path to output .jsonl file
        chunk_size: If specified, split long texts into chunks of this many characters
    """
    with open(input_file, 'r', encoding='utf-8') as fin, \
            open(output_file, 'w', encoding='utf-8') as fout:

        if chunk_size:
            # Read entire file and split into chunks
            content = fin.read()
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i + chunk_size].strip()
                if chunk:  # Skip empty chunks
                    json_obj = {"text": chunk}
                    fout.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
        else:
            # Process line by line
            for line_num, line in enumerate(fin, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    json_obj = {"text": line}
                    fout.write(json.dumps(json_obj, ensure_ascii=False) + '\n')

                if line_num % 10000 == 0:
                    print(f"Processed {line_num} lines...")


def main():
    parser = argparse.ArgumentParser(description="Convert text files to JSONL format")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing text files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save JSONL files")
    parser.add_argument("--chunk_size", type=int, default=None,
                        help="Split text into chunks of this size (characters)")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process all .txt files in input directory
    for filename in os.listdir(args.input_dir):
        if filename.endswith('.txt'):
            input_path = os.path.join(args.input_dir, filename)
            output_filename = filename.replace('.txt', '.jsonl')
            output_path = os.path.join(args.output_dir, output_filename)

            print(f"Converting {input_path} -> {output_path}")
            convert_txt_to_jsonl(input_path, output_path, args.chunk_size)
            print(f"Completed: {output_path}")


if __name__ == "__main__":
    main()