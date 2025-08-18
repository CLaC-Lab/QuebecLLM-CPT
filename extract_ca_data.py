#!/usr/bin/env python3
"""
Script to extract French Canadian (CA) data from paste.txt and append to train.txt and val.txt

Supports two formats:
1. Prefixed format: Lines starting with 'CA\t' (extracts only CA lines)
2. Raw format: All content treated as French Canadian text

Auto-detects format and processes accordingly.
"""

import os
import random

def extract_ca_data(input_file, train_file, val_file, train_split=0.8, prefixed_format=True):
    """
    Extract CA lines from input file and append to train/val files
    
    Args:
        input_file: Path to the input file with mixed data
        train_file: Path to training data file
        val_file: Path to validation data file
        train_split: Proportion of data to go to training (default 80%)
        prefixed_format: If True, look for 'CA\t' prefixed lines. If False, treat all lines as CA data
    """
    
    # Read the input file and extract CA lines
    ca_lines = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                if prefixed_format:
                    if line.startswith('CA\t'):
                        # Extract content after the tab
                        content = line[3:]  # Remove 'CA\t'
                        if content.strip():  # Only add non-empty lines
                            ca_lines.append(content)
                else:
                    # Treat all non-empty lines as CA data
                    ca_lines.append(line)
        
        if not ca_lines:
            print("No valid lines found in the input file.")
            return
        
        print(f"Found {len(ca_lines)} {'CA lines' if prefixed_format else 'lines of French Canadian content'}")
        
        # Shuffle the lines for random distribution
        random.shuffle(ca_lines)
        
        # Split into train and validation
        split_idx = int(len(ca_lines) * train_split)
        train_lines = ca_lines[:split_idx]
        val_lines = ca_lines[split_idx:]
        
        print(f"Splitting: {len(train_lines)} to train, {len(val_lines)} to validation")
        
        # Append to train file
        with open(train_file, 'a', encoding='utf-8') as f:
            for line in train_lines:
                f.write(line + '\n')
        
        # Append to validation file
        with open(val_file, 'a', encoding='utf-8') as f:
            for line in val_lines:
                f.write(line + '\n')
        
        print(f"Successfully appended {'CA data' if prefixed_format else 'French Canadian content'} to {train_file} and {val_file}")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Error processing files: {e}")

def main():
    # File paths
    input_file = "/home/k_ammade/Projects/QuebecCPT/CPT_scratch/data/FreCDO_train.txt"
    train_file = "/home/k_ammade/Projects/QuebecCPT/CPT_scratch/23M_data/train.txt"
    val_file = "/home/k_ammade/Projects/QuebecCPT/CPT_scratch/23M_data/train.txt"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        print("Please make sure the paste.txt file is in the current directory.")
        return
    
    # Check if target directories exist
    train_dir = os.path.dirname(train_file)
    if not os.path.exists(train_dir):
        print(f"Error: Directory '{train_dir}' does not exist.")
        return
    
    # Create target files if they don't exist
    for file_path in [train_file, val_file]:
        if not os.path.exists(file_path):
            print(f"Creating {file_path}")
            with open(file_path, 'w', encoding='utf-8') as f:
                pass  # Create empty file
    
    # Auto-detect format by checking first few lines
    prefixed_format = False
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            first_lines = [f.readline().strip() for _ in range(5)]
            # Check if any line starts with country codes like 'CA\t', 'FR\t', etc.
            for line in first_lines:
                if line and len(line) > 3 and line[2:4] == '\t' and line[:2].isalpha():
                    prefixed_format = True
                    break
    except:
        pass
    
    if prefixed_format:
        print("Detected prefixed format (CA\\t, FR\\t, etc.) - extracting CA lines only")
    else:
        print("Detected raw text format - treating all content as French Canadian")
    
    # Extract and append CA data
    extract_ca_data(input_file, train_file, val_file, prefixed_format=prefixed_format)

if __name__ == "__main__":
    main()