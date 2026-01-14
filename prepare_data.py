#!/usr/bin/env python
"""
Script to prepare and clean Quebec French text data for training
Supports both plain text (.txt) and JSONL (.jsonl) formats
"""

import re
import json
import argparse
import random
from pathlib import Path
from typing import List, Optional, Tuple
import unicodedata


class QuebecFrenchDataPreparer:
    """Prepare and clean Quebec French text data"""
    
    def __init__(self, min_length: int = 10, max_length: int = 10000):
        self.min_length = min_length
        self.max_length = max_length

    def normalize_text(self, text: str) -> str:
        """Normalize Unicode and clean text"""
        # Normalize Unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Fix common encoding issues
        text = text.replace('œ', 'oe').replace('Œ', 'OE')
        text = text.replace('æ', 'ae').replace('Æ', 'AE')
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'([.!?])\1+', r'\1', text)
        
        # Normalize text
        text = self.normalize_text(text)
        
        return text
    
    def filter_text(self, text: str) -> bool:
        """Filter text based on quality criteria"""
        # Check length
        if len(text) < self.min_length or len(text) > self.max_length:
            return False
        
        # Check for minimum word count
        words = text.split()
        if len(words) < 3:
            return False
        
        # Check for excessive repetition
        if len(set(words)) / len(words) < 0.3:
            return False
        
        # Check for too many special characters or numbers
        alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / len(text)
        if alpha_ratio < 0.7:
            return False
        
        return True
    
    def read_input_file(self, file_path: Path) -> List[str]:
        """Read input file - supports both .txt and .jsonl formats"""
        lines = []
        file_ext = file_path.suffix.lower()
        
        if file_ext in ['.jsonl', '.json']:
            # Handle JSONL format
            print(f"Reading JSONL file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        # Handle different possible field names
                        text = data.get('text') or data.get('content') or data.get('sentence')
                        if text:
                            lines.append(text.strip())
                        else:
                            print(f"Warning: Line {line_num} has no 'text' field, skipping")
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse JSON at line {line_num}: {e}")
                        continue
        
        elif file_ext in ['.txt', '.text']:
            # Handle plain text format
            print(f"Reading text file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        lines.append(line)
        
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: .txt, .text, .jsonl, .json")
        
        return lines
    
    def process_file(self, input_path: str, output_path: str, val_split: float = 0.1) -> Tuple[Path, Path]:
        """Process input file and create train/val splits"""
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Read input file based on format
        lines = self.read_input_file(input_path)
        print(f"Read {len(lines)} text samples from input file")
        
        # Process lines
        processed_texts = []
        quebec_french_count = 0
        filtered_count = 0
        
        for text in lines:
            # Clean text
            cleaned = self.clean_text(text)
            
            # Filter
            if self.filter_text(cleaned):
                processed_texts.append(cleaned)
            else:
                filtered_count += 1
        
        print(f"Processed {len(processed_texts)} texts (filtered out {filtered_count})")
        print(f"Quebec French markers found in {quebec_french_count} texts")
        
        if len(processed_texts) == 0:
            raise ValueError("No valid texts found after processing. Check your input file and filters.")
        
        # Shuffle for random split
        random.seed(42)  # For reproducibility
        random.shuffle(processed_texts)
        
        # Split into train and validation
        val_size = int(len(processed_texts) * val_split)
        train_texts = processed_texts[val_size:]
        val_texts = processed_texts[:val_size]
        
        # Save train file
        train_file = output_path / "train_1.txt"
        with open(train_file, 'w', encoding='utf-8') as f:
            for text in train_texts:
                f.write(text + '\n')
        print(f"Saved {len(train_texts)} training samples to {train_file}")
        
        # Save validation file
        val_file = output_path / "val_1.txt"
        with open(val_file, 'w', encoding='utf-8') as f:
            for text in val_texts:
                f.write(text + '\n')
        print(f"Saved {len(val_texts)} validation samples to {val_file}")
        
        # Print sample statistics
        print("\n=== Dataset Statistics ===")
        print(f"Total samples: {len(processed_texts)}")
        print(f"Training samples: {len(train_texts)} ({100*(1-val_split):.0f}%)")
        print(f"Validation samples: {len(val_texts)} ({100*val_split:.0f}%)")
        
        if len(train_texts) > 0:
            avg_length = sum(len(text.split()) for text in train_texts[:100]) / min(100, len(train_texts))
            print(f"Average words per text (sample): {avg_length:.1f}")
        
        # Show sample texts
        print("\n=== Sample processed texts ===")
        for i, text in enumerate(train_texts[:3], 1):
            preview = text[:150] + "..." if len(text) > 150 else text
            print(f"{i}. {preview}")
        
        return train_file, val_file
    
    def merge_multiple_files(self, input_files: List[str], output_path: str, val_split: float = 0.1) -> Tuple[Path, Path]:
        """Process and merge multiple input files"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_texts = []
        
        # Read all input files
        for input_file in input_files:
            input_path = Path(input_file)
            if not input_path.exists():
                print(f"Warning: File {input_path} does not exist, skipping...")
                continue
            
            lines = self.read_input_file(input_path)
            print(f"Read {len(lines)} samples from {input_path.name}")
            all_texts.extend(lines)
        
        print(f"\nTotal samples from all files: {len(all_texts)}")
        
        # Process all texts
        processed_texts = []
        quebec_french_count = 0
        filtered_count = 0
        
        for text in all_texts:
            cleaned = self.clean_text(text)
            
            if self.filter_text(cleaned):
                processed_texts.append(cleaned)
            else:
                filtered_count += 1
        
        print(f"Processed {len(processed_texts)} texts (filtered out {filtered_count})")
        print(f"Quebec French markers found in {quebec_french_count} texts")
        
        if len(processed_texts) == 0:
            raise ValueError("No valid texts found after processing.")
        
        # Shuffle and split
        random.shuffle(processed_texts)
        
        val_size = int(len(processed_texts) * val_split)
        train_texts = processed_texts[val_size:]
        val_texts = processed_texts[:val_size]
        
        # Save files
        train_file = output_path / "train.txt"
        with open(train_file, 'w', encoding='utf-8') as f:
            for text in train_texts:
                f.write(text + '\n')
        
        val_file = output_path / "val.txt"
        with open(val_file, 'w', encoding='utf-8') as f:
            for text in val_texts:
                f.write(text + '\n')
        
        print(f"\n✅ Merged {len(input_files)} files successfully!")
        print(f"Training samples: {len(train_texts)}")
        print(f"Validation samples: {len(val_texts)}")
        
        return train_file, val_file


def main():
    """Main function for data preparation"""
    parser = argparse.ArgumentParser(
        description="Prepare Quebec French data for training (supports .txt and .jsonl formats)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single text file
  python prepare_data.py --input corpus.txt --output ./data
  
  # Process a JSONL file
  python prepare_data.py --input data.jsonl --output ./data
  
  # Merge multiple files (mix of txt and jsonl)
  python prepare_data.py --input file1.txt file2.jsonl file3.txt --output ./data --merge
  
  # Custom validation split
  python prepare_data.py --input corpus.jsonl --output ./data --val_split 0.2
  
  # Adjust text length filters
  python prepare_data.py --input corpus.txt --output ./data --min_length 20 --max_length 5000
        """
    )
    
    parser.add_argument(
        "--input", 
        type=str,
        nargs='+',  # Allow multiple input files
        required=True, 
        help="Input file(s) - supports .txt and .jsonl formats (can specify multiple files)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="./data", 
        help="Output directory for processed files (default: ./data)"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge multiple input files into single train/val sets"
    )
    parser.add_argument(
        "--val_split", 
        type=float, 
        default=0.1, 
        help="Validation split ratio (default: 0.1)"
    )
    parser.add_argument(
        "--min_length", 
        type=int, 
        default=10, 
        help="Minimum text length in characters (default: 10)"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=10000, 
        help="Maximum text length in characters (default: 10000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.val_split < 0 or args.val_split > 0.5:
        print(f"Error: val_split must be between 0 and 0.5 (got {args.val_split})")
        return 1
    
    if args.min_length < 1:
        print(f"Error: min_length must be positive (got {args.min_length})")
        return 1
    
    if args.max_length <= args.min_length:
        print(f"Error: max_length must be greater than min_length")
        return 1
    
    # Set random seed
    random.seed(args.seed)
    
    # Create data preparer
    preparer = QuebecFrenchDataPreparer(
        min_length=args.min_length,
        max_length=args.max_length
    )
    
    try:
        # Check if we're processing multiple files or single file
        if len(args.input) > 1 or args.merge:
            # Multiple files - merge them
            print(f"Processing and merging {len(args.input)} files...")
            train_file, val_file = preparer.merge_multiple_files(
                args.input,
                args.output,
                args.val_split
            )
        else:
            # Single file
            input_file = args.input[0]
            if not Path(input_file).exists():
                print(f"Error: Input file '{input_file}' does not exist")
                return 1
            
            train_file, val_file = preparer.process_file(
                input_file,
                args.output,
                args.val_split
            )
        
        print(f"\n✅ Data preparation complete!")
        print(f"Train file: {train_file}")
        print(f"Validation file: {val_file}")
        print(f"\nYou can now use these files to train your model:")
        print(f"  python train.py --train_file {train_file} --val_file {val_file}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
