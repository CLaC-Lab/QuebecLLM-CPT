#!/usr/bin/env python
"""
build_cpt_corpus.py

Create train corpus for Continual Pretraining (CPT) from MANY .txt inputs.
Designed for Quebec French adaptation pipelines.

Features
- Accepts multiple input files and/or directories (recursive *.txt crawl)
- Split by paragraphs (blank-line separated), lines, or auto
- Cleans whitespace, normalizes paragraphs, min length filter
- Global shuffle with seed; uses 100% of data for training
- Append (default) or overwrite outputs
- Optional exact de-duplication across ALL inputs post-cleaning
- Optional source map TSV with precise appended line numbers

Examples
1) Basic (paragraphs, all data to train, append):
   python build_cpt_corpus.py \
     --inputs /data/wiki_qc.txt /data/forums_qc.txt \
     --train /home/k_ammade/CPT_scratch/data/new_data/train.txt

2) From a directory (recursive), keep non-empty lines, dedup, overwrite:
   python build_cpt_corpus.py \
     --inputs /home/k_ammade/Projects/QuebecCPT/CPT_scratch/data/ALL_DATA/raw_data \
     --unit line --dedup --overwrite \
     --train /home/k_ammade/CPT_scratch/data/new_data/train.txt

3) With a source map for auditability:
   python build_cpt_corpus.py \
     --inputs /data/qc_corpus \
     --train /.../train.txt \
     --source-map /.../source_map.tsv

Notes
- Outputs are UTF-8. Inputs try utf-8, utf-8-sig, latin-1 (then utf-8 ignore).
- De-dup uses exact string match after cleaning/normalization.
- All data goes to training - no validation split for continual pretraining.
"""

import os
import re
import argparse
import random
from typing import List, Tuple, Iterable, Dict, Set

# ---------- IO & Normalization ----------

def read_text_any(input_path: str) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(input_path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def normalize_paragraph(p: str) -> str:
    if p is None:
        return ""
    
    # MINIMAL cleaning for CPT - preserve authentic Quebec French patterns
    # Only fix encoding issues and truly problematic characters
    
    # Fix Windows line endings
    p = p.replace("\r\n", "\n")
    p = p.replace("\r", "\n")
    
    # Remove null bytes and other control characters that break tokenizers
    p = p.replace("\x00", "")
    p = p.replace("\x01", "")
    p = p.replace("\x02", "")
    
    # Only strip leading/trailing whitespace, preserve internal formatting
    return p.strip()

def split_units(text: str, unit: str) -> List[str]:
    # Debug: Check if text is valid
    if text is None:
        print("[DEBUG] Text is None!")
        return []
    
    if unit == "paragraph":
        raw_paras = re.split(r"\n\s*\n+", text.strip(), flags=re.MULTILINE)
        return [normalize_paragraph(p) for p in raw_paras if normalize_paragraph(p)]
    elif unit == "line":
        lines = []
        splitlines_result = text.splitlines()
        print(f"[DEBUG] text.splitlines() returned {len(splitlines_result)} lines")
        
        for i, line in enumerate(splitlines_result):
            if line is None:
                print(f"[DEBUG] Line {i} is None! This should never happen.")
                continue
            # print(f"[DEBUG] Processing line {i}: type={type(line)}, repr={repr(line[:50])}")
            normalized = normalize_paragraph(line)
            if normalized:
                lines.append(normalized)
        return lines
    else:  # auto
        if re.search(r"\n\s*\n+", text):
            return split_units(text, "paragraph")
        else:
            return split_units(text, "line")

# --- add this helper ---
def ensure_parent_dir(path: str):
    dirpath = os.path.dirname(os.path.abspath(path))
    os.makedirs(dirpath, exist_ok=True)

def safe_touch(path: str):
    ensure_parent_dir(path)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as _:
            pass

def count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f)

def append_lines(path: str, lines: List[str]):
    with open(path, "a", encoding="utf-8") as f:
        for x in lines:
            f.write(x + "\n")

def overwrite_lines(path: str, lines: List[str]):
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for x in lines:
            f.write(x + "\n")

# ---------- File Discovery ----------

def gather_input_files(inputs: List[str], ext: str = ".txt") -> List[str]:
    files: List[str] = []
    for p in inputs:
        if os.path.isfile(p):
            if p.lower().endswith(ext):
                files.append(os.path.abspath(p))
        elif os.path.isdir(p):
            for root, _, fnames in os.walk(p):
                for fn in fnames:
                    if fn.lower().endswith(ext):
                        files.append(os.path.abspath(os.path.join(root, fn)))
        else:
            # ignore non-existing / non-txt silently
            pass
    # de-dup while preserving order
    seen: Set[str] = set()
    uniq: List[str] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq

# ---------- Mapping (optional) ----------

def write_source_map(map_path: str,
                     start_line_index_1based: int,
                     units_with_src: List[Tuple[str, str]],
                     append: bool):
    """
    Write a TSV with columns: split, line_no, chars, src_path
    line_no is the line number in the OUTPUT FILE after this write.
    """
    header = "split\tline_no\tchars\tsrc_path\n"
    exists = os.path.exists(map_path)
    mode = "a" if append and exists else "w"
    ensure_parent_dir(map_path)
    with open(map_path, mode, encoding="utf-8") as f:
        if mode == "w":
            f.write(header)
        line_no = start_line_index_1based
        for text, src in units_with_src:
            f.write(f"train\t{line_no}\t{len(text)}\t{src}\n")  # Always "train" since no validation
            line_no += 1

# ---------- Main Pipeline ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="List of .txt files and/or directories (recursive *.txt).")
    ap.add_argument("--train", required=True, help="Path to train.txt")
    # Removed --val argument - no validation file needed
    ap.add_argument("--unit", choices=["auto", "paragraph", "line"], default="paragraph",
                    help="Split strategy: paragraph (default), line, or auto")
    # Removed --train-split argument - always use 100% for training
    ap.add_argument("--min-chars", type=int, default=10,
                    help="Drop units shorter than this many chars after cleaning")
    ap.add_argument("--dedup", action="store_true",
                    help="Exact de-duplication after cleaning across ALL inputs")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse/report counts, but do not write")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite the train file instead of appending")
    ap.add_argument("--source-map", default=None,
                    help="Optional TSV path to record (split, line_no, chars, src_path) for appended lines")
    args = ap.parse_args()

    train_abs = os.path.abspath(args.train)

    # Discover files
    files = gather_input_files(args.inputs, ext=".txt")
    if not files:
        raise SystemExit("No input .txt files found from --inputs.")

    # Exclude output if it accidentally sits inside input trees
    files = [f for f in files if os.path.abspath(f) != train_abs]

    print(f"[DEBUG] Found {len(files)} .txt files to process")

    # Parse & collect units with source
    all_units: List[Tuple[str, str]] = []
    total_raw_files = 0
    for fp in files:
        total_raw_files += 1
        print(f"[DEBUG] Processing file: {fp}")
        try:
            text = read_text_any(fp)
            print(f"[DEBUG] File {fp} read successfully, length: {len(text)} chars")
        except Exception as e:
            print(f"[warn] Skipping {fp}: {e}")
            continue
        units = split_units(text, args.unit)
        units = [u for u in units if len(u) >= args.min_chars]
        print(f"[DEBUG] File {fp} yielded {len(units)} units after filtering")
        all_units.extend((u, fp) for u in units)

    if not all_units:
        raise SystemExit("No usable text units found after cleaning across inputs.")

    # Optional dedup
    pre_dedup_count = len(all_units)
    if args.dedup:
        seen_texts: Set[str] = set()
        dedupd: List[Tuple[str, str]] = []
        for u, src in all_units:
            if u not in seen_texts:
                seen_texts.add(u)
                dedupd.append((u, src))
        all_units = dedupd

    # Shuffle reproducibly
    random.seed(args.seed)
    random.shuffle(all_units)

    # Use ALL data for training (no validation split)
    train_pairs = all_units

    # Stats
    print("========================================")
    print("CPT Corpus Builder (Quebec French)")
    print("Train-Only Mode (No Validation Split)")
    print("========================================")
    print(f"Files scanned:                 {total_raw_files}")
    print(f"Usable units before de-dup:    {pre_dedup_count}")
    if args.dedup:
        print(f"Usable units after de-dup:     {len(all_units)}")
    print(f"Unit='{args.unit}', min_chars={args.min_chars}")
    print(f"Total units for training:      {len(train_pairs)}")

    if args.dry_run:
        print("Dry-run only; no files written.")
        return

    # Prepare output file
    if args.overwrite:
        overwrite_lines(args.train, [u for u, _ in train_pairs])
        print(f"Overwrote train file: {args.train}")

        if args.source_map:
            # When overwriting, line numbers start at 1
            write_source_map(args.source_map, 1, train_pairs, append=False)
            print(f"Wrote source map: {args.source_map}")
    else:
        # Append mode (default) — compute starting line index
        safe_touch(args.train)
        train_start = count_lines(args.train) + 1

        append_lines(args.train, [u for u, _ in train_pairs])
        print(f"Appended to train file: {args.train}")

        if args.source_map:
            write_source_map(args.source_map, train_start, train_pairs, append=True)
            print(f"Updated source map: {args.source_map}")

    print(f"Final train file contains {count_lines(args.train)} lines.")

if __name__ == "__main__":
    main()
