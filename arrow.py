#!/usr/bin/env python3
"""
arrow_dataset_tools.py

Work with a Hugging Face "saved to disk" Arrow dataset folder that contains
files like data-00000-of-00037.arrow (your screenshot), plus the usual
dataset_info.json/state.json. If only raw .arrow shards are present, this
script can still load them and concatenate safely.

Requires: datasets, pyarrow
pip install -U datasets pyarrow
"""

import argparse
import glob
import json
import os
from typing import Optional, List, Union

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk


def load_arrow_dataset(path: str, split: Optional[str] = None) -> Dataset:
    """
    Load a dataset from a directory that contains Arrow shards.
    First tries `load_from_disk`. If that fails, falls back to manually
    concatenating *.arrow shards.
    """
    # Try the standard HF path (works when the folder was made by save_to_disk)
    try:
        obj = load_from_disk(path)
        if isinstance(obj, DatasetDict):
            if split is None:
                # Heuristic: prefer "train" if present; otherwise first split.
                split = "train" if "train" in obj else list(obj.keys())[0]
            ds = obj[split]
        else:
            ds = obj
        return ds
    except Exception:
        pass

    # Fallback: raw Arrow shards without the HF metadata
    arrow_files = sorted(
        glob.glob(os.path.join(path, "data-*.arrow"))
        + glob.glob(os.path.join(path, "*.arrow"))
    )
    if not arrow_files:
        raise FileNotFoundError(
            f"No dataset found at {path}. "
            "Expected a HF saved dataset folder or *.arrow shards."
        )

    from datasets import Dataset as HFDataset  # avoid shadowing
    parts = [HFDataset.from_file(f) for f in arrow_files]
    return concatenate_datasets(parts)


def peek(ds: Dataset, n: int = 5, columns: Optional[List[str]] = None) -> None:
    rows = ds.select_columns(columns) if columns else ds
    samples = rows.select(range(min(n, len(rows))))
    # Pretty print
    for i, ex in enumerate(samples):
        print(f"--- sample {i} ---")
        print(json.dumps(ex, ensure_ascii=False, indent=2))


def export_jsonl(ds: Dataset, out_path: str, columns: Optional[List[str]] = None) -> None:
    subset = ds.select_columns(columns) if columns else ds
    subset.to_json(out_path, orient="records", lines=True, force_ascii=False)
    print(f"Wrote JSONL → {out_path}")


def export_parquet(ds: Dataset, out_path: str, columns: Optional[List[str]] = None) -> None:
    subset = ds.select_columns(columns) if columns else ds
    subset.to_parquet(out_path)
    print(f"Wrote Parquet → {out_path}")


def main():
    p = argparse.ArgumentParser(description="Work with HF Arrow dataset shards.")
    p.add_argument("--data-dir", required=True, help="Path to folder like croissant_data/")
    p.add_argument("--split", default=None, help="If it's a DatasetDict, which split to use (e.g., train).")
    p.add_argument("--columns", nargs="*", default=None, help="Optional subset of columns to keep.")
    p.add_argument("--peek", type=int, default=0, help="Show N example rows and exit.")
    p.add_argument("--export-jsonl", default=None, help="Path to write JSONL (e.g., out.jsonl).")
    p.add_argument("--export-parquet", default=None, help="Path to write Parquet (e.g., out.parquet).")
    p.add_argument("--shuffle", action="store_true", help="Shuffle before exporting/peeking.")
    p.add_argument("--seed", type=int, default=42, help="Seed for shuffling.")
    args = p.parse_args()

    ds = load_arrow_dataset(args.data_dir, split=args.split)
    if args.shuffle:
        ds = ds.shuffle(seed=args.seed)

    # Print quick stats
    print("=== DATASET ===")
    print(f"path: {args.data_dir}")
    print(f"num_rows: {len(ds)}")
    print(f"columns: {list(ds.features.keys()) if ds.features else 'unknown'}")

    if args.peek > 0:
        peek(ds, n=args.peek, columns=args.columns)
        return

    if args.export_jsonl:
        export_jsonl(ds, args.export_jsonl, columns=args.columns)

    if args.export_parquet:
        export_parquet(ds, args.export_parquet, columns=args.columns)

    if not (args.export_jsonl or args.export_parquet or args.peek):
        print("\nNo action requested. Use --peek, --export-jsonl, or --export-parquet.")


if __name__ == "__main__":
    main()
