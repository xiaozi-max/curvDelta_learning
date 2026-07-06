#!/usr/bin/env python3
"""Save a FineFineWeb-validation curvature anchor set as text-only JSONL."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import list_repo_files
from tqdm import tqdm

DATASET_ID = "m-a-p/FineFineWeb-validation"
DEFAULT_NUM_SAMPLES = 16_384
DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "lora-dataset" / "curvature"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET_ID, help="Hugging Face dataset id to sample.")
    parser.add_argument("--num_samples", type=int, default=DEFAULT_NUM_SAMPLES, help="Number of non-empty text records to save.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Shuffle seed before sampling.")
    parser.add_argument("--output_path", default=None, help="Output JSONL path. Defaults under run-lora/lora-dataset/curvature/.")
    return parser.parse_args()


def default_output_path(dataset: str, num_samples: int, seed: int) -> Path:
    name = dataset.split("/")[-1].replace("-", "_").lower()
    return DEFAULT_OUTPUT_DIR / f"{name}_{num_samples}_seed{seed}.jsonl"


def main():
    args = parse_args()

    output_path = Path(args.output_path) if args.output_path else default_output_path(args.dataset, args.num_samples, args.seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data_files = sorted(
        file_name
        for file_name in list_repo_files(args.dataset, repo_type="dataset")
        if file_name.endswith(".jsonl")
    )

    dataset = load_dataset(args.dataset, data_files=data_files, split="train")
    dataset = dataset.filter(lambda row: isinstance(row.get("text"), str) and bool(row["text"].strip()))
    dataset = dataset.shuffle(seed=args.seed)
    dataset = dataset.select(range(args.num_samples))

    with output_path.open("w", encoding="utf-8") as f:
        for row in tqdm(dataset, desc=f"Writing {output_path}"):
            f.write(json.dumps({"text": row["text"]}, ensure_ascii=False) + "\n")

    print(f"Saved {len(dataset)} records to {output_path}")


if __name__ == "__main__":
    main()
