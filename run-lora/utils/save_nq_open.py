#!/usr/bin/env python3
"""
Save local NQ-open question-only samples for Corda KPM and CurvatureLoRA calibration.

Output contains only questions (no answers), safe for KFAC/Fisher collection.
FineFineWeb is the preferred calibration data source; this script provides
an NQ-open alternative.

   python utils/save_finefineweb.py
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

DATASET_ID = "nq_open"
DEFAULT_SPLIT = "train"
DEFAULT_NUM_SAMPLES = 256
DEFAULT_SEED = 42
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "lora-dataset"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--num_samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--curvature_output", default=None)
    parser.add_argument("--corda_kpm_output", default=None)
    return parser.parse_args()


def default_curvature_path(data_dir: Path, split: str, num_samples: int, seed: int) -> Path:
    return data_dir / "curvature" / f"nq_open_{split}_{num_samples}_seed{seed}.jsonl"


def default_corda_kpm_path(data_dir: Path) -> Path:
    return data_dir / "corda" / "kpm" / "train.jsonl"


def record_to_text(record: dict) -> str:
    question = str(record.get("question", "")).strip()
    return f"Question: {question}\nAnswer:"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in tqdm(rows, desc=f"Writing {path}"):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    curvature_output = Path(args.curvature_output) if args.curvature_output else default_curvature_path(
        data_dir, args.split, args.num_samples, args.seed
    )
    corda_kpm_output = Path(args.corda_kpm_output) if args.corda_kpm_output else default_corda_kpm_path(data_dir)

    dataset = load_dataset(args.dataset, split=args.split)
    dataset = dataset.shuffle(seed=args.seed).select(range(args.num_samples))
    rows = [{"text": record_to_text(row)} for row in dataset]

    write_jsonl(curvature_output, rows)
    write_jsonl(corda_kpm_output, rows)
    print(f"Saved {len(rows)} NQ-open records to {curvature_output}")
    print(f"Saved {len(rows)} NQ-open records to {corda_kpm_output}")


if __name__ == "__main__":
    main()
