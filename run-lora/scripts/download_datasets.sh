#!/usr/bin/env bash
# Download fine-tuning datasets for LoRA experiments.
# Run from the run-lora/ directory: `bash ./scripts/download_datasets.sh`
set -euo pipefail

# 1. Mathematical Reasoning and Code Generation
# Preprocessed datasets from the PiSSA authors.
huggingface-cli download --repo-type dataset --resume-download fxmeng/pissa-dataset \
    --include "metamath/*" "python/*" \
    --local-dir ./lora-dataset/pissa-dataset

# 2. Commonsense Reasoning (LLM-Adapters)

# Training dataset
mkdir -p ./lora-dataset/llm-adapters-dataset
wget -O ./lora-dataset/llm-adapters-dataset/commonsense_15k.json \
    https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/main/ft-training_set/commonsense_15k.json

# Testing dataset
for dataset in boolq piqa social_i_qa hellaswag winogrande ARC-Challenge ARC-Easy openbookqa; do
    mkdir -p ./lora-dataset/llm-adapters-dataset/${dataset}
    wget -O ./lora-dataset/llm-adapters-dataset/${dataset}/test.json \
        https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/main/dataset/${dataset}/test.json
done

# 3. Instruction Following

# Training dataset (alpaca-cleaned)
mkdir -p ./lora-dataset/instruction-dataset
python3 -c '
from datasets import load_dataset
import json
d = load_dataset("yahma/alpaca-cleaned", split="train")
res = [{"instruction": i["instruction"], "input": i["input"], "output": i["output"]} for i in d]
json.dump(res, open("./lora-dataset/instruction-dataset/alpaca_clean.json", "w"), indent=2)
'

# Testing dataset (IFEval)
wget -O ./lora-dataset/instruction-dataset/input_data.jsonl \
    https://raw.githubusercontent.com/google-research/google-research/master/instruction_following_eval/data/input_data.jsonl



