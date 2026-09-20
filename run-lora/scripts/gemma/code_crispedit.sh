#!/bin/bash

source "common_utils/server_env.sh"
set -e
source "common_utils/feishu_notify.sh"
feishu_trap_begin "$(basename "$0")"
feishu_notify_start "gemma-code-crispEdit" 
# Gemma-3-1B code (python) — CrispEdit full-model baseline (fixed K-FAC cache).
output_home="./"
master_port=16961
model="${ROOT_DIR}/models/google/gemma-3-1b-pt"
model_abbr="gemma-3-1b"

grab_gpus 1 40000 100 3
train_gpus="$CUDA_VISIBLE_DEVICES"
test_gpus="$CUDA_VISIBLE_DEVICES"
inference_bs=800
per_bs=4
delete_adapter_after_test=false
run_world_knowledge=true
world_knowledge_limit=50
model_max_length=512

seeds=(1)
rank=128
bs=64
task="python"
peft_methods=("CrispEdit")

# Fixed FineFineWeb K-FAC cache (prepare with `python utils/save_finefineweb.py`).
crispedit_data_path="./lora-dataset/corda/finefineweb_validation.jsonl"
crispedit_energy_ratio=0.9
crispedit_num_chunks=1

declare -A peft_lrs
peft_lrs["CrispEdit"]="2.0000e-5 3.5566e-5 6.3246e-5 1.1247e-4"

source "$(dirname "${BASH_SOURCE[0]}")/../run_best_common.sh"
