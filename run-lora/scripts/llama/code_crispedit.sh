#!/bin/bash

source "common_utils/server_env.sh"
set -e
source "common_utils/feishu_notify.sh"
feishu_trap_begin "$(basename "$0")"
feishu_notify_start "llama-code-crispEdit, now start" 
# Llama-2-7b code (python) — CrispEdit full-model baseline (fixed K-FAC cache).
output_home="./"
master_port=16962
model="${ROOT_DIR}/models/meta-llama/Llama-2-7b-hf"
model_abbr="Llama-2-7b"

# 4 GPUs for 7B; train.sh picks ZeRO-3 for llama like FFT.
grab_gpus 4 40000 100 3
train_gpus="$CUDA_VISIBLE_DEVICES"
test_gpus="$CUDA_VISIBLE_DEVICES"
inference_bs=800
per_bs=4
delete_adapter_after_test=false
run_world_knowledge=true
world_knowledge_limit=100
model_max_length=512
# Keep the inactive chunks' K-FAC A/S accumulators on the CPU (lower GPU peak).
crispedit_num_chunks=4

seeds=(1)
rank=128
bs=16
task="python"
peft_methods=("CrispEdit")

# Fixed FineFineWeb K-FAC cache (prepare with `python utils/save_finefineweb.py`).
crispedit_data_path="./lora-dataset/corda/finefineweb_validation.jsonl"
crispedit_energy_ratio=0.9

declare -A peft_lrs
peft_lrs["CrispEdit"]="2.0000e-5 3.5566e-5 6.3246e-5 1.1247e-4"

source "$(dirname "${BASH_SOURCE[0]}")/../run_best_common.sh"
