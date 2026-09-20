#!/bin/bash

source "common_utils/server_env.sh"
set -e
source "common_utils/feishu_notify.sh"
feishu_trap_begin "$(basename "$0")"
feishu_notify_start "gemma-math-crispEdit" 
# Runtime defaults mirror the corresponding FFT best script.
output_home="./"
master_port=16965
model="${ROOT_DIR}/models/google/gemma-3-1b-pt"
model_abbr="gemma-3-1b"

# Grab 1 GPU for small models (1B)
grab_gpus 1 40000 100 3
train_gpus="$CUDA_VISIBLE_DEVICES"
test_gpus="$CUDA_VISIBLE_DEVICES"
inference_bs=800
per_bs=8
delete_adapter_after_test=false
run_world_knowledge=true
world_knowledge_limit=50
model_max_length=512

# Configuration: Gemma math — CrispEdit baseline (full model, fixed K-FAC cache).
seeds=(1)
rank=128
bs=64
task="metamath"
peft_methods=("CrispEdit")

# CrispEdit: fixed FineFineWeb K-FAC cache (prepared via
# `python utils/save_finefineweb.py`); target modules default to
# q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj.
crispedit_energy_ratio=0.9
crispedit_num_chunks=1
crispedit_data_path="./lora-dataset/corda/finefineweb_validation.jsonl"

declare -A peft_lrs
peft_lrs["CrispEdit"]="2.0000e-5 3.5566e-5 6.3246e-5 1.1247e-4"

source "$(dirname "${BASH_SOURCE[0]}")/../run_best_common.sh"
