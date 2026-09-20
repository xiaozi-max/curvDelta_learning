#!/bin/bash

source "common_utils/server_env.sh"
set -e
source "common_utils/feishu_notify.sh"
feishu_trap_begin "$(basename "$0")"
feishu_notify_start "llama-math-crispEdit, start" 
# Runtime defaults mirror the corresponding FFT best script.
output_home="./"
master_port=16964
model="${ROOT_DIR}/models/meta-llama/Llama-2-7b-hf"
model_abbr="Llama-2-7b"

# Grab 4 GPUs for 7B models; train.sh picks ZeRO-3 when world_size >= 2.
grab_gpus 4 40000 100 3
train_gpus="$CUDA_VISIBLE_DEVICES"
test_gpus="$CUDA_VISIBLE_DEVICES"
inference_bs=800
per_bs=2
delete_adapter_after_test=false
run_world_knowledge=true
world_knowledge_limit=100
model_max_length=512
# crispedit_num_chunks: split each rank's target modules into 4 chunks and keep
# the K-FAC A/S accumulators of the inactive chunks on the CPU. This lowers the
# collection-time GPU peak, which is what overlaps with the training-resident
# state at the collection -> training transition.
crispedit_num_chunks=4
# Configuration: Llama math — CrispEdit baseline (full model, fixed K-FAC cache).
seeds=(1)
rank=128
bs=16
task="metamath"
peft_methods=("CrispEdit")

# CrispEdit: fixed FineFineWeb K-FAC cache (prepared via
# `python utils/save_finefineweb.py`); target modules default to
# q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj.
crispedit_energy_ratio=0.9
crispedit_data_path="./lora-dataset/corda/finefineweb_validation.jsonl"

declare -A peft_lrs
peft_lrs["CrispEdit"]="2.0000e-5 3.5566e-5 6.3246e-5 1.1247e-4"

source "$(dirname "${BASH_SOURCE[0]}")/../run_best_common.sh"
