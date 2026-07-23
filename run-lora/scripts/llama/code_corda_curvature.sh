#!/bin/bash

source "common_utils/server_env.sh"
set -e

# Runtime defaults mirror the corresponding best script.
output_home="./"
master_port=16930
model="${ROOT_DIR}/models/meta-llama/Llama-2-7b-hf"
model_abbr="Llama-2-7b"

# Grab 2 GPUs for large models (7B)
grab_gpus 2 40000 100 3
train_gpus="$CUDA_VISIBLE_DEVICES"
test_gpus="$CUDA_VISIBLE_DEVICES"
inference_bs=800
per_bs=4
delete_adapter_after_test=false
run_world_knowledge=true
world_knowledge_limit=100
model_max_length=512

# Configuration: Llama code — Corda (KPM/IPM) and CurvatureLoRA with LR sweep.
seeds=(1)
rank=32
bs=16
task="python"
# peft_methods=("Corda-KPM" "Corda-IPM" "CurvatureLoRA")
peft_methods=("CurvatureLoRA")
curvature_num_chunks=2  # split KFAC into 4 chunks to fit 2×L40
curvature_energy_ratio=0.7

declare -A peft_lrs
peft_lrs["Corda-KPM"]="2.0000e-5 6.3246e-5 1.1247e-4 2.0000e-4"
peft_lrs["Corda-IPM"]="2.0000e-5 6.3246e-5 1.1247e-4 2.0000e-4"
peft_lrs["CurvatureLoRA"]="1.1247e-4 2.0000e-4 3.5566e-4 6.3246e-4 1.1247e-3"

source "$(dirname "${BASH_SOURCE[0]}")/../run_best_common.sh"
