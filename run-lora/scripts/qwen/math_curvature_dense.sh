#!/bin/bash

source "common_utils/server_env.sh"
set -e

# Runtime defaults mirror the corresponding best script.
output_home="./"
master_port=16950
model="${ROOT_DIR}/models/Qwen/Qwen3-0.6B-Base"
model_abbr="qwen-3-0.6b"

# Grab 1 GPU for small models (1B)
grab_gpus 1 40000 100 3
train_gpus="$CUDA_VISIBLE_DEVICES"
test_gpus="$CUDA_VISIBLE_DEVICES"
inference_bs=800
per_bs=4
delete_adapter_after_test=false
run_world_knowledge=true
world_knowledge_limit=50
model_max_length=512

# Configuration: Qwen math — CurvatureLoRA vs CurvatureDense with LR sweep.
seeds=(1)
rank=128
bs=64
task="metamath"
peft_methods=("CurvatureDense")
curvature_energy_ratio=0.7

declare -A peft_lrs
peft_lrs["CurvatureDense"]="1.1247e-4 2.0000e-4 3.5566e-4 6.3246e-4"

source "$(dirname "${BASH_SOURCE[0]}")/../run_best_common.sh"
