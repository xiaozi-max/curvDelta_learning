#!/usr/bin/env bash

source "$(git rev-parse --show-toplevel)/common_utils/server_env.sh"
set -e

output_home="./"
master_port=17001
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
peft_methods=("LoRA" "PiSSA" "MiLoRA")
curvature_energy_ratio=0.7

declare -A peft_lrs
peft_lrs["LoRA"]="1.1247e-5 2.0000e-5 3.5566e-5 6.3246e-5 1.1247e-4 2.0000e-4 3.5566e-4 6.3246e-4"
peft_lrs["PiSSA"]="1.1247e-5 2.0000e-5 3.5566e-5 6.3246e-5 1.1247e-4 2.0000e-4 3.5566e-4 6.3246e-4"
peft_lrs["MiLoRA"]="1.1247e-5 2.0000e-5 3.5566e-5 6.3246e-5 1.1247e-4 2.0000e-4 3.5566e-4 6.3246e-4"

source "$(dirname "${BASH_SOURCE[0]}")/../run_best_common.sh"

