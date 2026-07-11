#!/bin/bash

source "common_utils/server_env.sh"
set -e

# Runtime defaults mirror the corresponding non-best script.
output_home="./"
master_port=16928
model="${ROOT_DIR}/models/Qwen/Qwen3-0.6B-Base"
model_abbr="qwen-3-0.6b"

# Grab 1 GPU for small models (0.6B)
grab_gpus 1 40000 100 3
train_gpus="$CUDA_VISIBLE_DEVICES"
test_gpus="$CUDA_VISIBLE_DEVICES"
inference_bs=800
per_bs=4
delete_adapter_after_test=false
run_world_knowledge=true
world_knowledge_limit=100
model_max_length=512

# Paper-best configuration: Qwen math, Table 7 / README.
seeds=(1)
rank=128
bs=64
task="metamath"
peft_methods=("LoRA" "PiSSA" "MiLoRA")

declare -A peft_lrs
peft_lrs["LoRA"]="2.0000e-4"
peft_lrs["PiSSA"]="2.0000e-5"
peft_lrs["MiLoRA"]="6.3246e-5"

source "$(dirname "${BASH_SOURCE[0]}")/../run_best_common.sh"
