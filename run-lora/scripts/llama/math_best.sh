#!/bin/bash

source "common_utils/server_env.sh"
set -e

# Runtime defaults mirror the corresponding non-best script.
output_home="./"
master_port=16927
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

# Paper-best configuration: Llama math, Table 10.
seeds=(1)
rank=32
bs=16
task="metamath"
peft_methods=("LoRA" "PiSSA" "MiLoRA")

declare -A peft_lrs
peft_lrs["LoRA"]="6.3246e-4"
peft_lrs["PiSSA"]="1.1247e-4"
peft_lrs["MiLoRA"]="3.5566e-4"

source "$(dirname "${BASH_SOURCE[0]}")/../run_best_common.sh"
