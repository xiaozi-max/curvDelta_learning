#!/bin/bash

source "common_utils/server_env.sh"
set -e

# Runtime defaults mirror the corresponding best script.
output_home="./"
master_port=16962
model="${ROOT_DIR}/models/meta-llama/Llama-2-7b-hf"
model_abbr="Llama-2-7b"

# Grab 4 GPUs for large models (7B) — ZeRO-3 (optimizer offload) for FFT.
# 7B BF16 params (~13GB) + gradients (~13GB) sharded over 4 GPUs;
# fp32 master + AdamW states (~80GB) offloaded to CPU.
grab_gpus 4 40000 100 3
train_gpus="$CUDA_VISIBLE_DEVICES"
test_gpus="$CUDA_VISIBLE_DEVICES"
inference_bs=800
per_bs=1
delete_adapter_after_test=false
run_world_knowledge=true
world_knowledge_limit=100
model_max_length=512

# Configuration: Llama math — Full Fine-Tuning baseline.
# per_bs=1 because 7B FFT with ZeRO-3 is memory-intensive.
seeds=(1)
bs=16
task="metamath"
peft_methods=("FFT")

declare -A peft_lrs
peft_lrs["FFT"]="2.0000e-5 3.5566e-5 6.3246e-5 1.1247e-4"

source "$(dirname "${BASH_SOURCE[0]}")/../run_best_common.sh"
