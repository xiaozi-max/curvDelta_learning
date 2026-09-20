#!/bin/bash

source "common_utils/server_env.sh"
set -e

# Runtime defaults mirror math_best.sh / curvature scripts.
output_home="./"
master_port=16970
model="${ROOT_DIR}/models/google/gemma-3-1b-pt"
model_abbr="gemma-3-1b"

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


seeds=(1)
rank=128
bs=64
task="python"
peft_methods=("FFT")
curvature_energy_ratio=0.7
replay_data_path="./lora-dataset/corda/finefineweb_validation.jsonl"
replay_num_samples=256

declare -A peft_lrs
peft_lrs["FFT"]="2.0000e-5 6.3246e-5"
source "$(dirname "${BASH_SOURCE[0]}")/../run_best_common.sh"
