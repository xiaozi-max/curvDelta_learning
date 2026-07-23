#!/bin/bash

source "common_utils/server_env.sh"
set -e

# Runtime defaults mirror the corresponding best script.
output_home="./"
master_port=16925
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

# Configuration: Gemma math — Corda (KPM/IPM) and CurvatureLoRA with LR sweep.
seeds=(1)
rank=128
bs=64
task="metamath"
peft_methods=("Corda-KPM" "Corda-IPM" "CurvatureLoRA")
curvature_energy_ratio=0.7

declare -A peft_lrs
peft_lrs["Corda-KPM"]="2.0000e-5 6.3246e-5 1.1247e-4 2.0000e-4"
peft_lrs["Corda-IPM"]="2.0000e-5 6.3246e-5 1.1247e-4 2.0000e-4"
peft_lrs["CurvatureLoRA"]="2.0000e-5 6.3246e-5 1.1247e-4 2.0000e-4 3.5566e-4 6.3246e-4"

source "$(dirname "${BASH_SOURCE[0]}")/../run_best_common.sh"
