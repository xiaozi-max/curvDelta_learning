#!/bin/bash

source "common_utils/server_env.sh"
set -e

# Runtime defaults mirror the corresponding best script.
output_home="./"
master_port=16942
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

# Trick 1+2: constant_with_warmup + trust-region (alpha_tr sweep: 0.001, 0.01, 0.1, 0.25)
curvature_lr_scheduler="constant_with_warmup"
curvature_energy_ratio=0.7

# Configuration: Gemma math — CurvatureLoRA with cww + trust-region.
seeds=(1)
rank=128
bs=64
task="metamath"
peft_methods=("CurvatureLoRA")

declare -A peft_lrs
peft_lrs["CurvatureLoRA"]="2.0000e-4 3.5566e-4 6.3246e-4"

# TR sweep: each alpha_tr value is a separate experiment.
# Run via: curvature_alpha_tr=0.001 bash scripts/gemma/math_curvature_cww_tr.sh
curvature_alpha_tr="${curvature_alpha_tr:-0.001}"
curvature_delta_tr=1e-12

source "$(dirname "${BASH_SOURCE[0]}")/../run_best_common.sh"
