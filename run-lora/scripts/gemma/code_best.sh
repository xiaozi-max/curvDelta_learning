#!/bin/bash

source "common_utils/server_env.sh"
set -e

# Comprehensive Gemma-3-1B code script: all PEFT methods in one place.
# Unlike llama (7B), Gemma-3-1B is small -> 1 GPU, rank=128, bs=64
# (see practical-heuristics Figure P4: Gemma-3-1B code, B=64).
output_home="./"
master_port=16955
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

# Configuration: Gemma code (python) — LR sweep over all PEFT methods.
# rank=128 / bs=64 mirror gemma math (same model, same B). The Corda and
# CurvatureLoRA sweeps are the model-specific ranges from gemma
# math_corda_curvature.sh; CurvatureDense / FFT keep their fixed grid.
seeds=(1)
rank=128
bs=64
task="python"
peft_methods=("LoRA" "PiSSA" "MiLoRA" "Corda-KPM" "Corda-IPM" "CurvatureLoRA" "CurvatureDense" "FFT")
curvature_energy_ratio=0.7

declare -A peft_lrs
# LoRA / PiSSA / MiLoRA: code optimum unknown for gemma, sweep around the
# gemma-math paper-best lrs (LoRA=6.3246e-4, PiSSA=2.0000e-4, MiLoRA=3.5566e-4).
peft_lrs["LoRA"]="2.0000e-4 3.5566e-4 6.3246e-4"
peft_lrs["PiSSA"]="1.1247e-4 2.0000e-4 3.5566e-4"
peft_lrs["MiLoRA"]="1.1247e-4 2.0000e-4 3.5566e-4 "
# Corda (KPM/IPM) — same sweep as gemma math_corda_curvature.sh.
peft_lrs["Corda-KPM"]="6.3246e-5 1.1247e-4 2.0000e-4"
peft_lrs["Corda-IPM"]="6.3246e-5 1.1247e-4 2.0000e-4"
# CurvatureLoRA — gemma uses a lower range than llama (7B).
peft_lrs["CurvatureLoRA"]="1.1247e-4 2.0000e-4 3.5566e-4 6.3246e-4"
# CurvatureDense / FFT — shared fixed grid (task- and model-independent).
peft_lrs["CurvatureDense"]="2.0000e-5 3.5566e-5 6.3246e-5"
peft_lrs["FFT"]="2.0000e-5 3.5566e-5 6.3246e-5"

source "$(dirname "${BASH_SOURCE[0]}")/../run_best_common.sh"
