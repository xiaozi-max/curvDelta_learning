#!/bin/bash

source "$(git rev-parse --show-toplevel)/common_utils/server_env.sh"
set -e

# Runtime defaults mirror the corresponding non-best script.
output_home="./"
master_port=16925
model="${ROOT_DIR}/models/meta-llama/Llama-2-7b-hf"
model_abbr="Llama-2-7b"
train_gpus="0,1,2,3"
test_gpus="0,1,2,3"
inference_bs=800
per_bs=4
delete_adapter_after_test=true
model_max_length=512

# Paper-best configuration: Llama code, Table 13.
seeds=(1)
rank=32
bs=16
task="python"
peft_methods=("LoRA" "PiSSA" "MiLoRA")

declare -A peft_lrs
peft_lrs["LoRA"]="3.5566e-4"
peft_lrs["PiSSA"]="2.0000e-4"
peft_lrs["MiLoRA"]="6.3246e-4"

source "$(dirname "${BASH_SOURCE[0]}")/../run_best_common.sh"
