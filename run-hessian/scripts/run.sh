#!/bin/bash

source "$(git rev-parse --show-toplevel)/common_utils/server_env.sh"

# Default values
CUDA_DEVICES="0,1"
MODEL="${ROOT_DIR}/models/Qwen/Qwen3-0.6B-Base"
DATA_PATH="../run-lora/lora-dataset/pissa-dataset/metamath/train.json"
DATA_FIELD="instruction,output"
SCAN_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
SAMPLES=500
MODEL_INPUT_BS=5
ITER=300
RANK=128
TOL="5e-3"
PEFT="lora"

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --model          MODEL_ID      HuggingFace model ID or local path (default: $MODEL)"
    echo "  --data_path      PATH          Path to dataset JSON file (default: $DATA_PATH)"
    echo "  --dataset_field  FIELDS        Comma-separated query,response field names (default: $DATA_FIELD)"
    echo "  --scan_modules   MODULES       Comma-separated module names to scan (default: $SCAN_MODULES)"
    echo "  --b              N             Number of data samples (default: $SAMPLES)"
    echo "  --model_input_bs N             Per-device batch size for inference (default: $MODEL_INPUT_BS)"
    echo "  --max_iter       N             Maximum Lanczos iterations (default: $ITER)"
    echo "  --rank           N             LoRA rank (default: $RANK)"
    echo "  --tol            FLOAT         Convergence tolerance (default: $TOL)"
    echo "  --peft           METHOD        PEFT method, e.g. lora, olora, pissa, milora, loraga (default: $PEFT)"
    echo "  --cuda_devices   IDS           CUDA_VISIBLE_DEVICES value (default: $CUDA_DEVICES)"
    echo "  -h, --help                     Show this help message"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)          MODEL="$2";          shift 2 ;;
        --data_path)      DATA_PATH="$2";      shift 2 ;;
        --dataset_field)  DATA_FIELD="$2";     shift 2 ;;
        --scan_modules)   SCAN_MODULES="$2";   shift 2 ;;
        --b)              SAMPLES="$2";        shift 2 ;;
        --model_input_bs) MODEL_INPUT_BS="$2"; shift 2 ;;
        --max_iter)       ITER="$2";           shift 2 ;;
        --rank)           RANK="$2";           shift 2 ;;
        --tol)            TOL="$2";            shift 2 ;;
        --peft)           PEFT="$2";           shift 2 ;;        # ← 新增
        --cuda_devices)   CUDA_DEVICES="$2";   shift 2 ;;
        -h|--help)        usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

python3 lanczos.py \
    --model "$MODEL" \
    --data_path "$DATA_PATH" \
    --dataset_field "$DATA_FIELD" \
    --scan_modules "$SCAN_MODULES" \
    --b $SAMPLES \
    --model_input_bs $MODEL_INPUT_BS \
    --max_iter $ITER \
    --peft "$PEFT" \
    --rank $RANK \
    --tol $TOL
