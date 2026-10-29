#!/usr/bin/env bash
# Portable storage, credentials, and GPU allocation for public launchers.
_ENV_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_DATA_ROOT="${PROJECT_DATA_ROOT:-${_ENV_REPO_ROOT}/data}"
export ROOT_DIR="$PROJECT_DATA_ROOT"
export HF_HOME="${HF_HOME:-${PROJECT_DATA_ROOT}/.cache/huggingface}"
export DEFAULT_GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"
export HF_TOKEN="${HF_TOKEN:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-curvdelta-learning}"

grab_gpus() {
    local count="${1:-1}"
    export CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES="$(python "${_ENV_REPO_ROOT}/common_utils/get_gpu_ids.py" \
        -N "$count" -M "${2:-30000}" -U "${3:-100}" \
        -P "${4:-0}" -I ${5:-$DEFAULT_GPU_IDS})"
    [[ -n "$CUDA_VISIBLE_DEVICES" ]] || {
        echo "No requested GPU is currently available. Set GPU_IDS explicitly." >&2
        return 1
    }
}
