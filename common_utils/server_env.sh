#!/usr/bin/env bash
# common_utils/server_env.sh — centralised server detection + common env vars.
#
# Source this at the top of any script with one line:
#
#   source "$(git rev-parse --show-toplevel)/common_utils/server_env.sh"
#
# After sourcing, ROOT_DIR points to your home/data root on the current server.

# ── Server / ROOT_DIR detection ───────────────────────────────────────────────
DEFAULT_GPU_IDS="0 1 2 3 4 5 6 7"
DEFAULT_MAX_PROC=0

if   [[ -d "/nfsshare/home/xiaozeguan" ]]; then
    ROOT_DIR="/nfsshare/home/xiaozeguan"
elif [[ -d "/volume/pt-train/users/wzhang/ghchen/xzg" ]]; then
    ROOT_DIR="/volume/pt-train/users/wzhang/ghchen/xzg"
elif [[ -d "/data/xzg" ]]; then
    ROOT_DIR="/data/xzg"
elif [[ -d "/data1/xzg" ]]; then
    ROOT_DIR="/data1/xzg"
elif [[ -d "/data/laipeng/xzg" ]]; then
    ROOT_DIR="/data/laipeng/xzg"
    DEFAULT_GPU_IDS="0 1"
    DEFAULT_MAX_PROC=8
    # export NCCL_P2P_DISABLE=1  # need this for some unknown reason to avoid hangs
    export NCCL_P2P_LEVEL=PXB  # maybe faster than NCCL_P2P_DISABLE=1
    export HF_HOME="/data/laipeng/xzg/huggingface_cache"
elif [[ -d "/HOME/sustc_ghchen/sustc_ghchen_6" ]]; then
    ROOT_DIR="/HOME/sustc_ghchen/sustc_ghchen_6/HDD_POOL"
elif [[ -d "/home/xzg" ]]; then
    ROOT_DIR="/home/xzg"
elif [[ -d "/root" ]]; then
    ROOT_DIR="/data"
elif [[ -d "/Users/xiaozg" ]]; then
    ROOT_DIR="/Users/xiaozg"
else
    echo "server_env.sh: Error: cannot detect ROOT_DIR for this server" >&2
    exit 1
fi

export ROOT_DIR
export DEFAULT_GPU_IDS
export DEFAULT_MAX_PROC

# ── Common environment variables ─────────────────────────────────────────────
export WANDB_BASE_URL="https://api.bandw.top"

LOCAL_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/local_env.sh"
if [[ -f "${LOCAL_ENV}" ]]; then
    source "${LOCAL_ENV}"
fi
unset LOCAL_ENV

# ── GPU grabber ───────────────────────────────────────────────────────────────
# Usage: grab_gpus [N=1] [MEM_MB=30000] [MAX_UTIL=10] [MAX_PROC=2] [GPU_IDS="$DEFAULT_GPU_IDS"] [WAIT_SEC=60]
# Sets and exports CUDA_VISIBLE_DEVICES once N qualifying GPUs are found.
# Example: grab_gpus 2 40000              # wait for 2 GPUs with ≥40 GB free
#          grab_gpus 1 30000 10 0         # only idle GPUs (0 processes)
#          grab_gpus 1 30000 10 0 "0 1 2" # restrict to GPUs 0,1,2
#          grab_gpus 2 40000 10 2 "4 5 6 7" 120  # custom GPU set, 120s retry
grab_gpus() {
    local n=${1:-1}
    local mem=${2:-30000}
    local util=${3:-10}
    local proc=${4:-$DEFAULT_MAX_PROC}
    local ids=${5:-"$DEFAULT_GPU_IDS"}
    local wait=${6:-60}
    local _repo; _repo="$(git rev-parse --show-toplevel)"
    local res_ids

    while true; do
        # shellcheck disable=SC2086
        res_ids=$(python "$_repo/common_utils/get_gpu_ids.py" \
            -N "$n" -M "$mem" -U "$util" -I $ids -P "$proc" -v)
        if [[ -n "$res_ids" ]]; then
            break
        fi
        printf "\r%s: No GPU available. Checking again in %d second(s). " "$(date)" "$wait"
        sleep "$wait"
    done
    echo "GPU(id=$res_ids) is available!"
    export CUDA_VISIBLE_DEVICES="$res_ids"
}
