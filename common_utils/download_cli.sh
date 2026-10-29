#!/usr/bin/env bash
# Download the public or access-authorized checkpoints used in the paper.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/server_env.sh"

for repo_id in \
    google/gemma-3-1b-pt \
    meta-llama/Llama-2-7b-hf; do
    hf download "$repo_id" --local-dir "${PROJECT_DATA_ROOT}/models/${repo_id}" \
        --exclude "*.bin" "*.pth"
done
