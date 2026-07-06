#!/usr/bin/env bash

source "$(git rev-parse --show-toplevel)/common_utils/server_env.sh"

: "${HF_TOKEN:?Set HF_TOKEN before running download_cli.sh}"

# Downloads a HuggingFace repo.
# Usage: hf_download <repo_type> <local_dir> <repo_id> [exclude_pattern] [revision]
hf_download() {
  local repo_type="$1"   # "model" or "dataset"
  local local_dir="$2"
  local repo_id="$3"
  local exclude="${4:-}"  # optional glob pattern, e.g. "*.pth" or "*.{pth,bin}"
  local revision="${5:-}" # optional revision, e.g. "pytorch_model.260000"


  echo "==> Downloading ${repo_type}: ${repo_id}"

  local cmd=(hf download --token "${HF_TOKEN}" "${repo_id}" --local-dir "${local_dir}/${repo_id}")

  [[ "${repo_type}" == "dataset" ]] && cmd+=(--repo-type dataset)
  [[ -n "${exclude}"  ]] && cmd+=(--exclude "${exclude}")
  [[ -n "${revision}" ]] && cmd+=(--revision "${revision}")

  "${cmd[@]}"
}

# ─────────────────────────────────────────────
### Models ###
# ─────────────────────────────────────────────
MODEL_DIR="${ROOT_DIR}/models"

# Each entry: "REPO_ID|EXCLUDE|REVISION"
# Leave EXCLUDE or REVISION empty to skip that flag.
MODEL_ENTRIES=(
  "Qwen/Qwen3-0.6B-Base||"
  "google/gemma-3-1b-pt||"
  # "meta-llama/Llama-2-7b-hf|*.{bin}|"
)

for entry in "${MODEL_ENTRIES[@]}"; do
  IFS='|' read -r repo_id exclude revision <<< "${entry}"
  hf_download "model" "${MODEL_DIR}" "${repo_id}" "${exclude}" "${revision}"
done

# ─────────────────────────────────────────────
### Datasets ###
# ─────────────────────────────────────────────
DATASET_DIR="${ROOT_DIR}/datasets"

# Each entry: "REPO_ID|EXCLUDE"
DATASET_ENTRIES=(
  # "cais/mmlu|"
  # "cais/wmdp|"
)

for entry in "${DATASET_ENTRIES[@]}"; do
  IFS='|' read -r repo_id exclude <<< "${entry}"
  hf_download "dataset" "${DATASET_DIR}" "${repo_id}" "${exclude}"
done
