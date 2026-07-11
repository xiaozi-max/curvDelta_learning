#!/bin/bash
source "common_utils/server_env.sh"
set -e

# Minimal base-model evaluator (no training).
# Usage:
#   bash scripts/eval_base.sh --model "gemma-3-1b-pt" --model_abbr gemma-3-1b --task metamath
#   bash scripts/eval_base.sh --model Qwen/Qwen3-0.6B-Base --model_abbr qwen-3-0.6b --task python

model=""; model_abbr=""; task=""; inference_bs=400; gpus=""
while [[ $# -gt 0 ]]; do
  case "$1" in --model) model="$2"; shift 2;; --model_abbr) model_abbr="$2"; shift 2;;
    --task) task="$2"; shift 2;; --bs) inference_bs="$2"; shift 2;; --gpus) gpus="$2"; shift 2;;
    *) echo "Unknown: $1"; exit 1;; esac
done

[[ -z "$gpus" ]] && { grab_gpus 1 40000 100 3; gpus="$CUDA_VISIBLE_DEVICES"; }

data="./lora-dataset/pissa-dataset"
output_path=$(readlink -m "./output_base/${task}-Base-${model_abbr}")
mkdir -p "$output_path"
resp_file="${output_path}/${task}_response.jsonl"
echo ">>> Base eval: $model_abbr / $task -> $output_path"

# ===== Math / Code =====
python3 utils/gen_vllm_lora/math_code.py \
  --model "${ROOT_DIR}/models/$model" --sub_task "$task" --data_path "$data" \
  --batch_size $inference_bs --output_file "$resp_file" \
  --gpus "$gpus" --temp_path "${output_path}/temp_merged"

if [ "$task" = "python" ]; then
  python3 utils/code_process.py --path "$resp_file"
  evalplus.evaluate --dataset humaneval --samples "${output_path}/humaneval.jsonl" 2>&1 | grep pass@1
  evalplus.evaluate --dataset mbpp --samples "${output_path}/mbpp.jsonl" 2>&1 | grep pass@1
else
  python3 utils/test_acc.py --input_file "$resp_file"
fi

# ===== World Knowledge =====
echo ">>> World Knowledge (lm-eval)"
python3 -c "
import lm_eval, json
r = lm_eval.simple_evaluate(model='vllm',
  model_args='pretrained=${model},tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.7,max_model_len=1024,max_num_seqs=8',
  tasks=['triviaqa','nq_open','webqs'], num_fewshot=5, batch_size='auto', device='cuda:0')
s = {t: r['results'][t] for t in ['triviaqa','nq_open','webqs']}
json.dump(s, open('${output_path}/world_knowledge.json','w'), indent=2)
for t,v in s.items(): print(f'  {t}: {v}')
"
