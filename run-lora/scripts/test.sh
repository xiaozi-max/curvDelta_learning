#!/bin/bash

source "common_utils/server_env.sh"

# =====================
# default values
# =====================
model="${ROOT_DIR}/models/meta-llama/Llama-2-7b-hf"   # model path
model_abbr="Llama-2-7b"            # model abbr
data_path="pissa-dataset"          # data path (task-dependent; see --help)
task="metamath"                    # task name
peft="LoRA"                        # peft method
rank=128                           # lora rank
bs=400                             # inference batch size
output_home="./"                   # reroute the whole "output" folder to other place
output_path=""                     # path for trained adapter  (REQUIRED)
temp_path=""                       # path for saving temporary merged model (REQUIRED)
gpus="0,1,2,3"                     # GPU ids
timestamp=""                       # timestamp (auto-generate if empty)

# =====================
# help function
# =====================
show_help() {
cat << EOF
Usage: $0 [options]

Options:
  --model NAME       Model name (default: $model)
  --model_abbr NAME  Model abbreviation (default: $model_abbr)
  --data_path PATH   Data path; task-dependent:
                       math/code  → HuggingFace dataset name, e.g. pissa-dataset
                       commonsense→ local dir containing {task}/test.json, e.g. lora-dataset/llm-adapters-dataset
                       instruction→ local dir containing IFEval/input_data.jsonl
                     (default: $data_path)
  --task NAME        Task name: metamath, python, commonsense, instruction (default: "$task")
  --peft NAME        PEFT method: LoRA, CurvatureLoRA, Corda-KPM, Corda-IPM, PiSSA, MiLoRA, DoRA, OLoRA, InitAB, LoRA-GA, GraLoRA, RandLoRA, LoFT (default: "$peft")
  --rank INT         LoRA rank (default: $rank)
  --bs INT           Inference batch size (default: $bs)
  --output_home PATH Reroute the whole "output" folder to other place (default: $output_home)
  --output_path PATH Output path for trained adapter (REQUIRED)
  --temp_path PATH   Path for saving temporary merged model (REQUIRED)
  --gpus LIST        Comma-separated GPU ids (default: $gpus)
  --timestamp STR    Timestamp string; if omitted, auto-generates as YYYYmmdd-HHMMSS
  --help             Show this help message and exit
EOF
}

# =====================
# parse args
# =====================
TEMP=$(getopt -o '' \
  --long model:,model_abbr:,data_path:,task:,peft:,rank:,bs:,output_home:,output_path:,temp_path:,gpus:,timestamp:,help \
  -n "$0" -- "$@")
if [ $? != 0 ]; then
    echo "Error parsing options." >&2
    exit 1
fi

eval set -- "$TEMP"
while true; do
  case "$1" in
    --model)        model="$2"; shift 2 ;;
    --model_abbr)   model_abbr="$2"; shift 2 ;;
    --data_path)    data_path="$2"; shift 2 ;;
    --task)         task="$2"; shift 2 ;;
    --peft)         peft="$2"; shift 2 ;;
    --rank)         rank="$2"; shift 2 ;;
    --bs)           bs="$2"; shift 2 ;;
    --output_home)  output_home="$2"; shift 2 ;;
    --output_path)  output_path="$2"; shift 2 ;;
    --temp_path)    temp_path="$2"; shift 2 ;;
    --gpus)         gpus="$2"; shift 2 ;;
    --timestamp)    timestamp="$2"; shift 2 ;;
    --help)         show_help; exit 0 ;;
    --) shift; break ;;
    *) echo "Internal error!"; exit 1 ;;
  esac
done

# =====================
# validations
# =====================
if [[ -z "$output_path" ]]; then
  echo "Error: --output_path is required." >&2; exit 2
fi
if [[ ! -d "$output_path" ]]; then
  echo "Error: output_path does not exist: $output_path" >&2; exit 1
fi
if [[ ! "$peft" =~ ^(LoRA|CurvatureLoRA|Corda-KPM|Corda-IPM|PiSSA|MiLoRA|DoRA|OLoRA|InitAB|LoRA-GA|GraLoRA|RandLoRA|LoFT)$ ]]; then
  echo "Error: --peft must be one of {LoRA, CurvatureLoRA, Corda-KPM, Corda-IPM, PiSSA, MiLoRA, DoRA, OLoRA, InitAB, LoRA-GA, GraLoRA, RandLoRA, LoFT}." >&2
  exit 2
fi

world_size=$(awk -F',' '{print NF}' <<< "$gpus")
mod=$(( 32 % world_size ))
if (( mod != 0 )); then
  echo "Error: total attention heads (32) is not divisible by num_GPUs ($world_size)" >&2
  exit 1
fi

# Clean task name (strip sub-task suffix and epoch tag)
echo "original task name: $task"
task=$(echo "${task%%:*}" | sed -E 's/-ep[0-9]+$//')
echo "cleaned task name: $task"

## echo settings ##
echo "===========Testing=============="
echo "Using output_path: $output_path"
echo "Using GPUs: $gpus (num_GPUs=$world_size)"
echo "Using PEFT method: $peft"
echo "Base model abbr: $model_abbr"
echo "Task: $task, Rank: $rank"
echo "Batch size = $bs"
echo "================================"

# =====================
# resolve base_model and lora_path
# =====================
resp_file="$output_path/${task}_response.jsonl"
lora_path="$output_path/adapter_model"   # default for all LoRA variants

if [[ "$peft" == "PiSSA" ]]; then
  base_model=$(readlink -m "${output_home}/output/PiSSA-${model_abbr}-r${rank}")

elif [[ "$peft" == "MiLoRA" ]]; then
  base_model=$(readlink -m "${output_home}/output/MiLoRA-${model_abbr}-r${rank}")

elif [[ "$peft" == "OLoRA" ]]; then
  base_model="$output_path/olora_residual_model"

elif [[ "$peft" == "InitAB" ]]; then
  base_model=$(readlink -m "${output_home}/output/InitAB-${model_abbr}-r${rank}-${timestamp}")

elif [[ "$peft" == "LoRA-GA" ]]; then
  base_model=$(readlink -m "${output_home}/output/LoRA-GA-${model_abbr}-r${rank}-${task}")
elif [[ "$peft" == "Corda-KPM" ]]; then
  base_model=$(readlink -m "${output_home}/output/Corda-KPM-${model_abbr}-r${rank}-${task}")
elif [[ "$peft" == "Corda-IPM" ]]; then
  base_model=$(readlink -m "${output_home}/output/Corda-IPM-${model_abbr}-r${rank}-${task}")
else
  # LoRA, CurvatureLoRA, DoRA, GraLoRA, LoFT — base model with standard adapter
  base_model="$model"
fi

# =====================
# vLLM inference (unified dispatch)
# =====================

# Build optional --lora argument
lora_args=()
if [[ -n "$lora_path" ]]; then
  lora_args=(--lora "$lora_path")
fi

if [ "$task" == "commonsense" ]; then
  python3 utils/gen_vllm_lora/cs.py \
    --model "$base_model" \
    "${lora_args[@]}" \
    --data_path "$data_path" \
    --batch_size $bs \
    --output_file "$resp_file" \
    --gpus "$gpus" \
    --temp_path "$temp_path"

elif [ "$task" == "instruction" ]; then
  python3 utils/gen_vllm_lora/inst.py \
    --model "$base_model" \
    "${lora_args[@]}" \
    --data_path "$data_path" \
    --output_file "$resp_file" \
    --batch_size $bs \
    --gpus "$gpus" \
    --temp_path "$temp_path"

else
  python3 utils/gen_vllm_lora/math_code.py \
    --model "$base_model" \
    "${lora_args[@]}" \
    --sub_task "$task" \
    --data_path "$data_path" \
    --batch_size $bs \
    --output_file "$resp_file" \
    --gpus "$gpus" \
    --temp_path "$temp_path"
fi

# =====================
# scoring
# =====================
if [ "$task" = "python" ]; then
  python3 utils/code_process.py --path "$resp_file"

  readarray -t scores < <(evalplus.evaluate --dataset humaneval --samples "$output_path/humaneval.jsonl" 2>&1 \
                          | grep "pass@1:" | awk '{print $2}')
  humaneval_base=${scores[0]}
  humaneval_extra=${scores[1]}

  readarray -t scores < <(evalplus.evaluate --dataset mbpp --samples "$output_path/mbpp.jsonl" 2>&1 \
                          | grep "pass@1:" | awk '{print $2}')
  mbpp_base=${scores[0]}
  mbpp_extra=${scores[1]}

  cat > "$output_path/perf.json" <<EOF
{
  "humaneval": [$humaneval_base, $humaneval_extra],
  "mbpp": [$mbpp_base, $mbpp_extra]
}
EOF
  echo "✅ Scores saved to $output_path/perf.json"

elif [ "$task" = "instruction" ]; then
  eval_output=$(python3 utils/IFEval/evaluation_main.py \
    --input_data="$data_path/IFEval/input_data.jsonl" \
    --input_response_data="$resp_file" \
    --output_dir="$output_path" 2>&1)

  echo "$eval_output"

  ifeval_strict_prompt=$(echo "$eval_output" | grep -m 1 "prompt-level:" | awk '{print $2}')
  ifeval_strict_inst=$(echo "$eval_output" | grep -m 1 "instruction-level:" | awk '{print $2}')
  cat > "$output_path/perf.json" <<EOF
{
  "ifeval_strict_prompt": $ifeval_strict_prompt,
  "ifeval_strict_inst": $ifeval_strict_inst
}
EOF
  echo "✅ IFEval Scores saved to $output_path/perf.json"

else
  python3 utils/test_acc.py \
    --input_file "$resp_file"
fi
