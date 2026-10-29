#!/bin/bash

source "$(git rev-parse --show-toplevel)/common_utils/server_env.sh"

# =====================
# default values
# =====================
model="${ROOT_DIR}/models/meta-llama/Llama-2-7b-hf"   # model path
model_abbr="Llama-2-7b"            # model name abbr
data="pissa-dataset"               # data path
task="metamath"                    # task name
peft="LoRA"                        # peft method: LoRA, CurvLoRA, CurvDelta, Corda-KPM, Corda-IPM, PiSSA, MiLoRA, DoRA, OLoRA, InitAB, LoRA-GA, GraLoRA, RandLoRA, LoFT, FFT, OPLoRA
rank=128                           # lora rank
lr=2e-5                            # learning rate
bs=128                             # total (global) batch size you want to simulate
per_bs=1                           # per device train batch size
model_max_length=512               # truncation length
gpus="0,1,2,3"                     # GPU ids
trial_id=1                         # experiment id
master_port=16971                  # if run two deepspeed exp simultaneously, set different master_port
timestamp=""                       # if empty, auto-generate later
output_home="./"                   # reroute the whole "output" folder to other place, default is "./"
output_path=""                     # the path to save exp results
gralora_k=2                        # GraLoRA k parameter
randlora_alpha=640.0               # RandLoRA alpha parameter

curvature_data_path="./lora-dataset/corda/finefineweb_validation.jsonl"  # CurvLoRA anchor JSONL (256 FineFineWeb, same as Corda)
curvature_energy_ratio=0.9  # KFAC top-eigenvector energy ratio for CurvLoRA projections
curvature_num_chunks=1  # KFAC chunk count (>1 stores matrices on CPU, lowering GPU VRAM)
curvature_alpha_tr=0.0  # trust-region alpha: ||Delta W_eff||_F <= alpha_tr * ||W_base||_F. 0 = disabled.
curvature_delta_tr=1e-12  # numerical stabilizer for trust-region denominator
curvature_lr_scheduler="cosine"  # lr_scheduler_type for CurvLoRA (cosine or constant_with_warmup)
crispedit_data_path="./lora-dataset/corda/finefineweb_validation.jsonl"  # FineFineWeb JSONL (field 'text') for CrispEdit fixed K-FAC cache
crispedit_energy_ratio=0.9  # Kronecker-eigenvalue energy ratio for the CrispEdit low-curvature mask
crispedit_collection_batch_size=1  # per-device batch size for CrispEdit KFAC collection
crispedit_num_chunks=1  # KFAC chunk count for CrispEdit (>1 stages matrices on CPU, lowering GPU VRAM)
corda_data_path="./lora-dataset/corda/finefineweb_validation.jsonl"  # Local CorDA calibration JSONL (FineFineWeb)
corda_num_samples=256                   # Number of local calibration samples for CorDA
replay_data_path=""                     # Optional raw-text JSONL (field "text") of pretraining data to mix into training (replay). Empty = disabled (original behaviour).
replay_num_samples=256                  # Number of replay samples to mix (used when replay_data_path is set)
logging_steps=20  # log every N steps to reduce terminal spam
op_lora_k=16                        # OPLoRA protected top-k singular directions of W0

# =====================
# help function
# =====================
show_help() {
cat << EOF
Usage: $0 [options]

Options:
  --model NAME       Model name (default: $model)
  --model_abbr NAME  Model abbreviation name (default: $model_abbr)
  --data PATH        Data path (default: $data)
  --task NAME        Task name, e.g., metamath, python, conversation (default: "$task")
  --peft NAME        PEFT method: LoRA, CurvLoRA, CurvDelta, CrispEdit, Corda-KPM, Corda-IPM, PiSSA, MiLoRA, DoRA, OLoRA, InitAB, LoRA-GA, GraLoRA, RandLoRA, LoFT, FFT, OPLoRA (default: "$peft")
  --rank INT         LoRA rank (default: $rank)
  --lr FLOAT         Learning rate (default: $lr)
  --bs INT           Global batch size (default: $bs)
  --per_bs INT       Per-device batch size (default: $per_bs)
  --model_max_length INT  Truncation length (default: $model_max_length)
  --gpus LIST        Comma-separated GPU ids (default: $gpus)
  --trial_id INT     Experiment ID (default: $trial_id)
  --master_port INT  Master port id (default: $master_port)
  --timestamp STR    Timestamp string; YYYYmmdd-HHMMSS
  --output_home PATH Output home directory (default: $output_home)
  --output_path PATH Specific output path for experiment results (default: auto-generated)
  --gralora_k INT    GraLoRA k parameter (default: $gralora_k)
  --randlora_alpha FLOAT  RandLoRA alpha parameter (default: $randlora_alpha)
  --curvature_data_path PATH  CurvLoRA anchor JSONL (default: $curvature_data_path)
  --curvature_energy_ratio FLOAT  KFAC energy ratio for CurvLoRA (default: $curvature_energy_ratio)
  --curvature_num_chunks INT  KFAC chunk count for CurvLoRA (default: $curvature_num_chunks)
  --curvature_alpha_tr FLOAT  Trust-region alpha (0=disabled) (default: $curvature_alpha_tr)
  --curvature_delta_tr FLOAT  Trust-region stabilizer (default: $curvature_delta_tr)
  --curvature_lr_scheduler STR  LR scheduler for CurvLoRA: cosine or constant_with_warmup (default: $curvature_lr_scheduler)
  --crispedit_data_path PATH  FineFineWeb JSONL (field 'text') for CrispEdit fixed K-FAC cache (default: $crispedit_data_path)
  --crispedit_energy_ratio FLOAT  Kronecker energy ratio for CrispEdit mask (default: $crispedit_energy_ratio)
  --crispedit_collection_batch_size INT  CrispEdit KFAC collection batch size (default: $crispedit_collection_batch_size)
  --crispedit_num_chunks INT  CrispEdit KFAC chunk count (default: $crispedit_num_chunks)
  --corda_data_path PATH  Local CorDA calibration data path (default: $corda_data_path)
  --corda_num_samples INT Number of CorDA calibration samples (default: $corda_num_samples)
  --replay_data_path PATH  Raw-text JSONL (field "text") of pretraining data to mix into training. Empty = disabled (default: $replay_data_path)
  --replay_num_samples INT Number of replay samples to mix (default: $replay_num_samples)
  --op_lora_k INT    OPLoRA protected top-k singular directions of W0 (default: $op_lora_k)
  --help             Show this help message and exit
EOF
}

# =====================
# parse long options
# =====================
TEMP=$(getopt -o '' \
  --long model:,model_abbr:,data:,task:,peft:,rank:,lr:,bs:,per_bs:,model_max_length:,gpus:,trial_id:,master_port:,timestamp:,output_home:,output_path:,gralora_k:,randlora_alpha:,curvature_data_path:,curvature_energy_ratio:,curvature_num_chunks:,curvature_alpha_tr:,curvature_delta_tr:,curvature_lr_scheduler:,crispedit_data_path:,crispedit_energy_ratio:,crispedit_collection_batch_size:,crispedit_num_chunks:,corda_data_path:,corda_num_samples:,replay_data_path:,replay_num_samples:,op_lora_k:,help \
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
    --data)         data="$2"; shift 2 ;;
    --task)         task="$2"; shift 2 ;;
    --peft)         peft="$2"; shift 2 ;;
    --rank)         rank="$2"; shift 2 ;;
    --lr)           lr="$2"; shift 2 ;;
    --bs)           bs="$2"; shift 2 ;;
    --per_bs)       per_bs="$2"; shift 2 ;;
    --model_max_length)       model_max_length="$2"; shift 2 ;;
    --gpus)         gpus="$2"; shift 2 ;;
    --trial_id)     trial_id="$2"; shift 2 ;;
    --master_port)  master_port="$2"; shift 2 ;;
    --timestamp)    timestamp="$2"; shift 2 ;;
    --output_home)  output_home="$2"; shift 2 ;;
    --output_path)  output_path="$2"; shift 2 ;;
    --gralora_k)    gralora_k="$2"; shift 2 ;;
    --randlora_alpha) randlora_alpha="$2"; shift 2 ;;
    --curvature_data_path) curvature_data_path="$2"; shift 2 ;;
    --curvature_energy_ratio) curvature_energy_ratio="$2"; shift 2 ;;
    --curvature_num_chunks) curvature_num_chunks="$2"; shift 2 ;;
    --curvature_alpha_tr) curvature_alpha_tr="$2"; shift 2 ;;
    --curvature_delta_tr) curvature_delta_tr="$2"; shift 2 ;;
    --curvature_lr_scheduler) curvature_lr_scheduler="$2"; shift 2 ;;
    --crispedit_data_path) crispedit_data_path="$2"; shift 2 ;;
    --crispedit_energy_ratio) crispedit_energy_ratio="$2"; shift 2 ;;
    --crispedit_collection_batch_size) crispedit_collection_batch_size="$2"; shift 2 ;;
    --crispedit_num_chunks) crispedit_num_chunks="$2"; shift 2 ;;
    --corda_data_path) corda_data_path="$2"; shift 2 ;;
    --corda_num_samples) corda_num_samples="$2"; shift 2 ;;
    --replay_data_path) replay_data_path="$2"; shift 2 ;;
    --replay_num_samples) replay_num_samples="$2"; shift 2 ;;
    --op_lora_k)    op_lora_k="$2"; shift 2 ;;
    --help)         show_help; exit 0 ;;
    --) shift; break ;;
    *) echo "Internal error!"; exit 1 ;;
  esac
done

# =====================
# optional replay mixing (forwarded to train.py only when a path is given,
# so experiments without mixed data run exactly as before)
# =====================
_replay_train_args=()
if [[ -n "$replay_data_path" ]]; then
  _replay_train_args+=(--replay_data_path "$replay_data_path")
  _replay_train_args+=(--replay_num_samples "$replay_num_samples")
fi

# =====================
# validations (types & allowed sets)
# =====================
if [[ ! "$peft" =~ ^(LoRA|CurvLoRA|CurvDelta|CrispEdit|Corda-KPM|Corda-IPM|PiSSA|MiLoRA|DoRA|OLoRA|InitAB|LoRA-GA|GraLoRA|RandLoRA|LoFT|FFT|OPLoRA)$ ]]; then
  echo "Error: --peft must be one of {LoRA, CurvLoRA, CurvDelta, CrispEdit, Corda-KPM, Corda-IPM, PiSSA, MiLoRA, DoRA, OLoRA, InitAB, LoRA-GA, GraLoRA, RandLoRA, LoFT, FFT, OPLoRA}." >&2
  exit 2
fi
if [[ "$peft" != "FFT" ]] && [[ ! "$rank" =~ ^[0-9]+$ ]]; then
  echo "Error: --rank must be an integer." >&2; exit 2
fi
if [[ ! "$bs" =~ ^[0-9]+$ ]]; then
  echo "Error: --bs must be an integer." >&2; exit 2
fi
if [[ ! "$per_bs" =~ ^[0-9]+$ ]]; then
  echo "Error: --per_bs must be an integer." >&2; exit 2
fi
if [[ ! "$lr" =~ ^-?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][-+]?[0-9]+)?$ ]]; then
  echo "Error: --lr must be a number (supports scientific notation)." >&2
  exit 2
fi
if [[ ! "$gpus" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "Error: --gpus must be a comma-separated list of integers, e.g., 0,1,2,3." >&2
  exit 2
fi
if [[ ! "$trial_id" =~ ^[0-9]+$ ]]; then
  echo "Error: --trial_id must be an integer." >&2
  exit 2
fi
if [[ "$peft" == "OPLoRA" ]] && [[ ! "$op_lora_k" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: --op_lora_k must be a positive integer." >&2
  exit 2
fi


# =====================
# task parsing with optional num_samples / epochs
# =====================

# keep the raw input for parsing
task_input="$task"

# default epoch
num_train_epochs=1

# avialable format：
#   python:1000-ep3   → base=python, num_samples=1000, epochs=3
#   python-ep3        → base=python, num_samples="",   epochs=3
#   metamath:5000-ep2 → base=metamath, num_samples=5000, epochs=2
if [[ "$task_input" =~ ^([a-zA-Z0-9_-]+)(:([0-9]+))?-ep([0-9]+)$ ]]; then
  base="${BASH_REMATCH[1]}"
  num_samples="${BASH_REMATCH[3]}"
  epochs="${BASH_REMATCH[4]}"

  if [[ -n "$num_samples" ]]; then
    task="${base}:${num_samples}"
  else
    task="${base}"
  fi

  num_train_epochs="$epochs"
  task_base="$base"
  
  # metamath default use 100k subset for training if no num_samples specified
  if [[ "$task_base" == "metamath" && "$task" == "metamath" ]]; then
    task="metamath:100000"
  fi
  # ===================

else
  if [[ "$task_input" =~ ^([a-zA-Z0-9_-]+):([0-9]+)$ ]]; then
    task_base="${BASH_REMATCH[1]}"
    num_samples="${BASH_REMATCH[2]}"
    task="${task_base}:${num_samples}"
  else
    task_base="$task_input"
    task="$task_input"
  fi

  if [[ "$task_base" == "metamath" ]]; then
    if [[ "$task" != metamath:* ]]; then
      task="metamath:100000"
    fi
  fi
fi

# =====================
# derived args
# =====================
# GPU count
world_size=$(awk -F',' '{print NF}' <<< "$gpus")

# gradient_accumulation_steps = bs / (per_bs * world_size)
denom=$(( per_bs * world_size )) 
if (( denom <= 0 )); then
  echo "Error: per_bs * num_gpus must be > 0." >&2
  exit 2
fi
gradient_accumulation_steps=$(( bs / denom ))
if (( gradient_accumulation_steps < 1 )); then
  gradient_accumulation_steps=1
fi

# Total effective batch size (only for a reference to user)
eff_bs=$(( per_bs * world_size * gradient_accumulation_steps ))

# formulate learning rate: scientific notation with 3 decimal places
if lr_fmt=$(printf "%.4e" "${lr}" | sed -E 's/e([+-]?)0*([0-9]+)/e\1\2/' 2>/dev/null); then
    : 
else
    lr_fmt="$lr"
fi


# =====================
# echo settings
# =====================
echo "===========Training=============="
echo "Model: $model"
echo "Data: $data"
echo "Task: $task"
echo "PEFT: $peft"
if [[ "$peft" == "FFT" ]]; then
  echo "FFT (Full Fine-Tuning) — no LoRA rank/alpha"
else
  echo "Rank: $rank"
fi
echo "Number of train epochs: $num_train_epochs"
echo "Learning rate: $lr"
echo "Batch size (global target): $bs"
echo "Per-device batch size: $per_bs"
echo "Model max length: $model_max_length"
echo "GPUs: $gpus (world_size=$world_size)"
echo "Grad Accum Steps: $gradient_accumulation_steps"
echo "TOTAL_BATCH_SIZE (effective): $eff_bs"
echo "Output Path: $output_path"
if [[ "$peft" == "CurvLoRA" ]]; then
  echo "Curvature Data Path: $curvature_data_path"
  echo "Curvature Energy Ratio: $curvature_energy_ratio"
  if [[ "$curvature_alpha_tr" != "0.0" && "$curvature_alpha_tr" != "0" ]]; then
    echo "Curvature Trust Region: alpha_tr=$curvature_alpha_tr delta_tr=$curvature_delta_tr"
  fi
  echo "Curvature LR Scheduler: $curvature_lr_scheduler"
fi
if [[ "$peft" == "CurvDelta" ]]; then
  echo "Curvature Data Path: $curvature_data_path"
  echo "Curvature Energy Ratio: $curvature_energy_ratio"
  echo "Curvature LR Scheduler: $curvature_lr_scheduler"
fi
if [[ "$peft" == "CrispEdit" ]]; then
  echo "CrispEdit Data Path: $crispedit_data_path"
  echo "CrispEdit Energy Ratio: $crispedit_energy_ratio"
  echo "CrispEdit LR Scheduler: cosine (fixed)"
fi
if [[ "$peft" == Corda-* ]]; then
  echo "CorDA Data Path: $corda_data_path"
  echo "CorDA Num Samples: $corda_num_samples"
fi
if [[ "$peft" == "FFT" ]]; then
  echo "Full Fine-Tuning (FFT) — no LoRA config; model saved directly to output_path."
fi
if [[ "$peft" == "OPLoRA" ]]; then
  echo "OPLoRA protected rank k: $op_lora_k"
fi
echo "Time stamp: $timestamp"
echo "Trial ID: $trial_id"
echo "================================"

# =====================
# start training
# =====================
run_name=$(basename "$output_path")

if [[ "$peft" == "PiSSA" ]]; then
  res_model=$(readlink -m "${output_home}/output/PiSSA-${model_abbr}-r${rank}")
  if [[ -e "$res_model" ]]; then
    echo "Use pre-initialized residual model at: $res_model"
  else
    echo "Perform PiSSA initialization…"
    python utils/init_pissa.py \
      --device "cuda:0" \
      --base_model_path "$model" \
      --output_dir "$res_model" \
      --init_weights "pissa_niter_16" \
      --lora_r "$rank" \
      --lora_alpha "$rank" \
      --lora_dropout 0 \
      --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
  fi

  deepspeed --master_port=$master_port --include=localhost:"$gpus" train.py \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --model_name_or_path "$res_model" \
    --adapter_name_or_path "pissa_init" \
    --full_finetune False \
    --bf16 \
    --data_path "$data" \
    --sub_task "$task" \
    --dataset_split train \
    --dataset_field instruction output \
    --output_dir "$output_path" \
    --num_train_epochs "$num_train_epochs" \
    --model_max_length "$model_max_length" \
    --per_device_train_batch_size "$per_bs" \
    --gradient_accumulation_steps "$gradient_accumulation_steps" \
    --learning_rate "$lr" \
    --save_strategy "no" \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --report_to "wandb" \
    --trial_id $trial_id \
    --logging_steps $logging_steps \
    "${_replay_train_args[@]}"

elif [[ "$peft" == "MiLoRA" ]]; then
  res_model=$(readlink -m "${output_home}/output/MiLoRA-${model_abbr}-r${rank}")
  if [[ -e "$res_model" ]]; then
    echo "Use pre-initialized residual model at: $res_model"
  else
    echo "Perform MiLoRA initialization…"
    python utils/init_milora.py \
      --device "cuda:0" \
      --base_model_path "$model" \
      --output_dir "$res_model" \
      --init_weights "milora" \
      --lora_r "$rank" \
      --lora_alpha "$rank" \
      --lora_dropout 0 \
      --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
  fi

  deepspeed --master_port=$master_port --include=localhost:"$gpus" train.py \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --model_name_or_path "$res_model" \
    --adapter_name_or_path "milora_init" \
    --full_finetune False \
    --bf16 \
    --data_path "$data" \
    --sub_task "$task" \
    --dataset_split train \
    --dataset_field instruction output \
    --output_dir "$output_path" \
    --num_train_epochs "$num_train_epochs" \
    --model_max_length "$model_max_length" \
    --per_device_train_batch_size "$per_bs" \
    --gradient_accumulation_steps "$gradient_accumulation_steps" \
    --learning_rate "$lr" \
    --save_strategy "no" \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --report_to "wandb" \
    --trial_id $trial_id \
    --logging_steps $logging_steps \
    "${_replay_train_args[@]}"

elif [[ "$peft" == "LoRA" ]]; then
  deepspeed --master_port=$master_port --include=localhost:"$gpus" train.py \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --model_name_or_path "$model" \
    --full_finetune False \
    --bf16 \
    --init_weights True \
    --target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
    --lora_rank "$rank" \
    --lora_alpha "$rank" \
    --lora_dropout 0 \
    --data_path "$data" \
    --sub_task "$task" \
    --dataset_split train \
    --dataset_field instruction output \
    --output_dir "$output_path" \
    --num_train_epochs "$num_train_epochs" \
    --model_max_length "$model_max_length" \
    --per_device_train_batch_size "$per_bs" \
    --gradient_accumulation_steps "$gradient_accumulation_steps" \
    --learning_rate "$lr" \
    --save_strategy "no" \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --report_to "wandb" \
    --trial_id $trial_id \
    --logging_steps $logging_steps \
    "${_replay_train_args[@]}"

elif [[ "$peft" == "CurvLoRA" ]]; then
  deepspeed --master_port=$master_port --include=localhost:"$gpus" train.py \
    --use_curv_lora True \
    --curvature_data_path "$curvature_data_path" \
    --curvature_energy_ratio "$curvature_energy_ratio" \
    --curvature_collection_batch_size 1 \
    --curvature_num_chunks "$curvature_num_chunks" \
    --curvature_alpha_tr "$curvature_alpha_tr" \
    --curvature_delta_tr "$curvature_delta_tr" \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --model_name_or_path "$model" \
    --full_finetune False \
    --bf16 \
    --init_weights True \
    --target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
    --lora_rank "$rank" \
    --lora_alpha "$rank" \
    --lora_dropout 0 \
    --data_path "$data" \
    --sub_task "$task" \
    --dataset_split train \
    --dataset_field instruction output \
    --output_dir "$output_path" \
    --num_train_epochs "$num_train_epochs" \
    --model_max_length "$model_max_length" \
    --per_device_train_batch_size "$per_bs" \
    --gradient_accumulation_steps "$gradient_accumulation_steps" \
    --learning_rate "$lr" \
    --save_strategy "no" \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "$curvature_lr_scheduler" \
    --report_to "wandb" \
    --trial_id $trial_id \
    --logging_steps $logging_steps \
    --gradient_checkpointing True \
    "${_replay_train_args[@]}"

elif [[ "$peft" == "CurvDelta" ]]; then
 _curv_delta_ds_config="configs/ds_config_zero3_no_offload.json"
  if (( world_size >= 2 )); then
    # CurvDelta with full-rank ΔW needs more memory than LoRA.
    # For multi-GPU (typically 7B models), use ZeRO-3 with **optimizer-only**
    # offload: params stay sharded on GPU (fast), only AdamW states go to CPU.
    # Full param+optim offload (ds_config_zero3.json) is ~2-3x slower because
    # every forward/backward fetches params from CPU.
    _curv_delta_ds_config="configs/ds_config_zero3_offload_optim.json"
  fi
  deepspeed --master_port=$master_port --include=localhost:"$gpus" train.py \
    --use_curv_delta True \
    --curvature_data_path "$curvature_data_path" \
    --curvature_energy_ratio "$curvature_energy_ratio" \
    --curvature_collection_batch_size 1 \
    --curvature_num_chunks "$curvature_num_chunks" \
    --deepspeed "$_curv_delta_ds_config" \
    --model_name_or_path "$model" \
    --full_finetune False \
    --bf16 \
    --target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
    --data_path "$data" \
    --sub_task "$task" \
    --dataset_split train \
    --dataset_field instruction output \
    --output_dir "$output_path" \
    --num_train_epochs "$num_train_epochs" \
    --model_max_length "$model_max_length" \
    --per_device_train_batch_size "$per_bs" \
    --gradient_accumulation_steps "$gradient_accumulation_steps" \
    --learning_rate "$lr" \
    --save_strategy "no" \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "$curvature_lr_scheduler" \
    --report_to "wandb" \
    --trial_id $trial_id \
    --logging_steps $logging_steps \
    --gradient_checkpointing True \
    "${_replay_train_args[@]}"

elif [[ "$peft" == Corda-* ]]; then
  corda_method="kpm"
  corda_label="KPM"
  if [[ "$peft" == "Corda-IPM" ]]; then
    corda_method="ipm"
    corda_label="IPM"
  fi
  res_model=$(readlink -m "${output_home}/output/Corda-${corda_label}-${model_abbr}-r${rank}-${task_base}")
  if [[ -e "$res_model" ]]; then
    echo "Use pre-initialized CorDA ${corda_method} model at: $res_model"
  else
    echo "Perform CorDA ${corda_method} initialization..."
    python utils/init_corda.py \
      --device "cuda:0" \
      --base_model_path "$model" \
      --output_dir "$res_model" \
      --data_path "$corda_data_path" \
      --sub_task "$task_base" \
      --corda_method "$corda_method" \
      --corda_num_samples "$corda_num_samples" \
      --corda_cache_dir "${output_home}/output/corda_cache/${model_abbr}" \
      --bits "bf16" \
      --lora_r "$rank" \
      --lora_alpha "$rank" \
      --lora_dropout 0 \
      --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
      --model_max_length "$model_max_length"
  fi

  deepspeed --master_port=$master_port --include=localhost:"$gpus" train.py \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --model_name_or_path "$res_model" \
    --adapter_name_or_path "corda_init" \
    --full_finetune False \
    --bf16 \
    --data_path "$data" \
    --sub_task "$task" \
    --dataset_split train \
    --dataset_field instruction output \
    --output_dir "$output_path" \
    --num_train_epochs "$num_train_epochs" \
    --model_max_length "$model_max_length" \
    --per_device_train_batch_size "$per_bs" \
    --gradient_accumulation_steps "$gradient_accumulation_steps" \
    --learning_rate "$lr" \
    --save_strategy "no" \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --report_to "wandb" \
    --trial_id $trial_id \
    --logging_steps $logging_steps \
    "${_replay_train_args[@]}"

elif [[ "$peft" == "DoRA" ]]; then
  deepspeed --master_port=$master_port --include=localhost:"$gpus" train.py \
    --use_dora True \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --model_name_or_path "$model" \
    --full_finetune False \
    --bf16 \
    --init_weights True \
    --target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
    --lora_rank "$rank" \
    --lora_alpha "$rank" \
    --lora_dropout 0 \
    --data_path "$data" \
    --sub_task "$task" \
    --dataset_split train \
    --dataset_field instruction output \
    --output_dir "$output_path" \
    --num_train_epochs "$num_train_epochs" \
    --model_max_length "$model_max_length" \
    --per_device_train_batch_size "$per_bs" \
    --gradient_accumulation_steps "$gradient_accumulation_steps" \
    --learning_rate "$lr" \
    --save_strategy "no" \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --report_to "wandb" \
    --trial_id $trial_id \
    --logging_steps $logging_steps \
    "${_replay_train_args[@]}"

elif [[ "$peft" == "OLoRA" ]]; then
  deepspeed --master_port=$master_port --include=localhost:"$gpus" train.py \
    --use_olora True \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --model_name_or_path "$model" \
    --full_finetune False \
    --bf16 \
    --init_weights True \
    --target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
    --lora_rank "$rank" \
    --lora_alpha "$rank" \
    --lora_dropout 0 \
    --data_path "$data" \
    --sub_task "$task" \
    --dataset_split train \
    --dataset_field instruction output \
    --output_dir "$output_path" \
    --num_train_epochs "$num_train_epochs" \
    --model_max_length "$model_max_length" \
    --per_device_train_batch_size "$per_bs" \
    --gradient_accumulation_steps "$gradient_accumulation_steps" \
    --learning_rate "$lr" \
    --save_strategy "no" \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --report_to "wandb" \
    --trial_id $trial_id \
    --logging_steps $logging_steps \
    "${_replay_train_args[@]}"

elif [[ "$peft" == "LoRA-GA" ]]; then
  res_model=$(readlink -m "${output_home}/output/LoRA-GA-${model_abbr}-r${rank}-${task_base}")
  if [[ -e "$res_model" ]]; then
    echo "Use pre-initialized LoRA-GA model at: $res_model"
  else
    echo "Perform LoRA-GA initialization…"
    python utils/init_loraga.py \
      --device "cuda:0" \
      --base_model_path "$model" \
      --output_dir "$res_model" \
      --data_path "$data" \
      --sub_task "$task_base" \
      --bits "bf16" \
      --lora_r "$rank" \
      --lora_alpha "$rank" \
      --lora_dropout 0 \
      --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
      --stable_gamma 16 \
      --grad_num_samples 32 \
      --grad_batch_size 2 \
      --model_max_length "$model_max_length"
  fi

  deepspeed --master_port=$master_port --include=localhost:"$gpus" train.py \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --model_name_or_path "$res_model" \
    --adapter_name_or_path "loraga_init" \
    --full_finetune False \
    --bf16 \
    --data_path "$data" \
    --sub_task "$task" \
    --dataset_split train \
    --dataset_field instruction output \
    --output_dir "$output_path" \
    --num_train_epochs "$num_train_epochs" \
    --model_max_length "$model_max_length" \
    --per_device_train_batch_size "$per_bs" \
    --gradient_accumulation_steps "$gradient_accumulation_steps" \
    --learning_rate "$lr" \
    --save_strategy "no" \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --report_to "wandb" \
    --trial_id $trial_id \
    --logging_steps $logging_steps \
    "${_replay_train_args[@]}"

elif [[ "$peft" == "InitAB" ]]; then
  initAB_res_model=$(readlink -m "${output_home}/output/InitAB-${model_abbr}-r${rank}-${timestamp}")
  deepspeed --master_port=$master_port --include=localhost:"$gpus" train.py \
    --use_initAB True \
    --initAB_res_model "$initAB_res_model"  \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --model_name_or_path "$model" \
    --full_finetune False \
    --bf16 \
    --init_weights True \
    --target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
    --lora_rank "$rank" \
    --lora_alpha "$rank" \
    --lora_dropout 0 \
    --data_path "$data" \
    --sub_task "$task" \
    --dataset_split train \
    --dataset_field instruction output \
    --output_dir "$output_path" \
    --num_train_epochs "$num_train_epochs" \
    --model_max_length "$model_max_length" \
    --per_device_train_batch_size "$per_bs" \
    --gradient_accumulation_steps "$gradient_accumulation_steps" \
    --learning_rate "$lr" \
    --save_strategy "no" \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --report_to "wandb" \
    --trial_id $trial_id \
    --logging_steps $logging_steps \
    "${_replay_train_args[@]}"

elif [[ "$peft" == "GraLoRA" ]]; then
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  deepspeed --master_port=$master_port --include=localhost:"$gpus" train.py \
    --use_gralora True \
    --gralora_k "$gralora_k" \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --model_name_or_path "$model" \
    --full_finetune False \
    --bf16 \
    --target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
    --lora_rank "$rank" \
    --lora_alpha "$rank" \
    --lora_dropout 0 \
    --data_path "$data" \
    --sub_task "$task" \
    --dataset_split train \
    --dataset_field instruction output \
    --output_dir "$output_path" \
    --num_train_epochs "$num_train_epochs" \
    --model_max_length "$model_max_length" \
    --per_device_train_batch_size "$per_bs" \
    --gradient_accumulation_steps "$gradient_accumulation_steps" \
    --learning_rate "$lr" \
    --save_strategy "no" \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --report_to "wandb" \
    --trial_id $trial_id \
    --logging_steps $logging_steps \
    "${_replay_train_args[@]}"

elif [[ "$peft" == "RandLoRA" ]]; then
  deepspeed --master_port=$master_port --include=localhost:"$gpus" train.py \
    --use_randlora True \
    --randlora_alpha "$randlora_alpha" \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --model_name_or_path "$model" \
    --full_finetune False \
    --bf16 \
    --target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
    --lora_rank "$rank" \
    --lora_dropout 0 \
    --data_path "$data" \
    --sub_task "$task" \
    --dataset_split train \
    --dataset_field instruction output \
    --output_dir "$output_path" \
    --num_train_epochs "$num_train_epochs" \
    --model_max_length "$model_max_length" \
    --per_device_train_batch_size "$per_bs" \
    --gradient_accumulation_steps "$gradient_accumulation_steps" \
    --learning_rate "$lr" \
    --save_strategy "no" \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --report_to "wandb" \
    --trial_id $trial_id \
    --logging_steps $logging_steps \
    "${_replay_train_args[@]}"

elif [[ "$peft" == "LoFT" ]]; then
  deepspeed --master_port=$master_port --include=localhost:"$gpus" train.py \
    --use_loft True \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --model_name_or_path "$model" \
    --full_finetune False \
    --bf16 \
    --init_weights True \
    --target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
    --lora_rank "$rank" \
    --lora_alpha "$rank" \
    --lora_dropout 0 \
    --data_path "$data" \
    --sub_task "$task" \
    --dataset_split train \
    --dataset_field instruction output \
    --output_dir "$output_path" \
    --num_train_epochs "$num_train_epochs" \
    --model_max_length "$model_max_length" \
    --per_device_train_batch_size "$per_bs" \
    --gradient_accumulation_steps "$gradient_accumulation_steps" \
    --learning_rate "$lr" \
    --save_strategy "no" \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --report_to "wandb" \
    --trial_id $trial_id \
    --logging_steps $logging_steps \
    "${_replay_train_args[@]}"

elif [[ "$peft" == "CrispEdit" ]]; then
  # CrispEdit updates every target projection weight in place (not a PEFT
  # adapter), so a 7B base model trains ~6.5B parameters: ZeRO-2 would keep an
  # fp32 master copy of all of them on every rank, which alone is ~26GB.
  # ZeRO-3 shards params + grads + optimizer states. `offload_optimizer` moves
  # the AdamW states (and the fp32 masters) to the CPU, which is the recipe this
  # repo already uses for 7B CurvDelta; the tighter reduce/prefetch/max-live
  # buffers in that config bound the extra ZeRO-3 buffers. Its optimizer block is
  # required for the offload and it deliberately has no scheduler block, so the
  # Hugging Face cosine schedule is used (a DeepSpeed "scheduler" key would make
  # Transformers build a DummyScheduler and let WarmupLR replace cosine).
  # If VRAM is still short, `configs/ds_config_zero3.json` adds param offload
  # (~2-3x slower per the CurvDelta comment).
  _ds_config="configs/ds_config_zero2_no_offload.json"
  if (( world_size >= 2 )); then
    _ds_config="configs/ds_config_zero3_crispedit.json"
  fi
  deepspeed --master_port=$master_port --include=localhost:"$gpus" train.py \
    --deepspeed "$_ds_config" \
    --use_crispedit True \
    --crispedit_data_path "$crispedit_data_path" \
    --crispedit_energy_ratio "$crispedit_energy_ratio" \
    --crispedit_collection_batch_size "$crispedit_collection_batch_size" \
    --crispedit_num_chunks "$crispedit_num_chunks" \
    --target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
    --model_name_or_path "$model" \
    --full_finetune False \
    --bf16 \
    --data_path "$data" \
    --sub_task "$task" \
    --dataset_split train \
    --dataset_field instruction output \
    --output_dir "$output_path" \
    --num_train_epochs "$num_train_epochs" \
    --model_max_length "$model_max_length" \
    --per_device_train_batch_size "$per_bs" \
    --gradient_accumulation_steps "$gradient_accumulation_steps" \
    --learning_rate "$lr" \
    --save_strategy "no" \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --report_to "wandb" \
    --trial_id $trial_id \
    --logging_steps $logging_steps \
    --gradient_checkpointing True \
    "${_replay_train_args[@]}"

elif [[ "$peft" == "FFT" ]]; then
  # ZeRO-3 + optimizer offload for 7B models, ZeRO-2 for small models (0.6B/1B).
  # - no offload: ~30-35GB/GPU on 4x40GB — too tight, OOMs with other procs.
  # - optim offload: ~10-15GB/GPU, only ~30-50% slower — recommended.
  # - param+optim offload: ~5GB but ~60h/epoch — too slow.
  _ds_config="configs/ds_config_zero2_no_offload.json"
  if [[ "$model_abbr" == Llama-* ]] || [[ "$model_abbr" == llama-* ]]; then
    _ds_config="configs/ds_config_zero3_no_offload.json"
  fi

  deepspeed --master_port=$master_port --include=localhost:"$gpus" train.py \
    --deepspeed "$_ds_config" \
    --model_name_or_path "$model" \
    --full_finetune True \
    --bf16 \
    --data_path "$data" \
    --sub_task "$task" \
    --dataset_split train \
    --dataset_field instruction output \
    --output_dir "$output_path" \
    --num_train_epochs "$num_train_epochs" \
    --model_max_length "$model_max_length" \
    --per_device_train_batch_size "$per_bs" \
    --gradient_accumulation_steps "$gradient_accumulation_steps" \
    --learning_rate "$lr" \
    --save_strategy "no" \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --report_to "wandb" \
    --trial_id $trial_id \
    --logging_steps $logging_steps \
    --gradient_checkpointing True \
    "${_replay_train_args[@]}"

elif [[ "$peft" == "OPLoRA" ]]; then
  deepspeed --master_port=$master_port --include=localhost:"$gpus" train.py \
    --use_op_lora True \
    --op_lora_k "$op_lora_k" \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --model_name_or_path "$model" \
    --full_finetune False \
    --bf16 \
    --init_weights True \
    --target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
    --lora_rank "$rank" \
    --lora_alpha "$rank" \
    --lora_dropout 0 \
    --data_path "$data" \
    --sub_task "$task" \
    --dataset_split train \
    --dataset_field instruction output \
    --output_dir "$output_path" \
    --num_train_epochs "$num_train_epochs" \
    --model_max_length "$model_max_length" \
    --per_device_train_batch_size "$per_bs" \
    --gradient_accumulation_steps "$gradient_accumulation_steps" \
    --learning_rate "$lr" \
    --save_strategy "no" \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --report_to "wandb" \
    --trial_id $trial_id \
    --logging_steps $logging_steps \
    "${_replay_train_args[@]}"

else
  echo "Error: Unsupported PEFT method: $peft" >&2
  echo "Supported methods: LoRA, CurvLoRA, CurvDelta, CrispEdit, Corda-KPM, Corda-IPM, PiSSA, MiLoRA, DoRA, OLoRA, InitAB, LoRA-GA, GraLoRA, RandLoRA, LoFT, FFT, OPLoRA" >&2
  exit 1
fi
