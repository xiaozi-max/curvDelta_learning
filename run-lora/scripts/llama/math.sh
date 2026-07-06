#!/bin/bash

source "$(git rev-parse --show-toplevel)/common_utils/server_env.sh"
set -e

### Yon can uncomment these lines to set HF cache and token globally
# export HF_HOME=/your/HuggingFace/cahce # Will be used to save HugginFace download model
# export HF_TOKEN= ## PLEASE ADD YOUR HF TOKEN HERE ##


# Automatically detect and configure CUDA_HOME for DeepSpeed compatibility
if command -v nvcc >/dev/null 2>&1; then
  CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  export CUDA_HOME CUDA_PATH="$CUDA_HOME"
  export PATH="$CUDA_HOME/bin:$PATH"
  if [ -d "$CUDA_HOME/lib64" ]; then
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  elif [ -d "$CUDA_HOME/lib" ]; then
    export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
  fi
  unset CUDAHOSTCXX
  unset CXX
  unset CC     
fi


##### Args Settings #####
output_home="./"
master_port=16925  # Change this if running multiple DeepSpeed jobs on the same node

# Model configuration
model="${ROOT_DIR}/models/meta-llama/Llama-2-7b-hf"
model_abbr="Llama-2-7b"

# GPU allocation
train_gpus="0,1,2,3" # Specify GPU IDs for training
test_gpus="0,1,2,3"  # Specify GPU IDs for testing

# Batch size settings
inference_bs=800   # VLLM inference batch size
per_bs=4           # Per-GPU training batch size for DeepSpeed (affects memory usage, set to lower values if you encounter OOM errors)

# Delete trained LoRA adapters after successful testing?
delete_adapter_after_test=true # set to false if you would like to do more custom analysis on the trained LoRA adapters

## All following args can have multiple values; if multiple values are given, the script will loop over them
seeds=(1 2 3)      # Training random seeds
ranks=(128)        # LoRA adapter rank
train_bs=(16 128)      # Training batch size
tasks=("metamath") # Task to fine-tune.
### Options: metamath, python, commonsense, instruction
### Use the ":XXX" postfix to train on only the first XXX samples of the dataset
### Use the "-epY" postfix to train for Y epochs; default is 1


peft_methods=("LoRA" "PiSSA" "MiLoRA" "InitAB" "DoRA") # assign the PEFT methods you would like to run
# ("LoRA-GA" "GraLoRA") # use custom conda enviroments to run these two PEFT methods per instruction in README
# ("RandLoRA") # assign custom LoRA rank to match its trainable parameter counts to other PEFT methods

## You can specify the learning rate search range for each PEFT method:
declare -A peft_lrs  
peft_lrs["LoRA"]="1.1247e-5 2.0000e-5 3.5566e-5 6.3246e-5 1.1247e-4 2.0000e-4 3.5566e-4 6.3246e-4 1.1247e-3 2.0000e-3 3.5566e-3 6.3246e-3"
peft_lrs["OLoRA"]="1.1247e-5 2.0000e-5 3.5566e-5 6.3246e-5 1.1247e-4 2.0000e-4 3.5566e-4 6.3246e-4 1.1247e-3 2.0000e-3 3.5566e-3 6.3246e-3"
peft_lrs["PiSSA"]="1.1247e-5 2.0000e-5 3.5566e-5 6.3246e-5 1.1247e-4 2.0000e-4 3.5566e-4 6.3246e-4 1.1247e-3 2.0000e-3 3.5566e-3 6.3246e-3"
peft_lrs["MiLoRA"]="1.1247e-5 2.0000e-5 3.5566e-5 6.3246e-5 1.1247e-4 2.0000e-4 3.5566e-4 6.3246e-4 1.1247e-3 2.0000e-3 3.5566e-3 6.3246e-3"
peft_lrs["InitAB"]="1.1247e-5 2.0000e-5 3.5566e-5 6.3246e-5 1.1247e-4 2.0000e-4 3.5566e-4 6.3246e-4 1.1247e-3 2.0000e-3 3.5566e-3 6.3246e-3"
peft_lrs["DoRA"]="1.1247e-5 2.0000e-5 3.5566e-5 6.3246e-5 1.1247e-4 2.0000e-4 3.5566e-4 6.3246e-4 1.1247e-3 2.0000e-3 3.5566e-3 6.3246e-3"
peft_lrs["LoFT"]="1.1247e-5 2.0000e-5 3.5566e-5 6.3246e-5 1.1247e-4 2.0000e-4 3.5566e-4 6.3246e-4 1.1247e-3 2.0000e-3 3.5566e-3 6.3246e-3"
peft_lrs["LoRA-GA"]="1.1247e-5 2.0000e-5 3.5566e-5 6.3246e-5 1.1247e-4 2.0000e-4 3.5566e-4 6.3246e-4 1.1247e-3 2.0000e-3 3.5566e-3 6.3246e-3"
peft_lrs["GraLoRA"]="1.1247e-5 2.0000e-5 3.5566e-5 6.3246e-5 1.1247e-4 2.0000e-4 3.5566e-4 6.3246e-4 1.1247e-3 2.0000e-3 3.5566e-3 6.3246e-3"
peft_lrs["RandLoRA"]="1.1247e-5 2.0000e-5 3.5566e-5 6.3246e-5 1.1247e-4 2.0000e-4 3.5566e-4 6.3246e-4 1.1247e-3 2.0000e-3 3.5566e-3 6.3246e-3"

##### Main Experiment Loop #####
model_max_length=512

for seed in "${seeds[@]}"; do
  for rank in "${ranks[@]}"; do
    for bs in "${train_bs[@]}"; do
      for task in "${tasks[@]}"; do  
      
        ## Fine-tuning dataset path setup
        if [[ "$task" == commonsense* ]]; then
          data="./lora-dataset/llm-adapters-dataset"
        elif [[ "$task" == instruction* ]]; then
          data="./lora-dataset/instruction-dataset"
        elif [[ "$task" == metamath* || "$task" == python* ]]; then
          data="./lora-dataset/pissa-dataset"
        else
          echo "Error: Unsupported task '$task'"
          exit 1
        fi
        for peft in "${peft_methods[@]}"; do

          lr_string="${peft_lrs[$peft]}"
          
          if [[ -z "$lr_string" ]]; then
            echo "Warning: No learning rates defined for '$peft'"
            exit 1
          fi
          
          read -ra curr_lrs <<< "$lr_string"
          echo ">>> Finetuning $peft with ${#curr_lrs[@]} learning rates: ${curr_lrs[*]}"
          
          for lr in "${curr_lrs[@]}"; do
            
            echo ">>> Running experiment: seed=${seed}, rank=${rank}, bs=${bs}, task=${task}, peft=${peft}, lr=${lr}"
            timestamp=$(date +"%Y%m%d-%H%M%S")
            output_path=$(readlink -m "${output_home}/output/${task}-${peft}-${model_abbr}-r${rank}/bs${bs}-lr${lr}-trial${seed}")
            adapter_path="${output_path}/adapter_model"
            perf_json_path="${output_path}/perf.json"
            temp_path=$(readlink -m "${output_home}/output/temp_merged_model-${timestamp}")
                
            echo ">>> Experiment output path: $output_path"
            echo ">>> Temp merged model save path: $temp_path"
            
            if [[ -f "$perf_json_path" ]] && grep -E '[0-9]+\.?[0-9]*' "$perf_json_path" >/dev/null; then
              echo "✓ perf.json exists and contains numbers"
              echo "Skip this experiment"
              continue
            fi

            # Training
            if [[ -d "$adapter_path" ]]; then
              echo ">>> Skipping: output path already exists an adapter_model: $adapter_path"
            else
              echo ">>> Start Training, adapters will be saved at $adapter_path"
              bash ./scripts/train.sh \
                --data "$data" \
                --master_port "$master_port" \
                --model "$model" \
                --model_abbr "$model_abbr" \
                --output_path "$output_path" \
                --task "$task" \
                --peft "$peft" \
                --rank "$rank" \
                --gpus "$train_gpus" \
                --trial_id "$seed" \
                --lr "$lr" \
                --bs "$bs" \
                --model_max_length "$model_max_length" \
                --timestamp "$timestamp" \
                --per_bs "$per_bs" \
                --output_home "$output_home"
            fi

            # Testing
            echo ">>> Start Testing, performance will be saved at $perf_json_path"   
            bash ./scripts/test.sh \
              --model "$model" \
              --data_path "$data" \
              --model_abbr "$model_abbr" \
              --task "$task" \
              --peft "$peft" \
              --rank "$rank" \
              --bs "$inference_bs" \
              --gpus "$test_gpus" \
              --output_home "$output_home" \
              --output_path "$output_path" \
              --temp_path "$temp_path" \
              --timestamp "$timestamp"

            # Always delete the temporary merged model to free up disk space
            if [[ -d "$temp_path" ]]; then
              rm -rf "$temp_path"
                echo "✓ Successfully deleted: $temp_path"
              else
                echo "✗ Directory not found: $temp_path, the merged model should already be deleted by python file"
            fi
            
            # If sucessfully get the `perf.json`, delete the LoRA adapters to free up disk space
            if [[ -f "$perf_json_path" ]] && grep -E '[0-9]+\.?[0-9]*' "$perf_json_path" >/dev/null; then
              echo "✓ perf.json exists and contains numbers"
              if [[ "$delete_adapter_after_test" == true ]]; then
                adapter_dir=$(readlink -m "${output_path}/adapter_model")
                if [[ -d "$adapter_dir" ]]; then
                  rm -rf "$adapter_dir"
                  echo "✓ Successfully deleted: $adapter_dir"
                else
                  echo "✗ Directory not found: $adapter_dir"
                fi
              else
                echo ">>> delete_adapter_after_test=false, keeping adapter"
              fi
            else
              echo "✗ perf.json missing or contains no numbers"
              echo "Keep the adapter!"            
            fi

            # For Init[AB] that initialize different residual models each run, delete the residual model to save disk space  
            if [ "$peft" == "InitAB" ]; then
              initAB_res_model=$(readlink -m "${output_home}/output/InitAB-${model_abbr}-r${rank}-${timestamp}")
              if [ -d "$initAB_res_model" ]; then
                  echo "deleting init AB res model after each experiment"
                  rm -rf "$initAB_res_model"
              fi
            fi
          done
            
        done  
      done
    done
  done
done
