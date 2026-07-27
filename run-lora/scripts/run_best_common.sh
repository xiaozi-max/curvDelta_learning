#!/bin/bash

# Shared execution body for *_best.sh wrappers.
# Each wrapper defines fixed config, then sources this file.


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

for seed in "${seeds[@]}"; do
  for peft in "${peft_methods[@]}"; do
    lr_string="${peft_lrs[$peft]}"

    if [[ -z "$lr_string" ]]; then
      echo "Error: No learning rate defined for '$peft'"
      exit 1
    fi

    # Support LR sweep: if multiple space-separated LRs, iterate over each.
    # A single LR (no spaces) works as before.
    read -ra curr_lrs <<< "$lr_string"

    for lr in "${curr_lrs[@]}"; do
      echo ">>> Running best experiment: seed=${seed}, rank=${rank}, bs=${bs}"
      echo ">>> task=${task}, peft=${peft}, lr=${lr}"

      timestamp=$(date +"%Y%m%d-%H%M%S")
      _er_suffix=""
      if [[ "$peft" == "CurvatureLoRA" || "$peft" == "CurvatureDense" ]]; then
        _er_suffix="-er${curvature_energy_ratio//./}"
        if [[ "${curvature_lr_scheduler:-cosine}" == "constant_with_warmup" ]]; then
          _er_suffix="${_er_suffix}-cww"
        fi
        if [[ "$peft" == "CurvatureLoRA" ]]; then
          _tr_alpha="${curvature_alpha_tr:-0.0}"
          if [[ "$_tr_alpha" != "0.0" && "$_tr_alpha" != "0" ]]; then
            _tr_suffix="-tr${_tr_alpha//./}"
            _er_suffix="${_er_suffix}${_tr_suffix}"
          fi
        fi
      fi
      output_path=$(readlink -m \
        "${output_home}/output/${task}-${peft}-${model_abbr}-r${rank}/bs${bs}-lr${lr}-trial${seed}${_er_suffix}")
      adapter_path="${output_path}/adapter_model"
      perf_json_path="${output_path}/perf.json"
      temp_path=$(readlink -m "${output_home}/output/temp_merged_model-${timestamp}")

      echo ">>> Experiment output path: $output_path"
      echo ">>> Temp merged model save path: $temp_path"

      perf_exists=false
      if [[ -f "$perf_json_path" ]] \
        && grep -E '[0-9]+\.?[0-9]*' "$perf_json_path" >/dev/null; then
        echo "✓ perf.json already exists"
        perf_exists=true
      fi

      if [[ "$perf_exists" == false ]]; then
        if [[ -d "$adapter_path" ]]; then
          echo ">>> Skipping: adapter_model already exists: $adapter_path"
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
            --output_home "$output_home" \
            --curvature_energy_ratio "${curvature_energy_ratio:-0.9}" \
            --curvature_num_chunks "${curvature_num_chunks:-1}" \
            --curvature_alpha_tr "${curvature_alpha_tr:-0.0}" \
            --curvature_delta_tr "${curvature_delta_tr:-1e-12}" \
            --curvature_lr_scheduler "${curvature_lr_scheduler:-cosine}"
        fi

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

        if [[ -d "$temp_path" ]]; then
          rm -rf "$temp_path"
          echo "✓ Successfully deleted: $temp_path"
        else
          echo "✗ Directory not found: $temp_path"
          echo "  The merged model may already have been deleted by Python."
        fi
      fi

      # Re-check perf.json after possible test run
      if [[ -f "$perf_json_path" ]] \
        && grep -E '[0-9]+\.?[0-9]*' "$perf_json_path" >/dev/null; then
        echo "✓ perf.json exists and contains numbers"
        if [[ "${run_world_knowledge:-false}" == "true" ]]; then
          wk_file="${output_path}/world_knowledge.json"
          if [[ -f "$wk_file" ]]; then
            echo "✓ world_knowledge.json exists"
          else
            echo ">>> Start World Knowledge Evaluation"
            wk_gpu=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d',' -f1)
            python utils/eval_world_knowledge.py \
              --model "$model" \
              --model_abbr "$model_abbr" \
              --lora "$adapter_path" \
              --peft "$peft" \
              --rank "$rank" \
              --task "$task" \
              --output_path "$output_path" \
              --output_home "$output_home" \
              --gpus "$wk_gpu" \
              --timestamp "$timestamp" \
              --batch_size "${world_knowledge_batch_size:-auto}"
            echo ">>> World Knowledge Evaluation done"
          fi

          # MMLU patch: if world_knowledge.json exists but lacks "mmlu", eval only MMLU
          if [[ -f "$wk_file" ]]; then
            if python3 -c "import json; d=json.load(open('$wk_file')); exit(0 if 'mmlu' in d else 1)" 2>/dev/null; then
              echo "✓ mmlu already in world_knowledge.json, skipping MMLU patch"
            else
              echo ">>> MMLU patch: world_knowledge.json lacks mmlu, running MMLU eval"
              wk_gpu=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d',' -f1)
              python utils/eval_world_knowledge.py \
                --model "$model" \
                --model_abbr "$model_abbr" \
                --lora "$adapter_path" \
                --peft "$peft" \
                --rank "$rank" \
                --task "$task" \
                --output_path "$output_path" \
                --output_home "$output_home" \
                --gpus "$wk_gpu" \
                --timestamp "$timestamp" \
                --batch_size "${world_knowledge_batch_size:-auto}" \
                --tasks "mmlu"
              echo ">>> MMLU patch done"
            fi
          fi
        fi
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
    done  # end for lr in "${curr_lrs[@]}"
  done  # end for peft in "${peft_methods[@]}"
done  # end for seed in "${seeds[@]}"
