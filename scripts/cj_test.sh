#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

## ============================================================
## Baseline and MEC+BPC experiments configuration
## Format: m1 m2 eval_model bpc k t
## ============================================================
experiment_matrix=(
  "gpt35 vicuna gpt-3.5-turbo  0 1 0"
  "gpt35 vicuna gpt-4          0 1 0"
  "gpt35 vicuna gpt-3.5-turbo  1 3 1"
  "gpt35 vicuna gpt-4          1 3 1"
)

# ============================================================
# Run baseline and MEC+BPC experiments (40 repetitions)
# ============================================================
for i in $(seq 1 40); do
  exp_dir="gathered_data/baseline_mec_results/${i}"
  mkdir -p "$exp_dir"

  for exp_config in "${!experiment_matrix[@]}"; do
    read -r m1 m2 eval_model bpc k t <<< "${experiment_matrix[$exp_config]}"

    python code_components/fair_eval.py \
      -q "datasets/vicuna/sampled_data/cj_sampled/questions/${exp_config}.jsonl" \
      -a "datasets/vicuna/sampled_data/cj_sampled/answer_${m1}/${exp_config}.jsonl" \
         "datasets/vicuna/sampled_data/cj_sampled/answer_${m2}/${exp_config}.jsonl" \
      -o "${exp_dir}/review_${m1}_${m2}_${eval_model}_mec${k}_bpc${bpc}.jsonl" \
      -m "$eval_model" \
      --bpc "$bpc" \
      -k "$k" \
      -t "$t"
  done
done

# ============================================================
# Data types for CascadedEval
# ============================================================
data_types=(
  "vicuna"
  "vicuna-mec"
  "vicuna-gpt4"
  "vicuna-mec-gpt4"
)

# ============================================================
# Model configurations for CascadedEval
# Format: model_path base_model_path model_type
# ============================================================
model_types=(
  "../models/autoj-13b ../models/Mistral-7B-v0.1 auto-j"
  "../models/judgelm-7b ../models/vicuna-7b judgelm"
)

# ============================================================
# Mapping: data_type -> baseline experiment configuration
# ============================================================
data_type_to_baseline=(
  "vicuna:gpt35 vicuna gpt-3.5-turbo 0 1 0"
  "vicuna-mec:gpt35 vicuna gpt-4 0 1 0"
  "vicuna-gpt4:gpt35 vicuna gpt-3.5-turbo 1 3 1"
  "vicuna-mec-gpt4:gpt35 vicuna gpt-4 1 3 1"
)

get_value() {
  local key="$1"
  for entry in "${data_type_to_baseline[@]}"; do
    local k="${entry%%:*}"
    local v="${entry#*:}"
    if [[ "$k" == "$key" ]]; then
      echo "$v"
      return 0
    fi
  done
  return 1
}

# ============================================================
# Iterate over all model configurations and data types
# Extract judgments, calculate reliability, and run CascadedEval
# ============================================================
for model_type_config in "${model_types[@]}"; do
  read -r model_path base_model_path model_type <<< "$model_type_config"

  for data_type in "${data_types[@]}"; do
    relia_scores_dir="gathered_data/cascaded-eval-results/relia_scores/${model_type}/${data_type}"
    result_dir="gathered_data/cascaded-eval-results/final_results/${data_type}"

    mkdir -p "$relia_scores_dir" "$result_dir"

    for i in $(seq 1 40); do
      # Calculate reliability score
      python3 -u code_components/cal_reliability.py \
        --model-name-or-path "$model_path" \
        --cali-model-name-or-path "$base_model_path" \
        --model-type "$model_type" \
        --data-type "$data_type" \
        --max-new-token 1024 \
        --logit-file "${relia_scores_dir}/${i}-logit.jsonl" \
        --output-file "${relia_scores_dir}/${i}-relia.json"

      # Get baseline config for current data_type
      read -r m1 m2 eval_model bpc k t <<< "$(get_value "$data_type")"

      # Apply CascadedEval methodology
      python3 -u src/cascaded_eval.py \
        --data-type "$data_type" \
        --logit-file1 "${relia_scores_dir}/${i}-logit.jsonl" \
        --output-file1 "${relia_scores_dir}/${i}-relia.json" \
        --logit-file-gpt "gathered_data/baseline_mec_results/${i}/review_${m1}_${m2}_${eval_model}_mec${k}_bpc${bpc}.jsonl" \
        --final-output-file "${result_dir}/${i}-${model_type}-${data_type}-final.json"
    done
  done
done
