#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

# ============================================================
# Data types for CascadedEval
# ============================================================
data_types=(
  "verbosity"
)

# ============================================================
# Model configurations for CascadedEval
# Format: model_path base_model_path model_type
# ============================================================
model_types=(
  "../models/autoj-13b ../models/Mistral-7B-v0.1 auto-j"
)

# ============================================================
# Iterate over all model configurations and data types
# Extract judgments, calculate reliability, and run CascadedEval
# ============================================================
for model_type_config in "${model_types[@]}"; do
  read -r model_path base_model_path model_type <<< "$model_type_config"

  for data_type in "${data_types[@]}"; do
    relia_scores_dir="gathered_data/cascaded-eval-results/relia_scores/${model_type}/${data_type}"

      # Calculate reliability score
      python3 -u code_components/cal_reliability.py \
        --model-name-or-path "$model_path" \
        --cali-model-name-or-path "$base_model_path" \
        --model-type "$model_type" \
        --data-type "$data_type" \
        --max-new-token 1024 \
        --logit-file "${relia_scores_dir}/${i}-logit.jsonl" \
        --output-file "${relia_scores_dir}/${i}-relia.json"
  done
done
