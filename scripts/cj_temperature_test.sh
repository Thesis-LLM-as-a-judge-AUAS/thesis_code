#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Experiment configuration:
# m1 m2 eval_model bpc k t
# ============================================================
experiment_matrix=(
  "gpt35 vicuna gpt-3.5-turbo 0 1 0"
  "gpt35 vicuna gpt-4          0 1 0"
  "gpt35 vicuna gpt-3.5-turbo  1 3 0.1"
  "gpt35 vicuna gpt-4          1 3 0.1"
)

# ============================================================
# Run experiments (40 repetitions)
# ============================================================
for run_idx in $(seq 1 40); do
  run_dir="gathered_data/temperature_investigation_results/${run_idx}"
  mkdir -p "$run_dir"

  for exp_idx in "${!experiment_matrix[@]}"; do
    read -r m1 m2 eval_model bpc k t <<< "${experiment_matrix[$exp_idx]}"

    python code_components/fair_eval.py \
      -q "datasets/vicuna/sampled_data/temperature_investigation_sampled/questions.jsonl" \
      -a "datasets/vicuna/sampled_data/temperature_investigation_sampled/answer_${m1}.jsonl" \
         "datasets/cj_merged/sampled_data/temperature_investigation_sampled/answer_${m2}.jsonl" \
      -o "${run_dir}/review_${m1}_${m2}_${eval_model}_mec${k}_bpc${bpc}.jsonl" \
      -m "$eval_model" \
      --bpc "$bpc" \
      -k "$k" \
      -t "$t"
  done
done