#!/bin/bash

experiment_matrix=(
  "gpt35 vicuna gpt-3.5-turbo 0 1 0"
  "gpt35 vicuna gpt-4 0 1 0"
  "gpt35 vicuna gpt-3.5-turbo 1 3 0.1"
  "gpt35 vicuna gpt-4 1 3 0.1"
)
for i in $(seq 1 40); do
  mkdir -p ./gathered_data/temperature_investigation_results/"${i}"

  for j in "${!experiment_matrix[@]}"; do
      row="${experiment_matrix[$j]}"
      read -r m1 m2 eval_model bpc k t <<< "$row"

      python code_components/fair_eval.py \
          -q datasets/cj_merged/sampled_data/temperature_investigation_sampled/questions.jsonl \
          -a datasets/cj_merged/sampled_data/temperature_investigation_sampled/answer_"$m1".jsonl datasets/cj_merged/sampled_data/temperature_investigation_sampled/answer_"$m2".jsonl \
          -o gathered_data/temperature_investigation_results/"${i}"/"review_${m1}_${m2}_${eval_model}_mec${k}_bpc${bpc}.jsonl" \
          -m "$eval_model" \
          --bpc "$bpc" \
          -k "$k" \
          -t "$t"
  done
done
