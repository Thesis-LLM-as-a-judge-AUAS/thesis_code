export CUDA_VISIBLE_DEVICES=0

# Baseline and MEC+BPC experiments configuration
experiment_matrix=(
  "gpt35 vicuna gpt-3.5-turbo 0 1 0"
  "gpt35 vicuna gpt-4 0 1 0"
  "gpt35 vicuna gpt-3.5-turbo 1 3 1"
  "gpt35 vicuna gpt-4 1 3 1"
)

# Baseline and MEC+BPC experiments run (each experiment is repeated 40 times)
for i in $(seq 1 40); do
  mkdir -p ./gathered_data/baseline_mec_results/"${i}"

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

MODEL_PATH=./models/autoj-13b
BASE_MODEL_PATH=./models/Mistral-7B-v0.1
MODEL_TYPE=auto-j

DATA_TYPE=verbosity

data_types=(
  "vicuna"
  "vicuna-mec"
  "vicuna-gpt4"
  "vicuna-mec-gpt4"
)


#python3 -u src/cal_reliability.py \
#      --model-name-or-path $MODEL_PATH \
#      --cali-model-name-or-path $BASE_MODEL_PATH \
#      --model-type ${MODEL_TYPE} \
#      --data-type ${DATA_TYPE} \
#      --max-new-token 1024 \
#      --logit-file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-logit.jsonl" \
#      --output-file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-relia.json"
#
#
#MODEL_TYPE="auto-j"
#DATA_TYPE="cj_sampled_merged"
#
#for i in $(seq 1 47); do
#  python3 -u src/cascaded_eval.py \
#      --data-type $DATA_TYPE \
#      --logit-file1 "relia_scores/${MODEL_TYPE}/${DATA_TYPE}/${i}-logit.jsonl" \
#      --output-file1 "relia_scores/${MODEL_TYPE}/${DATA_TYPE}/${i}-relia.json" \
#      --logit-file-gpt "outputs/expanded_review/review_gpt35_vicuna_gpt-4_mec3_bpc1/${i}_review_gpt35_vicuna_gpt-4_mec3_bpc1.jsonl" \
#      --final-output-file "outputs/final-outputs/${DATA_TYPE}/${i}-${MODEL_TYPE}-${DATA_TYPE}-final.json"
#done

#for i in $(seq 1 47); do
#  python3 -u src/cascaded_eval.py \
#      --data-type $DATA_TYPE \
#      --logit-file1 "relia_scores/${MODEL_TYPE}/${DATA_TYPE}/${i}-logit.jsonl" \
#      --output-file1 "relia_scores/${MODEL_TYPE}/${DATA_TYPE}/${i}-relia.json" \
#      --logit-file-gpt "outputs/expanded_review/review_gpt35_vicuna_gpt-4_mec3_bpc1/${i}_review_gpt35_vicuna_gpt-4_mec3_bpc1.jsonl" \
#      --final-output-file "outputs/final-outputs/${DATA_TYPE}/${i}-${MODEL_TYPE}-${DATA_TYPE}-final.json"
#done