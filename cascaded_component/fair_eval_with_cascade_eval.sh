export CUDA_VISIBLE_DEVICES=0

MODEL_TYPE="auto-j"
DATA_TYPE="vicuna"

for i in $(seq 1 47); do
  python3 -u src/cascaded_eval.py \
      --data-type $DATA_TYPE \
      --logit-file1 "relia_scores/${MODEL_TYPE}/${DATA_TYPE}/${i}-logit.jsonl" \
      --output-file1 "relia_scores/${MODEL_TYPE}/${DATA_TYPE}/${i}-relia.json" \
      --logit-file-gpt "outputs/expanded_review/review_gpt35_vicuna_gpt-4_mec3_bpc1/${i}_review_gpt35_vicuna_gpt-4_mec3_bpc1.jsonl" \
      --final-output-file "outputs/final-outputs/${DATA_TYPE}/${i}-${MODEL_TYPE}-${DATA_TYPE}-final.json"
done

#for i in $(seq 1 47); do
#  python3 -u src/cascaded_eval.py \
#      --data-type $DATA_TYPE \
#      --logit-file1 "relia_scores/${MODEL_TYPE}/${DATA_TYPE}/${i}-logit.jsonl" \
#      --output-file1 "relia_scores/${MODEL_TYPE}/${DATA_TYPE}/${i}-relia.json" \
#      --logit-file-gpt "outputs/expanded_review/review_gpt35_vicuna_gpt-4_mec3_bpc1/${i}_review_gpt35_vicuna_gpt-4_mec3_bpc1.jsonl" \
#      --final-output-file "outputs/final-outputs/${DATA_TYPE}/${i}-${MODEL_TYPE}-${DATA_TYPE}-final.json"
#done