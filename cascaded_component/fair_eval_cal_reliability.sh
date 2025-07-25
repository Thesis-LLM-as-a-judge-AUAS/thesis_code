export CUDA_VISIBLE_DEVICES=0

MODEL_PATH=./models/autoj-13b
BASE_MODEL_PATH=./models/Mistral-7B-v0.1
MODEL_TYPE=auto-j
DATA_TYPE=verbosity


python3 -u src/cal_reliability.py \
      --model-name-or-path $MODEL_PATH \
      --cali-model-name-or-path $BASE_MODEL_PATH \
      --model-type ${MODEL_TYPE} \
      --data-type ${DATA_TYPE} \
      --max-new-token 1024 \
      --logit-file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-logit.jsonl" \
      --output-file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-relia.json"
