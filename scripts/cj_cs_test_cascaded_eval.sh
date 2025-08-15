# ============================================================
# Run MEC + BPC with GPT-3.5-turbo on separated verbosity dataset
# ============================================================

mec_dir="datasets/alpaca/cj-cs-test"
gpt_results_dir="gathered_data/cj_cs_test_result"


python FairEval.py \
          -q "${mec_dir}/mec_bpc_questions.jsonl" \
          -a "${mec_dir}/mec_bpc_first_answer.jsonl" "${mec_dir}/mec_bpc_second_answer.jsonl" \
          -o "${gpt_results_dir}/gpt_judgement.json" \
          -m "gpt-3.5-turbo" \
          --bpc "1" \
          -k "3"

# ============================================================
# Execute CascadedEval on the previously gained judgements
# ============================================================

MODEL_TYPE="auto-j"
DATA_TYPE="verbosity"

relia_scores_dir="gathered_data/cascaded-eval-results/relia_scores/${MODEL_TYPE}"
result_dir="gathered_data/cascaded-eval-results/final_results"

python3 -u src/cascaded_eval.py \
      --data-type $DATA_TYPE \
      --logit-file1 "${relia_scores_dir}/${DATA_TYPE}-logit.jsonl" \
      --output-file1 "${relia_scores_dir}/${DATA_TYPE}-relia.json" \
      --logit-file-gpt "${gpt_results_dir}/gpt_judgement.json" \
      --final-output-file "${result_dir}/${DATA_TYPE}.json"