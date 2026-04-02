
conda activate LLMHealthFramingEffect

models=(
  "gpt-5.1"
  "claude_4.5_sonnet"
  "llama3.3_instruct_70B"
  "llama4_maverick"
  "huatuo-7B"
  "huatuo-8B"
  "qwen3-4B"
  "qwen3-30B"
)

echo "Running evaluation for FRAMED CONDITION (technical questions)"
for model in "${models[@]}"; do
    python3 -u ../code/run_evaluation.py \
        --file_path "../code/outputs/responses/${model}/question_responses.json" \
        --output_path "../code/outputs/evaluation/${model}_eval_results.json" \
        --eval_path "../code/outputs/questions/qwen3_thinking-4B/extracted/evidence_direction_questions_final.json" \
        --data_type "framing"
done

echo "Running evaluation for BASELINE CONDITION (technical questions)"
for model in "${models[@]}"; do
    python3 -u ../code/run_evaluation.py \
        --file_path "../code/outputs/baseline_responses/${model}/positive_question_responses.json" \
        --output_path "../code/outputs/baseline_evaluation/${model}_eval_results.json" \
        --eval_path "../code/outputs/questions/qwen3_thinking-4B/extracted/evidence_direction_questions_final.json" \
        --data_type "baseline"
done

echo "Running evaluation for FRAMED CONDITION (plain language questions)"
for model in "${models[@]}"; do
    python3 -u ../code/run_evaluation.py \
        --file_path "../code/outputs/responses/${model}/simplified_question_responses.json" \
        --output_path "../code/outputs/evaluation/${model}_eval_results_simplified.json" \
        --eval_path "../code/outputs/questions/qwen3_thinking-4B/simplified/evidence_direction_questions_raw.json" \
        --data_type "framing"
done

echo "Running evaluation for BASELINE CONDITION (plain language questions)"
for model in "${models[@]}"; do
    python3 -u ../code/run_evaluation.py \
        --file_path "../code/outputs/baseline_responses/${model}/positive_simplified_question_responses.json" \
        --output_path "../code/outputs/baseline_evaluation/${model}_eval_results_simplified.json" \
        --eval_path "../code/outputs/questions/qwen3_thinking-4B/simplified/evidence_direction_questions_raw.json" \
        --data_type "baseline"
done

conda deactivate
