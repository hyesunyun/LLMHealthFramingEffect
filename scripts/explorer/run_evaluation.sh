#!/bin/bash
#SBATCH --time=99:00:00
#SBATCH --partition=frink
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=eval
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40G
#SBATCH -o output_%j.txt                     # Standard output file
#SBATCH -e error_%j.txt                      # Standard error file

module purge
module load explorer anaconda3/2024.06

source activate base
source activate llm_health_framing_effect
conda activate llm_health_framing_effect

export HF_HOME="/scratch/yun.hy/.cache"
export HUGGINGFACE_HUB_CACHE="/scratch/yun.hy/.cache"
export XDG_CACHE_HOME="/scratch/yun.hy/.cache"
export GOOGLE_APPLICATION_CREDENTIALS="/scratch/yun.hy/question-framing-fd1030433dda.json"

models=(
  "gpt-5.1"
  "claude_4.5_sonnet"
  "api-llama3.3"
  "api-llama4"
  "huatuo-7B"
  "huatuo-8B"
  "qwen3-4B"
  "qwen3-30B"
)

# FOR POSITIVE vs NEGATIVE
# for model in "${models[@]}"; do
#     python3 -u ../code/run_evaluation.py \
#         --file_path "../code/outputs/responses/${model}/question_responses.json" \
#         --output_path "../code/outputs/evaluation/${model}_eval_results.json" \
#         --eval_path "../code/outputs/questions/qwen3_thinking-4B/extracted/evidence_direction_questions_final.json" \
#         --embedding_path "../code/outputs/evaluation/${model}_embeddings" \
#         --data_type "framing"
# done

# # FOR BASELINE (TWO SAMPLES OF POSITIVE)
# for model in "${models[@]}"; do
#     python3 -u ../code/run_evaluation.py \
#         --file_path "../code/outputs/baseline_responses/${model}/positive_question_responses.json" \
#         --output_path "../code/outputs/baseline_evaluation/${model}_eval_results.json" \
#         --eval_path "../code/outputs/questions/qwen3_thinking-4B/extracted/evidence_direction_questions_final.json" \
#         --embedding_path "../code/outputs/baseline_evaluation/${model}_embeddings" \
#         --data_type "basic_baseline"
# done

# echo "Running evaluation for PARAPHRASED BASELINE CONDITION (technical questions)"
# for model in "${models[@]}"; do
#     python3 -u ../code/run_evaluation.py \
#         --file_path "../code/outputs/paraphrased_baseline_responses/${model}/positive_question_responses.json" \
#         --output_path "../code/outputs/paraphrased_baseline_evaluation/${model}_eval_results.json" \
#         --eval_path "../code/outputs/questions/qwen3_thinking-4B/extracted/evidence_direction_questions_final.json" \
#         --embedding_path "../code/outputs/paraphrased_baseline_evaluation/${model}_embeddings" \
#         --data_type "para_baseline"
# done

# # FOR POSITIVE vs NEGATIVE
# for model in "${models[@]}"; do
#     python3 -u ../code/run_evaluation.py \
#         --file_path "../code/outputs/responses/${model}/simplified_question_responses.json" \
#         --output_path "../code/outputs/evaluation/${model}_eval_results_simplified.json" \
#         --eval_path "../code/outputs/questions/qwen3_thinking-4B/simplified/evidence_direction_questions_raw.json" \
#         --embedding_path "../code/outputs/evaluation/${model}_embeddings_simplified" \
#         --data_type "framing"
# done

# # FOR BASELINE (TWO SAMPLES OF POSITIVE)
# for model in "${models[@]}"; do
#     python3 -u ../code/run_evaluation.py \
#         --file_path "../code/outputs/baseline_responses/${model}/positive_simplified_question_responses.json" \
#         --output_path "../code/outputs/baseline_evaluation/${model}_eval_results_simplified.json" \
#         --eval_path "../code/outputs/questions/qwen3_thinking-4B/simplified/evidence_direction_questions_raw.json" \
#         --embedding_path "../code/outputs/baseline_evaluation/${model}_embeddings_simplified" \
#         --data_type "basic_baseline"
# done

# echo "Running evaluation for PARAPHRASED BASELINE CONDITION (plain language questions)"
# for model in "${models[@]}"; do
#     python3 -u ../code/run_evaluation.py \
#         --file_path "../code/outputs/paraphrased_baseline_responses/${model}/positive_simplified_question_responses.json" \
#         --output_path "../code/outputs/paraphrased_baseline_evaluation/${model}_eval_results_simplified.json" \
#         --eval_path "../code/outputs/questions/qwen3_thinking-4B/simplified/evidence_direction_questions_raw.json" \
#         --embedding_path "../code/outputs/paraphrased_baseline_evaluation/${model}_embeddings_simplified" \
#         --data_type "para_baseline"
# done

conda deactivate