#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=12-00:00:00
#SBATCH --job-name=response
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem=180G
#SBATCH --partition=177huntington
#SBATCH --gres=gpu:2
#SBATCH -o output_%j.txt                     # Standard output file
#SBATCH -e error_%j.txt                      # Standard error file

module purge
module load explorer anaconda3/2024.06

source activate base
source activate llm_health_framing_effect
conda activate llm_health_framing_effect

conda info

export HF_HOME="/scratch/yun.hy/.cache"
export HUGGINGFACE_HUB_CACHE="/scratch/yun.hy/.cache"
export XDG_CACHE_HOME="/scratch/yun.hy/.cache"

models=(
  # "gpt-5.1"
  # "claude_4.5_sonnet"
  # "qwen3-4B"
#   "qwen3_thinking-4B"
  # "qwen3-30B"
#   "qwen3_thinking-30B"
  # "llama3.3_instruct_70B"
  # "huatuo-7B"
  # "huatuo-8B"
  # "huatuo-70B"
)

# echo "Running LLM response generation"
# for model in "${models[@]}"; do
#     python3 ../code/generate_responses.py \
#         --model "$model" \
#         --input_path ../code/outputs/questions/qwen3_thinking-4B/cochrane_review_data_final_with_questions.json \
#         --output_path "../code/outputs/responses/$model/question_responses.json" 

# python3 ../code/generate_responses.py \
#         --model qwen3_thinking-30B \
#         --input_path ../code/outputs/questions/qwen3_thinking-4B/cochrane_review_data_final_with_questions.json \
#         --output_path ../code/outputs/responses/qwen3_thinking-30B/question_responses.json

# python3 ../code/generate_responses.py \
#         --model huatuo-70B \
#         --input_path ../code/outputs/questions/qwen3_thinking-4B/cochrane_review_data_final_with_questions.json \
#         --output_path ../code/outputs/responses/huatuo-70B/question_responses.json

python3 ../code/generate_responses.py \
        --model qwen3-4B \
        --input_path ../code/outputs/questions/qwen3_thinking-4B/cochrane_review_data_final_with_questions.json \
        --output_path ../code/outputs/responses/qwen3-4B/test.json \
        --batch_size 8

conda deactivate
