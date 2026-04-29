#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --job-name=test
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=5G
#SBATCH --partition=177huntington
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

# for default questions
# python3 -u ../../code/generate_baseline_responses.py \
#         --model gpt-5.1 \
#         --input_path ../../code/outputs/questions/qwen3_thinking-4B/extracted/cochrane_review_data_final_with_questions.json \
#         --output_path ../../code/outputs/baseline_responses/gpt-5.1/TEST_positive_question_responses.json \
#         --debug

# for simplified questions
# python3 -u ../../code/generate_baseline_responses.py \
#         --model gpt-5.1 \
#         --input_path ../../code/outputs/questions/qwen3_thinking-4B/simplified/cochrane_review_data_final_with_questions.json \
#         --output_path ../../code/outputs/baseline_responses/gpt-5.1/TEST_positive_simplified_question_responses.json \
#         --debug


# for paraphrased baseline (default questions)
# python3 -u ../../code/generate_paraphrased_baseline_responses.py \
#         --model gpt-5.1 \
#         --input_path ../../code/outputs/questions/qwen3_thinking-4B/extracted/cochrane_review_data_final_with_questions.json \
#         --output_path "../../code/outputs/paraphrased_baseline_responses/gpt-5.1/positive_question_responses.json" \
#         --debug

conda deactivate
