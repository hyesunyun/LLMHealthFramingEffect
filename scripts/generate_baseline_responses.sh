#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=12-00:00:00
#SBATCH --job-name=baseline
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80G
#SBATCH --partition=177huntington
#SBATCH --gres=gpu:2
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

models=(
  # "huatuo-7B"
  "huatuo-8B"
  "qwen3-4B"
  "qwen3-30B"
  # "huatuo-70B"
)

# echo "Running LLM baseline response generation for default questions"
# for model in "${models[@]}"; do
#     python3 -u ../code/generate_baseline_responses.py \
#         --model "$model" \
#         --input_path ../code/outputs/questions/qwen3_thinking-4B/extracted/cochrane_review_data_final_with_questions.json \
#         --output_path "../code/outputs/baseline_responses/$model/positive_question_responses.json" \
#         --batch_size 2 # change this based on the model size and GPU memory
# done

echo "Running LLM baseline response generation for simplified questions"
for model in "${models[@]}"; do
    python3 -u ../code/generate_baseline_responses.py \
        --model "$model" \
        --input_path ../code/outputs/questions/qwen3_thinking-4B/simplified/cochrane_review_data_final_with_questions.json \
        --output_path "../code/outputs/baseline_responses/$model/positive_simplified_question_responses.json" \
        --batch_size 16
done

conda deactivate
