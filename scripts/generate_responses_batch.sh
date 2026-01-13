#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=batch_response
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=10G
#SBATCH --partition=short
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
  "claude_4.5_sonnet"
)

echo "Running LLM response generation"
for model in "${models[@]}"; do
    python3 ../code/generate_responses.py \
        --model "$model" \
        --input_path ../code/outputs/questions/qwen3_thinking-4B/cochrane_review_data_final_with_questions_new.json \
        --output_path "../code/outputs/responses/$model/question_responses.json"
done

conda deactivate
