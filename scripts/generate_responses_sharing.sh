#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --job-name=response
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=120G
#SBATCH --partition=sharing
#SBATCH --gres=gpu:h100:1
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

model="qwen3_thinking-4B"

python3 ../code/generate_responses.py \
    --model "$model" \
    --input_path "../code/outputs/questions/$model/cochrane_review_data_final_with_questions.json" \
    --output_path "../code/outputs/responses/$model/question_responses.json" \
    --debug

conda deactivate
