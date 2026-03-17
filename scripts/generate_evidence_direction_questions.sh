#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=60:00:00
#SBATCH --job-name=eval_q_gen
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem=180G
#SBATCH --partition=177huntington
#SBATCH --gres=gpu:1
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

model="qwen3_thinking-4B"

# for default questions
# python3 ../code/generate_evidence_direction_questions.py \
#     --model "$model" \
#     --input_path "../code/outputs/questions/$model/extracted/cochrane_review_data_final_with_questions.json" \
#     --output_path "../code/outputs/questions/$model/extracted/evidence_direction_questions.json"

# for simplified questions
python3 ../code/generate_evidence_direction_questions.py \
    --model "$model" \
    --input_path "../code/outputs/questions/$model/simplified/cochrane_review_data_final_with_questions.json" \
    --output_path "../code/outputs/questions/$model/simplified/evidence_direction_questions.json" \
    --intervention_condition_key "SimplifiedExtractedText" \
    --debug

conda deactivate
