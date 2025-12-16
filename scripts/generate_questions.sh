#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --job-name=q_gen
#SBATCH --cpus-per-task=8
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

# CHANGE MODEL
model="qwen3_thinking-4B"

python3 ../code/extract_intervention_condition.py \
    --model "$model" \
    --input_path ../data/cochrane_review_data_final.jsonl \
    --output_path "../code/outputs/extracted_text/$model/extracted_interventions_conditions.json" \
    --debug

python3 ../code/apply_question_template.py \
    --input_path "../code/outputs/extracted_text/$model/extracted_interventions_conditions.json" \
    --output_path "../code/outputs/questions_outputs/$model/generated_questions.json"

conda deactivate
