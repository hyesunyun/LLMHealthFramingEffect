#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=q_gen
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G
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

##### DEFAULT QUESTIONS #####
# Step 1: Extract Interventions and Conditions
# python3 ../../code/extract_intervention_condition.py \
#     --model "$model" \
#     --input_path ../../data/cochrane_review_data_cleaned.jsonl \
#     --output_path "../../code/outputs/extracted_text/$model/extracted_interventions_conditions.json"

# Step 2: Generate Questions using the extracted interventions and conditions
python3 ../../code/apply_question_templates.py \
    --input_path "../../code/outputs/extracted_text/$model/extracted_interventions_conditions.json" \
    --output_path "../../code/outputs/questions/$model/extracted/cochrane_review_data_final_with_questions.json" \
    --run_scoring

##### END - DEFAULT QUESTIONS #####

##### SIMPLIFIED QUESTIONS #####

# STEP 1: simplify the extracted interventions and conditions
# USING Qwen3-4B (instruct and not thinking)
# python3 ../../code/simplify_intervention_condition.py \
    # --model qwen3-4B \
    # --input_path ../../code/outputs/extracted_text/qwen3_thinking-4B/extracted_interventions_conditions.json \
    # --output_path ../../code/outputs/extracted_text/qwen3-4B/simplified_extracted_interventions_conditions.json

# STEP 2: Generate questions using the simplified interventions and conditions
python3 ../../code/apply_question_templates.py \
    --input_path "../../code/outputs/extracted_text/extracted_interventions_conditions_with_simplified.json" \
    --output_path "../../code/outputs/questions/$model/simplified/cochrane_review_data_final_with_questions.json" \
    --intervention_condition_key "SimplifiedExtractedText" \
    --question_types "effectiveness" "efficacy" \
    --run_scoring
##### END - SIMPLIFIED QUESTIONS #####

conda deactivate
