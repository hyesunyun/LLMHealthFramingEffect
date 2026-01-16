#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=eval
#SBATCH --cpus-per-task=28
#SBATCH --ntasks-per-node=1
#SBATCH --mem=10G
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

# python3 ../code/run_lexical_analysis.py \
#     --file_path ../code/outputs/responses/gpt-5.1/batch_question_responses.jsonl \
#     --output_path ../code/outputs/analysis/gpt-5.1_analysis_results.json

# python3 ../code/run_lexical_analysis.py \
#     --file_path ../code/outputs/responses/claude_4.5_sonnet/batch_question_responses.jsonl \
#     --output_path ../code/outputs/analysis/claude_4.5_sonnet_analysis_results.json

# python3 ../code/run_lexical_analysis.py \
#     --file_path ../code/outputs/responses/qwen3-4B/question_responses.json \
#     --output_path ../code/outputs/analysis/qwen3-4B_analysis_results.json

conda deactivate
