#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --job-name=filter_topic
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=180G
#SBATCH --partition=177huntington
#SBATCH --gres=gpu:2
#SBATCH -o output_%j.txt  # Standard output file
#SBATCH -e error_%j.txt  # Standard error file

# Your program/command here
module purge
module load explorer anaconda3/2024.06 cuda/12.8.0

source activate base
source activate llm_health_framing_effect
conda activate llm_health_framing_effect

conda info

export HF_HOME="/scratch/yun.hy/.cache"
export HUGGINGFACE_HUB_CACHE="/scratch/yun.hy/.cache"
export XDG_CACHE_HOME="/scratch/yun.hy/.cache"

 python3 ../code/filter_reviews_by_topic_type.py \
 --model qwen3_thinking-30B \
 --input_path ../data/cochrane_review_data_filtered_mismatched_studies.jsonl \
 --output_path ../code/outputs/filtering_data/cochrane_review_data_filtered_topic_type_qwen3thinking_30B.jsonl

python3 ../code/filter_reviews_by_topic_type.py \
 --model qwen3_thinking-4B \
 --input_path ../data/cochrane_review_data_filtered_mismatched_studies.jsonl \
 --output_path ../code/outputs/filtering_data/cochrane_review_data_filtered_topic_type_qwen3thinking_4B.jsonl

conda deactivate
