#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=extract
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

# python3 ../code/extract_num_studies_from_abstract.py \
#  --model gpt-5.1 \
#  --input_path ../data/cochrane_review_data_300_samples_hl_annotated.json \
#  --output_path ../code/outputs/extract_num_studies/cochrane_review_data_300_samples_hl_annotated_gpt.json \
#  --plot_path ../code/outputs/extract_num_studies/300_sample_abs_diff_histogram_gpt.pdf \
#  --debug

#  python3 ../code/extract_num_studies_from_abstract.py \
#  --model deepseek_distill-qwen32B \
#  --input_path ../data/cochrane_review_data_300_samples_hl_annotated.json \
#  --output_path ../code/outputs/extract_num_studies/cochrane_review_data_300_samples_hl_annotated_deepseek_qwen32B.json \
#  --plot_path ../code/outputs/extract_num_studies/300_sample_abs_diff_histogram_deepseek_qwen32B.pdf \
#  --debug

 python3 ../code/extract_num_studies_from_abstract.py \
 --model deepseek_distill-llama70B \
 --input_path ../data/cochrane_review_data_300_samples_hl_annotated.json \
 --output_path ../code/outputs/extract_num_studies/cochrane_review_data_300_samples_hl_annotated_deepseek_llama70B.json \
 --plot_path ../code/outputs/extract_num_studies/300_sample_abs_diff_histogram_deepseek_llama70B.pdf \
#  --debug
 
#  python3 ../code/extract_num_studies_from_abstract.py \
#  --model qwen3_thinking-4B \
#  --input_path ../data/cochrane_review_data_300_samples_hl_annotated.json \
#  --output_path ../code/outputs/extract_num_studies/cochrane_review_data_300_samples_hl_annotated_qwen3_thinking_4B.json \
#  --plot_path ../code/outputs/extract_num_studies/300_sample_abs_diff_histogram_qwen3_thinking_4B.pdf \
#  --debug

#  python3 ../code/extract_num_studies_from_abstract.py \
#  --model qwen3_thinking-30B \
#  --input_path ../data/cochrane_review_data_300_samples_hl_annotated.json \
#  --output_path ../code/outputs/extract_num_studies/cochrane_review_data_300_samples_hl_annotated_qwen3_thinking_30B.json \
#  --plot_path ../code/outputs/extract_num_studies/300_sample_abs_diff_histogram_qwen3_thinking_30B.pdf \
#  --debug

# python3 ../code/extract_num_studies_from_abstract.py \
#  --model llama3.3_instruct_70B \
#  --input_path ../data/cochrane_review_data.jsonl \
#  --output_path ../data/cochrane_review_data_with_num_study_llama3.jsonl

conda deactivate
