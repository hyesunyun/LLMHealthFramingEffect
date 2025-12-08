#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=find
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G
#SBATCH --partition=short
#SBATCH -o output_%j.txt  # Standard output file
#SBATCH -e error_%j.txt  # Standard error file

# Your program/command here
module purge
module load explorer anaconda3/2024.06

source activate base
source activate llm_health_framing_effect
conda activate llm_health_framing_effect

conda info

python3 ../code/find_num_included_studies.py \
 --input_path ../data/cochrane_review_data_300_samples_hl_annotated.json \
 --output_path ../code/outputs/extract_num_studies/cochrane_review_data_300_samples_hl_annotated_webscraping.json \
#  --debug

python3 ../code/find_num_included_studies.py \
 --input_path ../data/cochrane_review_data_300_samples_ac_annotated.json \
 --output_path ../code/outputs/extract_num_studies/cochrane_review_data_300_samples_ac_annotated_webscraping.json \
#  --debug

conda deactivate
