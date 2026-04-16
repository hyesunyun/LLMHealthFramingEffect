#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --job-name=response
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G
#SBATCH --partition=177huntington
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

for model in "${models[@]}"; do
    python3 -u ../../code/generate_paraphrased_baseline_templates.py \
        --model qwen3_thinking-4B \
        --input_path ../../code/prompts/question_templates.json \
        --output_path ../../code/outputs/questions/qwen3_thinking-4B/paraphrased/paraphrased_question_templates.json
done

conda deactivate
