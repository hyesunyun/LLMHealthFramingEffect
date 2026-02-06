#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --partition=frink
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=eval
#SBATCH --cpus-per-task=28
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80G
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
  # "gpt-5.1" # DONE
  # "claude_4.5_sonnet"
  "api_llama-3.3"
)

for model in "${models[@]}"; do
    python3 ../code/run_analysis.py \
        --file_path "../code/outputs/responses/${model}/question_responses.json" \
        --output_path "../code/outputs/analysis/${model}_analysis_results.json"
done

conda deactivate
