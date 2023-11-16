#!/bin/bash
#SBATCH --job-name=chatgpt_scd
#SBATCH --output=chatgpt_scd_out.txt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd /local/scratch/pwu54/python/text-based-diarization/modeling/speaker-change-detection/sentence-level/chatgpt/
python3 chatgpt_scd.py