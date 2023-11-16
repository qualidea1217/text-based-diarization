#!/bin/bash
#SBATCH --job-name=chatgpt_sd
#SBATCH --output=chatgpt_sd_out.txt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd /local/scratch/pwu54/python/text-based-diarization/modeling/speaker-diarization/sentence-level/chatgpt/
python3 chatgpt_sd.py