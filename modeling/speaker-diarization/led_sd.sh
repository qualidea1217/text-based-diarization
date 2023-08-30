#!/bin/bash
#SBATCH --job-name=led_sd
#SBATCH --output=led_sd_out.txt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd /local/scratch/pwu54/python/text-based-diarization/modeling/speaker-diarization/
python3 ledmodel.py