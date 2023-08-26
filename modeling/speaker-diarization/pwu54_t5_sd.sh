#!/bin/bash
#SBATCH --job-name=pwu54_t5_sd
#SBATCH --output=pwu54_t5_sd_out.txt
#SBATCH --gres=gpu:1

cd /local/scratch/pwu54/python/text-based-diarization/modeling/speaker-diarization/
python3 t5model.py