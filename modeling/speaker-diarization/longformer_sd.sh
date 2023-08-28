#!/bin/bash
#SBATCH --job-name=longformer_sd
#SBATCH --output=longformer_sd_out.txt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd /local/scratch/pwu54/python/text-based-diarization/modeling/speaker-diarization/
python3 longformermodel.py