#!/bin/bash
#SBATCH --job-name=t5_sd
#SBATCH --output=t5_sd_out.txt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd /local/scratch/pwu54/python/text-based-diarization/modeling/speaker-diarization/sentence-level/seq2seq/
python3 t5model.py