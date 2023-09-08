#!/bin/bash
#SBATCH --job-name=t5_scd
#SBATCH --output=t5_scd_out.txt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd /local/scratch/pwu54/python/text-based-diarization/modeling/speaker-change-detection/sentence-level/seq2seq/t5/
python3 t5model.py