#!/bin/bash
#SBATCH --job-name=roberta_scd_001
#SBATCH --output=roberta_scd_001_out.txt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd /local/scratch/pwu54/python/text-based-diarization/modeling/speaker-change-detection/sentence-level/sentence-classification/roberta/
python3 robertamodel.py