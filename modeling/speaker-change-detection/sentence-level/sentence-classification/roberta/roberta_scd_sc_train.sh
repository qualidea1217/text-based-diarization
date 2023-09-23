#!/bin/bash
#SBATCH --job-name=roberta-d7-u4-s1-21
#SBATCH --output=roberta-d7-u4-s1-21.txt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd /local/scratch/pwu54/python/text-based-diarization/modeling/speaker-change-detection/sentence-level/sentence-classification/roberta/
python3 roberta_scd_sc_train.py