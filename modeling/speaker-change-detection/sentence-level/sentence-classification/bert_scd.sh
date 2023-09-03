#!/bin/bash
#SBATCH --job-name=bert_scd
#SBATCH --output=bert_scd_out.txt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd /local/scratch/pwu54/python/text-based-diarization/modeling/speaker-change-detection/sentence-level/sentence-classification/
python3 bertmodel.py