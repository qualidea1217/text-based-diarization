#!/bin/bash
#SBATCH --job-name=bert_scd_011
#SBATCH --output=bert_scd_011_out.txt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd /local/scratch/pwu54/python/text-based-diarization/modeling/speaker-change-detection/sentence-level/sentence-classification/bert/
python3 bertmodel.py