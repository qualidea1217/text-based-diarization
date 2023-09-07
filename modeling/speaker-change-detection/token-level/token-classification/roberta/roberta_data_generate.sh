#!/bin/bash
#SBATCH --job-name=bert_data_generate
#SBATCH --output=bert_data_generate.txt
#SBATCH --cpus-per-task=32

cd /local/scratch/pwu54/python/text-based-diarization/modeling/speaker-change-detection/token-level/token-classification/roberta/
python3 robert_data_generate.py