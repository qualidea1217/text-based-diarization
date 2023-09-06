#!/bin/bash
#SBATCH --job-name=roberta_data_generate
#SBATCH --output=roberta_data_generate.txt
#SBATCH --cpus-per-task=16

cd /local/scratch/pwu54/python/text-based-diarization/modeling/speaker-change-detection/sentence-level/sentence-classification/roberta/
python3 roberta_data_generate.py