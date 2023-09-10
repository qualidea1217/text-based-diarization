#!/bin/bash
#SBATCH --job-name=roberta_scd_sc_data
#SBATCH --output=roberta_scd_sc_data.txt
#SBATCH --cpus-per-task=16

cd /local/scratch/pwu54/python/text-based-diarization/modeling/speaker-change-detection/sentence-level/sentence-classification/roberta/
python3 roberta_scd_sc_data.py