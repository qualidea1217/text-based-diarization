#!/bin/bash
#SBATCH --job-name=roberta_scd_tc_data
#SBATCH --output=roberta_scd_tc_data_out.txt
#SBATCH --cpus-per-task=32

cd /local/scratch/pwu54/python/text-based-diarization/modeling/speaker-change-detection/token-level/token-classification/roberta/
python3 roberta_scd_tc_data.py