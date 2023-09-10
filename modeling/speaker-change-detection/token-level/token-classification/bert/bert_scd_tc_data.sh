#!/bin/bash
#SBATCH --job-name=bert_scd_tc_data
#SBATCH --output=bert_scd_tc_data_out.txt
#SBATCH --cpus-per-task=16

cd /local/scratch/pwu54/python/text-based-diarization/modeling/speaker-change-detection/token-level/token-classification/bert/
python3 bert_scd_tc_data.py