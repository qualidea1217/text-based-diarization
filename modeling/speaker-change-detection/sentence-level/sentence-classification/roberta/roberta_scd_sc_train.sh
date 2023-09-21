#!/bin/bash
#SBATCH --job-name=dataset7_roberta_scd_512
#SBATCH --output=dataset7_roberta_scd_512_out.txt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd /local/scratch/pwu54/python/text-based-diarization/modeling/speaker-change-detection/sentence-level/sentence-classification/roberta/
python3 roberta_scd_sc_train.py