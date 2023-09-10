#!/bin/bash
#SBATCH --job-name=t5_scd_seq_train
#SBATCH --output=t5_scd_seq_train_out.txt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd /local/scratch/pwu54/python/text-based-diarization/modeling/speaker-change-detection/sentence-level/seq2seq/t5/
python3 t5_scd_seq_train.py