#!/bin/bash
#SBATCH --job-name=led_sd_seq_train
#SBATCH --output=led_sd_seq_train_out.txt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd /local/scratch/pwu54/python/text-based-diarization/modeling/speaker-diarization/sentence-level/seq2seq/led/
python3 led_sd_seq_train.py