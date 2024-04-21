#!/bin/bash
#SBATCH --job-name=t5-3b-interview-scd-28-5e5
#SBATCH --output=t5-3b-interview-scd-28-5e5.txt
#SBATCH --mem=512G
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --account csai

cd /local/scratch/pwu54/python/text-based-diarization/modeling/speaker-change-detection/sentence-level/seq2seq/t5/
deepspeed t5_scd_seq_train.py