#!/bin/bash
#SBATCH --job-name=t5-11b-d7-scd-26-6e5
#SBATCH --output=t5-11b-d7-scd-26-6e5.txt
#SBATCH --mem=512G
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --account csai

cd /local/scratch/pwu54/python/text-based-diarization/modeling/speaker-change-detection/sentence-level/seq2seq/t5/
deepspeed t5_scd_seq_train.py