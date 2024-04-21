#!/bin/bash
#SBATCH --job-name=llama2-7b-d7-scd-28-3e5
#SBATCH --output=llama2-7b-d7-scd-28-3e5.txt
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48
#SBATCH --account csai

cd /local/scratch/pwu54/python/text-based-diarization/modeling/speaker-change-detection/sentence-level/seq2seq/llama/
deepspeed llama_scd_seq_train.py