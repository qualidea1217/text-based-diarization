#!/bin/bash
#SBATCH --job-name=t5-3b-interview-0-sd-8-6e5
#SBATCH --output=t5-3b-interview-0-sd-8-6e5.txt
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=512G

cd /local/scratch/pwu54/python/text-based-diarization/modeling/speaker-diarization/sentence-level/seq2seq/t5/
python3 t5_sd_seq_train.py