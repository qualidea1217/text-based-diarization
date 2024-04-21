#!/bin/bash
#SBATCH --job-name=t5_sd_seq_test
#SBATCH --output=t5_sd_seq_test_out.txt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

cd /local/scratch/pwu54/python/text-based-diarization/modeling/speaker-diarization/sentence-level/seq2seq/t5/
python3 t5_sd_seq_test.py