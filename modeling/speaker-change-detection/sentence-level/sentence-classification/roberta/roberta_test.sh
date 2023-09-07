#!/bin/bash
#SBATCH --job-name=roberta_test
#SBATCH --output=roberta_test_out.txt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd /local/scratch/pwu54/python/text-based-diarization/modeling/speaker-change-detection/sentence-level/sentence-classification/roberta/
python3 roberta_test.py