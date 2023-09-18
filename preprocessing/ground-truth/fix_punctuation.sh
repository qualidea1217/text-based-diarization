#!/bin/bash
#SBATCH --job-name=fix_punctuation
#SBATCH --output=fix_punctuation_out.txt
#SBATCH --cpus-per-task=4

cd /local/scratch/pwu54/python/text-based-diarization/preprocessing/ground-truth/
python3 fix_punctuation.py