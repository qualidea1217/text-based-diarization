#!/bin/bash
#SBATCH --job-name=pwu54_pure_text
#SBATCH --output=pwu54_pure_text_out.txt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

cd /local/scratch/pwu54/python/text-based-diarization/preprocessing
python3 pure_text_transcript.py