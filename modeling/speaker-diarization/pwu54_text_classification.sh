#!/bin/bash
#SBATCH --job-name=pwu54_text_classification
#SBATCH --output=pwu54_text_classification_out.txt
#SBATCH --gres=gpu:1

cd /local/scratch/pwu54/python/text-based-diarization/modeling/speaker-diarization/
python3 text_classification.py