#!/bin/bash
#SBATCH --job-name=pwu54_whisper
#SBATCH --output=pwu54_whisper_out.txt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd /local/scratch/pwu54/python/text-based-diarization/preprocessing
python3 whisper_transcribe.py