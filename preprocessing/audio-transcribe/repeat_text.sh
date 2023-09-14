#!/bin/bash
#SBATCH --job-name=repeat_text
#SBATCH --output=repeat_text_out.txt
#SBATCH --cpus-per-task=32

cd /local/scratch/pwu54/python/text-based-diarization/preprocessing/audio-transcribe/
python3 repeat_text.py