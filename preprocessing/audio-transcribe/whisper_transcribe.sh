#!/bin/bash
#SBATCH --job-name=whisper_transcribe
#SBATCH --output=whisper_transcribe_out.txt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd /local/scratch/pwu54/python/text-based-diarization/preprocessing/audio-transcribe/
python3 whisper_transcribe.py