#!/bin/bash
#SBATCH --job-name=gold_script
#SBATCH --output=gold_script_out.txt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd /local/scratch/pwu54/python/text-based-diarization/preprocessing/
python3 gold_script_preprocessing.py