#!/bin/bash
#SBATCH --job-name=gt_process
#SBATCH --output=gt_process_out.txt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

cd /local/scratch/pwu54/python/text-based-diarization/preprocessing/ground-truth/
python3 gt_process.py