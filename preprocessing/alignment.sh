#!/bin/bash
#SBATCH --job-name=alignment
#SBATCH --output=alignment_out.txt
#SBATCH --cpus-per-task=8

cd /local/scratch/pwu54/python/text-based-diarization/preprocessing
python3 alignment.py