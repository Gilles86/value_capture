#!/bin/bash
#SBATCH --job-name=glmsingle_upsample_exp
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/glmsingle_upsample_exp_%A-%a.txt
#SBATCH --array=1-3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16

$HOME/.conda/envs/value_capture/bin/python -u \
    /home/gdehol/git/value_capture/notes/experiments/glmsingle_upsample_exp.py
