#!/bin/bash
#SBATCH --job-name=glmsingle_upsample_exp
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/glmsingle_upsample_exp_%A-%a.txt
#SBATCH --array=1-3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00

REPO=$HOME/git/value_capture

. $HOME/init_conda.sh
conda activate value_capture
export PYTHONUNBUFFERED=1

python -u "$REPO/notes/experiments/glmsingle_upsample_exp.py"
