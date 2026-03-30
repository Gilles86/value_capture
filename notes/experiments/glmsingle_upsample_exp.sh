#!/bin/bash
#SBATCH --job-name=glmsingle_upsample_exp
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/glmsingle_upsample_exp_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00

# Run the temporal-upsampling GLMSingle experiment for one subject.
#
# Usage:
#   sbatch --array=1-2 glmsingle_upsample_exp.sh
#   sbatch --export=SLURM_ARRAY_TASK_ID=1 glmsingle_upsample_exp.sh   # single sub

REPO=$HOME/git/value_capture

echo "glmsingle_upsample_exp: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

. $HOME/init_conda.sh
conda activate value_capture
export PYTHONUNBUFFERED=1

python -u "$REPO/notes/experiments/glmsingle_upsample_exp.py"
