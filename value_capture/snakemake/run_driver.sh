#!/bin/bash
#SBATCH --job-name=vc_snake_driver
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --qos=medium
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/gdehol/git/value_capture/value_capture/snakemake/logs/driver_%j.log
# Snakemake driver for value_capture (see Snakefile header). Resubmit if it
# hits walltime — .snakemake/ persistence resumes where it left off.
#   ssh sciencecluster 'cd ~/git/value_capture && sbatch value_capture/snakemake/run_driver.sh [target]'
set -eo pipefail
echo "Host: $(hostname)  JobID: ${SLURM_JOB_ID}  Started: $(date)"
cd "$HOME/git/value_capture"
source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate retsupp_snake   # has snakemake + executor-plugin-slurm
ARGS=(--snakefile value_capture/snakemake/Snakefile
      --workflow-profile value_capture/snakemake/profile
      --configfile value_capture/snakemake/config.yaml)
snakemake "${ARGS[@]}" --unlock || true
exec snakemake "${ARGS[@]}" --rerun-incomplete "$@"
