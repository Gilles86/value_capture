#!/bin/bash
#SBATCH --job-name=vc_surf_neuropythy
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --output=/home/%u/logs/value_capture/surf_neuropythy_%A_%a.txt
#SBATCH --cpus-per-task=16   # neuropythy's Java registration scales with threads (retsupp)
#SBATCH --mem=24G
#SBATCH --time=03:00:00
# Sample the volumetric PRF maps to fsnative and run neuropythy's Bayesian retinotopy
# (inferred_varea / _angle / _eccen into the FreeSurfer subject's surf/).
#   PRF_DERIV=prf_glmsingle_items sbatch ... (default); neuropythy outputs in surf/ are overwritten
#   mkdir -p ~/logs/value_capture && sbatch --array=4-15 value_capture/prf/slurm_jobs/surface_neuropythy.sh
set -eo pipefail
SUB=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"
export PYTHONUNBUFFERED=1
source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
cd "$HOME/git/value_capture"
BIDS=/shares/zne.uzh/gdehol/ds-valuecapture
conda activate value_capture
export NEUROPYTHY_PYTHON=$HOME/data/conda/envs/retsupp_neuropythy/bin/python   # has java
PRF_DERIV="${PRF_DERIV:-prf_glmsingle_items}"   # the bar + search-items fits (2026-11-17)
python -u value_capture/prf/sample_prf_to_surface.py "$SUB" --bids-folder "$BIDS" --prf-deriv "$PRF_DERIV"
python -u value_capture/prf/register_retinotopy.py "$SUB" --bids-folder "$BIDS" --prf-deriv "$PRF_DERIV"
