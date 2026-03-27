#!/bin/bash
#SBATCH --job-name=fit_second_level
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/fit_second_level_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# Fit nilearn second-level (group) GLM for the value capture fMRI task.
# Generates group z-maps and a PDF report.
#
# Usage:
#   sbatch fit_second_level.sh
#
# Optional overrides (--export key=value):
#   SPACE             BIDS space entity (default: MNI152NLin2009cAsym)
#   RES               BIDS res entity (default: 2)
#   FDR_ALPHA         FDR significance level (default: 0.05)
#   CLUSTER_THRESHOLD minimum cluster size in voxels (default: 10)

SPACE="${SPACE:-MNI152NLin2009cAsym}"
RES="${RES:-2}"
FDR_ALPHA="${FDR_ALPHA:-0.05}"
CLUSTER_THRESHOLD="${CLUSTER_THRESHOLD:-10}"

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-valuecapture
REPO=$HOME/git/value_capture

ARGS=(
    --bids-folder "$BIDS_FOLDER"
    --space "$SPACE"
    --fdr-alpha "$FDR_ALPHA"
    --cluster-threshold "$CLUSTER_THRESHOLD"
)

[ -n "$RES" ] && ARGS+=(--res "$RES")

echo "fit_second_level: space=${SPACE}  res=${RES}  fdr_alpha=${FDR_ALPHA}  cluster_threshold=${CLUSTER_THRESHOLD}"
echo "Args: ${ARGS[*]}"

. $HOME/init_conda.sh
conda activate value_capture
export PYTHONUNBUFFERED=1

python -u "$REPO/value_capture/glm/fit_second_level.py" "${ARGS[@]}"
