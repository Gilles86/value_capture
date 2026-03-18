#!/bin/bash
#SBATCH --job-name=setup_env
#SBATCH --output=/home/gdehol/logs/setup_env_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Build the value_capture conda environment on the cluster.
# Requires internet access (pip downloads) — run once per cluster setup.
#
# Usage:
#   sbatch setup_env.sh
#
# After this completes, analysis jobs can use:
#   conda run -n value_capture python ...

source /etc/profile.d/lmod.sh

# ── locate mamba/conda ────────────────────────────────────────────────────────
if command -v mamba &>/dev/null; then
    CONDA_CMD=mamba
elif command -v conda &>/dev/null; then
    CONDA_CMD=conda
else
    echo "ERROR: neither mamba nor conda found. Install miniforge first."
    exit 1
fi

REPO=/home/$USER/git/value_capture

# ── create or update environment ──────────────────────────────────────────────
if $CONDA_CMD env list | grep -q "^value_capture "; then
    echo "Environment value_capture exists — updating..."
    $CONDA_CMD env update -n value_capture -f "$REPO/environment_linux.yml" --prune
else
    echo "Creating environment value_capture..."
    $CONDA_CMD env create -f "$REPO/environment_linux.yml"
fi

# ── install braincoder from local submodule ───────────────────────────────────
# libs/braincoder is on the keras-backend branch (pinned in .gitmodules)
conda run -n value_capture pip install -e "$REPO/libs/braincoder"

# ── install value_capture package itself ──────────────────────────────────────
conda run -n value_capture pip install -e "$REPO"

# ── verify ────────────────────────────────────────────────────────────────────
echo ""
echo "=== Verification ==="
conda run -n value_capture python -c "
import nibabel; print('nibabel:', nibabel.__version__)
import nilearn; print('nilearn:', nilearn.__version__)
import glmsingle; print('glmsingle OK')
import braincoder; print('braincoder OK')
from value_capture.utils.data import Subject
print('value_capture OK')
"

echo ""
echo "Setup complete."
