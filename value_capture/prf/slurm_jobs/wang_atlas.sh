#!/bin/bash
#SBATCH --job-name=vc_wang_atlas
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --output=/home/%u/logs/value_capture/wang_atlas_%A_%a.txt
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
# Wang 2015 maximum-probability atlas (IPS0-5, SPL1, FEF, ...) on each subject's surface and volume, as in
# retsupp/neuropythy/infer_wang_atlas.py. Neuropythy's register_retinotopy labels only V1-V3, hV4, VO, LO, TO, V3A/B.
# Writes surf/{lh,rh}.wang15_mplbl.mgz (+ fplbl, volume) into the FreeSurfer subject.
#   sbatch --array=4-15 value_capture/prf/slurm_jobs/wang_atlas.sh
set -eo pipefail
SUB=$(printf "sub-%02d_ses-1" "$SLURM_ARRAY_TASK_ID")
export SUBJECTS_DIR=/shares/zne.uzh/gdehol/ds-valuecapture/derivatives/fmriprep/sourcedata/freesurfer
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"
PY=$HOME/data/conda/envs/retsupp_neuropythy/bin/python
export PATH="$(dirname $PY):$PATH"
$PY -W ignore -c 'import numpy as _np; _np.complex = complex; from neuropythy.__main__ import main; import sys; sys.exit(main(sys.argv[1:]))' \
    atlas "$SUB" --atlases wang15 --volume-export --surface-export --output-path "$SUBJECTS_DIR/$SUB/surf" --verbose
ls "$SUBJECTS_DIR/$SUB/surf" | grep -i wang
