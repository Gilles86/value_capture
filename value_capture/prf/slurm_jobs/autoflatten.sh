#!/bin/bash
#SBATCH --job-name=vc_autoflatten
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/home/%u/logs/value_capture/autoflatten_%A_%a.txt
# Cut and flatten each subject's cortex with autoflatten (pycortex skill, references/autoflatten.md), from the
# fMRIPrep FreeSurfer recon (sub-XX_ses-1) that the pycortex subjects valuecapture.sub-XX were imported from.
# Writes surf/{lh,rh}.autoflatten.flat.patch.3d (+ .png distortion QC). Then pull the patches back and import with
# value_capture/visualize/pycortex/import_flatmaps.py.
#   sbatch --array=4-15 value_capture/prf/slurm_jobs/autoflatten.sh
set -eo pipefail
SUB=$(printf "sub-%02d_ses-1" "$SLURM_ARRAY_TASK_ID")
export FREESURFER_HOME=/shares/zne.uzh/containers/fmriprep-25.2.5/opt/freesurfer
export PATH=$FREESURFER_HOME/bin:$PATH
export FS_LICENSE=/shares/zne.uzh/containers/freesurfer/license.txt   # mri_label2label silently maps 0 vertices without it
export SUBJECTS_DIR=/shares/zne.uzh/gdehol/ds-valuecapture/derivatives/fmriprep/sourcedata/freesurfer   # has fsaverage
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"
source ~/data/miniforge3/etc/profile.d/conda.sh
conda activate autoflatten
# jaxlib >= 0.11 aborts on an XLA flag autoflatten sets: sitecustomize.py patch (pycortex skill)
export PYTHONPATH=/shares/zne.uzh/gdehol/ds-tmsrisk/derivatives/autoflatten/xla_fix${PYTHONPATH:+:$PYTHONPATH}
echo "$SUB on $(hostname)"
time autoflatten run "$SUBJECTS_DIR/$SUB" --parallel --overwrite \
    --output-dir "$SUBJECTS_DIR/$SUB/surf" --n-cores "$SLURM_CPUS_PER_TASK"
