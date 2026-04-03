#!/bin/bash -l
#SBATCH --job-name=fmriprep_valuecapture
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/valuecapture_fmriprep_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=36:00:00

module load apptainer/1.4.1

export APPTAINERENV_FS_LICENSE=$HOME/freesurfer/license.txt
export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

# Optional env vars for experimental runs:
#   FMRIPREP_SUFFIX      — appended to output dir, e.g. "romeo" → fmriprep_romeo
#   FMRIPREP_EXTRA_FLAGS — extra CLI flags, e.g. "--fmap-bspline --force syn-sdc"
#   FMRIPREP_WORKDIR     — scratch dir for workflow cache (default: /scratch/gdehol)
#   FMRIPREP_BIDS_FILTER — bids filter filename in cluster_preproc/ (default: bids_filter.json)
BIDS_FILTER_FILE="/bids_input/${FMRIPREP_BIDS_FILTER:-bids_filter.json}"
OUTDIR="/data/derivatives/fmriprep${FMRIPREP_SUFFIX:+_${FMRIPREP_SUFFIX}}"
WORKDIR="${FMRIPREP_WORKDIR:-/scratch/gdehol}"

apptainer run \
  -B /shares/zne.uzh/containers/templateflow:/opt/templateflow \
  -B /shares/zne.uzh/gdehol/ds-valuecapture:/data \
  -B ${WORKDIR}:/workflow \
  -B ${PWD}:/bids_input \
  --cleanenv /shares/zne.uzh/containers/fmriprep-25.2.5 \
    /data ${OUTDIR} participant \
  --participant_label $PARTICIPANT_LABEL \
  --output-spaces T1w MNI152NLin2009cAsym:res-2 fsnative \
  --dummy-scans 4 \
  --skip_bids_validation \
  -w /workflow \
  --nthreads 16 \
  --omp-nthreads 16 \
  --low-mem \
  --no-submm-recon \
  --bids-filter-file $BIDS_FILTER_FILE \
  ${FMRIPREP_EXTRA_FLAGS}
