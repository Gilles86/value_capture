#!/bin/bash
#SBATCH --job-name=fmriprep_valuecapture
#SBATCH --output=/home/gdehol/logs/valuecapture_fmriprep_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=36:00:00

source /etc/profile.d/lmod.sh
module --ignore_cache load apptainer/1.4.1

export APPTAINERENV_FS_LICENSE=$HOME/freesurfer/license.txt
export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

BIDS_FILTER_FILE="/bids_input/bids_filter.json"

apptainer run \
  -B /shares/zne.uzh/containers/templateflow:/opt/templateflow \
  -B /shares/zne.uzh/gdehol/ds-valuecapture:/data \
  -B /scratch/gdehol:/workflow \
  -B ${PWD}:/bids_input \
  --cleanenv /shares/zne.uzh/containers/fmriprep-25.2.5 \
    /data /data/derivatives/fmriprep participant \
  --participant_label $PARTICIPANT_LABEL \
  --output-spaces T1w MNI152NLin2009cAsymm:res-2 fsnative \
  --dummy-scans 4 \
  --skip_bids_validation \
  -w /workflow \
  --nthreads 16 \
  --omp-nthreads 16 \
  --low-mem \
  --no-submm-recon \
  --bids-filter-file $BIDS_FILTER_FILE
