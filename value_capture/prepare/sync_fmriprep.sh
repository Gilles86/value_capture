#!/bin/bash
# Sync fmriprep derivatives from sciencecluster to local, keeping only
# T1w-space functional outputs (excludes fsnative, fsaverage, MNI BOLD
# and all surface .gii files).

rsync -av --progress \
  --exclude '*_space-fsnative_*' \
  --exclude '*_space-fsaverage*' \
  --exclude '*_space-MNI*' \
  --exclude '*_hemi-*' \
  sciencecluster:/shares/zne.uzh/gdehol/ds-valuecapture/derivatives/fmriprep/ \
  /data/ds-valuecapture/derivatives/fmriprep/
