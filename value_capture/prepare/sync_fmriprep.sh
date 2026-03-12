#!/bin/bash
# Sync fmriprep derivatives from sciencecluster to local, keeping only
# T1w-space functional outputs (excludes fsnative, fsaverage, MNI BOLD
# and all surface .gii files).
#
# Two passes: figures/HTML first (fast QC), then everything else.

SRC="sciencecluster:/shares/zne.uzh/gdehol/ds-valuecapture/derivatives/fmriprep/"
DST="/data/ds-valuecapture/derivatives/fmriprep/"

EXCLUDES=(
  --exclude '*_space-fsnative_*'
  --exclude '*_space-fsaverage*'
  --exclude '*_space-MNI*'
  --exclude '*_hemi-*'
)

echo "=== Pass 1: figures and reports ==="
rsync -av --progress "${EXCLUDES[@]}" \
  --include '*/' \
  --include '*.html' \
  --include '*.svg' \
  --include '*.png' \
  --include '*.css' \
  --include '*.js' \
  --exclude '*' \
  "$SRC" "$DST"

echo "=== Pass 2: everything else ==="
rsync -av --progress "${EXCLUDES[@]}" \
  "$SRC" "$DST"
