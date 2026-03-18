#!/bin/bash
# Sync selected fmriprep derivatives from sciencecluster to local.
# Copies: figures/reports, T1w-space boldref + preproc BOLD + brain masks,
#         confounds TSVs, and T1w anatomicals (NIfTI + .gii).
# Excludes: MNI, fsnative, fsaverage, hemi-specific files.
#
# --no-perms: avoids fchmodat errors when /data dst dirs are root-owned.

SRC="sciencecluster:/shares/zne.uzh/gdehol/ds-valuecapture/derivatives/fmriprep/"
DST="/data/ds-valuecapture/derivatives/fmriprep/"

EXCLUDES=(
  --exclude '*_space-fsnative_*'
  --exclude '*_space-fsaverage*'
  --exclude '*_space-MNI*'
  --exclude '*_hemi-*'
)

RSYNC="rsync -av --no-perms --progress"

echo "=== Pass 1: figures and reports ==="
$RSYNC "${EXCLUDES[@]}" \
  --include '*/' \
  --include '*.html' \
  --include '*.svg' \
  --include '*.png' \
  --include '*.css' \
  --include '*.js' \
  --exclude '*' \
  "$SRC" "$DST"

echo "=== Pass 2: T1w-space boldref ==="
$RSYNC "${EXCLUDES[@]}" \
  --include '*/' \
  --include '*_space-T1w_boldref*' \
  --exclude '*' \
  "$SRC" "$DST"

echo "=== Pass 3: T1w-space brain masks ==="
$RSYNC "${EXCLUDES[@]}" \
  --include '*/' \
  --include '*_space-T1w_desc-brain_mask.nii.gz' \
  --exclude '*' \
  "$SRC" "$DST"

echo "=== Pass 4: confounds timeseries ==="
$RSYNC "${EXCLUDES[@]}" \
  --include '*/' \
  --include '*_desc-confounds_timeseries.tsv' \
  --exclude '*' \
  "$SRC" "$DST"

echo "=== Pass 5: T1w anatomicals (NIfTI + .gii) ==="
# Note: hemi exclusion deliberately omitted here — all anat .gii surfaces have hemi-L/R in name.
# func/ excluded entirely to keep this anat-only.
$RSYNC \
  --exclude '*_space-fsnative_*' \
  --exclude '*_space-fsaverage*' \
  --exclude '*_space-MNI*' \
  --exclude 'func/' \
  --include '*/' \
  --include '*_T1w*' \
  --include '*.gii' \
  --exclude '*' \
  "$SRC" "$DST"
