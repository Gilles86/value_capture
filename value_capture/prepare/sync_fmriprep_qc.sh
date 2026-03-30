#!/bin/bash
# Sync QC-relevant fmriprep outputs from all fmriprep_* derivatives on the cluster.
# Copies: HTML reports + figures, T1w anatomicals, T1w-space boldref.
# Useful for comparing SDC variants (fmriprep, fmriprep_romeo, fmriprep_bspline, etc.)
#
# Usage: bash sync_fmriprep_qc.sh [derivative_suffix ...]
#   No args: syncs all fmriprep* derivatives found on the cluster
#   With args: syncs only specified suffixes, e.g. "romeo bspline_syn"

CLUSTER="sciencecluster"
REMOTE_BASE="/shares/zne.uzh/gdehol/ds-valuecapture/derivatives"
LOCAL_BASE="/data/ds-valuecapture/derivatives"

RSYNC="rsync -av --no-perms --progress"

EXCLUDES=(
  --exclude '*_space-fsnative_*'
  --exclude '*_space-fsaverage*'
  --exclude '*_space-MNI*'
  --exclude '*_hemi-*'
)

sync_derivative() {
    local name=$1
    local SRC="${CLUSTER}:${REMOTE_BASE}/${name}/"
    local DST="${LOCAL_BASE}/${name}/"

    echo ""
    echo "=== ${name} ==="

    echo "  -- figures and reports --"
    $RSYNC "${EXCLUDES[@]}" \
      --include '*/' \
      --include '*.html' \
      --include '*.svg' \
      --include '*.png' \
      --include '*.css' \
      --include '*.js' \
      --exclude '*' \
      "$SRC" "$DST"

    echo "  -- T1w-space boldref --"
    $RSYNC "${EXCLUDES[@]}" \
      --include '*/' \
      --include '*_space-T1w_boldref*' \
      --exclude '*' \
      "$SRC" "$DST"

    echo "  -- T1w anatomicals --"
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
}

# Determine which derivatives to sync
if [ $# -gt 0 ]; then
    for suffix in "$@"; do
        sync_derivative "fmriprep_${suffix}"
    done
else
    # Auto-discover all fmriprep* dirs on the cluster
    echo "Discovering fmriprep* derivatives on cluster..."
    DIRS=$(ssh "$CLUSTER" "ls -d ${REMOTE_BASE}/fmriprep*/ 2>/dev/null | xargs -n1 basename")
    for name in $DIRS; do
        sync_derivative "$name"
    done
fi

echo ""
echo "Done."
