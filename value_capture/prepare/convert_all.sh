#!/bin/bash
# Run BIDS conversion sequentially for all subjects and sessions.
# Usage: bash convert_all.sh [--bids_dir /path/to/bids]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIDS_DIR="/data/ds-valuecapture"

while [[ $# -gt 0 ]]; do
    case $1 in
        --bids_dir) BIDS_DIR="$2"; shift 2;;
        *) echo "Unknown argument: $1"; exit 1;;
    esac
done

SUBJECTS=(1 2)
SESSIONS=(1 2)

for SUB in "${SUBJECTS[@]}"; do
    for SES in "${SESSIONS[@]}"; do
        echo ""
        echo "============================================================"
        echo "  Converting sub-$(printf '%02d' $SUB) ses-${SES}"
        echo "============================================================"
        python "$SCRIPT_DIR/convert_raw_mri_data.py" $SUB $SES --bids_dir "$BIDS_DIR"
        if [ $? -ne 0 ]; then
            echo "ERROR: conversion failed for sub-$SUB ses-$SES"
            exit 1
        fi
    done
done

echo ""
echo "All conversions complete."
