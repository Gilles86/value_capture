#!/bin/bash
# Submit NORDIC jobs for all runs of a subject/session.
# Prints all job IDs (colon-separated) so callers can build SLURM dependencies.
#
# Usage:
#   bash submit_all_runs.sh SUBJECT SESSION
#   JOB_IDS=$(bash submit_all_runs.sh 01 1)

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 SUBJECT SESSION"
    exit 1
fi

SUBJECT=$1
SESSION=$2
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

JOB_IDS=""

# 8 valuecapture runs
for RUN in {1..8}; do
    JOB_ID=$(sbatch --parsable --export=ALL,NORDIC_DIR="$SCRIPT_DIR" \
        "$SCRIPT_DIR/submit_nordic_job.sh" \
        -s "$SUBJECT" -r "$RUN" -e "$SESSION" -t "valuecapture")
    echo "Submitted NORDIC valuecapture run $RUN → job $JOB_ID" >&2
    JOB_IDS="${JOB_IDS}:${JOB_ID}"
done

# 1 deepmreye run
JOB_ID=$(sbatch --parsable --export=ALL,NORDIC_DIR="$SCRIPT_DIR" \
    "$SCRIPT_DIR/submit_nordic_job.sh" \
    -s "$SUBJECT" -r 1 -e "$SESSION" -t "deepmreye")
echo "Submitted NORDIC deepmreye run 1 → job $JOB_ID" >&2
JOB_IDS="${JOB_IDS}:${JOB_ID}"

# Strip leading colon and print for use as dependency string
echo "${JOB_IDS#:}"
