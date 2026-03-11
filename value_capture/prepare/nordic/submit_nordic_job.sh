#!/bin/bash
#SBATCH --job-name=nordic
#SBATCH --output=/home/gdehol/logs/nordic_%j.log
#SBATCH --ntasks=32
#SBATCH --time=02:00:00
#SBATCH --mem=64gb
#SBATCH --constraint=INTEL

source /etc/profile.d/lmod.sh
module load matlab

BASE_PATH="/shares/zne.uzh/gdehol/ds-valuecapture"
SUBJECT=""
RUN=""
SESSION=""
TASK="valuecapture"

while getopts b:s:r:e:t: flag; do
    case "${flag}" in
        b) BASE_PATH=${OPTARG};;
        s) SUBJECT=${OPTARG};;
        r) RUN=${OPTARG};;
        e) SESSION=${OPTARG};;
        t) TASK=${OPTARG};;
    esac
done

if [ -z "$SUBJECT" ] || [ -z "$RUN" ]; then
    echo "Usage: sbatch submit_nordic_job.sh -s SUBJECT -r RUN [-e SESSION] [-t TASK] [-b BASE_PATH]"
    exit 1
fi

# Build the MATLAB command
if [ -z "$SESSION" ]; then
    MATLAB_CMD="addpath('$(dirname "$0")'); run_nordic('$SUBJECT', $RUN, '$BASE_PATH', [], '$TASK'); exit"
else
    MATLAB_CMD="addpath('$(dirname "$0")'); run_nordic('$SUBJECT', $RUN, '$BASE_PATH', $SESSION, '$TASK'); exit"
fi

matlab -nodisplay -r "$MATLAB_CMD"
