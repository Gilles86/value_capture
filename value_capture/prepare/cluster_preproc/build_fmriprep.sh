#!/bin/bash -l
#SBATCH --job-name=build_fmriprep
#SBATCH --account=zne.uzh
#SBATCH --output=/home/%u/logs/value_capture/build_fmriprep_%A.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00

source /etc/profile.d/lmod.sh
module load apptainer/1.4.1
# Usage: FMRIPREP_VERSION=25.2.7 sbatch build_fmriprep.sh
VERSION=${FMRIPREP_VERSION:-25.2.7}
OUT=/shares/zne.uzh/containers/fmriprep-${VERSION}
set -e
apptainer build --sandbox "$OUT" docker://nipreps/fmriprep:${VERSION}
touch "$OUT/.build_done"   # fmriprep.sh waits for this marker
