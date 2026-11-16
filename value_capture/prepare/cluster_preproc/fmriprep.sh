#!/bin/bash -l
#SBATCH --job-name=fmriprep_valuecapture
#SBATCH --account=zne.uzh
#SBATCH --output=/home/%u/logs/value_capture/valuecapture_fmriprep_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00

module load apptainer/1.4.1

# fmriprep container version. 25.2.7 for the full cohort (sub-04+ onward);
# sub-01..03 were originally run with 25.2.5.
FMRIPREP_VERSION=${FMRIPREP_VERSION:-25.2.7}
CONTAINER=/shares/zne.uzh/containers/fmriprep-${FMRIPREP_VERSION}
# If the container is still being built (build_fmriprep.sh), wait up to 12 h.
for i in $(seq 144); do [ -f "$CONTAINER/.build_done" ] && break; echo "waiting for $CONTAINER"; sleep 300; done
[ -f "$CONTAINER/.build_done" ] || { echo "no built container at $CONTAINER"; exit 1; }

export APPTAINERENV_FS_LICENSE=$HOME/freesurfer/license.txt
export PARTICIPANT_LABEL=${PARTICIPANT_LABEL:-$(printf "%02d" $SLURM_ARRAY_TASK_ID)}

# Optional env vars for experimental runs:
#   FMRIPREP_SUFFIX      — appended to output dir, e.g. "romeo" → fmriprep_romeo
#   FMRIPREP_EXTRA_FLAGS — extra CLI flags, e.g. "--fmap-bspline --force syn-sdc"
#   FMRIPREP_WORKDIR     — scratch dir for workflow cache (default: /scratch/gdehol)
#   FMRIPREP_BIDS_FILTER — bids filter filename in cluster_preproc/ (default: bids_filter.json)
BIDS_FILTER_FILE="/bids_input/${FMRIPREP_BIDS_FILTER:-bids_filter.json}"
OUTDIR="/data/derivatives/fmriprep${FMRIPREP_SUFFIX:+_${FMRIPREP_SUFFIX}}"
WORKDIR="${FMRIPREP_WORKDIR:-/scratch/gdehol}"

# Tell fMRIPrep its allocation (otherwise nipype plans with the whole node's RAM and cores).
NCPU=${SLURM_CPUS_PER_TASK:-8}
MEM_MB=$(( ${SLURM_MEM_PER_NODE:-65536} - 8000 ))
STDOUT_COPY="${WORKDIR}/fmriprep_stdout_sub-${PARTICIPANT_LABEL}.log"

set -o pipefail
apptainer run \
  -B /shares/zne.uzh/containers/templateflow:/opt/templateflow \
  -B /shares/zne.uzh/gdehol/ds-valuecapture:/data \
  -B ${WORKDIR}:/workflow \
  -B ${PWD}:/bids_input \
  --cleanenv "$CONTAINER" \
    /data ${OUTDIR} participant \
  --participant_label $PARTICIPANT_LABEL \
  --output-spaces T1w MNI152NLin2009cAsym:res-2 fsnative \
  --dummy-scans 4 \
  --skip_bids_validation \
  -w /workflow \
  --nthreads "$NCPU" \
  --omp-nthreads "$NCPU" \
  --mem-mb "$MEM_MB" \
  --low-mem \
  --no-submm-recon \
  --bids-filter-file $BIDS_FILTER_FILE \
  ${FMRIPREP_EXTRA_FLAGS} 2>&1 | tee "$STDOUT_COPY"
RC=${PIPESTATUS[0]}

# apptainer >= 1.4 can exit 1 after a run that finished cleanly (2026-11-14: all 12 subjects
# printed "fMRIPrep finished successfully!" and exited 1, failing the Snakemake rule). fMRIPrep
# prints that line only at the very end of a successful run, and writes a failure-report HTML on
# real failures, so the HTML alone is no proof. Trust the line, not the exit code.
if [[ $RC -ne 0 ]]; then
    if grep -q "fMRIPrep finished successfully" "$STDOUT_COPY"; then
        echo "apptainer exited $RC but fMRIPrep reported success: treating as a clean run."
    else
        echo "apptainer exited $RC and fMRIPrep did not report success: failed."
        exit $RC
    fi
fi
