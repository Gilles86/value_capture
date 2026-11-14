#!/bin/bash
# Submit the Snakemake driver. Always use this instead of a bare `sbatch run_driver.sh`:
# it creates the log directories first (SLURM kills a job with no log at all when the
# --output directory is missing), via the sciencecluster skill's sbatch_checked.sh if present.
#   ssh sciencecluster 'cd ~/git/value_capture && bash value_capture/snakemake/submit_driver.sh [target]'
set -eo pipefail
cd "$HOME/git/value_capture"
mkdir -p "$HOME/logs/value_capture/snakemake/slurm"
CHECKED="$HOME/git/ruffgroup-claude-skills/skills/sciencecluster/scripts/sbatch_checked.sh"
if [ -x "$CHECKED" ]; then exec "$CHECKED" value_capture/snakemake/run_driver.sh "$@"; fi
exec sbatch value_capture/snakemake/run_driver.sh "$@"
