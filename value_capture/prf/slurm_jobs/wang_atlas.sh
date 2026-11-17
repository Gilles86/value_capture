#!/bin/bash
#SBATCH --job-name=vc_wang_atlas
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --output=/home/%u/logs/value_capture/wang_atlas_%A_%a.txt
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
# Wang 2015 maximum-probability atlas (IPS0-5, SPL1, FEF, ...) on each subject's surface: neuropythy's fsaverage
# copy of the atlas, resampled to the subject with FreeSurfer's mri_surf2surf (sphere.reg, nearest neighbour).
# (`neuropythy atlas --atlases wang15` wrote no surface files here.) Neuropythy's register_retinotopy labels only
# V1-V3, hV4, VO, LO, TO, V3A/B. Output: surf/{lh,rh}.wang15_mplbl.mgz, as in retsupp.
#   sbatch --array=4-15 value_capture/prf/slurm_jobs/wang_atlas.sh
set -eo pipefail
SUB=$(printf "sub-%02d_ses-1" "$SLURM_ARRAY_TASK_ID")
export SUBJECTS_DIR=/shares/zne.uzh/gdehol/ds-valuecapture/derivatives/fmriprep/sourcedata/freesurfer
export FREESURFER_HOME=/shares/zne.uzh/containers/fmriprep-25.2.5/opt/freesurfer
export FS_LICENSE=/shares/zne.uzh/containers/freesurfer/license.txt
export PATH="$FREESURFER_HOME/bin:$PATH"
source "$FREESURFER_HOME/SetUpFreeSurfer.sh" >/dev/null 2>&1 || true
export SUBJECTS_DIR=/shares/zne.uzh/gdehol/ds-valuecapture/derivatives/fmriprep/sourcedata/freesurfer
ATLAS=$HOME/data/conda/envs/retsupp_neuropythy/lib/python3.10/site-packages/neuropythy/lib/data/fsaverage/surf
for h in lh rh; do
    mri_surf2surf --srcsubject fsaverage --trgsubject "$SUB" --hemi $h --mapmethod nnf \
        --sval "$ATLAS/$h.wang15_mplbl.v1_0.mgz" --tval "$SUBJECTS_DIR/$SUB/surf/$h.wang15_mplbl.mgz"
done
ls -la "$SUBJECTS_DIR/$SUB/surf/"*wang15*
