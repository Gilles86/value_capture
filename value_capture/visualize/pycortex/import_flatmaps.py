"""Import the autoflatten flat maps into the pycortex subjects ``valuecapture.sub-XX``.

Patches come from prf/slurm_jobs/autoflatten.sh (cluster, same recon as the pycortex subjects). Pull them first:

    for s in 04 05 ... 15; do rsync -a sciencecluster:/shares/zne.uzh/gdehol/ds-valuecapture/derivatives/fmriprep/sourcedata/freesurfer/sub-${s}_ses-1/surf/{lh,rh}.autoflatten.flat.patch.3d \\
        /data/ds-valuecapture/derivatives/fmriprep/sourcedata/freesurfer/sub-${s}_ses-1/surf/; done
    ~/mambaforge/envs/pycortex2/bin/python value_capture/visualize/pycortex/import_flatmaps.py

``import_flat`` appends ``.flat`` itself, so ``patch='autoflatten'`` (pycortex skill). The subject's cache is cleared
so the next viewer build uses the flat surface.
"""
import argparse
from pathlib import Path

import cortex
from cortex import freesurfer

FS_DIR = Path('/data/ds-valuecapture/derivatives/fmriprep/sourcedata/freesurfer')


def main(subjects, overwrite=False):
    store = Path(cortex.database.default_filestore)
    for s in subjects:
        fs, cx = f'sub-{s:02d}_ses-1', f'valuecapture.sub-{s:02d}'
        if not all((FS_DIR / fs / 'surf' / f'{h}.autoflatten.flat.patch.3d').exists() for h in ('lh', 'rh')):
            print(f'{cx}: no flat patches yet, skipped')
            continue
        if (store / cx / 'surfaces' / 'flat_rh.gii').exists() and not overwrite:
            print(f'{cx}: already has flat surfaces')
            continue
        freesurfer.import_flat(fs, patch='autoflatten', hemis=['lh', 'rh'], cx_subject=cx,
                               freesurfer_subject_dir=str(FS_DIR), auto_overwrite=True)
        for f in (store / cx / 'cache').glob('*'):
            if f.is_file():
                f.unlink()
        print(f'{cx}: flat surfaces imported', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('subjects', nargs='*', type=int, default=None)
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    main(a.subjects or list(range(4, 16)), a.overwrite)
