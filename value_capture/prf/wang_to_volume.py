"""Project the Wang 2015 surface labels (surf/{lh,rh}.wang15_mplbl.mgz) into the FreeSurfer volume.

Each vertex's label is written to the voxels along the line from the white to the pial surface (7 steps), on the
grid of mri/inferred_varea.mgz, so the volume atlas can carry IPS0-5 / FEF where neuropythy has no area
(``Subject.get_retinotopic_atlas(include_wang=True)``). Output: mri/wang15_mplbl.mgz (int16).

    python value_capture/prf/wang_to_volume.py 04 05 ...
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.freesurfer import read_geometry

ap = argparse.ArgumentParser()
ap.add_argument('subjects', nargs='+')
ap.add_argument('--fs-dir', default='/shares/zne.uzh/gdehol/ds-valuecapture/derivatives/fmriprep/sourcedata/freesurfer')
a = ap.parse_args()

for s in a.subjects:
    fs = Path(a.fs_dir) / f'sub-{s}_ses-1'
    ref = nib.load(fs / 'mri' / 'inferred_varea.mgz')
    out = np.zeros(ref.shape[:3], np.int16)
    ras2vox = np.linalg.inv(ref.header.get_vox2ras_tkr())
    for h in ('lh', 'rh'):
        lab = np.round(np.asarray(nib.load(fs / 'surf' / f'{h}.wang15_mplbl.mgz').dataobj).squeeze()).astype(int)
        white, _ = read_geometry(fs / 'surf' / f'{h}.white')
        pial, _ = read_geometry(fs / 'surf' / f'{h}.pial')
        on = lab > 0
        for f in np.linspace(0, 1, 7):
            p = white[on] + f * (pial[on] - white[on])
            ijk = np.round(nib.affines.apply_affine(ras2vox, p)).astype(int)
            ok = np.all((ijk >= 0) & (ijk < np.array(out.shape)), 1)
            out[tuple(ijk[ok].T)] = lab[on][ok]
    img = nib.MGHImage(out, ref.affine, ref.header)
    img.set_data_dtype(np.int16)
    img.to_filename(fs / 'mri' / 'wang15_mplbl.mgz')
    print(s, {int(k): int(v) for k, v in zip(*np.unique(out[out > 0], return_counts=True)) if k in range(18, 26)},
          flush=True)
