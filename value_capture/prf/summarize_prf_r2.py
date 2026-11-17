"""Summarise event-related PRF fits on GLMsingle betas: dtype check and R² distribution per subject.

    python value_capture/prf/summarize_prf_r2.py --bids-folder /shares/zne.uzh/gdehol/ds-valuecapture
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-valuecapture')
ap.add_argument('--subjects', nargs='+', default=[f'{s:02d}' for s in range(4, 16)])
a = ap.parse_args()
d = Path(a.bids_folder) / 'derivatives' / 'prf_glmsingle'
print('sub  dtype    n_unique  n_fit   R2>.05  R2>.10  R2>.20  R2_p99  R2_max  ecc_med(R2>.1)  sd_med(R2>.1)')
for s in a.subjects:
    f = lambda k: d / f'sub-{s}' / 'ses-all' / 'func' / f'sub-{s}_ses-all_task-valuecapture_space-T1w_desc-prf{k}_pe.nii.gz'  # noqa: E731
    img = nib.load(f('R2'))
    r2 = img.get_fdata(dtype=np.float32)
    fit = np.isfinite(r2) & (r2 != 0)
    v = r2[fit]
    good = fit & (r2 > .1)
    ecc = nib.load(f('ecc')).get_fdata(dtype=np.float32)[good]
    sd = nib.load(f('sd')).get_fdata(dtype=np.float32)[good]
    print(f'{s}   {img.get_data_dtype()!s:8s} {len(np.unique(np.round(v, 4))):8d}  {fit.sum():6d}  '
          f'{(v > .05).sum():6d}  {(v > .1).sum():6d}  {(v > .2).sum():6d}  {np.percentile(v, 99):.3f}   '
          f'{v.max():.3f}   {np.median(ecc) if good.any() else np.nan:6.2f}          '
          f'{np.median(sd) if good.any() else np.nan:6.2f}')
