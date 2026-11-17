"""Rewrite the PRF parameter NIfTIs of an existing fit from its full-precision params TSV, as float32.

fit_prf.py wrote the maps through the boolean GM mask, so they came out uint8 (~250 distinct values per map;
fixed 2026-11-17). The TSV has every parameter at full precision in mask order, so no refit is needed.
Rebuilds the mask exactly as fit_prf.py does (GM probseg >= threshold, first session), checks the voxel count
matches the TSV, and overwrites the *_pe.nii.gz maps.

    python value_capture/prf/rewrite_prf_niftis.py 04 05 ... --gm-threshold 0.1
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from value_capture.prf.fit_prf import _get_gm_mask, save_param_nifti
from value_capture.utils.data import Subject

ap = argparse.ArgumentParser()
ap.add_argument('subjects', nargs='+')
ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-valuecapture')
ap.add_argument('--gm-threshold', type=float, default=0.1)
ap.add_argument('--glmsingle-deriv', default='glmsingle')
a = ap.parse_args()

for s in a.subjects:
    sub = Subject(s, bids_folder=a.bids_folder)
    sessions = sub.get_sessions()
    ses_label = f'ses-{sessions[0]}' if len(sessions) == 1 else 'ses-all'
    out = Path(a.bids_folder) / 'derivatives' / f'prf_{a.glmsingle_deriv}' / f'sub-{s}' / ses_label / 'func'
    base = f'sub-{s}_{ses_label}_task-valuecapture_space-T1w_desc-{{}}'
    pars = pd.read_csv(out / (base.format('prf') + '_params.tsv'), sep='\t', index_col=0)
    mask = _get_gm_mask(sub, sessions[0], a.bids_folder, a.gm_threshold)
    n = int(np.asarray(mask.dataobj).astype(bool).sum())
    assert n == len(pars), f'sub-{s}: mask has {n} voxels, TSV {len(pars)} rows'
    for col, desc in [('x', 'prfx'), ('y', 'prfy'), ('sd', 'prfsd'), ('baseline', 'prfbaseline'),
                      ('amplitude', 'prfamplitude'), ('ecc', 'prfecc'), ('angle', 'prfangle'), ('r2', 'prfR2')]:
        p = out / (base.format(desc) + '_pe.nii.gz')
        save_param_nifti(pars[col].values, mask, p)
    back = nib.load(out / (base.format('prfR2') + '_pe.nii.gz'))
    v = back.get_fdata(dtype=np.float32)[np.asarray(mask.dataobj).astype(bool)]
    print(f'sub-{s}: {n} voxels, dtype {back.get_data_dtype()}, {len(np.unique(np.round(v, 4)))} unique R² '
          f'values, max |R² - TSV| {np.abs(v - pars.r2.values).max():.2e}')
