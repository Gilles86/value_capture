"""Collect fsnative surface PRF fits of all subjects into one table for the validation figures.

One row per vertex in a neuropythy visual area (inferred_varea 1-12) with R² > --r2-keep in either fit:
the bar + search-items fit (``prf_glmsingle_items``: x, y, sd, amplitude, r2) and the bar-only fit's R²
(``prf_glmsingle``: r2_bar). ROI names group the neuropythy labels as the renders do.

    python value_capture/prf/collect_prf_surface.py --bids-folder /shares/zne.uzh/gdehol/ds-valuecapture \\
        --out notes/data/prf_surface_vertices.tsv
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.freesurfer import read_morph_data

ROIS = {1: 'V1', 2: 'V2', 3: 'V3', 4: 'hV4', 5: 'VO', 6: 'VO', 7: 'LO', 8: 'LO', 9: 'TO', 10: 'TO',
        11: 'V3AB', 12: 'V3AB'}

ap = argparse.ArgumentParser()
ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-valuecapture')
ap.add_argument('--subjects', nargs='+', default=[f'{s:02d}' for s in range(4, 16)])
ap.add_argument('--prf-deriv', default='prf_glmsingle_items')
ap.add_argument('--bar-deriv', default='prf_glmsingle')
ap.add_argument('--r2-keep', type=float, default=.02)
ap.add_argument('--out', default='notes/data/prf_surface_vertices.tsv')
a = ap.parse_args()
d = Path(a.bids_folder) / 'derivatives'


def gii(deriv, s, hemi, par):
    fn = (d / deriv / f'sub-{s}' / 'ses-all' / 'func' /
          f'sub-{s}_ses-all_task-valuecapture_hemi-{hemi}_space-fsnative_desc-prf{par}_pe.func.gii')
    return nib.load(fn).darrays[0].data.astype(np.float32)


rows = []
for s in a.subjects:
    for h, H in (('lh', 'L'), ('rh', 'R')):
        va = np.round(read_morph_data(d / 'fmriprep' / 'sourcedata' / 'freesurfer' / f'sub-{s}_ses-1' / 'surf'
                                      / f'{h}.inferred_varea')).astype(int)
        t = pd.DataFrame({p: gii(a.prf_deriv, s, H, p) for p in ('x', 'y', 'sd', 'amplitude', 'R2')})
        t['r2_bar'] = gii(a.bar_deriv, s, H, 'R2')
        t = t.rename(columns={'R2': 'r2'}).assign(subject=s, hemi=H, vertex=np.arange(len(t)), varea=va)
        t = t[t.varea.isin(ROIS) & ((t.r2 > a.r2_keep) | (t.r2_bar > a.r2_keep))]
        rows.append(t)
    print(s, sum(len(r) for r in rows), flush=True)
out = pd.concat(rows)
out['roi'] = out.varea.map(ROIS)
out['ecc'] = np.hypot(out.x, out.y)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
out.round(5).to_csv(a.out, sep='\t', index=False)
print(a.out, len(out))
