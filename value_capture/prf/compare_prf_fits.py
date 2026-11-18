"""Compare two PRF fits voxel by voxel (e.g. betas with vs without the points displays modelled out).

Per subject × neuropythy ROI × eccentricity bin (of fit A): median R² of both fits, number of voxels with R² ≥ .1,
and agreement of x, y, σ (r, median |Δ|) over voxels with R² ≥ .1 in either fit.

    python value_capture/prf/compare_prf_fits.py prf_glmsingle_items prf_glmsingle_fbreg_items \\
        --out notes/data/prf_compare_fbreg.tsv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import masking

from value_capture.prf import fit_prf as fp
from value_capture.utils.data import Subject

ROIS = {1: 'V1', 2: 'V2', 3: 'V3', 4: 'hV4', 7: 'LO', 8: 'LO', 11: 'V3AB', 12: 'V3AB'}
ECC = [0, .5, 1, 2, 3, 5]

ap = argparse.ArgumentParser()
ap.add_argument('a')
ap.add_argument('b')
ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-valuecapture')
ap.add_argument('--subjects', nargs='+', default=[f'{i:02d}' for i in range(4, 16)])
ap.add_argument('--out', default='notes/data/prf_compare.tsv')
args = ap.parse_args()
d = Path(args.bids_folder) / 'derivatives'
rows = []
for s in args.subjects:
    sub = Subject(s, bids_folder=args.bids_folder)
    tsv = lambda n: pd.read_csv(d / n / f'sub-{s}' / 'ses-all' / 'func' /  # noqa: E731
                                f'sub-{s}_ses-all_task-valuecapture_space-T1w_desc-prf_params.tsv', sep='\t', index_col=0)
    A, B = tsv(args.a), tsv(args.b)
    mask = fp._get_gm_mask(sub, sub.get_sessions()[0], args.bids_folder, .1)
    va = np.round(masking.apply_mask(sub.get_retinotopic_atlas(bold_space=True), mask)).astype(int)
    t = pd.DataFrame({'roi': [ROIS.get(v) for v in va], 'ecc': np.hypot(A.x, A.y).to_numpy(),
                      'r2_a': A.r2.to_numpy(), 'r2_b': B.r2.to_numpy()})
    for p in ('x', 'y', 'sd'):
        t[f'{p}_a'], t[f'{p}_b'] = A[p].to_numpy(), B[p].to_numpy()
    t = t[t.roi.notna()]
    t['ecc_bin'] = pd.cut(t.ecc, ECC)
    for (roi, eb), g in t.groupby(['roi', 'ecc_bin'], observed=True):
        good = g[(g.r2_a >= .1) | (g.r2_b >= .1)]
        row = dict(subject=s, roi=roi, ecc_bin=str(eb), n=len(g), med_r2_a=g.r2_a.median(), med_r2_b=g.r2_b.median(),
                   n_good_a=int((g.r2_a >= .1).sum()), n_good_b=int((g.r2_b >= .1).sum()))
        for p in ('x', 'y', 'sd'):
            row[f'r_{p}'] = good[f'{p}_a'].corr(good[f'{p}_b']) if len(good) > 5 else np.nan
            row[f'mad_{p}'] = (good[f'{p}_a'] - good[f'{p}_b']).abs().median() if len(good) > 5 else np.nan
        rows.append(row)
    print(s, 'done', flush=True)
out = pd.DataFrame(rows)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
out.round(4).to_csv(args.out, sep='\t', index=False)
pd.set_option('display.width', 220)
g = out.groupby(['roi', 'ecc_bin'])
print(pd.concat([g[['med_r2_a', 'med_r2_b', 'r_x', 'r_y', 'r_sd', 'mad_x', 'mad_sd']].median(),
                 g[['n_good_a', 'n_good_b']].sum()], axis=1).round(3).to_string())
