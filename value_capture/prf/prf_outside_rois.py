"""Where are well-fitted PRFs on the cortex, inside and outside the visual areas?

Per subject and hemisphere, every fsnative vertex with the bar + items PRF R² (``prf_glmsingle_items``) above
--r2: its neuropythy visual area (with Wang IPS0-5 / FEF where neuropythy has none; 0 = outside all of them) and its
FreeSurfer Destrieux label (aparc.a2009s). Output: one row per subject x hemisphere x area x Destrieux label with the
number of vertices, their median R² and PRF eccentricity.

    python value_capture/prf/prf_outside_rois.py --out notes/data/prf_outside_rois.tsv
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.freesurfer import read_annot, read_morph_data

AREA = {0: 'outside', 1: 'V1', 2: 'V2', 3: 'V3', 4: 'hV4', 5: 'VO', 6: 'VO', 7: 'LO', 8: 'LO', 9: 'TO', 10: 'TO',
        11: 'V3AB', 12: 'V3AB', 101: 'IPS', 102: 'FEF'}
WANG = {101: (18, 19, 20, 21, 22, 23), 102: (25,)}

ap = argparse.ArgumentParser()
ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-valuecapture')
ap.add_argument('--subjects', nargs='+', default=[f'{s:02d}' for s in range(4, 16)])
ap.add_argument('--prf-deriv', default='prf_glmsingle_items')
ap.add_argument('--r2', type=float, default=.1)
ap.add_argument('--out', default='notes/data/prf_outside_rois.tsv')
a = ap.parse_args()
d = Path(a.bids_folder) / 'derivatives'


def gii(s, hemi, par):
    fn = (d / a.prf_deriv / f'sub-{s}' / 'ses-all' / 'func' /
          f'sub-{s}_ses-all_task-valuecapture_hemi-{hemi}_space-fsnative_desc-prf{par}_pe.func.gii')
    return nib.load(fn).darrays[0].data.astype(np.float32)


rows = []
for s in a.subjects:
    fs = d / 'fmriprep' / 'sourcedata' / 'freesurfer' / f'sub-{s}_ses-1'
    for h, H in (('lh', 'L'), ('rh', 'R')):
        va = np.round(read_morph_data(fs / 'surf' / f'{h}.inferred_varea')).astype(int)
        wf = fs / 'surf' / f'{h}.wang15_mplbl.mgz'
        if wf.exists():
            wang = np.round(np.asarray(nib.load(wf).dataobj).squeeze()).astype(int)
            for code, labels in WANG.items():
                va[(va == 0) & np.isin(wang, labels)] = code
        lab, _, names = read_annot(fs / 'label' / f'{h}.aparc.a2009s.annot')
        names = np.array([n.decode() for n in names] + ['unknown'])
        lab = np.where(lab < 0, len(names) - 1, lab)
        t = pd.DataFrame({p: gii(s, H, p) for p in ('x', 'y', 'R2')})
        t['area'] = pd.Series(va).map(AREA).fillna('outside').to_numpy()
        t['destrieux'] = names[lab]
        t = t[t.R2 > a.r2]
        t['ecc'] = np.hypot(t.x, t.y)
        g = t.groupby(['area', 'destrieux'])
        rows.append(g.agg(n=('R2', 'size'), r2=('R2', 'median'), ecc=('ecc', 'median')).reset_index()
                    .assign(subject=s, hemi=H))
    print(s, flush=True)
out = pd.concat(rows)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
out.round(4).to_csv(a.out, sep='\t', index=False)
tot = out.groupby('area').n.sum().sort_values(ascending=False)
print(tot.to_string())
print(out[out.area == 'outside'].groupby('destrieux').agg(n=('n', 'sum'), subjects=('subject', 'nunique'))
      .sort_values('n', ascending=False).head(20).to_string())
