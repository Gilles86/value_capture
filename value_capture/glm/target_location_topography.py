"""Is there attention-driven topography in IPS (and the other visual areas)? Target location in single-trial betas.

The bar does not map PRFs in IPS (R² ≈ 0 there). What moves across trials and draws attention is the target (and
the distractor) on one of the 4 diagonals at 4°. Per subject, on the fsnative surface:
  - GLMsingle betas (glmsingle_fbreg: points displays modelled out) sampled to the cortical surface,
    z-scored within run per vertex;
  - per vertex, mean beta per target location (1 = upper right, 3 = upper left, 5 = lower left, 7 = lower right);
  - contralateral index = mean(contralateral-field targets) − mean(ipsilateral-field targets), left hemisphere:
    right field contralateral;
  - reliability of the location preference: per vertex, correlation of the centred 4-location profile between
    sessions 1 and 2 (independent data); the ROI mean is compared with 0 across subjects.
Computed on all trials and on distractor-absent trials (target is the only singleton).
ROIs: neuropythy V1-V3, hV4, LO, V3AB; Wang IPS (IPS0-5) and FEF where neuropythy has no label.

    python value_capture/glm/target_location_topography.py --out notes/data/target_location_topography.tsv
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.freesurfer import read_morph_data
from nilearn import image, surface

from value_capture.prf import fit_prf as fp
from value_capture.utils.data import Subject

ROIS = {1: 'V1', 2: 'V2', 3: 'V3', 4: 'hV4', 7: 'LO', 8: 'LO', 11: 'V3AB', 12: 'V3AB', 101: 'IPS', 102: 'FEF'}
WANG = {101: (18, 19, 20, 21, 22, 23), 102: (25,)}
RIGHT_FIELD, LEFT_FIELD = (1, 7), (3, 5)


def labels(fsdir, h):
    va = np.round(read_morph_data(fsdir / 'surf' / f'{h}.inferred_varea')).astype(int)
    wang = np.round(np.asarray(nib.load(fsdir / 'surf' / f'{h}.wang15_mplbl.mgz').dataobj).squeeze()).astype(int)
    for code, labs in WANG.items():
        va[(va == 0) & np.isin(wang, labs)] = code
    return va


def main(subjects, bids, deriv, out):
    rows = []
    for s in subjects:
        sub = Subject(s, bids_folder=bids)
        ses = sub.get_sessions()
        tr = fp.add_item_info(sub, sub.get_trial_metadata(ses, glmsingle_deriv=deriv).reset_index(drop=True))
        betas = image.load_img(str(Path(bids) / 'derivatives' / deriv / f'sub-{s}' / 'ses-all' / 'func' /
                                   f'sub-{s}_ses-all_task-valuecapture_space-T1w_desc-target_pe.nii.gz'))
        surf = sub.get_surf_info()
        fsdir = Path(sub.freesurfer_subjects_dir) / sub.freesurfer_subject_id
        for h, H, contra in (('lh', 'L', RIGHT_FIELD), ('rh', 'R', LEFT_FIELD)):
            va = labels(fsdir, h)
            keep = np.isin(va, list(ROIS))
            X = surface.vol_to_surf(betas, surf[H]['outer'], inner_mesh=surf[H]['inner']).T[:, keep]  # trials × vertices
            X = pd.DataFrame(X).groupby([tr.session.values, tr.run.values]).transform(lambda c: (c - c.mean()) / (c.std() + 1e-9)).to_numpy()
            roi = np.array([ROIS[v] for v in va[keep]])
            for subset, m in (('all', np.ones(len(tr), bool)), ('absent', ~tr.distractor_present.isin([True, 'True']).to_numpy())):
                loc = tr.target_location.to_numpy()
                prof = {}
                for se in (1, 2):
                    sm = m & (tr.session.to_numpy() == se)
                    prof[se] = np.stack([X[sm & (loc == l)].mean(0) for l in (1, 3, 5, 7)])   # 4 × vertices
                allm = np.stack([X[m & (loc == l)].mean(0) for l in (1, 3, 5, 7)])
                idx = {1: 0, 3: 1, 5: 2, 7: 3}
                ci = allm[[idx[l] for l in contra]].mean(0) - allm[[idx[l] for l in (set((1, 3, 5, 7)) - set(contra))]].mean(0)
                a, b = prof[1] - prof[1].mean(0), prof[2] - prof[2].mean(0)
                rel = (a * b).sum(0) / np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0) + 1e-12)
                for r in np.unique(roi):
                    sel = roi == r
                    rows.append(dict(subject=s, hemi=H, roi=r, subset=subset, n_vertices=int(sel.sum()),
                                     contra_index=float(np.nanmean(ci[sel])), reliability=float(np.nanmean(rel[sel]))))
        print(s, 'done', flush=True)
    t = pd.DataFrame(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    t.round(5).to_csv(out, sep='\t', index=False)
    from scipy import stats
    g = t.groupby(['subset', 'roi', 'subject'])[['contra_index', 'reliability']].mean().reset_index()
    summ = []
    for (subset, roi), x in g.groupby(['subset', 'roi']):
        summ.append(dict(subset=subset, roi=roi, n=len(x), contra=x.contra_index.mean(),
                         p_contra=stats.ttest_1samp(x.contra_index, 0).pvalue, reliability=x.reliability.mean(),
                         p_rel=stats.ttest_1samp(x.reliability, 0).pvalue))
    print(pd.DataFrame(summ).round(4).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--subjects', nargs='+', default=[f'{i:02d}' for i in range(4, 16)])
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-valuecapture')
    ap.add_argument('--glmsingle-deriv', default='glmsingle_fbreg')
    ap.add_argument('--out', default='notes/data/target_location_topography.tsv')
    a = ap.parse_args()
    main(a.subjects, a.bids_folder, a.glmsingle_deriv, a.out)
