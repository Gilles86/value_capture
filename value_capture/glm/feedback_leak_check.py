"""Do the points displays (feedback, 1 s, ~0.5-4 s after the search display, on half the scanner trials) leak
into the GLMsingle single-trial betas? GLMsingle models only the search-phase onset, so the feedback response
can only end up in the betas of the same trial (it follows the search display within ~2-6 s) or the next one.

Per voxel: residualise the betas on the stimulus image (bar position × orientation, 20 means), then regress the
residuals on feedback_shown(this trial) and feedback_shown(previous trial, within run), both centred. Summarised
per neuropythy ROI and PRF eccentricity bin (bar-only fit, R² ≥ .1): mean t and the effect in % of the voxel's
mean bar-driven response. A foveal effect (the points are shown at fixation) would mean the feedback inflates or
deflates PRF-driven betas.

    python value_capture/glm/feedback_leak_check.py --out notes/data/feedback_leak.tsv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import masking

from value_capture.prf import fit_prf as fp
from value_capture.utils.data import Subject

ROIS = {1: 'V1', 2: 'V2', 3: 'V3', 4: 'hV4', 7: 'LO', 8: 'LO', 11: 'V3AB', 12: 'V3AB'}

ap = argparse.ArgumentParser()
ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-valuecapture')
ap.add_argument('--subjects', nargs='+', default=[f'{i:02d}' for i in range(4, 16)])
ap.add_argument('--prf-deriv', default='prf_glmsingle')
ap.add_argument('--out', default='notes/data/feedback_leak.tsv')
a = ap.parse_args()
rows = []
for s in a.subjects:
    sub = Subject(s, bids_folder=a.bids_folder)
    ses = sub.get_sessions()
    tr = sub.get_trial_metadata(ses, glmsingle_deriv='glmsingle').reset_index(drop=True)
    fb = pd.concat([sub.get_onsets(int(se), int(r)).query("event_type == 'feedback'")
                    [['trial_nr', 'show_feedback']].assign(session=se, run=r)
                    for (se, r), _ in tr.groupby(['session', 'run'])]).drop_duplicates(['session', 'run', 'trial_nr'])
    tr = tr.merge(fb, on=['session', 'run', 'trial_nr'], how='left', validate='one_to_one')
    tr['fb'] = tr.show_feedback.isin([True, 'True']).astype(float)
    tr = tr.sort_values(['session', 'run', 'trial_nr'])
    tr['fb_prev'] = tr.groupby(['session', 'run']).fb.shift(1).fillna(tr.fb.mean())
    tr = tr.sort_index()
    mask = fp._get_gm_mask(sub, ses[0], a.bids_folder, .1)
    pars = pd.read_csv(Path(a.bids_folder) / 'derivatives' / a.prf_deriv / f'sub-{s}' / 'ses-all' / 'func' /
                       f'sub-{s}_ses-all_task-valuecapture_space-T1w_desc-prf_params.tsv', sep='\t', index_col=0)
    keep = np.flatnonzero(pars.r2.to_numpy() >= .1)
    Y = masking.apply_mask(sub.get_single_trial_estimates(ses, glmsingle_deriv='glmsingle'), mask)[:, keep]
    varea = np.round(masking.apply_mask(sub.get_retinotopic_atlas(bold_space=True, session=ses[0]), mask))[keep]
    img = tr.bar_orientation + tr.bar_position.round(3).astype(str)
    means = pd.DataFrame(Y).groupby(img.values).transform('mean').to_numpy()
    R = Y - means
    X = np.column_stack([np.ones(len(tr)), tr.fb - tr.fb.mean(), tr.fb_prev - tr.fb_prev.mean()])
    B, res, *_ = np.linalg.lstsq(X, R, rcond=None)
    dof = len(tr) - X.shape[1]
    s2 = ((R - X @ B) ** 2).sum(0) / dof
    se = np.sqrt(np.outer(np.diag(np.linalg.inv(X.T @ X)), s2))
    t = B / se
    drive = np.abs(means).mean(0)                    # mean |bar-driven beta| per voxel, for scale
    ecc = np.hypot(pars.x, pars.y).to_numpy()[keep]
    d = pd.DataFrame({'roi': [ROIS.get(int(v), 'other') for v in varea], 'ecc': ecc,
                      't_fb': t[1], 't_fb_prev': t[2], 'pct_fb': 100 * B[1] / drive, 'pct_fb_prev': 100 * B[2] / drive})
    d['ecc_bin'] = pd.cut(d.ecc, [0, .5, 1, 1.5, 2, 3, 5])
    g = d.groupby(['roi', 'ecc_bin'], observed=True).agg(n=('t_fb', 'size'), t_fb=('t_fb', 'mean'),
                                                         t_fb_prev=('t_fb_prev', 'mean'), pct_fb=('pct_fb', 'median'),
                                                         pct_fb_prev=('pct_fb_prev', 'median')).reset_index()
    rows.append(g.assign(subject=s))
    print(s, f'{len(keep)} voxels; mean t(fb) {d.t_fb.mean():+.2f}, t(fb_prev) {d.t_fb_prev.mean():+.2f}', flush=True)
out = pd.concat(rows)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
out.round(4).to_csv(a.out, sep='\t', index=False)
pd.set_option('display.width', 200)
print(out.groupby(['roi', 'ecc_bin'], observed=True)[['n', 't_fb', 't_fb_prev', 'pct_fb', 'pct_fb_prev']].mean().round(2))
