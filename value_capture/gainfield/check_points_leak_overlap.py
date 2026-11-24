"""Is the points leak retinotopic? Leak per voxel against how much of its pRF covers the points text.

For every voxel in the visual areas (incl. Wang IPS) with pRF R² >= --min-r2:

- overlap: the fraction of its Gaussian pRF (x, y, sd; ``prf_glmsingle_items``) that falls on the points text,
  using the ink of the texts actually shown, averaged over the subject's points trials (``points._render``).
- leak: mean single-trial beta (``glmsingle_fbevents``) on trials with points minus trials without, each relative to
  the trials without points at the same bar position (% signal).
- leak_slope: regression of that per-trial difference on the HRF overlap c(delay) of the points and search events
  over the points trials (positive = more leak the closer the points follow the search).
- leak_z: leak divided by the SD of the no-points residuals (comparable across voxels).
- leak_big / leak_small: leak on points trials with >= 100 points (3-4 characters; high multiplier only) and the rest.

One row per voxel. If the leak is retinotopic it falls to ~0 as overlap -> 0; if it is a global (reward / arousal)
signal, peripheral voxels carry it too.

    python value_capture/gainfield/check_points_leak_overlap.py 04 --out <derivatives>/points_leak_overlap/sub-04.tsv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import masking

from value_capture.gainfield import points as pts
from value_capture.gainfield.fit_gain_fields import ROIS, trial_table
from value_capture.prf import fit_prf as fp
from value_capture.utils.data import Subject


def main(subject, bids, glmsingle_deriv, prf_deriv, min_r2, out):
    sub = Subject(subject, bids_folder=bids)
    sessions = sub.get_sessions()
    trials = pts.add_points_info(sub, trial_table(sub, sessions, glmsingle_deriv))
    grid = fp.make_grid_coordinates()
    ink = pts.ink_map(trials, grid)
    mask = fp._get_gm_mask(sub, sessions[0], bids, .1)
    varea = np.round(masking.apply_mask(sub.get_retinotopic_atlas(bold_space=True, session=sessions[0],
                                                                  include_wang=True), mask))
    pars = pd.read_csv(Path(bids) / 'derivatives' / prf_deriv / f'sub-{subject}' / 'ses-all' / 'func' /
                       f'sub-{subject}_ses-all_task-valuecapture_space-T1w_desc-prf_params.tsv', sep='\t', index_col=0)
    roi_of = {v: r for r, vs in ROIS.items() for v in vs}
    sel = np.flatnonzero(np.isin(varea, list(roi_of)) & (pars.r2.to_numpy() >= min_r2))
    p = pars.iloc[sel]
    gx, gy = grid.x.to_numpy()[None], grid.y.to_numpy()[None]
    rf = np.exp(-((gx - p.x.to_numpy()[:, None]) ** 2 + (gy - p.y.to_numpy()[:, None]) ** 2)
                / (2 * p.sd.to_numpy()[:, None] ** 2))
    overlap = (rf * ink[None]).sum(1) / rf.sum(1)

    Y = masking.apply_mask(sub.get_single_trial_estimates(sessions, glmsingle_deriv=glmsingle_deriv), mask)[:, sel]
    shown = trials.show_feedback.astype(bool).to_numpy()
    bar = (trials.session.astype(str) + trials.bar_orientation.astype(str)
           + trials.bar_position.round(4).astype(str)).to_numpy()
    ref = pd.DataFrame(Y[~shown]).groupby(bar[~shown]).mean()
    D = Y - ref.loc[bar].to_numpy()
    c = trials.delay.map(pts.overlap).to_numpy()
    leak = D[shown].mean(0)
    cs = c[shown] - c[shown].mean()
    slope = (cs[:, None] * (D[shown] - D[shown].mean(0))).sum(0) / (cs ** 2).sum()
    resid_sd = D[~shown].std(0)
    big = shown & (trials.earned_points.fillna(0).to_numpy() >= 100)      # 3-4 characters, high multiplier only
    small = shown & ~big
    t = pd.DataFrame(dict(subject=subject, roi=[roi_of[int(v)] for v in varea[sel]], x=p.x.to_numpy(),
                          y=p.y.to_numpy(), sd=p.sd.to_numpy(), r2=p.r2.to_numpy(), overlap=overlap, leak=leak,
                          leak_slope=slope, leak_z=leak / resid_sd,
                          leak_big=D[big].mean(0), leak_small=D[small].mean(0), n_big=int(big.sum()),
                          n_small=int(small.sum())))
    t['ecc'] = np.hypot(t.x, t.y)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    t.round(6).to_csv(out, sep='\t', index=False)
    bins = pd.cut(t.overlap, [-1, .001, .01, .05, .2, 1])
    print(t.groupby(bins, observed=True)[['leak', 'leak_z', 'leak_slope']].agg(['mean', 'count']).round(3).to_string(),
          out, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject')
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-valuecapture')
    ap.add_argument('--glmsingle-deriv', default='glmsingle_fbevents')
    ap.add_argument('--prf-deriv', default='prf_glmsingle_items')
    ap.add_argument('--min-r2', type=float, default=.1)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    out = a.out or (f'/shares/zne.uzh/gdehol/ds-valuecapture/derivatives/points_leak_overlap/'
                    f'sub-{a.subject}_points_leak_overlap.tsv')
    main(a.subject, a.bids_folder, a.glmsingle_deriv, a.prf_deriv, a.min_r2, out)
