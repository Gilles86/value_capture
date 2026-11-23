"""Single-trial responses at the item locations: target, low- and high-value distractor vs. a neutral item.

For each diagonal item location (where targets and distractors appear) the voxels whose pRF centre lies within
--radius deg of the item. pRFs come from the other session (``<prf-deriv>_train-ses<other>``), so the voxel selection
is independent of the betas it is applied to. Per trial, the mean single-trial beta of those voxels; minus the mean
over trials of the same session in which that location held a neutral item with the bar at the same position (the
bar drives these voxels far more than the items). One row per subject, session, area, points handling and role.

Points handling: ``exclude`` uses only trials without the points display (also for the reference), ``subtract``
removes the points response estimated on the no-distractor trials (``decode.aggregate_decoded.remove_points``),
``keep`` uses all trials as they are.

    python value_capture/gainfield/item_location_responses.py 04 --out <derivatives>/item_location_responses/sub-04.tsv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import masking

from value_capture.decode.aggregate_decoded import remove_points
from value_capture.decode.decode_trials import DIAGONALS
from value_capture.gainfield.fit_gain_fields import ROIS, trial_table
from value_capture.prf import fit_prf as fp
from value_capture.utils.data import Subject


def roles(trials, loc):
    r = np.full(len(trials), 'neutral', dtype=object)
    r[trials.target_location.astype(int).to_numpy() == loc] = 'target'
    dist = trials.distractor_present.to_numpy(bool) & (trials.distractor_location.fillna(-1).astype(int).to_numpy() == loc)
    r[dist & (trials.value_rank.to_numpy() == 2)] = 'distractor_high'
    r[dist & (trials.value_rank.to_numpy() != 2)] = 'distractor_low'
    return r


def main(subject, bids, glmsingle_deriv, prf_deriv, radius, min_r2, out):
    sub = Subject(subject, bids_folder=bids)
    sessions = sub.get_sessions()
    assert len(sessions) == 2, sessions
    trials = trial_table(sub, sessions, glmsingle_deriv)
    mask = fp._get_gm_mask(sub, sessions[0], bids, .1)
    Y_all = masking.apply_mask(sub.get_single_trial_estimates(sessions, glmsingle_deriv=glmsingle_deriv),
                               mask).astype(np.float32)
    varea = np.round(masking.apply_mask(sub.get_retinotopic_atlas(bold_space=True, session=sessions[0]), mask))
    bar = (trials.bar_orientation.astype(str) + trials.bar_position.round(4).astype(str)).to_numpy()
    rows = []
    for s, o in zip(sessions, sessions[::-1]):
        pars = pd.read_csv(Path(bids) / 'derivatives' / f'{prf_deriv}_train-ses{o}' / f'sub-{subject}' / 'ses-all' /
                           'func' / f'sub-{subject}_ses-all_task-valuecapture_space-T1w_desc-prf_params.tsv',
                           sep='\t', index_col=0)
        assert len(pars) == Y_all.shape[1]
        idx = np.flatnonzero(trials.session.to_numpy() == int(s))
        tr = trials.iloc[idx].reset_index(drop=True)
        shown = tr.show_feedback.to_numpy(bool)
        for points in ('exclude', 'subtract', 'keep'):
            Y = Y_all[idx]
            if points == 'subtract':
                Y = remove_points(tr, Y, shown)
            keep = ~shown if points == 'exclude' else np.ones(len(tr), bool)
            for roi, labels in ROIS.items():
                inroi = np.isin(varea, labels) & (pars.r2.to_numpy() >= min_r2)
                for loc in DIAGONALS:
                    cx, cy = fp.ITEM_XY[loc]
                    sel = inroi & (np.hypot(pars.x.to_numpy() - cx, pars.y.to_numpy() - cy) < radius)
                    if sel.sum() < 3:
                        continue
                    y = Y[:, sel].mean(1)
                    role = roles(tr, loc)
                    b = bar[idx]
                    ref = pd.Series(y[keep & (role == 'neutral')]).groupby(b[keep & (role == 'neutral')]).mean()
                    d = y - pd.Series(b).map(ref).to_numpy()
                    df = pd.DataFrame({'role': role, 'diff': d, 'beta': y})[keep]
                    for r, g in df.groupby('role'):
                        rows.append(dict(subject=subject, session=int(s), roi=roi, points=points, location=loc,
                                         role=r, n_voxels=int(sel.sum()), n_trials=int(g['diff'].notna().sum()),
                                         diff=g['diff'].mean(), beta=g.beta.mean()))
    t = pd.DataFrame(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    t.round(5).to_csv(out, sep='\t', index=False)
    print(t.groupby(['points', 'roi', 'role'])['diff'].mean().unstack().round(3).to_string(), out, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject')
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-valuecapture')
    ap.add_argument('--glmsingle-deriv', default='glmsingle_fbevents')
    ap.add_argument('--prf-deriv', default='prf_glmsingle_fbevents_items')
    ap.add_argument('--radius', type=float, default=2.)
    ap.add_argument('--min-r2', type=float, default=.1)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    out = a.out or (f'/shares/zne.uzh/gdehol/ds-valuecapture/derivatives/item_location_responses/'
                    f'sub-{a.subject}_item_location_responses.tsv')
    main(a.subject, a.bids_folder, a.glmsingle_deriv, a.prf_deriv, a.radius, a.min_r2, out)
