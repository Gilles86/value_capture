"""Does the points display leak into the search beta as a function of the delay between them?

The points display (``+8`` ... ``+850``, foveal, 1 s) follows the search display after iti1 (0.5-4 s). Even with
the points as their own GLMsingle events (``glmsingle_fbevents``), the search beta of a trial with points carries a
foveal response. If that leak comes from the overlap of the two HRF-convolved events, it should shrink with the
delay, roughly like the overlap of the two regressors.

Per subject: foveal voxels (PRF eccentricity < --max-ecc, V1-V3, PRF R² >= .1), the mean search beta per trial,
minus the mean of trials without points at the same bar position. One row per trial with the delay, the number of
characters shown, and the predicted overlap ``c`` = regression weight of the points regressor on the search regressor
(nilearn SPM HRF, boxcars of the real durations). The group analysis is in the notes.

    python value_capture/gainfield/check_points_leak.py 04 05 ... --out notes/data/points_leak_isi.tsv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import masking
from nilearn.glm.first_level import compute_regressor

from value_capture.prf import fit_prf as fp
from value_capture.utils.data import Subject

SEARCH_DUR, POINTS_DUR = 1.75, 1.0


def overlap(delay, dt=.05):
    """Regression weight of a points regressor (onset ``delay`` after search onset) on the search regressor."""
    t = np.arange(0, 40, dt)
    s = compute_regressor([[0.], [SEARCH_DUR], [1.]], 'spm', t)[0][:, 0]
    p = compute_regressor([[delay], [POINTS_DUR], [1.]], 'spm', t)[0][:, 0]
    return float(s @ p / (s @ s))


def subject_rows(subject, bids, glmsingle_deriv, prf_deriv, max_ecc):
    sub = Subject(subject, bids_folder=bids)
    sessions = sub.get_sessions()
    trials = sub.get_trial_metadata(sessions, glmsingle_deriv=glmsingle_deriv).reset_index(drop=True)
    trials[['session', 'run']] = trials[['session', 'run']].astype(int)
    ev = []
    for (s, r), _ in trials.groupby(['session', 'run']):
        o = sub.get_onsets(s, r)
        t = o[o.event_type == 'target'].drop_duplicates('trial_nr').set_index('trial_nr').onset
        f = o[o.event_type == 'feedback'].drop_duplicates('trial_nr').set_index('trial_nr').onset
        ev.append(pd.DataFrame({'session': s, 'run': r, 'delay': f - t}).rename_axis('trial_nr').reset_index())
    beh = sub.get_behavioral_data(exclude_outlier_rts=False).reset_index()
    beh[['session', 'run']] = beh[['session', 'run']].astype(int)
    trials = (trials.merge(pd.concat(ev), on=['session', 'run', 'trial_nr'], how='left', validate='one_to_one')
              .merge(beh[['session', 'run', 'trial_nr', 'show_feedback', 'earned_points']],
                     on=['session', 'run', 'trial_nr'], how='left', validate='one_to_one'))
    shown = trials.show_feedback.isin([True, 'True']).to_numpy()

    mask = fp._get_gm_mask(sub, sessions[0], bids, .1)
    pars = pd.read_csv(Path(bids) / 'derivatives' / prf_deriv / f'sub-{subject}' / 'ses-all' / 'func' /
                       f'sub-{subject}_ses-all_task-valuecapture_space-T1w_desc-prf_params.tsv', sep='\t', index_col=0)
    varea = np.round(masking.apply_mask(sub.get_retinotopic_atlas(bold_space=True, session=sessions[0]), mask))
    sel = (np.isin(varea, (1, 2, 3)) & (pars.r2.to_numpy() >= .1)
           & (np.hypot(pars.x, pars.y).to_numpy() < max_ecc))
    Y = masking.apply_mask(sub.get_single_trial_estimates(sessions, glmsingle_deriv=glmsingle_deriv), mask)[:, sel]
    y = Y.mean(1)
    bar = trials.bar_orientation.astype(str) + trials.bar_position.round(4).astype(str)
    ref = pd.Series(y[~shown]).groupby(bar[~shown].to_numpy()).mean()
    trials['foveal_beta'] = y - bar.map(ref).to_numpy()
    trials['shown'] = shown
    trials['n_chars'] = np.where(shown, trials.earned_points.fillna(0).astype(int).astype(str).str.len() + 1, 0)
    trials['overlap'] = trials.delay.map(overlap)
    trials['subject'] = subject
    trials['n_voxels'] = int(sel.sum())
    return trials[['subject', 'session', 'run', 'trial_nr', 'value_rank', 'distractor_present', 'shown', 'delay',
                   'overlap', 'n_chars', 'earned_points', 'foveal_beta', 'n_voxels']]


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subjects', nargs='+')
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-valuecapture')
    ap.add_argument('--glmsingle-deriv', default='glmsingle_fbevents')
    ap.add_argument('--prf-deriv', default='prf_glmsingle_items')
    ap.add_argument('--max-ecc', type=float, default=1.)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = pd.concat([subject_rows(s, a.bids_folder, a.glmsingle_deriv, a.prf_deriv, a.max_ecc) for s in a.subjects])
    out.to_csv(a.out, sep='\t', index=False)
    print(out.groupby(['subject']).n_voxels.first().to_dict(), a.out)
