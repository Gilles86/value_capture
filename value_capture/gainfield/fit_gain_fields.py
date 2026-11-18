"""Gain fields on single-trial betas: does the search display's target / (high- or low-value) distractor change how
strongly the stimulus near it drives visual cortex? Uses braincoder's GainFieldPRF2D (braincoder >= 0.9).

Model (per subject x ROI; braincoder docs: gain_fields.rst): each trial's input image (bar + 8 search items, the
PRF paradigm of prf/fit_prf.py) is multiplied by
    M_t(g) = max(1 + sum_k sum_l W[t,l,k] * gain_k * A_l(g; sigma), 0)
with Gaussian fields A_l at the 8 item locations and gain types k = target, distractor_low, distractor_high
(W: which item has which role on trial t; absent trials have no distractor). Each voxel's Gaussian PRF (fixed, from
the PRF fit) pools the modulated image. Shared per subject x ROI: sigma and the 3 gains (ParameterFitter shared_pars,
PRFs as fixed_pars). Gains at 0 = the PRF model.

Outputs (one row per subject x ROI x scope): scope 'all' = all trials; scope 'run-<s>-<r>' = gains refitted per run
with sigma fixed at the 'all' estimate (learning curves, for bauer). Plus R² of the gain model vs the PRF model.

    python value_capture/gainfield/fit_gain_fields.py 07 --rois V1 V2 V3 --per-run
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault('KERAS_BACKEND', 'jax')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from nilearn import masking  # noqa: E402

from value_capture.prf import fit_prf as fp  # noqa: E402
from value_capture.utils.data import Subject  # noqa: E402

ROIS = {'V1': (1,), 'V2': (2,), 'V3': (3,), 'hV4': (4,), 'LO': (7, 8), 'V3AB': (11, 12)}
GAIN_LABELS = ['target', 'distractor_low', 'distractor_high']
PRF_LABELS = ['x', 'y', 'sd', 'baseline', 'amplitude']


def trial_table(sub, sessions, glmsingle_deriv):
    trials = fp.add_item_info(sub, sub.get_trial_metadata(sessions, glmsingle_deriv=glmsingle_deriv)
                              .reset_index(drop=True))
    raw = pd.concat([sub.get_onsets(int(s), int(r)).query("event_type == 'target'")
                     [['trial_nr', 'distractor_location', 'distractor_present']].assign(session=s, run=r)
                     for (s, r), _ in trials.groupby(['session', 'run'])])
    trials = trials.drop(columns=['distractor_present'], errors='ignore').merge(
        raw.drop_duplicates(['session', 'run', 'trial_nr']), on=['session', 'run', 'trial_nr'], how='left',
        validate='one_to_one')
    trials['distractor_present'] = trials.distractor_present.isin([True, 'True'])
    return trials


def assignments(trials):
    out = []
    for _, t in trials.iterrows():
        a = {'target': int(t.target_location)}
        if t.distractor_present:
            a['distractor_high' if t.value_rank == 2 else 'distractor_low'] = int(t.distractor_location)
        out.append(a)
    return out


def r2(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    return 1 - ((y - yhat) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


def main(subject, rois, bids, prf_deriv, glmsingle_deriv, min_r2, per_run, n_iter, lr, out):
    from braincoder.models import GainFieldPRF2D
    from braincoder.optimize import ParameterFitter

    sub = Subject(subject, bids_folder=bids)
    sessions = sub.get_sessions()
    trials = trial_table(sub, sessions, glmsingle_deriv)
    images = fp.build_paradigm(trials, 'bar_items')                       # (trials, pixels)
    grid = fp.make_grid_coordinates()
    model = GainFieldPRF2D(grid_coordinates=grid, field_positions=fp.ITEM_XY, gain_labels=GAIN_LABELS)
    paradigm = model.make_paradigm(images, assignments(trials))

    pars_all = pd.read_csv(Path(bids) / 'derivatives' / prf_deriv / f'sub-{subject}' / 'ses-all' / 'func' /
                           f'sub-{subject}_ses-all_task-valuecapture_space-T1w_desc-prf_params.tsv',
                           sep='\t', index_col=0)
    mask = fp._get_gm_mask(sub, sessions[0], bids, .1)
    Y = masking.apply_mask(sub.get_single_trial_estimates(sessions, glmsingle_deriv=glmsingle_deriv),
                           mask).astype(np.float32)
    assert Y.shape[1] == len(pars_all), (Y.shape, len(pars_all))
    varea = np.round(masking.apply_mask(sub.get_retinotopic_atlas(bold_space=True, session=sessions[0]), mask))
    rows = []
    for roi in rois:
        sel = np.flatnonzero(np.isin(varea, ROIS[roi]) & (pars_all.r2.to_numpy() >= min_r2))
        if len(sel) < 10:
            print(f'{roi}: {len(sel)} voxels, skipped', flush=True)
            continue
        prf = pars_all.iloc[sel][PRF_LABELS].reset_index(drop=True).astype(np.float32)
        data = pd.DataFrame(Y[:, sel], index=paradigm.index)
        init = model.add_field_parameters(prf, field_sigma=1.5)

        def fit(d, p, init_pars, fixed):
            fitter = ParameterFitter(model=model, data=d, paradigm=p)
            est = fitter.fit(init_pars=init_pars, fixed_pars=fixed,
                             shared_pars=[l for l in model.field_parameter_labels(include_tied=False)
                                          if l not in fixed],
                             max_n_iterations=n_iter, learning_rate=lr, progressbar=False)
            return est, fitter

        est, fitter = fit(data, paradigm, init, model.prf_parameter_labels())
        pred = model.predict(paradigm=paradigm, parameters=est)
        pred0 = model.predict(paradigm=paradigm, parameters=init)
        row = dict(subject=subject, roi=roi, scope='all', n_voxels=len(sel), n_trials=len(data),
                   r2_prf=float(np.median(r2(data, pred0))), r2_gain=float(np.median(r2(data, pred))))
        row.update({l: float(est[l].iloc[0]) for l in model.field_parameter_labels()})
        rows.append(row)
        print(f"{roi}: {len(sel)} voxels, sigma={row['field_sigma']:.2f}, "
              + ', '.join(f"{k}={row['field_gain_' + k]:+.3f}" for k in GAIN_LABELS), flush=True)
        if per_run:
            fixed = model.prf_parameter_labels() + ['field_sigma']
            for (s, r), idx in trials.groupby(['session', 'run']).groups.items():
                idx = np.asarray(sorted(idx))
                e, _ = fit(data.iloc[idx], paradigm.iloc[idx], est, fixed)
                rr = dict(subject=subject, roi=roi, scope=f'run-{s}-{r}', session=s, run=r, n_voxels=len(sel),
                          n_trials=len(idx))
                rr.update({l: float(e[l].iloc[0]) for l in model.field_parameter_labels()})
                rows.append(rr)
    t = pd.DataFrame(rows)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t.round(5).to_csv(out, sep='\t', index=False)
    print(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject')
    ap.add_argument('--rois', nargs='+', default=list(ROIS))
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-valuecapture')
    ap.add_argument('--prf-deriv', default='prf_glmsingle_items')
    ap.add_argument('--glmsingle-deriv', default='glmsingle_fbreg')
    ap.add_argument('--min-r2', type=float, default=.1)
    ap.add_argument('--per-run', action='store_true')
    ap.add_argument('--n-iter', type=int, default=2000)
    ap.add_argument('--learning-rate', type=float, default=.01)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    out = a.out or (f'/shares/zne.uzh/gdehol/ds-valuecapture/derivatives/gain_fields/'
                    f'sub-{a.subject}_gain_fields.tsv')
    main(a.subject, a.rois, a.bids_folder, a.prf_deriv, a.glmsingle_deriv, a.min_r2, a.per_run, a.n_iter,
         a.learning_rate, out)
