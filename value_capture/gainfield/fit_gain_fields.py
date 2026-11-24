"""Gain fields on single-trial betas: does the search display's target / (high- or low-value) distractor change how
strongly the stimulus near it drives visual cortex? Uses braincoder's GainFieldPRF2D (braincoder >= 0.9).

Model (per subject x ROI; braincoder docs: gain_fields.rst): each trial's input image (bar + 8 search items, the
PRF paradigm of prf/fit_prf.py) is multiplied by
    M_t(g) = max(1 + sum_k sum_l W[t,l,k] * gain_k * A_l(g; sigma), 0)
with Gaussian fields A_l at the 8 item locations and gain types k = target, distractor_low, distractor_high
(W: which item has which role on trial t; absent trials have no distractor). Each voxel's Gaussian PRF (fixed, from
the PRF fit) pools the modulated image. Shared per subject x ROI: sigma and the 3 gains (ParameterFitter shared_pars,
PRFs as fixed_pars). Gains at 0 = the PRF model.

Cross-validation by session (--cross-session): the gain fields of session s sit on the PRFs fitted on the other
session (``<prf-deriv>_train-ses<other>``), as in decode/decode_trials.py. Use the PRFs fitted on the same betas.

Free PRFs (--free-prf): the PRF parameters are fitted together with the gain fields on the same betas (started from
the PRF fit), and the PRF-only comparison model is refitted on those betas too (gains at 0). Without it the PRFs stay
fixed at the PRF fit, which may come from other betas.

Colour (--split-colour): four distractor gains, colour x value (distractor_<green|orange>_<low|high>). Colour and
value are counterbalanced across sessions (CLAUDE.md), so a colour effect over and above value is identified within
each subject.

Points display (--points): the feedback display comes 2.5-4 s after the search display and leaks into its betas even
when GLMsingle models it. 'subtract' removes each voxel's points response, estimated on the no-distractor trials only
(decode/aggregate_decoded.remove_points: there the multiplier is random and nothing else depends on it); 'exclude'
keeps only trials without points; 'model' puts the points display into the stimulus (gainfield/points.py): each
trial's rendered points string, scaled by the overlap of its HRF-convolved event with the search event (a function of
the delay between them), in an extra foveal pixel layer on the same coordinates. One shared weight (``points_weight``,
an image scale) is fitted; the PRFs decide where it lands; the gain fields do not modulate that layer. A points term fitted on all trials would also absorb a value effect, since the
multiplier follows the distractor's value.

Medium value (pilot design: rank 1, x10 points): when distractor-present trials of rank 1 exist, a fourth gain
``distractor_medium`` is added (its own field with --separate-widths). Designs without rank 1 are unchanged.

Betas in another subject's space (--anat-subject, --betas): the grey-matter mask, visual-area atlas and pRF fits
of ``--anat-subject`` are used, and ``--betas`` is the 4D beta image already resampled into that subject's BOLD grid
(same trial order as the trial metadata of ``--glmsingle-deriv``). Trials, behaviour and outputs stay keyed by the
subject label given.

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

ROIS = {'V1': (1,), 'V2': (2,), 'V3': (3,), 'hV4': (4,), 'LO': (7, 8), 'V3AB': (11, 12), 'VO': (5, 6), 'TO': (9, 10),
        'IPS': (101,)}
GAIN_LABELS = ['target', 'distractor_low', 'distractor_high']
GAIN_LABELS_MEDIUM = ['target', 'distractor_low', 'distractor_medium', 'distractor_high']
VALUE_NAME = {2: 'high', 1: 'medium', 0: 'low'}
COLOUR_GAIN_LABELS = ['target'] + [f'distractor_{c}_{v}' for c in ('green', 'orange') for v in ('low', 'high')]
HIGH_COLOUR = {0: 'orange', 1: 'green'}          # value_condition -> colour of the high-value (rank 2) distractor


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
    shown = sub.get_behavioral_data(exclude_outlier_rts=False).reset_index()[['session', 'run', 'trial_nr', 'show_feedback',
                                                                           'value_condition']]
    shown[['session', 'run']] = shown[['session', 'run']].astype(int)
    trials[['session', 'run']] = trials[['session', 'run']].astype(int)
    trials = trials.merge(shown.astype({'show_feedback': bool}), on=['session', 'run', 'trial_nr'], how='left',
                          validate='one_to_one')
    assert trials.show_feedback.notna().all()
    return trials


def distractor_colour(t):
    high = HIGH_COLOUR[int(t.value_condition)]
    return high if t.value_rank == 2 else {'orange': 'green', 'green': 'orange'}[high]


def assignments(trials, split_colour=False):
    out = []
    for _, t in trials.iterrows():
        a = {'target': int(t.target_location)}
        if t.distractor_present:
            value = VALUE_NAME[int(t.value_rank)]
            label = f'distractor_{distractor_colour(t)}_{value}' if split_colour else f'distractor_{value}'
            a[label] = int(t.distractor_location)
        out.append(a)
    return out


def r2(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    return 1 - ((y - yhat) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


def main(subject, rois, bids, prf_deriv, glmsingle_deriv, min_r2, per_run, n_iter, lr, out, prf_model='gauss',
         cross_session=False, points='keep', free_prf=False, split_colour=False, min_ecc=0., separate_widths=False,
         max_points_overlap=None, anat_subject=None, betas=None):
    from braincoder.models import GainFieldDoGPRF2D, GainFieldPRF2D
    from braincoder.models.gain_field import GainField
    from braincoder.optimize import ParameterFitter

    from value_capture.decode.aggregate_decoded import remove_points

    sub = Subject(subject, bids_folder=bids)
    sessions = sub.get_sessions()
    anat = sub if anat_subject is None else Subject(anat_subject, bids_folder=bids)   # whose mask, atlas and pRFs
    anat_session = anat.get_sessions()[0]
    trials = trial_table(sub, sessions, glmsingle_deriv)
    has_medium = bool((trials.distractor_present & (trials.value_rank == 1)).any())
    images = fp.build_paradigm(trials, 'bar_items')                       # (trials, pixels)
    grid = fp.make_grid_coordinates()
    model_class = {'gauss': GainFieldPRF2D, 'dog': GainFieldDoGPRF2D}[prf_model]
    if split_colour and has_medium:
        raise NotImplementedError('--split-colour with a medium value level')
    gain_labels = COLOUR_GAIN_LABELS if split_colour else (GAIN_LABELS_MEDIUM if has_medium else GAIN_LABELS)
    extra = {}
    if points == 'model':
        from keras import ops
        from value_capture.gainfield import points as pts
        trials = pts.add_points_info(sub, trials)
        patch = pts.patch_coordinates(grid)
        images = np.hstack([images, pts.points_images(trials, patch)])
        layer = np.r_[np.zeros(len(grid)), np.ones(len(patch))].astype(np.float32)
        grid = pd.concat([grid, patch], ignore_index=True)

        def combine(imgs, field_sums):                    # fields modulate the display, not the points layer
            m = ops.maximum(1.0 + sum(field_sums), 0.0)
            keep = ops.convert_to_tensor(layer)[None, None, :]
            return imgs * (m * (1.0 - keep) + keep)
        extra = dict(image_scales={'points_weight': layer}, combination=combine)
        print(f'points layer: {len(patch)} pixels, {int(trials.show_feedback.sum())} trials with points', flush=True)
    if separate_widths:                               # one field per gain label, each with its own width
        extra['fields'] = [GainField(name=f'field_{k}', gain_labels=[k]) for k in gain_labels]
    model = model_class(grid_coordinates=grid, field_positions=fp.ITEM_XY, gain_labels=gain_labels, **extra)
    sigma_labels = [l for l in model.field_parameter_labels() if l.endswith('sigma')]

    def out_name(l):                                  # field_<k>_gain_<k> -> field_gain_<k>; field_<k>_sigma -> field_sigma_<k>
        if not separate_widths:
            return l
        for k in gain_labels:
            if l == f'field_{k}_gain_{k}':
                return f'field_gain_{k}'
            if l == f'field_{k}_sigma':
                return f'field_sigma_{k}'
        return l
    prf_labels = model.prf_parameter_labels()
    paradigm_all = model.make_paradigm(images, assignments(trials, split_colour))

    def read_prf(deriv):
        a = anat.subject_id
        return pd.read_csv(Path(bids) / 'derivatives' / deriv / f'sub-{a}' / 'ses-all' / 'func' /
                           f'sub-{a}_ses-all_task-valuecapture_space-T1w_desc-prf_params.tsv',
                           sep='\t', index_col=0)
    mask = fp._get_gm_mask(anat, anat_session, bids, .1)
    beta_img = betas if betas is not None else sub.get_single_trial_estimates(sessions, glmsingle_deriv=glmsingle_deriv)
    Y_all = masking.apply_mask(beta_img, mask).astype(np.float32)
    assert Y_all.shape[0] == len(trials), (Y_all.shape, len(trials))
    varea = np.round(masking.apply_mask(anat.get_retinotopic_atlas(bold_space=True, session=anat_session, include_wang=True), mask))

    # folds: (label, trial indices, PRF derivative)
    if cross_session:
        assert len(sessions) == 2, sessions
        folds = [(f'ses-{s}', np.flatnonzero(trials.session.astype(int).to_numpy() == int(s)),
                  f'{prf_deriv}_train-ses{o}') for s, o in zip(sessions, sessions[::-1])]
    else:
        folds = [('all', np.arange(len(trials)), prf_deriv)]

    rows = []
    for scope, idx, deriv in folds:
        pars_all = read_prf(deriv)            # voxels are selected on the Gaussian fit, for every PRF model
        pars_model = pars_all if prf_model == 'gauss' else read_prf(f'{deriv}_{prf_model}')
        assert len(pars_model) == len(pars_all) == Y_all.shape[1], (len(pars_model), len(pars_all), Y_all.shape)
        tr = trials.iloc[idx].reset_index(drop=True)
        Y = Y_all[idx]
        shown = tr.show_feedback.to_numpy(bool)
        if points == 'subtract':
            Y = remove_points(tr, Y, shown)
        keep = ~shown if points == 'exclude' else np.ones(len(tr), bool)
        tr, Y, paradigm = tr[keep].reset_index(drop=True), Y[keep], paradigm_all.iloc[idx[keep]]
        for roi in rois:
            ecc = np.hypot(pars_all.x.to_numpy(), pars_all.y.to_numpy())
            sel = np.flatnonzero(np.isin(varea, ROIS[roi]) & (pars_all.r2.to_numpy() >= min_r2) & (ecc >= min_ecc)
                                 & np.isfinite(pars_model[prf_labels].to_numpy()).all(1))
            if max_points_overlap is not None:
                from value_capture.gainfield import points as pts
                ink = pts.ink_map(pts.add_points_info(sub, tr), fp.make_grid_coordinates())
                ovl = pts.text_overlap(ink, fp.make_grid_coordinates(), pars_all.x.to_numpy()[sel],
                                       pars_all.y.to_numpy()[sel], pars_all.sd.to_numpy()[sel])
                sel = sel[ovl < max_points_overlap]
            if len(sel) < 10:
                print(f'{scope} {roi}: {len(sel)} voxels, skipped', flush=True)
                continue
            prf = pars_model.iloc[sel][prf_labels].reset_index(drop=True).astype(np.float32)
            data = pd.DataFrame(Y[:, sel], index=paradigm.index)
            init = model.add_field_parameters(prf, **{l: 1.5 for l in sigma_labels})

            def fit(d, p, init_pars, fixed):
                shared = [l for l in model.field_parameter_labels(include_tied=False) if l not in fixed]
                fitter = ParameterFitter(model=model, data=d, paradigm=p)
                est = fitter.fit(init_pars=init_pars, fixed_pars=fixed or None, shared_pars=shared or None,
                                 max_n_iterations=n_iter, learning_rate=lr, progressbar=False)
                return est, fitter

            gain_pars = [l for l in model.field_parameter_labels(include_tied=False) if l != 'points_weight']
            if free_prf:
                est, _ = fit(data, paradigm, init, [])
                base, _ = fit(data, paradigm, init, gain_pars)
            else:
                est, _ = fit(data, paradigm, init, model.prf_parameter_labels())
                base = init
            pred = model.predict(paradigm=paradigm, parameters=est)
            pred0 = model.predict(paradigm=paradigm, parameters=base)
            row = dict(subject=subject, roi=roi, scope=scope, points=points, free_prf=free_prf, n_voxels=len(sel), n_trials=len(data),
                       r2_prf=float(np.median(r2(data, pred0))), r2_gain=float(np.median(r2(data, pred))))
            row.update({out_name(l): float(est[l].iloc[0]) for l in model.field_parameter_labels()})
            rows.append(row)
            print(f"{scope} {roi}: {len(sel)} voxels, {len(data)} trials, "
                  + ', '.join(f'{out_name(l)}={est[l].iloc[0]:.2f}' for l in sigma_labels) + ', '
                  + ', '.join(f"{k}={row['field_gain_' + k]:+.3f}" for k in gain_labels), flush=True)
            if per_run:
                fixed = model.prf_parameter_labels() + sigma_labels + (['points_weight'] if points == 'model' else [])
                for (s, r), ridx in tr.groupby(['session', 'run']).groups.items():
                    ridx = np.asarray(sorted(ridx))
                    e, _ = fit(data.iloc[ridx], paradigm.iloc[ridx], est, fixed)
                    rr = dict(subject=subject, roi=roi, scope=f'run-{s}-{r}', session=s, run=r, points=points,
                              n_voxels=len(sel), n_trials=len(ridx))
                    rr.update({out_name(l): float(e[l].iloc[0]) for l in model.field_parameter_labels()})
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
    ap.add_argument('--prf-model', choices=['gauss', 'dog'], default='gauss',
                    help='PRF under the gain field; dog reads <prf-deriv>_dog and writes to gain_fields_dog')
    ap.add_argument('--cross-session', action='store_true',
                    help='gain fields of each session on the PRFs of <prf-deriv>_train-ses<other>')
    ap.add_argument('--points', choices=['keep', 'subtract', 'exclude', 'model'], default='keep')
    ap.add_argument('--free-prf', action='store_true', help='fit the PRFs together with the gain fields')
    ap.add_argument('--split-colour', action='store_true', help='distractor gains per colour x value')
    ap.add_argument('--min-ecc', type=float, default=0., help='only voxels with PRF eccentricity >= this (deg)')
    ap.add_argument('--max-points-overlap', type=float, default=None,
                    help='only voxels whose pRF covers less than this fraction of the points text')
    ap.add_argument('--out-suffix', default='', help='appended to the output file name, e.g. _extra-rois')
    ap.add_argument('--separate-widths', action='store_true', help='one field (own width) per gain label')
    ap.add_argument('--tag', default='', help='suffix of the output folder, e.g. fbevents_subtract')
    ap.add_argument('--anat-subject', default=None, help='use this subject\'s mask, atlas and pRFs')
    ap.add_argument('--betas', default=None, help='4D beta NIfTI in that subject\'s BOLD grid (with --anat-subject)')
    a = ap.parse_args()
    folder = 'gain_fields' if a.prf_model == 'gauss' else f'gain_fields_{a.prf_model}'
    folder += f'_{a.tag}' if a.tag else ''
    out = a.out or (f'/shares/zne.uzh/gdehol/ds-valuecapture/derivatives/{folder}/'
                    f'sub-{a.subject}_gain_fields{a.out_suffix}.tsv')
    main(a.subject, a.rois, a.bids_folder, a.prf_deriv, a.glmsingle_deriv, a.min_r2, a.per_run, a.n_iter,
         a.learning_rate, out, prf_model=a.prf_model, cross_session=a.cross_session, points=a.points,
         free_prf=a.free_prf, split_colour=a.split_colour, min_ecc=a.min_ecc, separate_widths=a.separate_widths,
         max_points_overlap=a.max_points_overlap, anat_subject=a.anat_subject, betas=a.betas)
