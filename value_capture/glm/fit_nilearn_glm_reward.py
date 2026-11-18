#!/usr/bin/env python3
"""First-level GLM aimed at reward valuation (vmPFC, striatum): cue value, outcome magnitude, prediction error.

The colour of the distractor predicts how many points a fast correct response earns (×100 high, ×1 low). On
distractor-ABSENT trials the multiplier is still high or low (random, 50/50) but nothing on screen predicts it,
so the outcome magnitude is a surprise. Feedback (points) is shown on half of the scanner trials; on the other
half a blank replaces it, which on a high-value trial omits a large expected reward.

Regressors (SPM canonical HRF):
  Search display (1.75 s):   search_high, search_low (distractor of that value), search_absent
  Feedback shown (1 s):      fb_cued (distractor present: multiplier predicted by its colour), fb_uncued
                             (distractor absent: multiplier not predicted), each with a parametric modulator
                             log(1 + points received), centred within cue type and run
  Feedback omitted (1 s):    omit_high, omit_low, omit_absent
  Errors are kept in their condition; a fast-response bonus scales points, so 'high/low' here is the multiplier.

Frame times: slice_time_ref = fMRIPrep's StartTime / TR (slice-timing correction to mid-TR; nilearn's default 0
would shift every onset by ~0.8 s).
Nuisance: 6 motion parameters, FD, DVARS, 7 aCompCor and fMRIPrep's DCT cosines (the older
fit_nilearn_glm.py turned nilearn's drift model off but did not pass the cosines, so it had no high-pass).

Contrasts:
  cue_value                search_high − search_low          anticipatory value of the distractor colour
  capture                  (search_high + search_low)/2 − search_absent
  outcome_logreward        mean of the two log-points modulators (reward received)
  outcome_logreward_cued   / _uncued: each cue type separately
  outcome_rpe              uncued − cued log-points slope    (prediction-error signature: larger when unpredicted)
  feedback_vs_omit         points shown vs omitted
  omission_value           omit_high − omit_low              omission of a large vs small expected reward (−RPE)

    python value_capture/glm/fit_nilearn_glm_reward.py 04 --bids-folder /shares/zne.uzh/gdehol/ds-valuecapture \\
        --space MNI152NLin2009cAsym --res 2 --smoothing-fwhm 6 --n-jobs 8
Output: <bids>/derivatives/nilearn_glm_reward/sub-<sub>/func/*_contrast-<name>_stat-{z,effect}_statmap.nii.gz
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import image
from nilearn.glm.first_level import FirstLevelModel

from value_capture.utils.data import BIDS_FOLDER, Subject

TR = 1.6
CONTRASTS = {
    'cue_value': 'search_high - search_low',
    'capture': '(search_high + search_low) / 2 - search_absent',
    # Outcome: one regressor per cue type (unmodulated) + log(1 + points) as parametric modulator, centred within
    # cue type. (v1 had high/low feedback regressors AND a log-points modulator centred over both: the modulator
    # was ~ a copy of high − low, the two split the variance and came out with opposite signs in striatum.)
    'outcome_logreward': '(fb_cued_logpoints + fb_uncued_logpoints) / 2',
    'outcome_logreward_cued': 'fb_cued_logpoints',
    'outcome_logreward_uncued': 'fb_uncued_logpoints',
    'outcome_rpe': 'fb_uncued_logpoints - fb_cued_logpoints',
    'feedback_vs_omit': '(fb_cued + fb_uncued) / 2 - (omit_high + omit_low + omit_absent) / 3',
    'omission_value': 'omit_high - omit_low',
}
# Design 'rpe': reward relative to what the cue predicts. Feedback split by cue (high colour, low colour, no
# distractor = unpredicted), each unmodulated, plus ONE modulator over all shown feedback:
#   rpe = log(1 + points) − mean log(1 + points) of that cue condition (in this run)
# For cued trials the multiplier is predicted, so the rpe is only the within-multiplier variation (speed);
# for uncued trials it also carries the unpredicted ×100 vs ×1. The condition means carry the predicted part.
CONTRASTS_RPE = {
    'cue_value': 'search_high - search_low',
    'rpe': 'fb_rpe',                                                      # reward over and above the cue's prediction
    'expected_reward': 'fb_cue_high - fb_cue_low',                       # outcome response, high vs low predicted
    'uncued_vs_cued': 'fb_uncued - (fb_cue_high + fb_cue_low) / 2',     # unpredicted vs predicted feedback, mean
    'feedback_vs_omit': '(fb_cue_high + fb_cue_low + fb_uncued) / 3 - (omit_high + omit_low + omit_absent) / 3',
    'omission_value': 'omit_high - omit_low',
}
# Design 'rpe_rt': within a multiplier, points depend only on speed and accuracy, so the rpe modulator of cued trials
# is also an RT / error regressor. This design adds, at the search display, an RT modulator (correct trials, centred
# within run) and an error regressor, and splits the rpe modulator: cued (speed only) vs uncued (also the unpredicted
# ×100 vs ×1, which does not depend on RT).
CONTRASTS_RPE_RT = {
    'cue_value': 'search_high - search_low',
    'rpe': '(fb_rpe_cued + fb_rpe_uncued) / 2',
    'rpe_cued': 'fb_rpe_cued',
    'rpe_uncued': 'fb_rpe_uncued',
    'expected_reward': 'fb_cue_high - fb_cue_low',
    'omission_value': 'omit_high - omit_low',
    'search_rt': 'search_rt',
}
CONFOUNDS = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'framewise_displacement', 'dvars'] + \
    [f'a_comp_cor_{i:02d}' for i in range(7)]


def build_events_rpe(ev, n_removed, rt_control=False):
    off = n_removed * TR
    rows = []
    if rt_control:      # RT and accuracy are logged from the phase after the response on, so read them at feedback
        beh = ev[ev.event_type == 'feedback'].drop_duplicates('trial_nr').set_index('trial_nr')
        ok = beh.correct.isin([True, 'True']) & pd.to_numeric(beh.rt, errors='coerce').notna()
        rt = pd.to_numeric(beh.rt, errors='coerce')[ok]
        rt = rt - rt.mean()
        for _, r in ev[ev.event_type == 'target'].drop_duplicates('trial_nr').iterrows():
            on = r.onset - off
            if r.trial_nr in rt.index:
                rows.append(dict(onset=on, duration=1.75, trial_type='search_rt', modulation=float(rt[r.trial_nr])))
            else:
                rows.append(dict(onset=on, duration=1.75, trial_type='search_error', modulation=1.))
    for _, r in ev[ev.event_type == 'target'].iterrows():
        lab = 'search_absent' if not r.distractor_present else ('search_high' if r.value_rank == 2 else 'search_low')
        rows.append(dict(onset=r.onset - off, duration=1.75, trial_type=lab, modulation=1.))
    fb = ev[ev.event_type == 'feedback'].copy()
    fb['shown'] = fb.show_feedback.isin([True, 'True'])
    cued = fb.distractor_present.isin([True, 'True'])
    fb['cue'] = np.where(cued, np.where(fb.value_rank == 2, 'cue_high', 'cue_low'), 'uncued')
    fb['logp'] = np.log1p(fb.earned_points.fillna(0).astype(float))
    m = fb[fb.shown].groupby('cue').logp.mean()
    fb['rpe'] = fb.logp - fb.cue.map(m).fillna(0)
    for _, r in fb.iterrows():
        on = r.onset - off
        if r.shown:
            rows.append(dict(onset=on, duration=1.0, trial_type=f'fb_{r.cue}', modulation=1.))
            name = 'fb_rpe' if not rt_control else ('fb_rpe_uncued' if r.cue == 'uncued' else 'fb_rpe_cued')
            rows.append(dict(onset=on, duration=1.0, trial_type=name, modulation=r.rpe))
        else:
            lab = {'cue_high': 'omit_high', 'cue_low': 'omit_low', 'uncued': 'omit_absent'}[r.cue]
            rows.append(dict(onset=on, duration=1.0, trial_type=lab, modulation=1.))
    return pd.DataFrame(rows).sort_values('onset').reset_index(drop=True)


def build_events(ev, n_removed):
    off = n_removed * TR
    rows = []
    for _, r in ev[ev.event_type == 'target'].iterrows():
        lab = 'search_absent' if not r.distractor_present else ('search_high' if r.value_rank == 2 else 'search_low')
        rows.append(dict(onset=r.onset - off, duration=1.75, trial_type=lab))
    fb = ev[ev.event_type == 'feedback'].copy()
    fb['shown'] = fb.show_feedback.isin([True, 'True'])
    fb['cued'] = fb.distractor_present.isin([True, 'True'])
    fb['logp'] = np.log1p(fb.earned_points.fillna(0).astype(float))
    # centre log points within cue type over the shown feedback of this run (nilearn does not centre)
    fb['logp_c'] = fb.logp - fb[fb.shown].groupby('cued').logp.transform('mean').reindex(fb.index).fillna(0)
    for _, r in fb.iterrows():
        if r.shown:
            lab = 'fb_cued' if r.cued else 'fb_uncued'
            rows.append(dict(onset=r.onset - off, duration=1.0, trial_type=f'{lab}_logpoints', modulation=r.logp_c))
        else:
            lab = ('omit_high' if r.value_rank == 2 else 'omit_low') if r.cued else 'omit_absent'
        rows.append(dict(onset=r.onset - off, duration=1.0, trial_type=lab, modulation=1.))
    return pd.DataFrame(rows).sort_values('onset').reset_index(drop=True)


def confounds(sub, session, run, fmriprep_deriv):
    fn = (Path(sub.bids_folder) / 'derivatives' / fmriprep_deriv / f'sub-{sub.subject_id}' / f'ses-{session}' /
          'func' / f'sub-{sub.subject_id}_ses-{session}_task-valuecapture_rec-NORDIC_run-{run}'
          '_desc-confounds_timeseries.tsv')
    df = pd.read_csv(fn, sep='\t')
    cols = [c for c in CONFOUNDS if c in df] + [c for c in df if c.startswith('cosine')]
    return df[cols].fillna(0)


def main(subject, bids_folder, space, res, smoothing_fwhm, n_jobs, fmriprep_deriv='fmriprep', design='logreward'):
    sub = Subject(subject, bids_folder=bids_folder)
    sessions = sub.get_sessions()
    space_tag = f'space-{space}' + (f'_res-{res}' if res else '')
    imgs, evs, confs, start_times = [], [], [], []
    for ses in sessions:
        runs = sub.get_runs(ses)
        for run, bold in zip(runs, sub.get_preprocessed_bold(ses, runs, fmriprep_deriv=fmriprep_deriv,
                                                             space=space, res=res)):
            raw = sub.get_onsets(ses, run)
            n_vols = image.load_img(str(bold)).shape[3]
            n_removed = len(raw[raw.event_type == 'pulse']) - n_vols
            ev = (build_events(raw, n_removed) if design == 'logreward'
                  else build_events_rpe(raw, n_removed, rt_control=design == 'rpe_rt'))
            imgs.append(str(bold))
            start_times.append(json.loads(Path(str(bold).replace('.nii.gz', '.json')).read_text()).get('StartTime', 0.))
            evs.append(ev)
            confs.append(confounds(sub, ses, run, fmriprep_deriv))
            print(f'ses-{ses} run-{run}: {n_vols} vols, ' +
                  ', '.join(f'{k} {v}' for k, v in ev.trial_type.value_counts().sort_index().items()), flush=True)
    # fMRIPrep's slice-timing correction resamples every slice to StartTime (here 0.77 s = mid-TR); nilearn
    # assumes frame times at the start of each TR (slice_time_ref=0) unless told, a 0.77 s shift.
    assert np.ptp(start_times) < 1e-6, start_times
    slice_time_ref = start_times[0] / TR
    print(f'StartTime {start_times[0]} s -> slice_time_ref {slice_time_ref:.3f}', flush=True)
    model = FirstLevelModel(t_r=TR, hrf_model='spm', slice_time_ref=slice_time_ref, drift_model=None, noise_model='ar1',
                            smoothing_fwhm=smoothing_fwhm, standardize=False, n_jobs=n_jobs,
                            mask_img=sub.get_brain_mask(sessions[0], fmriprep_deriv=fmriprep_deriv,
                                                        space=space, res=res))
    model.fit(imgs, events=evs, confounds=confs)
    deriv = {'logreward': 'nilearn_glm_reward', 'rpe': 'nilearn_glm_reward_rpe',
             'rpe_rt': 'nilearn_glm_reward_rpe_rt'}[design]
    out = Path(bids_folder) / 'derivatives' / deriv / f'sub-{subject}' / 'func'
    out.mkdir(parents=True, exist_ok=True)
    from nilearn.glm.contrasts import expression_to_contrast_vector
    import re
    for name, expr in {'logreward': CONTRASTS, 'rpe': CONTRASTS_RPE, 'rpe_rt': CONTRASTS_RPE_RT}[design].items():
        # Per-run contrast vectors: a run that lacks one of the contrast's regressors (e.g. no uncued
        # low-reward feedback in that run) gets a null vector and is left out of the fixed-effects combination,
        # instead of failing the whole subject (KeyError) or biasing the contrast.
        needed = set(re.findall(r'[A-Za-z_]\w*', expr))
        vecs = [expression_to_contrast_vector(expr, list(dm.columns)) if needed <= set(dm.columns)
                else np.zeros(dm.shape[1]) for dm in model.design_matrices_]
        n_used = sum(v.any() for v in vecs)
        if n_used == 0:
            print(f'{name}: skipped (no run has all regressors)')
            continue
        for kind in ('z_score', 'effect_size'):
            img = model.compute_contrast(vecs, output_type=kind)
            img.set_data_dtype(np.float32)
            img.to_filename(str(out / f'sub-{subject}_task-valuecapture_{space_tag}_contrast-{name}'
                                      f'_stat-{"z" if kind == "z_score" else "effect"}_statmap.nii.gz'))
        if n_used < len(vecs):
            print(f'{name}: {n_used}/{len(vecs)} runs', flush=True)
        print(f'{name}: saved', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject')
    ap.add_argument('--bids-folder', default=BIDS_FOLDER)
    ap.add_argument('--space', default='MNI152NLin2009cAsym')
    ap.add_argument('--res', default='2')
    ap.add_argument('--smoothing-fwhm', type=float, default=6.)
    ap.add_argument('--n-jobs', type=int, default=8)
    ap.add_argument('--design', choices=['logreward', 'rpe', 'rpe_rt'], default='logreward',
                    help="'rpe': reward relative to the cue's prediction -> nilearn_glm_reward_rpe; "
                         "'rpe_rt': the same with search RT and errors modelled -> nilearn_glm_reward_rpe_rt")
    a = ap.parse_args()
    main(a.subject, a.bids_folder, a.space, a.res, a.smoothing_fwhm, a.n_jobs, design=a.design)
