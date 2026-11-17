#!/usr/bin/env python3
"""First-level GLM aimed at reward valuation (vmPFC, striatum): cue value, outcome magnitude, prediction error.

The colour of the distractor predicts how many points a fast correct response earns (×100 high, ×1 low). On
distractor-ABSENT trials the multiplier is still high or low (random, 50/50) but nothing on screen predicts it,
so the outcome magnitude is a surprise. Feedback (points) is shown on half of the scanner trials; on the other
half a blank replaces it, which on a high-value trial omits a large expected reward.

Regressors (SPM canonical HRF):
  Search display (1.75 s):   search_high, search_low (distractor of that value), search_absent
  Feedback shown (1 s):      fb_high_cued, fb_low_cued        (distractor present: magnitude predicted)
                             fb_high_uncued, fb_low_uncued    (distractor absent: magnitude not predicted)
  Feedback omitted (1 s):    omit_high, omit_low, omit_absent
  Errors are kept in their condition; a fast-response bonus scales points, so 'high/low' here is the multiplier.

Frame times: slice_time_ref = fMRIPrep's StartTime / TR (slice-timing correction to mid-TR; nilearn's default 0
would shift every onset by ~0.8 s).
Nuisance: 6 motion parameters, FD, DVARS, 7 aCompCor and fMRIPrep's DCT cosines (the older
fit_nilearn_glm.py turned nilearn's drift model off but did not pass the cosines, so it had no high-pass).

Contrasts:
  cue_value          search_high − search_low               anticipatory value of the distractor colour
  capture            (search_high + search_low)/2 − search_absent
  outcome_magnitude  (fb_high_cued + fb_high_uncued − fb_low_cued − fb_low_uncued)/2
  outcome_rpe        (fb_high_uncued − fb_low_uncued) − (fb_high_cued − fb_low_cued)
                     magnitude effect larger when unpredicted (a prediction-error signature)
  outcome_uncued     fb_high_uncued − fb_low_uncued
  feedback_vs_omit   mean(fb_*) − mean(omit_*)
  omission_value     omit_high − omit_low                   omission of a large vs small expected reward (−RPE)
  outcome_logreward  parametric log(1 + points) on all shown feedback, centred (fb_logpoints)

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
    'outcome_magnitude': '(fb_high_cued + fb_high_uncued - fb_low_cued - fb_low_uncued) / 2',
    'outcome_rpe': '(fb_high_uncued - fb_low_uncued) - (fb_high_cued - fb_low_cued)',
    'outcome_uncued': 'fb_high_uncued - fb_low_uncued',
    'feedback_vs_omit': ('(fb_high_cued + fb_low_cued + fb_high_uncued + fb_low_uncued) / 4 '
                         '- (omit_high + omit_low + omit_absent) / 3'),
    'omission_value': 'omit_high - omit_low',
    # log(1 + points) received, over and above the condition means (cued/uncued × high/low): within-condition
    # reward variation (faster responses earn more)
    'outcome_logreward': 'fb_logpoints',
}
CONFOUNDS = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'framewise_displacement', 'dvars'] + \
    [f'a_comp_cor_{i:02d}' for i in range(7)]


def build_events(ev, n_removed):
    off = n_removed * TR
    rows = []
    for _, r in ev[ev.event_type == 'target'].iterrows():
        lab = 'search_absent' if not r.distractor_present else ('search_high' if r.value_rank == 2 else 'search_low')
        rows.append(dict(onset=r.onset - off, duration=1.75, trial_type=lab))
    fb = ev[ev.event_type == 'feedback']
    shown = fb.show_feedback.isin([True, 'True'])
    logp = np.log1p(fb.earned_points.fillna(0).astype(float))
    logp_c = logp - logp[shown].mean()           # centred over shown feedback (nilearn does not centre)
    for (_, r), lp in zip(fb.iterrows(), logp_c):
        mag = 'high' if r.value_rank == 2 else 'low'
        if r.show_feedback in (True, 'True'):
            lab = f'fb_{mag}_' + ('cued' if r.distractor_present in (True, 'True') else 'uncued')
            # parametric: log(1 + points) actually received, all shown feedback
            rows.append(dict(onset=r.onset - off, duration=1.0, trial_type='fb_logpoints', modulation=lp))
        else:
            lab = f'omit_{mag}' if r.distractor_present in (True, 'True') else 'omit_absent'
        rows.append(dict(onset=r.onset - off, duration=1.0, trial_type=lab, modulation=1.))
    return pd.DataFrame(rows).sort_values('onset').reset_index(drop=True)


def confounds(sub, session, run, fmriprep_deriv):
    fn = (Path(sub.bids_folder) / 'derivatives' / fmriprep_deriv / f'sub-{sub.subject_id}' / f'ses-{session}' /
          'func' / f'sub-{sub.subject_id}_ses-{session}_task-valuecapture_rec-NORDIC_run-{run}'
          '_desc-confounds_timeseries.tsv')
    df = pd.read_csv(fn, sep='\t')
    cols = [c for c in CONFOUNDS if c in df] + [c for c in df if c.startswith('cosine')]
    return df[cols].fillna(0)


def main(subject, bids_folder, space, res, smoothing_fwhm, n_jobs, fmriprep_deriv='fmriprep'):
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
            ev = build_events(raw, n_removed)
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
    out = Path(bids_folder) / 'derivatives' / 'nilearn_glm_reward' / f'sub-{subject}' / 'func'
    out.mkdir(parents=True, exist_ok=True)
    cols = set().union(*[dm.columns for dm in model.design_matrices_])
    for name, expr in CONTRASTS.items():
        import re
        if not set(re.findall(r'[A-Za-z_]\w*', expr)) <= cols:
            print(f'{name}: skipped (missing regressors)')
            continue
        for kind in ('z_score', 'effect_size'):
            img = model.compute_contrast(expr, output_type=kind)
            img.set_data_dtype(np.float32)
            img.to_filename(str(out / f'sub-{subject}_task-valuecapture_{space_tag}_contrast-{name}'
                                      f'_stat-{"z" if kind == "z_score" else "effect"}_statmap.nii.gz'))
        print(f'{name}: saved', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject')
    ap.add_argument('--bids-folder', default=BIDS_FOLDER)
    ap.add_argument('--space', default='MNI152NLin2009cAsym')
    ap.add_argument('--res', default='2')
    ap.add_argument('--smoothing-fwhm', type=float, default=6.)
    ap.add_argument('--n-jobs', type=int, default=8)
    a = ap.parse_args()
    main(a.subject, a.bids_folder, a.space, a.res, a.smoothing_fwhm, a.n_jobs)
