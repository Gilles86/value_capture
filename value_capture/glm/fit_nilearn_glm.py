#!/usr/bin/env python3
"""
Fit a nilearn first-level GLM for the value capture fMRI task.

What is modelled
----------------
Two sets of regressors are built from each run's events file and convolved
with the SPM canonical HRF:

  Target phase (1.75 s) — the visual-search array:
    dist_absent   — distractor-absent trials
    dist_rank0    — low-value distractor
    dist_rank1    — medium-value distractor
    dist_rank2    — high-value distractor

  Feedback phase (1.0 s):
    feedback_rank0_shown  — shown feedback, low-value distractor trial
    feedback_rank1_shown  — shown feedback, medium-value distractor trial
    feedback_rank2_shown  — shown feedback, high-value distractor trial
    feedback_absent_shown — shown feedback, distractor-absent trial
    feedback_points       — shown feedback, amplitude = mean-centred log(1 + earned_points)
                            (areas where BOLD scales with log reward magnitude)
    feedback_omitted      — blank fixation on no-feedback trials

Motion parameters and anatomical CompCor + DCT cosines from fmriprep confounds
are included as nuisance regressors.  The fmriprep cosines handle low-frequency
drift so nilearn's own drift model is disabled.

Contrasts
---------
  value_linear    : dist_rank2 - dist_rank0
                    (BOLD scales linearly with distractor value; proximate reward)
  value_capture   : dist_rank2 - dist_absent
                    (high-value distractor vs no distractor; signature effect)
  value_nonlinear : dist_rank1 - (dist_rank0 + dist_rank2) / 2
                    (nonlinear/quadratic value coding; tests whether medium-value
                    response deviates from linear interpolation between low/high)
  value_step_low  : dist_rank1 - dist_rank0
                    (low→medium value step)
  value_step_high : dist_rank2 - dist_rank1
                    (medium→high value step)
  distractor_any  : (dist_rank0 + dist_rank1 + dist_rank2) / 3 - dist_absent
                    (attentional capture regardless of value)
  feedback        : (feedback_rank0_shown + feedback_rank1_shown +
                     feedback_rank2_shown + feedback_absent_shown) / 4 - feedback_omitted
                    (any explicit feedback vs no feedback; RPE proxy)
  feedback_value  : feedback_points
                    (BOLD scales with log magnitude of reward received; RPE magnitude)
  feedback_step_low  : feedback_rank1_shown - feedback_rank0_shown
                       (medium vs low value at feedback; value-dependent RPE)
  feedback_step_high : feedback_rank2_shown - feedback_rank1_shown
                       (high vs medium value at feedback; value-dependent RPE)

Timing
------
get_onsets() returns times relative to the first scanner pulse (t=0), which
includes stabilisation volumes that fmriprep discards.  Onsets are shifted by
n_removed * TR so they are relative to the first fmriprep volume.

Output
------
Per-subject z-score and effect-size maps:

  <bids>/derivatives/nilearn_glm/sub-<sub>/func/
    sub-<sub>_task-valuecapture_space-<space>_contrast-<name>_stat-z_statmap.nii.gz
    sub-<sub>_task-valuecapture_space-<space>_contrast-<name>_stat-effect_statmap.nii.gz

  <bids>/derivatives/nilearn_glm/sub-<sub>/figures/
    sub-<sub>_task-valuecapture_space-<space>_design_matrix.png

Usage:
    python fit_nilearn_glm.py 01
    python fit_nilearn_glm.py 01 --sessions 2
    python fit_nilearn_glm.py 01 --space MNI152NLin2009cAsym --res 2
    python fit_nilearn_glm.py 01 --smoothing-fwhm 6
    python fit_nilearn_glm.py 01 --bids-folder /shares/zne.uzh/gdehol/ds-valuecapture
"""

import argparse
import warnings
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from nilearn import image
from nilearn.glm.first_level import FirstLevelModel
from nilearn.plotting import plot_design_matrix

from value_capture.utils.data import Subject, BIDS_FOLDER

warnings.filterwarnings('ignore')

TR = 1.6

CONTRASTS = {
    # Proximate reward: BOLD scales with distractor value (linear trend)
    'value_linear':      'dist_rank2 - dist_rank0',
    # Signature value-capture effect: high value vs no distractor
    'value_capture':     'dist_rank2 - dist_absent',
    # Nonlinear value: is medium value over/under-represented vs. linear?
    # Positive = inverted-U (more response to middle rank than expected from linear)
    # Negative = U-shaped (suppressed response to middle)
    'value_nonlinear':   'dist_rank1 - (dist_rank0 + dist_rank2) / 2',
    # Fine-grained steps: does each rank step drive equal BOLD increments?
    'value_step_low':    'dist_rank1 - dist_rank0',
    'value_step_high':   'dist_rank2 - dist_rank1',
    # Any distractor vs absent
    'distractor_any':    '(dist_rank0 + dist_rank1 + dist_rank2) / 3 - dist_absent',
    # RPE proxy: any explicit feedback vs. no feedback
    'feedback':          ('(feedback_rank0_shown + feedback_rank1_shown + '
                          'feedback_rank2_shown + feedback_absent_shown) / 4 - feedback_omitted'),
    # RPE magnitude: BOLD scales with log(earned_points) — reward prediction error proxy
    'feedback_value':    'feedback_points',
    # Value-dependent feedback: medium vs low rank, high vs medium rank
    'feedback_step_low':  'feedback_rank1_shown - feedback_rank0_shown',
    'feedback_step_high': 'feedback_rank2_shown - feedback_rank1_shown',
}


def build_events(onsets_df, n_removed):
    """Return nilearn events DataFrame for one run.

    Parameters
    ----------
    onsets_df : DataFrame from Subject.get_onsets() — times relative to first pulse
    n_removed : int — stabilisation volumes removed by fmriprep

    Returns
    -------
    DataFrame with columns onset, duration, trial_type (and modulation where used)
    """
    dummy_offset = n_removed * TR
    rows = []

    # --- Target phase: split by distractor value condition ---
    targets = onsets_df[onsets_df['event_type'] == 'target']
    for _, row in targets.iterrows():
        label = ('dist_absent' if not row['distractor_present']
                 else f'dist_rank{int(row["value_rank"])}')
        rows.append({
            'onset': row['onset'] - dummy_offset,
            'duration': 1.75,
            'trial_type': label,
        })

    # --- Feedback phase ---
    feedbacks = onsets_df[onsets_df['event_type'] == 'feedback']
    shown = feedbacks[feedbacks['show_feedback'] == True]

    # Log-transform points (log1p handles zeros) then mean-centre within run
    log_points = np.log1p(shown['earned_points'].fillna(0).values)
    mean_log_points = log_points.mean() if len(log_points) > 0 else 0.0

    for _, row in feedbacks.iterrows():
        onset = row['onset'] - dummy_offset
        if not row['show_feedback']:
            rows.append({'onset': onset, 'duration': 1.0,
                         'trial_type': 'feedback_omitted'})
        else:
            # Unmodulated: split by distractor rank on this trial so we can
            # contrast medium-vs-low and high-vs-medium at feedback
            if not row['distractor_present']:
                fb_label = 'feedback_absent_shown'
            else:
                fb_label = f'feedback_rank{int(row["value_rank"])}_shown'
            rows.append({'onset': onset, 'duration': 1.0,
                         'trial_type': fb_label})
            # Parametric: captures how BOLD scales with log reward magnitude
            log_val = np.log1p(float(row['earned_points'] or 0))
            rows.append({'onset': onset, 'duration': 1.0,
                         'trial_type': 'feedback_points',
                         'modulation': log_val - mean_log_points})

    return pd.DataFrame(rows).sort_values('onset').reset_index(drop=True)


def main(subject, sessions=None, bids_folder=BIDS_FOLDER, fmriprep_deriv='fmriprep',
         space='T1w', res=None, smoothing_fwhm=None, n_jobs=1):
    sub = Subject(subject, bids_folder=bids_folder)

    if sessions is None:
        sessions = sub.get_sessions()

    res_label = f'_res-{res}' if res else ''
    space_tag = f'space-{space}{res_label}'
    smooth_tag = f'  smoothing={smoothing_fwhm} mm' if smoothing_fwhm else ''
    print(f'sub-{subject}  sessions={sessions}  {space_tag}{smooth_tag}  [{fmriprep_deriv}]  n_jobs={n_jobs}')

    imgs, events_list, confounds_list = [], [], []

    for session in sessions:
        runs = sub.get_runs(session)
        bold_paths = sub.get_preprocessed_bold(session, runs,
                                               fmriprep_deriv=fmriprep_deriv,
                                               space=space, res=res)

        for run, bold_path in zip(runs, bold_paths):
            raw_events = sub.get_onsets(session, run)
            n_vols = image.load_img(str(bold_path)).shape[3]
            n_pulses = len(raw_events[raw_events['event_type'] == 'pulse'])
            n_removed = n_pulses - n_vols

            events = build_events(raw_events, n_removed)
            confounds = sub.get_confounds(session, run,
                                          fmriprep_deriv=fmriprep_deriv)

            imgs.append(str(bold_path))
            events_list.append(events)
            confounds_list.append(confounds)

            conditions = sorted(events['trial_type'].unique())
            print(f'  ses-{session} run-{run}: {n_vols} vols '
                  f'({n_removed} removed), conditions: {conditions}')

    mask = sub.get_brain_mask(sessions[0], fmriprep_deriv=fmriprep_deriv,
                              space=space, res=res)

    model = FirstLevelModel(
        t_r=TR,
        hrf_model='spm',
        drift_model=None,   # fmriprep DCT cosines in confounds handle drift
        noise_model='ar1',
        smoothing_fwhm=smoothing_fwhm,
        standardize=False,
        mask_img=mask,
        n_jobs=n_jobs,
    )

    print('\nFitting GLM...')
    model.fit(imgs, events=events_list, confounds=confounds_list)

    out_dir = (Path(bids_folder) / 'derivatives' / 'nilearn_glm'
               / f'sub-{subject}' / 'func')
    fig_dir = out_dir.parent / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    fn_base = f'sub-{subject}_task-valuecapture_{space_tag}'

    # Plot design matrix for every run (one subplot per run)
    n_runs = len(model.design_matrices_)
    fig, axes = plt.subplots(1, n_runs, figsize=(5 * n_runs, 8),
                             constrained_layout=True)
    if n_runs == 1:
        axes = [axes]
    run_labels = [f'ses-{ses} run-{run}'
                  for ses in sessions for run in sub.get_runs(ses)]
    for ax, dm, label in zip(axes, model.design_matrices_, run_labels):
        plot_design_matrix(dm, axes=ax)
        ax.set_title(label, fontsize=9)
    fig.suptitle(f'sub-{subject} design matrices', fontsize=11)
    dm_path = fig_dir / f'{fn_base}_design_matrix.png'
    fig.savefig(str(dm_path), dpi=150)
    plt.close(fig)
    print(f'\nDesign matrix plot → {dm_path.name}')

    print('\nSaving contrasts:')
    for name, expr in CONTRASTS.items():
        z_map = model.compute_contrast(expr, output_type='z_score')
        effect_map = model.compute_contrast(expr, output_type='effect_size')

        z_fname = out_dir / f'{fn_base}_contrast-{name}_stat-z_statmap.nii.gz'
        e_fname = out_dir / f'{fn_base}_contrast-{name}_stat-effect_statmap.nii.gz'

        z_map.to_filename(str(z_fname))
        effect_map.to_filename(str(e_fname))
        print(f'  {name}: {z_fname.name}')

    print(f'\nDone → {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('subject', help="Subject label without 'sub-', e.g. 01")
    parser.add_argument('--sessions', type=int, nargs='+', default=None,
                        help='Session number(s) to fit. Default: all available.')
    parser.add_argument('--space', default='T1w',
                        help="BIDS space entity (default: T1w). "
                             "Use 'MNI152NLin2009cAsym' for MNI output.")
    parser.add_argument('--res', default=None,
                        help="BIDS res entity, e.g. '2' for 2 mm MNI. "
                             "Omit for T1w native resolution.")
    parser.add_argument('--smoothing-fwhm', type=float, default=None,
                        help='Spatial smoothing kernel FWHM in mm applied before '
                             'fitting (e.g. 6). Default: no smoothing.')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Parallel jobs for GLM fitting across runs (default: 1). '
                             'Set to -1 to use all available CPUs.')
    parser.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    parser.add_argument('--fmriprep-deriv', default='fmriprep',
                        help='Name of the fmriprep derivatives folder.')
    args = parser.parse_args()

    main(args.subject,
         sessions=args.sessions,
         bids_folder=args.bids_folder,
         fmriprep_deriv=args.fmriprep_deriv,
         space=args.space,
         res=args.res,
         smoothing_fwhm=args.smoothing_fwhm,
         n_jobs=args.n_jobs)
