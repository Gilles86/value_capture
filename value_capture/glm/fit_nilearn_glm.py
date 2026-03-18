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
    feedback_shown    — points text visible
    feedback_omitted  — blank fixation (trial selected for no feedback)

Motion parameters and DCT cosines from fmriprep confounds are included as
nuisance regressors.  The fmriprep cosines handle low-frequency drift so
nilearn's own drift model is disabled.

Contrasts
---------
  value_linear   : dist_rank2 - dist_rank0          (high > low value distractor)
  value_any      : mean(dist_rank0..2) - dist_absent (any distractor > absent)
  feedback       : feedback_shown - feedback_omitted

Timing
------
get_onsets() returns times relative to the first scanner pulse (t=0), which
includes stabilisation volumes that fmriprep discards.  Onsets are shifted by
n_removed * TR so they are relative to the first fmriprep volume.

Output
------
Per-subject z-score and effect-size maps in T1w space:

  <bids>/derivatives/nilearn_glm/sub-<sub>/func/
    sub-<sub>_task-valuecapture_space-T1w_contrast-<name>_stat-z_statmap.nii.gz
    sub-<sub>_task-valuecapture_space-T1w_contrast-<name>_stat-effect_statmap.nii.gz

Usage:
    python fit_nilearn_glm.py 01
    python fit_nilearn_glm.py 01 --sessions 2
    python fit_nilearn_glm.py 01 --bids-folder /shares/zne.uzh/gdehol/ds-valuecapture
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from nilearn import image
from nilearn.glm.first_level import FirstLevelModel
from nilearn.plotting import plot_design_matrix

from value_capture.utils.data import Subject, BIDS_FOLDER

warnings.filterwarnings('ignore')

TR = 1.6

CONTRASTS = {
    'value_linear': 'dist_rank2 - dist_rank0',
    'value_any': '(dist_rank0 + dist_rank1 + dist_rank2) / 3 - dist_absent',
    'feedback': 'feedback_shown - feedback_omitted',
}


def build_events(onsets_df, n_removed):
    """Return nilearn events DataFrame for one run.

    Parameters
    ----------
    onsets_df : DataFrame from Subject.get_onsets() — times relative to first pulse
    n_removed : int — stabilisation volumes removed by fmriprep

    Returns
    -------
    DataFrame with columns onset, duration, trial_type
    """
    dummy_offset = n_removed * TR
    rows = []

    # Target phase: split by distractor value condition
    targets = onsets_df[onsets_df['event_type'] == 'target']
    for _, row in targets.iterrows():
        if not row['distractor_present']:
            label = 'dist_absent'
        else:
            label = f'dist_rank{int(row["value_rank"])}'
        rows.append({
            'onset': row['onset'] - dummy_offset,
            'duration': 1.75,
            'trial_type': label,
        })

    # Feedback phase: split by whether feedback was shown
    feedbacks = onsets_df[onsets_df['event_type'] == 'feedback']
    for _, row in feedbacks.iterrows():
        label = 'feedback_shown' if row['show_feedback'] else 'feedback_omitted'
        rows.append({
            'onset': row['onset'] - dummy_offset,
            'duration': 1.0,
            'trial_type': label,
        })

    return pd.DataFrame(rows).sort_values('onset').reset_index(drop=True)


def main(subject, sessions=None, bids_folder=BIDS_FOLDER, fmriprep_deriv='fmriprep'):
    sub = Subject(subject, bids_folder=bids_folder)

    if sessions is None:
        sessions = sub.get_sessions()

    print(f'sub-{subject}  sessions={sessions}  [{fmriprep_deriv}]')

    imgs, events_list, confounds_list = [], [], []

    for session in sessions:
        runs = sub.get_runs(session)
        bold_paths = sub.get_preprocessed_bold(session, runs,
                                               fmriprep_deriv=fmriprep_deriv)

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

    mask = sub.get_brain_mask(sessions[0], fmriprep_deriv=fmriprep_deriv)

    model = FirstLevelModel(
        t_r=TR,
        hrf_model='spm',
        drift_model=None,   # fmriprep DCT cosines in confounds handle drift
        noise_model='ar1',
        standardize=False,
        mask_img=mask,
    )

    print('\nFitting GLM...')
    model.fit(imgs, events=events_list, confounds=confounds_list)

    out_dir = (Path(bids_folder) / 'derivatives' / 'nilearn_glm'
               / f'sub-{subject}' / 'func')
    fig_dir = out_dir.parent / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    fn_base = f'sub-{subject}_task-valuecapture_space-T1w'

    # Plot design matrix for every run (one subplot per run)
    n_runs = len(model.design_matrices_)
    fig, axes = plt.subplots(1, n_runs, figsize=(5 * n_runs, 8),
                             constrained_layout=True)
    if n_runs == 1:
        axes = [axes]
    run_labels = [
        f'ses-{ses} run-{run}'
        for ses in sessions
        for run in sub.get_runs(ses)
    ]
    for ax, dm, label in zip(axes, model.design_matrices_, run_labels):
        plot_design_matrix(dm, ax=ax)
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
    parser.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    parser.add_argument('--fmriprep-deriv', default='fmriprep',
                        help='Name of the fmriprep derivatives folder.')
    args = parser.parse_args()

    main(args.subject,
         sessions=args.sessions,
         bids_folder=args.bids_folder,
         fmriprep_deriv=args.fmriprep_deriv)
