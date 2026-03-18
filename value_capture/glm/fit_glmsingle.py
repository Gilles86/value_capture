#!/usr/bin/env python3
"""
Fit GLMsingle single-trial betas for the value capture fMRI task.

What is modelled
----------------
One event per trial: the ``target`` phase onset — the moment the visual search
array appears on screen (phase 2, ~1.75 s, response collected here).  Each trial
contributes exactly one row to the design matrix at the TR nearest to its onset.

Condition labels (used for GLMsingle fracridge cross-validation)
----------------------------------------------------------------
GLMsingle uses design-matrix column indices to group same-condition trials
together when selecting the ridge regularisation parameter (lambda) via
leave-one-run-out cross-validation.  Trials are grouped into 4 conditions:

  absent       — distractor-absent trial (value rank N/A)
  low_value    — distractor present, value rank 0
  medium_value — distractor present, value rank 1
  high_value   — distractor present, value rank 2

The same 4-column design is used consistently across all runs and sessions so
column indices remain stable across the fit.  GLMsingle then expands this
condition-level design internally to yield one beta per trial per voxel.

Timing reference
----------------
t = 0 is the FIRST PULSE event in the events file — the very first TR sent by
the scanner (including the stabilisation TRs at the start of the run).  The
fmriprep output discards a fixed number of leading TRs (``--dummy-scans``), so
fmriprep volume 0 corresponds to acquisition TR ``n_removed``.  ``n_removed`` is
computed at run time as ``total_pulses − n_fmriprep_vols``, making the code
robust to any ``--dummy-scans`` setting.  The stabilisation TRs that remain in
the fmriprep output simply have all-zero rows in the design matrix (baseline).

Output
------
Two 4-D NIfTI images (x × y × z × n_total_trials):

  desc-target_pe.nii.gz  — single-trial betas for the target phase
  desc-R2_pe.nii.gz      — cross-validated R² of the final GLMsingle model

Plus a trial-metadata TSV (one row per trial, same order as the beta volumes):

  desc-target_trials.tsv — condition, value_rank, distractor_present,
                           session, run, trial_nr, onset_trigger

Output path:
  <bids_folder>/derivatives/glmsingle/sub-<sub>/<ses_label>/func/

Sessions
--------
By default all available sessions are fitted jointly (recommended: GLMsingle
uses the session indicator to handle session-to-session scaling differences).
Pass --sessions to restrict to specific session numbers.

Usage:
    python fit_glmsingle.py 01
    python fit_glmsingle.py 01 --sessions 2
    python fit_glmsingle.py 01 --sessions 1 2
    python fit_glmsingle.py 01 --bids-folder /shares/zne.uzh/gdehol/ds-valuecapture
    python fit_glmsingle.py 01 --debug   # write all 4 GLMsingle model steps + figures
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import image

from value_capture.utils.data import Subject, BIDS_FOLDER

warnings.filterwarnings('ignore')

TR = 1.6

CONDITIONS = ['absent', 'low_value', 'medium_value', 'high_value']
CONDITION_TO_IDX = {c: i for i, c in enumerate(CONDITIONS)}


def make_condition_label(row):
    """Map a target-event row to its condition label."""
    if not row['distractor_present']:
        return 'absent'
    rank = int(row['value_rank'])
    return ['low_value', 'medium_value', 'high_value'][rank]


def build_design_matrix(events, n_vols):
    """Return (dm, trial_order, trial_metadata) for one run.

    Parameters
    ----------
    events : DataFrame from Subject.get_onsets() — onsets relative to the first pulse.
    n_vols : int — number of volumes in the fmriprep output for this run.

    Returns
    -------
    dm             : (n_vols × 4) binary float array; one 1 per TR per trial
    trial_order    : list[str] — condition label per trial in onset order
    trial_metadata : list[dict] — per-trial info dicts
    """
    total_pulses = len(events[events['event_type'] == 'pulse'])
    # n_removed: number of leading TRs fmriprep discarded (--dummy-scans N)
    # inferred from the mismatch between total acquired pulses and output vols
    n_removed = total_pulses - n_vols

    target_events = events[events['event_type'] == 'target'].sort_values('onset')

    dm = np.zeros((n_vols, len(CONDITIONS)))
    trial_order = []
    trial_metadata = []

    for _, row in target_events.iterrows():
        # onset is relative to first pulse (t=0).  Subtract the removed TRs to
        # get the index into the fmriprep output time-series.
        dm_row = int(np.round(row['onset'] / TR)) - n_removed
        dm_row = max(0, min(dm_row, n_vols - 1))

        cond = make_condition_label(row)
        dm[dm_row, CONDITION_TO_IDX[cond]] = 1.0
        trial_order.append(cond)
        trial_metadata.append({
            'condition': cond,
            'value_rank': row['value_rank'],
            'distractor_present': row['distractor_present'],
            'trial_nr': row['trial_nr'],
            'onset': row['onset'],
        })

    return dm, trial_order, trial_metadata


def main(subject, sessions=None, bids_folder=BIDS_FOLDER, fmriprep_deriv='fmriprep',
         debug=False):
    sub = Subject(subject, bids_folder=bids_folder)

    if sessions is None:
        sessions = sub.get_sessions()

    ses_label = f'ses-{sessions[0]}' if len(sessions) == 1 else 'ses-all'
    print(f'sub-{subject}  {ses_label}  [{fmriprep_deriv}]')

    data = []
    X = []
    all_trial_types = []
    all_trial_metadata = []
    session_indicators = []
    ref_bold = None

    for session in sessions:
        runs = sub.get_runs(session)
        bold_paths = sub.get_preprocessed_bold(session, runs, fmriprep_deriv=fmriprep_deriv)
        print(f'  ses-{session}: {len(runs)} runs: {runs}')

        for run, bold_path in zip(runs, bold_paths):
            if ref_bold is None:
                ref_bold = bold_path
            img = image.load_img(str(bold_path))
            bold_data = img.get_fdata()
            n_vols = bold_data.shape[3]

            events = sub.get_onsets(session, run)
            dm, trial_order, trial_meta = build_design_matrix(events, n_vols)

            total_pulses = len(events[events['event_type'] == 'pulse'])
            n_removed = total_pulses - n_vols
            print(f'    run-{run}: {n_vols} vols (removed {n_removed}), '
                  f'{len(trial_order)} trials')
            cond_counts = {c: int(dm[:, i].sum()) for c, i in CONDITION_TO_IDX.items()}
            print(f'      condition counts: {cond_counts}')

            data.append(bold_data)
            X.append(dm)
            all_trial_types.extend(trial_order)
            for m in trial_meta:
                m.update({'session': session, 'run': run})
            all_trial_metadata.extend(trial_meta)
            session_indicators.append(session)

    out_dir = (Path(bids_folder) / 'derivatives' / 'glmsingle'
               / f'sub-{subject}' / ses_label / 'func')
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir.parent / 'figures'

    opt = dict(
        wantlibrary=1,
        wantglmdenoise=1,
        wantfracridge=1,
        wantfileoutputs=[1, 1, 1, 1] if debug else [0, 0, 0, 1],
        sessionindicator=np.array(session_indicators)[np.newaxis, :],
    )

    from glmsingle.glmsingle import GLM_single
    results = GLM_single(opt).fit(X, data, TR, TR,
                                  outputdir=str(out_dir),
                                  figuredir=str(fig_dir))

    betas = results['typed']['betasmd']  # (x, y, z, n_total_trials)

    fn_base = (f'sub-{subject}_{ses_label}_task-valuecapture'
               f'_space-T1w_desc-{{desc}}')
    image.new_img_like(ref_bold, betas).to_filename(
        str(out_dir / (fn_base.format(desc='target') + '_pe.nii.gz')))
    image.new_img_like(ref_bold, results['typed']['R2']).to_filename(
        str(out_dir / (fn_base.format(desc='R2') + '_pe.nii.gz')))

    trials_df = pd.DataFrame(all_trial_metadata)
    trials_df.to_csv(
        str(out_dir / (fn_base.format(desc='target') + '_trials.tsv')),
        sep='\t', index=False)

    print(f'\nSaved to {out_dir}')
    print(f'  beta volume: {betas.shape}  ({betas.shape[3]} total trials)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('subject', help="Subject label without 'sub-', e.g. 01")
    parser.add_argument('--sessions', type=int, nargs='+', default=None,
                        help='Session number(s) to fit. Default: all available sessions.')
    parser.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    parser.add_argument('--fmriprep-deriv', default='fmriprep',
                        help='Name of the fmriprep derivatives folder (default: fmriprep).')
    parser.add_argument('--debug', action='store_true',
                        help='Write outputs and figures for all 4 GLMsingle model steps.')
    args = parser.parse_args()

    main(args.subject, sessions=args.sessions,
         bids_folder=args.bids_folder,
         fmriprep_deriv=args.fmriprep_deriv,
         debug=args.debug)
