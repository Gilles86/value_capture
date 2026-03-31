#!/usr/bin/env python3
"""
Experiment: does 2.5x temporal upsampling before GLMSingle change the betas?

GLMSingle requires a TR-resolution design matrix, so sub-TR event onsets must be
rounded to the nearest volume. This experiment tests whether the rounding error
(up to ~0.8 s with TR=1.6 s) meaningfully affects single-trial betas compared
to running GLMSingle on 2.5x-upsampled data (new TR=0.64 s, max rounding error
~0.32 s).

Conditions
----------
A  TR-resolution (1.6 s)   events rounded to nearest 1.6 s volume   [baseline]
B  2.5x upsampled (0.64 s) events rounded to nearest 0.64 s volume  [test]

Dataset: sub-01, ses-1, all 8 valuecapture runs.

Outputs (in <bids>/derivatives/glmsingle_upsample_exp/sub-01/ses-1/func/)
--------------------------------------------------------------------------
  condA_betas.nii.gz       single-trial betas, TR=1.6 s
  condB_betas.nii.gz       single-trial betas, TR=0.64 s (upsampled)
  condA_R2.nii.gz          cross-validated R², condition A
  condB_R2.nii.gz          cross-validated R², condition B
  comparison_stats.json    summary: mean/median voxel correlation, R² stats
  comparison_figure.pdf    scatter + histogram plots
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from value_capture.utils.data import Subject

BIDS = Path('/shares/zne.uzh/gdehol/ds-valuecapture')
# Subject from SLURM array task ID (1→'01', 2→'02', ...) or first CLI arg
_task_id = os.environ.get('SLURM_ARRAY_TASK_ID') or (sys.argv[1] if len(sys.argv) > 1 else '1')
SUBJECT = f'{int(_task_id):02d}'
SESSION = 1
TR = 1.6
UPSAMPLE_FACTOR = 2.5
TR_UP = TR / UPSAMPLE_FACTOR        # 0.64 s
STIM_DUR = 1.75                      # target phase duration
RUNS = list(range(1, 9))             # runs 1–8


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def bold_path(run):
    return (BIDS / 'derivatives' / 'fmriprep'
            / f'sub-{SUBJECT}' / f'ses-{SESSION}' / 'func'
            / f'sub-{SUBJECT}_ses-{SESSION}_task-valuecapture'
              f'_rec-NORDIC_run-{run}_space-T1w_desc-preproc_bold.nii.gz')


def mask_path(run=1):
    return (BIDS / 'derivatives' / 'fmriprep'
            / f'sub-{SUBJECT}' / f'ses-{SESSION}' / 'func'
            / f'sub-{SUBJECT}_ses-{SESSION}_task-valuecapture'
              f'_rec-NORDIC_run-{run}_space-T1w_desc-brain_mask.nii.gz')


# ---------------------------------------------------------------------------
# Condition label (replicated from fit_glmsingle.py without importing it)
# ---------------------------------------------------------------------------

def bar_condition_label(row):
    pos = round(float(row['bar_position']), 2)
    return f'bar_{pos:.2f}_{row["bar_orientation"]}'


def build_condition_index(all_events):
    conditions = set()
    for ev in all_events:
        for _, row in ev[ev['event_type'] == 'target'].iterrows():
            conditions.add(bar_condition_label(row))

    def _sort_key(label):
        rest = label.split('_', 1)[1].rsplit('_', 1)
        return (rest[1], float(rest[0]))

    return {c: i for i, c in enumerate(sorted(conditions, key=_sort_key))}


# ---------------------------------------------------------------------------
# Design matrix builder — parameterised by TR
# ---------------------------------------------------------------------------

def build_design(events, n_vols, tr, condition_to_idx):
    """Return (n_vols × n_conds) design matrix, rounded to given TR."""
    total_pulses = len(events[events['event_type'] == 'pulse'])
    # n_removed must be in units of tr. total_pulses is in native TR units,
    # so convert via time: dummy_secs = (total_pulses - n_native_vols) * TR,
    # then n_removed = dummy_secs / tr.
    n_vols_native = int(round(n_vols * tr / TR))
    n_removed = int(round((total_pulses - n_vols_native) * TR / tr))

    n_conds = len(condition_to_idx)
    dm = np.zeros((n_vols, n_conds), dtype=np.float32)
    trial_meta = []

    for _, row in events[events['event_type'] == 'target'].sort_values('onset').iterrows():
        vol = int(np.round(row['onset'] / tr)) - n_removed
        vol = np.clip(vol, 0, n_vols - 1)
        cond = bar_condition_label(row)
        dm[vol, condition_to_idx[cond]] = 1.0
        trial_meta.append({
            'condition': cond,
            'value_rank': row['value_rank'],
            'distractor_present': row['distractor_present'],
            'trial_nr': row['trial_nr'],
            'onset': row['onset'],
            'vol_A': int(np.round(row['onset'] / TR)),
            'vol_B': int(np.round(row['onset'] / TR_UP)),
        })

    return dm, trial_meta


# ---------------------------------------------------------------------------
# Upsampling
# ---------------------------------------------------------------------------

def upsample_bold(bold_4d, factor):
    """Linearly upsample a (x,y,z,t) array along the time axis by `factor`."""
    x, y, z, t = bold_4d.shape
    t_orig = np.arange(t, dtype=np.float64)
    n_new = int(round(t * factor))
    t_new = np.linspace(0, t - 1, n_new)

    flat = bold_4d.reshape(-1, t).astype(np.float32)
    interp = interp1d(t_orig, flat, axis=1, kind='linear', fill_value='extrapolate',
                      assume_sorted=True)
    return interp(t_new).reshape(x, y, z, n_new)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from glmsingle.glmsingle import GLM_single

    out_dir = (BIDS / 'derivatives' / 'glmsingle_upsample_exp'
               / f'sub-{SUBJECT}' / f'ses-{SESSION}' / 'func')
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Load events and build global condition map --------------------------
    # Use Subject.get_onsets() — reads sourcedata behavioral TSVs (which include
    # pulse events) and normalises t=0 to the first scanner pulse.  The BIDS func/
    # events TSVs have no pulse rows, which breaks the n_removed calculation.
    print('Loading events...')
    sub = Subject(SUBJECT, bids_folder=str(BIDS))
    all_events = [sub.get_onsets(SESSION, r) for r in RUNS]
    condition_to_idx = build_condition_index(all_events)
    print(f'  {len(condition_to_idx)} bar conditions')

    # -- Load BOLD, build design matrices ------------------------------------
    print('Loading BOLD data...')
    mask_img = nib.load(mask_path())
    mask = mask_img.get_fdata().astype(bool)
    ref_img = None

    data_A, data_B, X_A, X_B = [], [], [], []
    all_meta = []

    for run, events in zip(RUNS, all_events):
        print(f'  run-{run}', flush=True)
        img = nib.load(bold_path(run))
        if ref_img is None:
            ref_img = img
        bold = img.get_fdata(dtype=np.float32)
        n_vols = bold.shape[3]

        # Condition A: native TR
        dm_A, meta = build_design(events, n_vols, TR, condition_to_idx)
        data_A.append(bold)
        X_A.append(dm_A)
        for m in meta:
            m['run'] = run
        all_meta.extend(meta)

        # Condition B: upsampled
        bold_up = upsample_bold(bold, UPSAMPLE_FACTOR)
        n_vols_up = bold_up.shape[3]
        dm_B, _ = build_design(events, n_vols_up, TR_UP, condition_to_idx)
        data_B.append(bold_up)
        X_B.append(dm_B)

        print(f'    A: {n_vols} vols  B: {n_vols_up} vols', flush=True)

    pd.DataFrame(all_meta).to_csv(out_dir / 'trials.tsv', sep='\t', index=False)

    # -- GLMSingle runs -------------------------------------------------------
    opt_base = dict(
        wantlibrary=1,
        wantglmdenoise=1,
        wantfracridge=1,
        wantfileoutputs=[0, 0, 0, 0],   # we save manually
        sessionindicator=np.ones((1, len(RUNS)), dtype=int),
    )

    print('\n=== Condition A (TR=1.6 s) ===', flush=True)
    res_A = GLM_single(dict(opt_base)).fit(X_A, data_A, STIM_DUR, TR)

    print('\n=== Condition B (TR=0.64 s, 2.5x upsampled) ===', flush=True)
    res_B = GLM_single(dict(opt_base)).fit(X_B, data_B, STIM_DUR, TR_UP)

    betas_A = res_A['typed']['betasmd']   # (x, y, z, n_trials)
    betas_B = res_B['typed']['betasmd']
    r2_A    = res_A['typed']['R2']        # (x, y, z)
    r2_B    = res_B['typed']['R2']

    # -- Save NIfTIs ----------------------------------------------------------
    nib.Nifti1Image(betas_A, ref_img.affine).to_filename(out_dir / 'condA_betas.nii.gz')
    nib.Nifti1Image(betas_B, ref_img.affine).to_filename(out_dir / 'condB_betas.nii.gz')
    nib.Nifti1Image(r2_A,    ref_img.affine).to_filename(out_dir / 'condA_R2.nii.gz')
    nib.Nifti1Image(r2_B,    ref_img.affine).to_filename(out_dir / 'condB_R2.nii.gz')

    # -- Comparison stats -----------------------------------------------------
    r2_A_masked = r2_A[mask]
    r2_B_masked = r2_B[mask]
    A_flat = betas_A[mask]   # (n_voxels, n_trials)
    B_flat = betas_B[mask]

    def voxel_corr(a, b):
        """Pearson r per voxel across trials."""
        az = a - a.mean(axis=1, keepdims=True)
        bz = b - b.mean(axis=1, keepdims=True)
        num = (az * bz).sum(axis=1)
        denom = np.sqrt((az**2).sum(axis=1) * (bz**2).sum(axis=1)) + 1e-10
        return num / denom

    def trial_corr(a, b):
        """Pearson r per trial across voxels (spatial pattern)."""
        az = a - a.mean(axis=0, keepdims=True)
        bz = b - b.mean(axis=0, keepdims=True)
        num = (az * bz).sum(axis=0)
        denom = np.sqrt((az**2).sum(axis=0) * (bz**2).sum(axis=0)) + 1e-10
        return num / denom

    # R²-stratified analysis: use condition A R² as the reference ranking
    r2_thresholds = {'all': 0.0, 'top20': 80.0, 'top10': 90.0, 'top5': 95.0}
    stats = {
        'n_trials':  int(betas_A.shape[3]),
        'r2_A_mean': float(r2_A_masked.mean()),
        'r2_B_mean': float(r2_B_masked.mean()),
    }
    corr_by_threshold = {}   # saved for figure

    for label, pct in r2_thresholds.items():
        if pct == 0.0:
            sel = np.ones(mask.sum(), dtype=bool)
        else:
            thresh = np.percentile(r2_A_masked, pct)
            sel = r2_A_masked >= thresh

        vc = voxel_corr(A_flat[sel], B_flat[sel])
        tc = trial_corr(A_flat[sel], B_flat[sel])
        corr_by_threshold[label] = vc

        stats[f'{label}_n_voxels']        = int(sel.sum())
        stats[f'{label}_voxel_corr_mean'] = float(np.nanmean(vc))
        stats[f'{label}_voxel_corr_med']  = float(np.nanmedian(vc))
        stats[f'{label}_voxel_corr_p5']   = float(np.nanpercentile(vc, 5))
        stats[f'{label}_trial_corr_mean'] = float(np.nanmean(tc))
        stats[f'{label}_trial_corr_med']  = float(np.nanmedian(tc))

    (out_dir / 'comparison_stats.json').write_text(json.dumps(stats, indent=2))
    print('\n--- Results ---')
    for k, v in stats.items():
        print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')

    # -- Figure ---------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Row 1: voxel-wise correlation histograms for all / top-10% / top-5%
    for ax, label, color in zip(axes[0],
                                 ['all', 'top10', 'top5'],
                                 ['steelblue', 'darkorange', 'firebrick']):
        vc = corr_by_threshold[label]
        med = float(np.nanmedian(vc))
        ax.hist(vc, bins=100, color=color, edgecolor='none', alpha=0.85)
        ax.axvline(med, color='k', ls='--', lw=1.5, label=f'median={med:.3f}')
        ax.set(xlabel='Pearson r (A vs B, per voxel)',
               ylabel='Voxel count',
               title=f'Voxel correlation — {label}\n(n={stats[f"{label}_n_voxels"]:,})')
        ax.legend(fontsize=8)

    # Row 2 left: median voxel correlation vs R² threshold
    labels_ord = ['all', 'top20', 'top10', 'top5']
    medians = [stats[f'{l}_voxel_corr_med'] for l in labels_ord]
    axes[1, 0].plot(range(len(labels_ord)), medians, 'o-', color='steelblue')
    axes[1, 0].set(xticks=range(len(labels_ord)), xticklabels=labels_ord,
                   ylabel='Median voxel correlation (A vs B)',
                   title='Correlation vs R² threshold', ylim=(0, 1))
    axes[1, 0].grid(axis='y', alpha=0.3)

    # Row 2 middle: scatter mean beta A vs B (top-10% R² voxels)
    thresh10 = np.percentile(r2_A_masked, 90)
    sel10 = r2_A_masked >= thresh10
    mA = A_flat[sel10].mean(axis=1)
    mB = B_flat[sel10].mean(axis=1)
    lim = np.percentile(np.abs(np.concatenate([mA, mB])), 99)
    axes[1, 1].scatter(mA, mB, s=0.8, alpha=0.3, rasterized=True, color='grey')
    axes[1, 1].plot([-lim, lim], [-lim, lim], 'r--', lw=1)
    axes[1, 1].set(xlim=(-lim, lim), ylim=(-lim, lim),
                   xlabel='Mean beta — A (1.6 s TR)',
                   ylabel='Mean beta — B (0.64 s TR)',
                   title='Mean beta scatter (top-10% R² voxels)')

    # Row 2 right: ΔR² distribution (B − A) for top-10% R² voxels
    dr2 = r2_B_masked[sel10] - r2_A_masked[sel10]
    axes[1, 2].hist(dr2, bins=80, color='mediumpurple', edgecolor='none')
    axes[1, 2].axvline(dr2.mean(), color='k', ls='--', lw=1.5,
                       label=f'mean={dr2.mean():.3f}')
    axes[1, 2].axvline(0, color='r', ls='-', lw=0.8)
    axes[1, 2].set(xlabel='ΔR² (B − A)',
                   ylabel='Voxel count',
                   title='R² improvement from upsampling\n(top-10% R² voxels)')
    axes[1, 2].legend(fontsize=8)

    fig.suptitle(f'sub-{SUBJECT} ses-{SESSION}  |  '
                 f'GLMSingle: TR=1.6 s (A) vs 0.64 s upsampled (B)',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / 'comparison_figure.pdf', dpi=150)
    print(f'\nSaved to {out_dir}')


if __name__ == '__main__':
    main()
