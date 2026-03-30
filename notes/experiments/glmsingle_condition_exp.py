#!/usr/bin/env python3
"""
Experiment: does Tomas's suggestion of encoding bar × task conditions (rather
than bar-only) change the GLMSingle single-trial betas?

Background
----------
GLMSingle uses design-matrix column indices (condition labels) *only* for
cross-validation to select:
  1. The number of noise PCs to regress out (GLMdenoise step)
  2. The ridge regularisation parameter λ (fracridge step)

The per-trial betas are then estimated with the chosen hyperparameters applied
uniformly to all trials.  So the condition labels do *not* directly encode
which trials are averaged together — they only influence how the
regularisation hyperparameters are tuned.

Conditions tested
-----------------
A  bar only      : 10 positions × 2 orientations = 20 conditions
                   ~2 reps / condition / run  →  well-suited for CV
B  bar × task    : bar_pos × bar_ori × distractor_present × value_rank
                   ≈80 conditions, ~1-2 reps / condition / run
                   (Tomas's suggestion — "a couple hundred different trial types")

Dataset: sub-XX, ses-1, all runs. Subject from SLURM_ARRAY_TASK_ID or argv[1].

Outputs (in <bids>/derivatives/glmsingle_condition_exp/sub-XX/ses-1/func/)
---------------------------------------------------------------------------
  condA_betas.nii.gz        single-trial betas, bar-only conditions
  condB_betas.nii.gz        single-trial betas, bar × task conditions
  condA_R2.nii.gz           cross-validated R², condition A
  condB_R2.nii.gz           cross-validated R², condition B
  hyperparams.json          GLMdenoise n_PCs and fracridge λ for both conditions
  comparison_stats.json     voxel/trial correlation and R² stats
  comparison_figure.pdf     scatter + histogram plots
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

BIDS = Path('/shares/zne.uzh/gdehol/ds-valuecapture')
_task_id = os.environ.get('SLURM_ARRAY_TASK_ID') or (sys.argv[1] if len(sys.argv) > 1 else '1')
SUBJECT = f'{int(_task_id):02d}'
SESSION = 1
TR = 1.6
STIM_DUR = 1.75
RUNS = list(range(1, 9))


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


def events_path(run):
    return (BIDS / f'sub-{SUBJECT}' / f'ses-{SESSION}' / 'func'
            / f'sub-{SUBJECT}_ses-{SESSION}_task-valuecapture'
              f'_run-{run}_events.tsv')


# ---------------------------------------------------------------------------
# Condition label functions
# ---------------------------------------------------------------------------

def bar_condition_label(row):
    """Condition A: bar position × orientation only (current scheme, 20 conds)."""
    pos = round(float(row['bar_position']), 2)
    return f'bar_{pos:.2f}_{row["bar_orientation"]}'


def combined_condition_label(row):
    """Condition B: bar × distractor_present × value_rank (Tomas's suggestion).

    value_rank is NaN for absent trials → encoded as 'absent'.
    """
    pos = round(float(row['bar_position']), 2)
    ori = row['bar_orientation']
    dp = int(bool(row['distractor_present']))
    vr_raw = row['value_rank']
    vr = 'absent' if (not dp or pd.isna(vr_raw)) else str(int(vr_raw))
    return f'bar_{pos:.2f}_{ori}_dp{dp}_vr{vr}'


# ---------------------------------------------------------------------------
# Condition index builder
# ---------------------------------------------------------------------------

def _bar_sort_key(label):
    """Sort bar_<pos>_<ori>[_...] by orientation then position."""
    parts = label.split('_', 1)          # ['bar', '-2.85_vertical_...']
    rest = parts[1].split('_')           # ['-2.85', 'vertical', ...]
    return (rest[1], float(rest[0]))


def build_condition_index(all_events, label_fn):
    conditions = set()
    for ev in all_events:
        for _, row in ev[ev['event_type'] == 'target'].iterrows():
            conditions.add(label_fn(row))
    return {c: i for i, c in enumerate(sorted(conditions, key=_bar_sort_key))}


# ---------------------------------------------------------------------------
# Design matrix builder
# ---------------------------------------------------------------------------

def build_design(events, n_vols, condition_to_idx, label_fn):
    """Return (n_vols × n_conds) design matrix for one run."""
    total_pulses = len(events[events['event_type'] == 'pulse'])
    n_removed = total_pulses - n_vols

    n_conds = len(condition_to_idx)
    dm = np.zeros((n_vols, n_conds), dtype=np.float32)

    for _, row in events[events['event_type'] == 'target'].sort_values('onset').iterrows():
        vol = int(np.round(row['onset'] / TR)) - n_removed
        vol = np.clip(vol, 0, n_vols - 1)
        cond = label_fn(row)
        dm[vol, condition_to_idx[cond]] = 1.0

    return dm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from glmsingle.glmsingle import GLM_single

    out_dir = (BIDS / 'derivatives' / 'glmsingle_condition_exp'
               / f'sub-{SUBJECT}' / f'ses-{SESSION}' / 'func')
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Load events ---------------------------------------------------------
    print('Loading events...')
    all_events = [pd.read_csv(events_path(r), sep='\t') for r in RUNS]

    cond_A = build_condition_index(all_events, bar_condition_label)
    cond_B = build_condition_index(all_events, combined_condition_label)
    print(f'  Condition A (bar only):     {len(cond_A)} conditions')
    print(f'  Condition B (bar × task):   {len(cond_B)} conditions')

    # Compute and report average reps/condition per run
    n_trials_per_run = []
    for ev in all_events:
        n_trials_per_run.append(len(ev[ev['event_type'] == 'target']))
    n_trials_per_run = np.array(n_trials_per_run)
    print(f'  Trials per run: {n_trials_per_run}  (mean {n_trials_per_run.mean():.1f})')
    print(f'  A reps/cond/run: {n_trials_per_run.mean() / len(cond_A):.2f}')
    print(f'  B reps/cond/run: {n_trials_per_run.mean() / len(cond_B):.2f}')

    # -- Load BOLD, build design matrices ------------------------------------
    print('\nLoading BOLD data...')
    mask_img = nib.load(mask_path())
    mask = mask_img.get_fdata().astype(bool)
    ref_img = None

    data, X_A, X_B = [], [], []

    for run, events in zip(RUNS, all_events):
        print(f'  run-{run}', flush=True)
        img = nib.load(bold_path(run))
        if ref_img is None:
            ref_img = img
        bold = img.get_fdata(dtype=np.float32)
        n_vols = bold.shape[3]

        dm_A = build_design(events, n_vols, cond_A, bar_condition_label)
        dm_B = build_design(events, n_vols, cond_B, combined_condition_label)

        data.append(bold)
        X_A.append(dm_A)
        X_B.append(dm_B)
        print(f'    {n_vols} vols  |  A: {dm_A.shape[1]} conds  B: {dm_B.shape[1]} conds',
              flush=True)

    # -- GLMSingle runs -------------------------------------------------------
    opt_base = dict(
        wantlibrary=1,
        wantglmdenoise=1,
        wantfracridge=1,
        wantfileoutputs=[0, 0, 0, 0],
        sessionindicator=np.ones((1, len(RUNS)), dtype=int),
        numpcstotry=20,
    )

    print('\n=== Condition A (bar only, 20 conds) ===', flush=True)
    res_A = GLM_single(dict(opt_base)).fit(X_A, data, STIM_DUR, TR)

    print('\n=== Condition B (bar × task, ~80 conds) ===', flush=True)
    res_B = GLM_single(dict(opt_base)).fit(X_B, data, STIM_DUR, TR)

    betas_A = res_A['typed']['betasmd']   # (x, y, z, n_trials)
    betas_B = res_B['typed']['betasmd']
    r2_A    = res_A['typed']['R2']        # (x, y, z)
    r2_B    = res_B['typed']['R2']

    # -- Save NIfTIs ----------------------------------------------------------
    nib.Nifti1Image(betas_A, ref_img.affine).to_filename(out_dir / 'condA_betas.nii.gz')
    nib.Nifti1Image(betas_B, ref_img.affine).to_filename(out_dir / 'condB_betas.nii.gz')
    nib.Nifti1Image(r2_A,    ref_img.affine).to_filename(out_dir / 'condA_R2.nii.gz')
    nib.Nifti1Image(r2_B,    ref_img.affine).to_filename(out_dir / 'condB_R2.nii.gz')

    # FRACvalue is per-voxel (x,y,z) — save as NIfTI, summarise in JSON
    for label, res in [('A', res_A), ('B', res_B)]:
        frac = res['typed'].get('FRACvalue')
        if frac is not None:
            nib.Nifti1Image(frac.astype(np.float32), ref_img.affine).to_filename(
                out_dir / f'cond{label}_FRACvalue.nii.gz')

    # -- Extract scalar hyperparameters for JSON ------------------------------
    hyperparams = {}
    for label, res in [('A', res_A), ('B', res_B)]:
        typed = res['typed']
        hp = {}
        # pcnum: scalar — number of noise PCs removed
        if 'pcnum' in typed:
            hp['pcnum'] = int(typed['pcnum'])
        # FRACvalue: per-voxel volume — summarise over brain mask
        if 'FRACvalue' in typed:
            frac_masked = typed['FRACvalue'][mask]
            hp['FRACvalue_mean']   = float(frac_masked.mean())
            hp['FRACvalue_median'] = float(np.median(frac_masked))
            hp['FRACvalue_std']    = float(frac_masked.std())
            hp['FRACvalue_min']    = float(frac_masked.min())
            hp['FRACvalue_max']    = float(frac_masked.max())
        hyperparams[f'cond_{label}'] = hp

    (out_dir / 'hyperparams.json').write_text(json.dumps(hyperparams, indent=2))
    print('\n--- Hyperparameters ---')
    print(json.dumps(hyperparams, indent=2))

    # -- Comparison stats -----------------------------------------------------
    r2_A_masked = r2_A[mask]
    r2_B_masked = r2_B[mask]
    A_flat = betas_A[mask]   # (n_voxels, n_trials)
    B_flat = betas_B[mask]

    def voxel_corr(a, b):
        az = a - a.mean(axis=1, keepdims=True)
        bz = b - b.mean(axis=1, keepdims=True)
        num = (az * bz).sum(axis=1)
        denom = np.sqrt((az**2).sum(axis=1) * (bz**2).sum(axis=1)) + 1e-10
        return num / denom

    def trial_corr(a, b):
        az = a - a.mean(axis=0, keepdims=True)
        bz = b - b.mean(axis=0, keepdims=True)
        num = (az * bz).sum(axis=0)
        denom = np.sqrt((az**2).sum(axis=0) * (bz**2).sum(axis=0)) + 1e-10
        return num / denom

    r2_thresholds = {'all': 0.0, 'top20': 80.0, 'top10': 90.0, 'top5': 95.0}
    stats = {
        'n_trials':   int(betas_A.shape[3]),
        'n_conds_A':  len(cond_A),
        'n_conds_B':  len(cond_B),
        'r2_A_mean':  float(r2_A_masked.mean()),
        'r2_B_mean':  float(r2_B_masked.mean()),
    }
    corr_by_threshold = {}

    for label, pct in r2_thresholds.items():
        if pct == 0.0:
            sel = np.ones(mask.sum(), dtype=bool)
        else:
            thresh = np.percentile(r2_A_masked, pct)
            sel = r2_A_masked >= thresh

        vc = voxel_corr(A_flat[sel], B_flat[sel])
        tc = trial_corr(A_flat[sel], B_flat[sel])
        corr_by_threshold[label] = vc

        # ΔBeta = B − A, normalised by the SD of A across trials (per voxel)
        delta = B_flat[sel] - A_flat[sel]                      # (n_sel, n_trials)
        a_sd = A_flat[sel].std(axis=1, keepdims=True) + 1e-10
        rel_delta = (np.abs(delta) / a_sd).mean()              # mean |ΔBeta| / SD_A

        stats[f'{label}_n_voxels']        = int(sel.sum())
        stats[f'{label}_voxel_corr_mean'] = float(np.nanmean(vc))
        stats[f'{label}_voxel_corr_med']  = float(np.nanmedian(vc))
        stats[f'{label}_voxel_corr_p5']   = float(np.nanpercentile(vc, 5))
        stats[f'{label}_trial_corr_mean'] = float(np.nanmean(tc))
        stats[f'{label}_trial_corr_med']  = float(np.nanmedian(tc))
        stats[f'{label}_delta_beta_rel']  = float(rel_delta)   # ~0 → betas unchanged

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

    # Row 2 left: ΔBeta histogram — directly shows betas are unchanged (all voxels)
    delta_all = B_flat - A_flat                          # (n_voxels, n_trials)
    a_sd_all = A_flat.std(axis=1, keepdims=True) + 1e-10
    rel_delta_all = (delta_all / a_sd_all).ravel()       # normalised ΔBeta, flat
    p1, p99 = np.percentile(rel_delta_all, [1, 99])
    axes[1, 0].hist(rel_delta_all, bins=200, color='steelblue', edgecolor='none',
                    alpha=0.85, range=(p1, p99))
    axes[1, 0].axvline(0, color='r', lw=1)
    axes[1, 0].axvline(np.nanmean(rel_delta_all), color='k', ls='--', lw=1.5,
                       label=f'mean={np.nanmean(rel_delta_all):.4f}')
    axes[1, 0].set(xlabel='(Beta_B − Beta_A) / SD(Beta_A)',
                   ylabel='Count',
                   title='ΔBeta (B − A), normalised\n(all brain-masked voxels × trials)')
    axes[1, 0].legend(fontsize=8)

    # Row 2 middle: scatter mean beta A vs B (top-10% R² voxels)
    thresh10 = np.percentile(r2_A_masked, 90)
    sel10 = r2_A_masked >= thresh10
    mA = A_flat[sel10].mean(axis=1)
    mB = B_flat[sel10].mean(axis=1)
    lim = np.percentile(np.abs(np.concatenate([mA, mB])), 99)
    axes[1, 1].scatter(mA, mB, s=0.8, alpha=0.3, rasterized=True, color='grey')
    axes[1, 1].plot([-lim, lim], [-lim, lim], 'r--', lw=1)
    axes[1, 1].set(xlim=(-lim, lim), ylim=(-lim, lim),
                   xlabel='Mean beta — A (bar only, 20 conds)',
                   ylabel='Mean beta — B (bar × task, ~80 conds)',
                   title='Mean beta scatter (top-10% R² voxels)')

    # Row 2 right: ΔR² distribution (B − A) for top-10% R² voxels
    dr2 = r2_B_masked[sel10] - r2_A_masked[sel10]
    axes[1, 2].hist(dr2, bins=80, color='mediumpurple', edgecolor='none')
    axes[1, 2].axvline(dr2.mean(), color='k', ls='--', lw=1.5,
                       label=f'mean={dr2.mean():.3f}')
    axes[1, 2].axvline(0, color='r', ls='-', lw=0.8)
    axes[1, 2].set(xlabel='ΔR² (B − A)',
                   ylabel='Voxel count',
                   title='R² change: bar×task vs bar-only\n(top-10% R² voxels)')
    axes[1, 2].legend(fontsize=8)

    fig.suptitle(
        f'sub-{SUBJECT} ses-{SESSION}  |  '
        f'GLMSingle condition encoding: bar-only (A, {len(cond_A)} conds) '
        f'vs bar×task (B, {len(cond_B)} conds)',
        fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / 'comparison_figure.pdf', dpi=150)
    print(f'\nSaved to {out_dir}')


if __name__ == '__main__':
    main()
