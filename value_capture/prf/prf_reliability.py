#!/usr/bin/env python3
"""
Check cross-run reliability of PRF fits.

Uses the existing ses-all PRF parameters (GaussianPRF2D.predict) to make
forward-model predictions for every single-trial beta, then computes R²
separately per run with braincoder.utils.get_rsq.

Output (saved to <bids>/derivatives/<prf_deriv>/qc/):
    sub-XX_prf_reliability.png              — pairwise scatter of per-voxel R²
                                              between runs (ses-all R² > --r2-min)
    sub-XX_run-<label>_prfR2run_pe.nii.gz   — per-run R² map in T1w space
    sub-XX_prfR2run_mean_pe.nii.gz          — mean R² across runs

Usage:
    python prf_reliability.py 01
    python prf_reliability.py 01 02 03
    python prf_reliability.py 01 --r2-min 0.1 --prf-deriv prf_glmsingle_s6mm
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nilearn import masking

import os
os.environ.setdefault('KERAS_BACKEND', 'jax')
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')   # Metal doesn't support large tensors

sys.path.insert(0, str(Path(__file__).parents[2]))
from value_capture.utils.data import Subject, BIDS_FOLDER
from value_capture.prf.fit_prf import build_paradigm, make_grid_coordinates

from braincoder.models import GaussianPRF2D
from braincoder.utils.stats import get_rsq


def run_subject(subject, bids_folder, glmsingle_deriv, prf_deriv, r2_min, out_dir):
    sub = Subject(subject, bids_folder=bids_folder)
    sessions = sub.get_sessions()
    print(f'\nsub-{subject}  sessions={sessions}')

    # ── Load betas + trial metadata ────────────────────────────────────────
    betas_img = sub.get_single_trial_estimates(sessions, glmsingle_deriv=glmsingle_deriv)
    trials = sub.get_trial_metadata(sessions, glmsingle_deriv=glmsingle_deriv).reset_index(drop=True)

    mask_img  = sub.get_brain_mask(sessions[0])
    betas_2d  = masking.apply_mask(betas_img, mask_img).astype(np.float32)
    n_trials, n_voxels = betas_2d.shape
    print(f'  betas: {n_trials} trials × {n_voxels} voxels')

    data_df = pd.DataFrame(betas_2d)
    data_df.index.name = 'frame'

    # ── Load PRF parameters ────────────────────────────────────────────────
    prf_imgs = sub.get_prf_parameters_volume(prf_deriv=prf_deriv)
    pars = pd.DataFrame({
        param: masking.apply_mask(img, mask_img).squeeze()
        for param, img in prf_imgs.items()
    })
    # GaussianPRF2D expects exactly these columns
    pars_model = pars[['x', 'y', 'sd', 'baseline', 'amplitude']].astype(np.float32)

    # ── Build paradigm + model ─────────────────────────────────────────────
    grid_coords  = make_grid_coordinates()
    paradigm_arr = build_paradigm(trials).astype(np.float32)
    paradigm_df  = pd.DataFrame(paradigm_arr)
    paradigm_df.index.name = 'frame'

    model = GaussianPRF2D(
        data=data_df,
        paradigm=paradigm_df,
        grid_coordinates=grid_coords,
    )

    print('  Computing predictions...')
    predictions_df = model.predict(paradigm=paradigm_df, parameters=pars_model)

    # ── R² per run ─────────────────────────────────────────────────────────
    run_r2 = {}   # label → pd.Series of shape (n_voxels,)
    for (ses, run), idx in trials.groupby(['session', 'run']).groups.items():
        label    = f's{ses}r{run}'
        data_run = data_df.iloc[idx]
        pred_run = predictions_df.iloc[idx]
        r2       = get_rsq(data_run, pred_run)
        run_r2[label] = r2
        print(f'  {label}  median R²={r2.median():.3f}  '
              f'n(>0.1)={int((r2 > 0.1).sum())}')

    # ── Save per-run R² NIfTIs ─────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    fn_base = f'sub-{subject}_task-valuecapture_space-T1w'

    for label, r2 in run_r2.items():
        r2_arr = np.where(np.isfinite(r2.values), r2.values, 0.0).astype(np.float32)
        masking.unmask(r2_arr, mask_img).to_filename(
            str(out_dir / f'{fn_base}_run-{label}_desc-prfR2run_pe.nii.gz'))

    r2_stack = np.stack([r2.values for r2 in run_r2.values()], axis=0)
    r2_mean  = np.nanmean(r2_stack, axis=0).astype(np.float32)
    masking.unmask(r2_mean, mask_img).to_filename(
        str(out_dir / f'{fn_base}_desc-prfR2runmean_pe.nii.gz'))
    print(f'  NIfTIs saved to {out_dir}')

    # ── Pairwise scatter — only visually responsive voxels ─────────────────
    keep = pars['R2'].values > r2_min
    print(f'  scatter: {keep.sum():,} voxels with ses-all R²>{r2_min}')

    labels = list(run_r2.keys())
    n = len(labels)
    fig, axes = plt.subplots(n, n, figsize=(2.2 * n, 2.2 * n))
    axes = np.atleast_2d(axes)

    lim = (-0.5, 0.8)
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            ax = axes[i, j]
            ri = run_r2[li].values[keep]
            rj = run_r2[lj].values[keep]
            if i == j:
                ax.hist(ri[np.isfinite(ri)], bins=60,
                        range=lim, color='steelblue', edgecolor='none')
                ax.set_title(li, fontsize=8)
            else:
                valid = np.isfinite(ri) & np.isfinite(rj)
                ax.scatter(ri[valid], rj[valid], s=1, alpha=0.15,
                           c='steelblue', rasterized=True)
                r = np.corrcoef(ri[valid], rj[valid])[0, 1]
                ax.text(0.05, 0.93, f'r={r:.2f}', transform=ax.transAxes,
                        fontsize=7, va='top')
                ax.set_xlim(lim)
                ax.set_ylim(lim)
                ax.plot(lim, lim, 'k--', lw=0.5)
                ax.set_aspect('equal')
            ax.tick_params(labelsize=6)
            if i == n - 1:
                ax.set_xlabel(lj, fontsize=8)
            if j == 0:
                ax.set_ylabel(li, fontsize=8)

    fig.suptitle(
        f'sub-{subject}  PRF cross-run R² reliability\n'
        f'ses-all model · {keep.sum():,} voxels with ses-all R²>{r2_min}',
        y=1.01, fontsize=10,
    )
    plt.tight_layout()

    fig_path = out_dir / f'sub-{subject}_prf_reliability.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {fig_path}')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('subjects', nargs='+', help="Subject label(s) without 'sub-'")
    parser.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    parser.add_argument('--glmsingle-deriv', default='glmsingle')
    parser.add_argument('--prf-deriv', default='prf_glmsingle')
    parser.add_argument('--r2-min', type=float, default=0.1,
                        help='Min ses-all R² to include in scatter plot (default: 0.1)')
    parser.add_argument('--out-dir', default=None,
                        help='Output directory (default: <bids>/derivatives/<prf-deriv>/qc/)')
    args = parser.parse_args()

    out_dir = (
        Path(args.out_dir) if args.out_dir
        else Path(args.bids_folder) / 'derivatives' / args.prf_deriv / 'qc'
    )

    for subject in args.subjects:
        run_subject(subject, args.bids_folder, args.glmsingle_deriv,
                    args.prf_deriv, args.r2_min, out_dir)


if __name__ == '__main__':
    main()
