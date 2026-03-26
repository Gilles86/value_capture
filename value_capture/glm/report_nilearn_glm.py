#!/usr/bin/env python3
"""
Per-subject PDF report of FDR-thresholded GLM contrast maps.

For each contrast: one page with a glass brain (top) and axial slices on the
subject's own T1w (bottom), thresholded at FDR q < alpha with a minimum
cluster size.

Output
------
  <bids>/derivatives/nilearn_glm/sub-<sub>/figures/
    sub-<sub>_task-valuecapture_space-<space>_fdr-report.pdf

Usage
-----
  python report_nilearn_glm.py 01
  python report_nilearn_glm.py 01 --fdr-alpha 0.05 --cluster-threshold 10
  python report_nilearn_glm.py 01 --bids-folder /shares/zne.uzh/gdehol/ds-valuecapture
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import numpy as np

from nilearn import image
from nilearn.image import threshold_stats_img
from nilearn.plotting import plot_stat_map, plot_glass_brain

from value_capture.utils.data import Subject, BIDS_FOLDER

warnings.filterwarnings('ignore')

# Ordered list: (contrast_name, formula_string, description)
CONTRAST_INFO = [
    ('value_linear',
     'dist_rank2 − dist_rank0',
     'Proximate reward — BOLD scales linearly with distractor value rank'),
    ('value_capture',
     'dist_rank2 − dist_absent',
     'Value-capture signature — high-value distractor vs no distractor'),
    ('value_nonlinear',
     'dist_rank1 − (dist_rank0 + dist_rank2) / 2',
     'Nonlinear value coding — mid-value response vs linear expectation\n'
     '(+) inverted-U: more to middle than expected    '
     '(−) U-shaped: suppressed mid-value response'),
    ('value_step_low',
     'dist_rank1 − dist_rank0',
     'Low → medium value step'),
    ('value_step_high',
     'dist_rank2 − dist_rank1',
     'Medium → high value step'),
    ('distractor_any',
     '(dist_rank0 + dist_rank1 + dist_rank2) / 3 − dist_absent',
     'Any distractor vs absent — attentional capture regardless of value'),
    ('feedback',
     'feedback_shown − feedback_omitted',
     'RPE proxy — any explicit feedback vs blank fixation'),
    ('feedback_value',
     'feedback_points  [parametric, log(1+pts), mean-centred per run]',
     'RPE magnitude — BOLD scales with log reward received at feedback'),
]


def _make_contrast_page(pdf, z_map, t1w, name, expr, desc,
                        fdr_thr, n_survived, fdr_alpha):
    """Render one PDF page for a single contrast."""
    survived_str = (f'FDR threshold: |z| ≥ {fdr_thr:.2f}  ({n_survived} voxels)'
                    if fdr_thr is not None and n_survived > 0
                    else 'No voxels survive FDR')
    display_thr = fdr_thr if (fdr_thr is not None and n_survived > 0) else 1e6

    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 1, figure=fig,
                  height_ratios=[0.13, 0.43, 0.43],
                  hspace=0.05, top=0.97, bottom=0.02)

    # --- Title block ---
    ax_title = fig.add_subplot(gs[0])
    ax_title.axis('off')
    ax_title.text(0.5, 0.85, name, transform=ax_title.transAxes,
                  ha='center', va='top', fontsize=14, fontweight='bold')
    ax_title.text(0.5, 0.52, f'Formula:  {expr}',
                  transform=ax_title.transAxes,
                  ha='center', va='top', fontsize=10, family='monospace')
    ax_title.text(0.5, 0.22, desc,
                  transform=ax_title.transAxes,
                  ha='center', va='top', fontsize=9)
    ax_title.text(0.5, 0.0, f'[{survived_str}  |  FDR q < {fdr_alpha}  |  min 10 voxels]',
                  transform=ax_title.transAxes,
                  ha='center', va='bottom', fontsize=8,
                  color='#555555')

    # --- Glass brain ---
    ax_glass = fig.add_subplot(gs[1])
    try:
        plot_glass_brain(
            z_map, threshold=display_thr, colorbar=True,
            plot_abs=False, display_mode='lyrz',
            axes=ax_glass, vmax=8, annotate=False,
            title='Glass brain (FDR-thresholded)')
    except Exception as e:
        ax_glass.text(0.5, 0.5, f'[glass brain unavailable: {e}]',
                      transform=ax_glass.transAxes, ha='center', va='center',
                      fontsize=9, color='red')

    # --- Axial slices on T1w ---
    ax_stat = fig.add_subplot(gs[2])
    try:
        plot_stat_map(
            z_map, bg_img=t1w, threshold=display_thr,
            display_mode='z', cut_coords=7,
            colorbar=True, axes=ax_stat, vmax=8,
            title='Axial slices on T1w (FDR-thresholded)')
    except Exception as e:
        ax_stat.text(0.5, 0.5, f'[stat map unavailable: {e}]',
                     transform=ax_stat.transAxes, ha='center', va='center',
                     fontsize=9, color='red')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def report_subject(subject, bids_folder=BIDS_FOLDER, space='T1w', res=None,
                   fdr_alpha=0.05, cluster_threshold=10):
    sub = Subject(subject, bids_folder=bids_folder)

    res_label = f'_res-{res}' if res else ''
    space_tag = f'space-{space}{res_label}'
    fn_base = f'sub-{subject}_task-valuecapture_{space_tag}'

    glm_dir = Path(bids_folder) / 'derivatives' / 'nilearn_glm' / f'sub-{subject}'
    func_dir = glm_dir / 'func'
    fig_dir = glm_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f'sub-{subject}: loading T1w...')
    t1w = sub.get_t1w()

    pdf_path = fig_dir / f'{fn_base}_fdr-report.pdf'
    print(f'sub-{subject}: writing {len(CONTRAST_INFO)}-contrast report → {pdf_path.name}')

    with PdfPages(str(pdf_path)) as pdf:
        # Cover page
        fig_cover = plt.figure(figsize=(11, 4))
        fig_cover.text(0.5, 0.72, f'GLM Contrast Report  —  sub-{subject}',
                       ha='center', va='center', fontsize=18, fontweight='bold')
        fig_cover.text(0.5, 0.52,
                       f'space: {space_tag}   |   FDR q < {fdr_alpha}'
                       f'   |   min cluster: {cluster_threshold} voxels',
                       ha='center', va='center', fontsize=12)
        fig_cover.text(0.5, 0.32,
                       'Sessions pooled  |  5 mm smoothing  |  SPM HRF  |  AR(1) noise model',
                       ha='center', va='center', fontsize=10, color='#555555')
        pdf.savefig(fig_cover, bbox_inches='tight')
        plt.close(fig_cover)

        for name, expr, desc in CONTRAST_INFO:
            z_path = func_dir / f'{fn_base}_contrast-{name}_stat-z_statmap.nii.gz'
            if not z_path.exists():
                print(f'  {name}: z-map not found, skipping')
                continue

            z_map = image.load_img(str(z_path))

            try:
                _, fdr_thr = threshold_stats_img(
                    z_map, alpha=fdr_alpha, height_control='fdr',
                    cluster_threshold=cluster_threshold)
                n_survived = int(np.sum(np.abs(z_map.get_fdata()) >= fdr_thr))
            except Exception:
                fdr_thr = None
                n_survived = 0

            print(f'  {name}: '
                  + (f'z ≥ {fdr_thr:.2f}, {n_survived} voxels'
                     if fdr_thr is not None else 'no voxels survive FDR'))

            _make_contrast_page(pdf, z_map, t1w, name, expr, desc,
                                fdr_thr, n_survived, fdr_alpha)

    print(f'\nDone → {pdf_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('subject', help="Subject label without 'sub-', e.g. 01")
    parser.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    parser.add_argument('--space', default='T1w')
    parser.add_argument('--res', default=None)
    parser.add_argument('--fdr-alpha', type=float, default=0.05,
                        help='FDR significance level (default: 0.05)')
    parser.add_argument('--cluster-threshold', type=int, default=10,
                        help='Minimum cluster size in voxels after FDR (default: 10)')
    args = parser.parse_args()

    report_subject(args.subject,
                   bids_folder=args.bids_folder,
                   space=args.space,
                   res=args.res,
                   fdr_alpha=args.fdr_alpha,
                   cluster_threshold=args.cluster_threshold)
