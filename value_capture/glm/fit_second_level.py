#!/usr/bin/env python3
"""
Fit a nilearn second-level (group) GLM for the value capture fMRI task.

One-sample t-test across subjects for each first-level contrast.  Input is
the per-subject effect-size maps produced by fit_nilearn_glm.py.

Output
------
  <bids>/derivatives/nilearn_glm/group/func/
    group_task-valuecapture_space-<space>_contrast-<name>_stat-z_statmap.nii.gz
    group_task-valuecapture_space-<space>_contrast-<name>_stat-effect_statmap.nii.gz

  <bids>/derivatives/nilearn_glm/group/figures/
    group_task-valuecapture_space-<space>_fdr-report.pdf

Usage
-----
  python fit_second_level.py
  python fit_second_level.py --subjects 01 02 03
  python fit_second_level.py --space MNI152NLin2009cAsymm --res 2
  python fit_second_level.py --bids-folder /shares/zne.uzh/gdehol/ds-valuecapture
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
import pandas as pd

from nilearn import image
from nilearn.glm import threshold_stats_img
from nilearn.glm.second_level import SecondLevelModel
from nilearn.plotting import plot_stat_map, plot_glass_brain

from value_capture.utils.data import get_all_subject_ids, BIDS_FOLDER

warnings.filterwarnings('ignore')

CONTRASTS = [
    ('value_linear',    'dist_rank2 − dist_rank0',
     'Proximate reward — BOLD scales linearly with distractor value rank'),
    ('value_capture',   'dist_rank2 − dist_absent',
     'Value-capture signature — high-value distractor vs no distractor'),
    ('value_nonlinear', 'dist_rank1 − (dist_rank0 + dist_rank2) / 2',
     'Nonlinear value coding — mid-value response vs linear expectation'),
    ('value_step_low',  'dist_rank1 − dist_rank0',
     'Low → medium value step'),
    ('value_step_high', 'dist_rank2 − dist_rank1',
     'Medium → high value step'),
    ('distractor_any',  '(dist_rank0 + dist_rank1 + dist_rank2) / 3 − dist_absent',
     'Any distractor vs absent — attentional capture regardless of value'),
    ('feedback',        'feedback_shown − feedback_omitted',
     'RPE proxy — any explicit feedback vs blank fixation'),
    ('feedback_value',  'feedback_points  [parametric]',
     'RPE magnitude — BOLD scales with log(1 + earned_points)'),
]


FALLBACK_Z = 2.3  # p < 0.01 uncorrected, used when FDR yields nothing


def _contrast_page(pdf, z_map, name, expr, desc, fdr_thr, n_survived,
                   n_subjects, fdr_alpha):
    if fdr_thr is not None and n_survived > 0:
        survived_str = f'FDR threshold: |z| ≥ {fdr_thr:.2f}  ({n_survived} voxels)'
        display_thr = fdr_thr
        thr_label = 'FDR-thresholded'
    else:
        survived_str = (f'No voxels survive FDR  —  '
                        f'showing uncorrected |z| > {FALLBACK_Z} (p < 0.01)')
        display_thr = FALLBACK_Z
        thr_label = f'uncorrected |z| > {FALLBACK_Z}'

    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 1, figure=fig,
                  height_ratios=[0.13, 0.43, 0.43],
                  hspace=0.05, top=0.97, bottom=0.02)

    ax_title = fig.add_subplot(gs[0])
    ax_title.axis('off')
    ax_title.text(0.5, 0.85, name,
                  transform=ax_title.transAxes,
                  ha='center', va='top', fontsize=14, fontweight='bold')
    ax_title.text(0.5, 0.52, f'Formula:  {expr}',
                  transform=ax_title.transAxes,
                  ha='center', va='top', fontsize=10, family='monospace')
    ax_title.text(0.5, 0.22, desc,
                  transform=ax_title.transAxes,
                  ha='center', va='top', fontsize=9)
    ax_title.text(0.5, 0.0,
                  f'[{survived_str}  |  FDR q < {fdr_alpha}  |  N={n_subjects} subjects]',
                  transform=ax_title.transAxes,
                  ha='center', va='bottom', fontsize=8, color='#555555')

    ax_glass = fig.add_subplot(gs[1])
    try:
        plot_glass_brain(
            z_map, threshold=display_thr, colorbar=True,
            plot_abs=False, display_mode='lyrz',
            axes=ax_glass, vmax=6, annotate=False,
            title=f'Glass brain ({thr_label})')
    except Exception as e:
        ax_glass.text(0.5, 0.5, f'[glass brain unavailable: {e}]',
                      transform=ax_glass.transAxes, ha='center', va='center',
                      fontsize=9, color='red')

    ax_stat = fig.add_subplot(gs[2])
    try:
        plot_stat_map(
            z_map, threshold=display_thr,
            display_mode='z', cut_coords=7,
            colorbar=True, axes=ax_stat, vmax=6,
            title=f'Axial slices ({thr_label})')
    except Exception as e:
        ax_stat.text(0.5, 0.5, f'[stat map unavailable: {e}]',
                     transform=ax_stat.transAxes, ha='center', va='center',
                     fontsize=9, color='red')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def main(subjects=None, bids_folder=BIDS_FOLDER, space='MNI152NLin2009cAsymm',
         res='2', fdr_alpha=0.05, cluster_threshold=10):

    if subjects is None:
        subjects = get_all_subject_ids()

    res_label = f'_res-{res}' if res else ''
    space_tag = f'space-{space}{res_label}'
    fn_base = f'group_task-valuecapture_{space_tag}'

    out_dir = Path(bids_folder) / 'derivatives' / 'nilearn_glm' / 'group' / 'func'
    fig_dir = out_dir.parent / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f'Second-level GLM  |  subjects: {subjects}  |  {space_tag}')

    # Design matrix: intercept only (one-sample t-test)
    design_matrix = pd.DataFrame({'intercept': np.ones(len(subjects))})

    pdf_path = fig_dir / f'{fn_base}_fdr-report.pdf'

    with PdfPages(str(pdf_path)) as pdf:
        # Cover page
        fig_cover = plt.figure(figsize=(11, 4))
        fig_cover.text(0.5, 0.72, 'Group GLM Report',
                       ha='center', va='center', fontsize=18, fontweight='bold')
        fig_cover.text(0.5, 0.52,
                       f'N = {len(subjects)} subjects  |  {space_tag}\n'
                       f'FDR q < {fdr_alpha}  |  min cluster: {cluster_threshold} voxels',
                       ha='center', va='center', fontsize=12)
        fig_cover.text(0.5, 0.28,
                       f'Subjects: {", ".join(subjects)}\n'
                       'One-sample t-test on first-level effect-size maps',
                       ha='center', va='center', fontsize=10, color='#555555')
        fig_cover.text(0.5, 0.08,
                       'Note: N=3 — maps are exploratory only. '
                       'FDR corrects for multiple voxels but df=2 inflates false positives.',
                       ha='center', va='center', fontsize=8,
                       color='#aa3333', style='italic')
        pdf.savefig(fig_cover, bbox_inches='tight')
        plt.close(fig_cover)

        for name, expr, desc in CONTRASTS:
            # Collect per-subject effect maps
            effect_imgs = []
            missing = []
            for sub in subjects:
                sub_fn_base = f'sub-{sub}_task-valuecapture_{space_tag}'
                effect_path = (Path(bids_folder) / 'derivatives' / 'nilearn_glm'
                               / f'sub-{sub}' / 'func'
                               / f'{sub_fn_base}_contrast-{name}_stat-effect_statmap.nii.gz')
                if effect_path.exists():
                    effect_imgs.append(str(effect_path))
                else:
                    missing.append(sub)

            if missing:
                print(f'  {name}: missing subjects {missing}, skipping')
                continue
            if len(effect_imgs) < 2:
                print(f'  {name}: need ≥2 subjects, got {len(effect_imgs)}, skipping')
                continue

            print(f'  {name}: fitting second-level (N={len(effect_imgs)})...')

            model = SecondLevelModel(smoothing_fwhm=None)
            model.fit(effect_imgs, design_matrix=design_matrix)

            z_map = model.compute_contrast('intercept', output_type='z_score')
            effect_map = model.compute_contrast('intercept', output_type='effect_size')

            # Save NIfTIs
            z_path = out_dir / f'{fn_base}_contrast-{name}_stat-z_statmap.nii.gz'
            e_path = out_dir / f'{fn_base}_contrast-{name}_stat-effect_statmap.nii.gz'
            z_map.to_filename(str(z_path))
            effect_map.to_filename(str(e_path))

            # FDR threshold
            try:
                _, fdr_thr = threshold_stats_img(
                    z_map, alpha=fdr_alpha, height_control='fdr',
                    cluster_threshold=cluster_threshold)
                n_survived = int(np.sum(np.abs(z_map.get_fdata()) >= fdr_thr))
            except Exception:
                fdr_thr = None
                n_survived = 0

            print(f'    → '
                  + (f'z ≥ {fdr_thr:.2f}, {n_survived} voxels'
                     if fdr_thr is not None else 'no voxels survive FDR'))

            _contrast_page(pdf, z_map, name, expr, desc,
                           fdr_thr, n_survived, len(effect_imgs), fdr_alpha)

    print(f'\nNIfTIs → {out_dir}')
    print(f'Report → {pdf_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--subjects', nargs='+', default=None,
                        help='Subject IDs (default: all in subjects.yml)')
    parser.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    parser.add_argument('--space', default='MNI152NLin2009cAsymm')
    parser.add_argument('--res', default='2')
    parser.add_argument('--fdr-alpha', type=float, default=0.05)
    parser.add_argument('--cluster-threshold', type=int, default=10)
    args = parser.parse_args()

    main(subjects=args.subjects,
         bids_folder=args.bids_folder,
         space=args.space,
         res=args.res,
         fdr_alpha=args.fdr_alpha,
         cluster_threshold=args.cluster_threshold)
