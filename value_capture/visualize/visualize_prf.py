#!/usr/bin/env python3
"""
Visualize PRF parameter maps in pycortex.

Requires the pycortex subject to exist in the pycortex database.
The subject name convention is ``valuecapture.sub-{id}``.

To add a new subject to pycortex (one-time setup):
    import cortex
    cortex.freesurfer.import_subj(
        fs_subject='sub-01_ses-1',
        pycortex_subject='valuecapture.sub-01',
        freesurfer_subject_dir='/data/ds-valuecapture/derivatives/fmriprep/sourcedata/freesurfer',
    )

Usage:
    python visualize_prf.py 01
    python visualize_prf.py 01 --prf-deriv prf_glmsingle_s6mm --r2-threshold 0.05
    python visualize_prf.py 01 --make-colorbars   # also writes colorbars.pdf
"""

import argparse
from pathlib import Path

import cortex
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from scipy.stats import norm

from value_capture.utils.data import BIDS_FOLDER, Subject

# Neuropythy Benson atlas ROI labels (index → name)
BENSON_ROIS = {
    1: 'V1', 2: 'V2', 3: 'V3', 4: 'hV4',
    5: 'VO1', 6: 'VO2', 7: 'LO1', 8: 'LO2',
    9: 'TO1', 10: 'TO2', 11: 'V3A', 12: 'V3B',
}


def r2_alpha(r2, thr, sigma=0.01):
    """Smooth alpha via Gaussian CDF centred on thr with width sigma."""
    return norm.cdf(r2, loc=thr, scale=sigma).astype(np.float32)


def get_masked_vertex(values, alpha, subject, vmin, vmax, cmap='RdBu_r'):
    """Create a pycortex Vertex blended onto curvature using a float alpha array."""
    v = cortex.Vertex(np.nan_to_num(values).astype(np.float32),
                      subject, vmin=vmin, vmax=vmax, cmap=cmap)
    return v.blend_curvature(alpha)


def save_colorbar_pdf(out_path, r2_threshold, r2_vmax, has_benson=True):
    """Write a PDF with colorbars for every PRF map and a Benson ROI legend."""
    continuous = [
        ('R²',                        'hot',           r2_threshold, r2_vmax, ''),
        ('Eccentricity',              'nipy_spectral', 0,            4,       '°'),
        ('Polar angle (contralateral hemifield)', 'hsv', 0,          1,       '← upper  |  lower →'),
        ('PRF size σ',                'nipy_spectral', 0,            3,       '°'),
        ('x position',                'coolwarm',      -4,           4,       '°  (left ← | → right)'),
        ('y position',                'coolwarm',      -4,           4,       '°  (down ← | → up)'),
    ]

    n_cont = len(continuous)
    n_rows = n_cont + (3 if has_benson else 0)  # benson_angle + benson_ecc + roi legend
    fig, axes = plt.subplots(n_rows, 1, figsize=(6, 1.4 * n_rows))
    if n_rows == 1:
        axes = [axes]

    for ax, (label, cmap, vmin, vmax, unit) in zip(axes[:n_cont], continuous):
        cb = ColorbarBase(ax, cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax),
                          orientation='horizontal')
        title = f'{label}  [{unit}]' if unit else label
        cb.set_label(title)

    if has_benson:
        benson_extra = [
            ('Benson polar angle (matched to PRF scale)', 'hsv', 0, 1,
             '← upper VM  |  HM  |  lower VM →'),
            ('Benson eccentricity', 'nipy_spectral', 0, 8, '°'),
        ]
        for ax, (label, cmap, vmin, vmax, unit) in zip(axes[n_cont:n_cont+2], benson_extra):
            cb = ColorbarBase(ax, cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax),
                              orientation='horizontal')
            cb.set_label(f'{label}  [{unit}]')

        ax = axes[n_cont + 2]
        ax.axis('off')
        cmap_tab20 = plt.cm.tab20
        patches = [
            mpatches.Patch(color=cmap_tab20(idx / 20), label=f'{idx}  {name}')
            for idx, name in BENSON_ROIS.items()
        ]
        ax.legend(handles=patches, title='Benson atlas ROIs',
                  loc='center', ncol=4, fontsize=9, title_fontsize=10,
                  frameon=False)

    fig.tight_layout()
    fig.savefig(str(out_path), bbox_inches='tight')
    plt.close(fig)
    print(f'Colorbars saved → {out_path}')


def visualize_prf(subject, prf_deriv='prf_glmsingle', sessions=None,
                  bids_folder=BIDS_FOLDER, r2_threshold=0.05, make_colorbars=False):
    sub = Subject(subject, bids_folder=bids_folder)
    cx_subject = f'valuecapture.sub-{sub.subject_id}'

    pars = sub.get_prf_parameters_surface(prf_deriv=prf_deriv, sessions=sessions,
                                          space='fsnative')

    r2 = np.nan_to_num(pars['R2'].values).astype(np.float32)
    alpha = r2_alpha(r2, thr=r2_threshold, sigma=0.01)

    ecc = np.sqrt(pars['x'].values ** 2 + pars['y'].values ** 2)
    theta = np.arctan2(pars['y'].values, pars['x'].values)

    # Polar angle: split by hemisphere for correct hemifield mapping
    n_lh = len(pars.loc['L'])
    theta_l = np.mod(theta[:n_lh] + np.pi, 2 * np.pi)
    theta_l = -np.clip((theta_l - 0.25 * np.pi) / (1.5 * np.pi), 0, 1) + 1
    theta_r = np.mod(theta[n_lh:], 2 * np.pi)
    theta_r = np.clip((theta_r - 0.25 * np.pi) / (1.5 * np.pi), 0, 1)
    theta_hemi = np.concatenate([theta_l, theta_r])

    r2_vmax = float(np.nanpercentile(r2[r2 > 0], 99.9)) if (r2 > 0).any() else 0.3

    ds = {
        f'{subject}.r2':          get_masked_vertex(r2,                 alpha, cx_subject, vmin=r2_threshold, vmax=r2_vmax, cmap='hot'),
        f'{subject}.ecc':         get_masked_vertex(ecc,                alpha, cx_subject, vmin=0,  vmax=4,  cmap='nipy_spectral'),
        f'{subject}.polar_angle': get_masked_vertex(theta_hemi,         alpha, cx_subject, vmin=0,  vmax=1,  cmap='hsv'),
        f'{subject}.sd':          get_masked_vertex(pars['sd'].values,  alpha, cx_subject, vmin=0,  vmax=3,  cmap='nipy_spectral'),
        f'{subject}.x':           get_masked_vertex(pars['x'].values,   alpha, cx_subject, vmin=-4, vmax=4,  cmap='coolwarm'),
        f'{subject}.y':           get_masked_vertex(pars['y'].values,   alpha, cx_subject, vmin=-4, vmax=4,  cmap='coolwarm'),
    }

    # Benson atlas ROIs, polar angle, eccentricity — native surface files.
    has_benson = False
    try:
        surf_dir = (Path(sub.freesurfer_subjects_dir) / sub.freesurfer_subject_id / 'surf')
        parts_varea, parts_angle, parts_ecc = [], [], []
        for fs_hemi in ['lh', 'rh']:
            parts_varea.append(nib.freesurfer.read_morph_data(
                str(surf_dir / f'{fs_hemi}.inferred_varea')).astype(np.float32))
            parts_angle.append(nib.freesurfer.read_morph_data(
                str(surf_dir / f'{fs_hemi}.inferred_angle')).astype(np.float32))
            parts_ecc.append(nib.freesurfer.read_morph_data(
                str(surf_dir / f'{fs_hemi}.inferred_eccen')).astype(np.float32))

        varea = np.concatenate(parts_varea)
        benson_angle = np.concatenate(parts_angle)
        benson_ecc = np.concatenate(parts_ecc)

        roi_alpha = (varea > 0).astype(np.float32)
        ds[f'{subject}.benson_rois']  = get_masked_vertex(varea,       roi_alpha, cx_subject,
                                                           vmin=0,   vmax=12,  cmap='tab20')
        # Normalise to same [0,1] HSV scale as fitted polar angle:
        # UVM (0°) → 1/6, HM (90°) → 1/2, LVM (180°) → 5/6
        benson_angle_norm = benson_angle / 270.0 + 1.0 / 6.0
        ds[f'{subject}.benson_angle'] = get_masked_vertex(benson_angle_norm, roi_alpha, cx_subject,
                                                           vmin=0,   vmax=1,   cmap='hsv')
        ds[f'{subject}.benson_ecc']   = get_masked_vertex(benson_ecc,   roi_alpha, cx_subject,
                                                           vmin=0,   vmax=8,   cmap='nipy_spectral')
        has_benson = True
    except Exception as e:
        print(f'Could not load Benson atlas (run register_retinotopy.py first): {e}')

    if make_colorbars:
        ses_label = 'ses-all' if sessions is None else f'ses-{sessions[0]}'
        out_dir = (Path(bids_folder) / 'derivatives' / prf_deriv
                   / f'sub-{sub.subject_id}' / ses_label / 'func')
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = out_dir / f'sub-{sub.subject_id}_{ses_label}_prf_colorbars.pdf'
        save_colorbar_pdf(pdf_path, r2_threshold=r2_threshold,
                          r2_vmax=r2_vmax, has_benson=has_benson)

    print(f'Launching pycortex viewer with {len(ds)} dataset(s)...')
    cortex.webgl.show(ds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize PRF maps in pycortex.')
    parser.add_argument('subject', help='Subject ID (e.g. 01)')
    parser.add_argument('--prf-deriv', default='prf_glmsingle')
    parser.add_argument('--sessions', nargs='+', type=int, default=None)
    parser.add_argument('--bids-folder', default=BIDS_FOLDER)
    parser.add_argument('--r2-threshold', type=float, default=0.05,
                        help='R² threshold for alpha masking (default: 0.05)')
    parser.add_argument('--make-colorbars', action='store_true',
                        help='Write a colorbars.pdf alongside the surface files')
    args = parser.parse_args()

    visualize_prf(
        args.subject,
        prf_deriv=args.prf_deriv,
        sessions=args.sessions,
        bids_folder=args.bids_folder,
        r2_threshold=args.r2_threshold,
        make_colorbars=args.make_colorbars,
    )
