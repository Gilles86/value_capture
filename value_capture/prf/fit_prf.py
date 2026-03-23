#!/usr/bin/env python3
"""
Fit a 2-D Gaussian PRF model to GLMsingle single-trial betas.

Each GLMsingle beta is treated as a single "frame".  The corresponding
stimulus is the binary bar image reconstructed from the trial metadata
(bar_position × bar_orientation).  No HRF is modelled because GLMsingle
already handles the haemodynamic convolution when estimating betas.

Geometry is derived from experiment/settings/default.yml so it stays in
sync with the experiment code, following the same logic as
ValueCaptureSession.__init__ in session.py:

  radius_bar_aperture = eccentricity_stimulus - size_stimuli / 1.8
  fov_size            = radius_bar_aperture * 2
  bar_width           = fov_size / n_bar_positions

Stimulus grid: 0.2°/pixel → ceil(fov_size / 0.2) ≈ 32 × 32 pixels.

Output (saved to <bids_folder>/derivatives/prf/sub-<sub>/<ses_label>/func/):
  *_desc-prf_params.nii.gz  — 5 volumes (x, y, sd, baseline, amplitude)
  *_desc-prfR2_pe.nii.gz    — R² from gradient descent
  *_desc-prf_params.tsv     — parameter table (voxels × parameters)

Usage:
    python fit_prf.py 01
    python fit_prf.py 01 --sessions 2
    python fit_prf.py 01 --sessions 1 2 --gm-threshold 0.5
    python fit_prf.py 01 --bids-folder /shares/zne.uzh/gdehol/ds-valuecapture
"""

import argparse
import os
import warnings
from pathlib import Path

# Must be set before keras/braincoder is imported
os.environ.setdefault('KERAS_BACKEND', 'jax')

# --cpu flag: disable MPS/CUDA so Keras/torch falls back to CPU.
# Parse just this one flag early, before any GPU-touching imports.
if '--cpu' in os.sys.argv:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    try:
        import torch
        torch.backends.mps.is_available = lambda: False
        torch.backends.mps.is_built = lambda: False
    except ImportError:
        pass

import numpy as np
import pandas as pd
import yaml

from nilearn import image, masking

from value_capture.utils.data import Subject, BIDS_FOLDER

warnings.filterwarnings('ignore')

# ── pixel resolution ──────────────────────────────────────────────────────────
PIXEL_SIZE_DEG = 0.2   # degrees per pixel

# ── derive geometry from default.yml (mirrors session.py) ────────────────────
_SETTINGS_FILE = (
    Path(__file__).parents[2] / 'experiment' / 'settings' / 'default.yml'
)


def _load_geometry(settings_file=_SETTINGS_FILE):
    with open(settings_file) as f:
        cfg = yaml.safe_load(f)
    e = cfg['experiment']['eccentricity_stimulus']   # 4.0 deg
    s = cfg['experiment']['size_stimuli']             # 1.5 deg
    n = cfg['design']['n_bar_positions']              # 10
    radius = e - s / 1.8
    fov_size = radius * 2
    bar_width = fov_size / n
    bar_positions = np.linspace(
        -fov_size / 2 + bar_width / 2,
         fov_size / 2 - bar_width / 2,
        n,
    )
    return dict(
        radius=float(radius),
        fov_size=float(fov_size),
        bar_width=float(bar_width),
        bar_positions=bar_positions,
        n_pos=n,
    )


GEOM = _load_geometry()
GRID_RES = int(np.ceil(GEOM['fov_size'] / PIXEL_SIZE_DEG))   # 32


# ── stimulus helpers ──────────────────────────────────────────────────────────

def make_grid_coordinates():
    """Return (GRID_RES², 2) float32 DataFrame of (x, y) in visual degrees.

    Pixel ordering matches make_bar_image().ravel():
      - x increases left → right
      - y increases bottom → top
    """
    half = GEOM['fov_size'] / 2
    xs = np.linspace(-half, half, GRID_RES, dtype=np.float32)
    ys = np.linspace(-half, half, GRID_RES, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    return pd.DataFrame({'x': xx.ravel(), 'y': yy.ravel()})


def make_bar_image(position, orientation):
    """Return (GRID_RES, GRID_RES) float32 binary bar image.

    Parameters
    ----------
    position    : float — bar centre in deg (along the axis ⊥ to bar length)
    orientation : 'horizontal' (bar spans X; positioned on Y-axis)
                  'vertical'   (bar spans Y; positioned on X-axis)
    """
    half = GEOM['fov_size'] / 2
    xs = np.linspace(-half, half, GRID_RES, dtype=np.float32)
    ys = np.linspace(-half, half, GRID_RES, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)

    aperture = (xx ** 2 + yy ** 2) <= GEOM['radius'] ** 2
    hw = GEOM['bar_width'] / 2
    if orientation == 'horizontal':
        bar = np.abs(yy - position) <= hw
    else:
        bar = np.abs(xx - position) <= hw

    return (bar & aperture).astype(np.float32)


def build_paradigm(trials):
    """Return (n_trials, GRID_RES²) float32 stimulus matrix.

    One row per trial in the same order as the GLMsingle beta volumes.
    """
    imgs = [
        make_bar_image(row.bar_position, row.bar_orientation).ravel()
        for _, row in trials.iterrows()
    ]
    return np.array(imgs, dtype=np.float32)


# ── gray matter mask helper ───────────────────────────────────────────────────

def _get_gm_mask(sub, session, bids_folder, threshold):
    """Load fmriprep GM probseg, resample to bold space, threshold.

    Returns a NIfTI image or None if the file cannot be found.
    """
    anat_dir = (
        Path(bids_folder) / 'derivatives' / 'fmriprep'
        / f'sub-{sub.subject_id}' / f'ses-{session}' / 'anat'
    )
    # fmriprep writes native-T1w-space probsegs without a space entity;
    # try space-T1w first (some versions include it), then fall back.
    candidates = sorted(anat_dir.glob(
        f'sub-{sub.subject_id}*space-T1w*label-GM_probseg.nii.gz'
    ))
    if not candidates:
        candidates = sorted(anat_dir.glob(
            f'sub-{sub.subject_id}*label-GM_probseg.nii.gz'
        ))
        # exclude any MNI-space files that slipped through
        candidates = [c for c in candidates if 'space-MNI' not in c.name]
    if not candidates:
        return None

    gm_img = image.load_img(str(candidates[0]))
    ref_img = sub.get_brain_mask(session)
    gm_res = image.resample_to_img(gm_img, ref_img, interpolation='linear')
    return image.math_img(f'img >= {threshold}', img=gm_res)


# ── main ──────────────────────────────────────────────────────────────────────

def main(subject, sessions=None, bids_folder=BIDS_FOLDER,
         gm_threshold=None, gd_iterations=1000, glmsingle_deriv='glmsingle',
         voxel_chunk_size=5000):

    sub = Subject(subject, bids_folder=bids_folder)
    if sessions is None:
        sessions = sub.get_sessions()

    ses_label = f'ses-{sessions[0]}' if len(sessions) == 1 else 'ses-all'
    print(f'sub-{subject}  {ses_label}  [{glmsingle_deriv}]')
    print(f'  FOV: {GEOM["fov_size"]:.3f}° diam,  bar_width: {GEOM["bar_width"]:.3f}°')
    print(f'  Grid: {GRID_RES}×{GRID_RES} px @ {PIXEL_SIZE_DEG}°/px  '
          f'({GRID_RES ** 2} pixels total)')

    # ── load GLMsingle betas and trial metadata ───────────────────────────────
    betas_img = sub.get_single_trial_estimates(sessions, glmsingle_deriv=glmsingle_deriv)
    trials = sub.get_trial_metadata(sessions, glmsingle_deriv=glmsingle_deriv).reset_index(drop=True)
    assert len(trials) == betas_img.shape[3], (
        f'Trial count mismatch: {len(trials)} rows in TSV but '
        f'{betas_img.shape[3]} beta volumes'
    )
    print(f'  {len(trials)} trials loaded')

    # ── brain / gray-matter mask ──────────────────────────────────────────────
    if gm_threshold is not None:
        mask_img = _get_gm_mask(sub, sessions[0], bids_folder, gm_threshold)
        if mask_img is None:
            print(f'  WARNING: GM probseg not found for ses-{sessions[0]}, '
                  'falling back to brain mask')
            mask_img = sub.get_brain_mask(sessions[0])
        else:
            print(f'  Using GM mask (threshold={gm_threshold})')
    else:
        mask_img = sub.get_brain_mask(sessions[0])
        print('  Using brain mask')

    # ── flatten betas → (n_trials, n_voxels) ─────────────────────────────────
    betas_2d = masking.apply_mask(betas_img, mask_img)   # (n_trials, n_voxels)
    n_trials, n_voxels = betas_2d.shape
    print(f'  Betas: {n_trials} trials × {n_voxels} voxels')

    data_df = pd.DataFrame(betas_2d.astype(np.float32))
    data_df.index.name = 'frame'

    # ── build stimulus paradigm ───────────────────────────────────────────────
    grid_coords = make_grid_coordinates()
    paradigm_arr = build_paradigm(trials)          # (n_trials, GRID_RES²)
    paradigm_df = pd.DataFrame(paradigm_arr)
    paradigm_df.index.name = 'frame'

    # ── PRF model (no HRF — betas are already in amplitude space) ────────────
    from braincoder.models import GaussianPRF2D
    from braincoder.optimize import ParameterFitter

    model = GaussianPRF2D(
        data=data_df,
        paradigm=paradigm_df,
        grid_coordinates=grid_coords,
    )
    par_fitter = ParameterFitter(model=model, data=data_df, paradigm=paradigm_df)

    # ── coarse grid search (correlation cost → amplitude/baseline-free) ───────
    r = GEOM['radius'] * 0.9
    x_grid = np.linspace(-r, r, 10, dtype=np.float32)
    y_grid = np.linspace(-r, r, 10, dtype=np.float32)
    sd_grid = np.linspace(0.1, GEOM['radius'], 10, dtype=np.float32)

    print('  Grid search...')
    pars_grid = par_fitter.fit_grid(
        x=x_grid,
        y=y_grid,
        sd=sd_grid,
        baseline=[np.float32(0.0)],
        amplitude=[np.float32(1.0)],
        use_correlation_cost=True,
    )
    pars_ols = par_fitter.refine_baseline_and_amplitude(pars_grid)

    # ── gradient descent (chunked to avoid GPU OOM) ───────────────────────────
    print(f'  Gradient descent ({gd_iterations} iterations, '
          f'chunk_size={voxel_chunk_size})...')
    n_vox = data_df.shape[1]
    chunks = range(0, n_vox, voxel_chunk_size)
    pars_chunks = []
    for i, start in enumerate(chunks):
        end = min(start + voxel_chunk_size, n_vox)
        print(f'    chunk {i+1}/{len(chunks)}  voxels {start}–{end}')
        chunk_data = data_df.iloc[:, start:end]
        chunk_init = pars_ols.iloc[start:end]
        chunk_model = GaussianPRF2D(
            data=chunk_data,
            paradigm=paradigm_df,
            grid_coordinates=grid_coords,
        )
        chunk_fitter = ParameterFitter(
            model=chunk_model, data=chunk_data, paradigm=paradigm_df)
        pars_chunks.append(
            chunk_fitter.fit(init_pars=chunk_init, max_n_iterations=gd_iterations))
        # free GPU memory between chunks
        del chunk_model, chunk_fitter, chunk_data, chunk_init
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    pars_gd = pd.concat(pars_chunks)
    r2 = par_fitter.get_rsq(pars_gd)
    print(f'  Median R²: {r2.median():.3f}  '
          f'(n voxels R²>0.1: {(r2 > 0.1).sum()})')

    # ── save ──────────────────────────────────────────────────────────────────
    # Put results under prf/<glmsingle_deriv>/ so smoothed and unsmoothed
    # fits live in separate directories.
    prf_deriv = f'prf_{glmsingle_deriv}'
    out_dir = (
        Path(bids_folder) / 'derivatives' / prf_deriv
        / f'sub-{subject}' / ses_label / 'func'
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fn_base = (
        f'sub-{subject}_{ses_label}_task-valuecapture'
        f'_space-T1w_desc-{{desc}}'
    )
    ref_img = sub.get_brain_mask(sessions[0])

    def _save(arr, desc):
        masking.unmask(arr.astype(np.float32), mask_img).to_filename(
            str(out_dir / (fn_base.format(desc=desc) + '_pe.nii.gz')))

    # Individual parameter NIfTIs
    for col in ['x', 'y', 'sd', 'baseline', 'amplitude']:
        _save(pars_gd[col].values, f'prf{col}')

    # Derived maps
    ecc = np.sqrt(pars_gd['x'].values ** 2 + pars_gd['y'].values ** 2)
    angle = np.degrees(np.arctan2(pars_gd['y'].values, pars_gd['x'].values))
    _save(ecc,   'prfecc')
    _save(angle, 'prfangle')

    # R²
    _save(r2.values, 'prfR2')

    # Parameters TSV (includes ecc + angle for convenience)
    pars_out = pars_gd.copy()
    pars_out['ecc']   = ecc
    pars_out['angle'] = angle
    pars_out['r2']    = r2.values
    pars_out.to_csv(
        str(out_dir / (fn_base.format(desc='prf') + '_params.tsv')),
        sep='\t',
    )

    print(f'\nSaved to {out_dir}')
    for col in ['x', 'y', 'sd', 'baseline', 'amplitude', 'ecc', 'angle', 'R2']:
        print(f'  {fn_base.format(desc=f"prf{col}")}_pe.nii.gz')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('subject', help="Subject label without 'sub-', e.g. 01")
    parser.add_argument('--sessions', type=int, nargs='+', default=None,
                        help='Session number(s). Default: all available.')
    parser.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    parser.add_argument(
        '--gm-threshold', type=float, default=None,
        help='GM probability threshold (e.g. 0.5) for gray-matter masking. '
             'Default: use the whole-brain mask from fmriprep.',
    )
    parser.add_argument('--gd-iterations', type=int, default=1000,
                        help='Max gradient descent iterations (default: 1000).')
    parser.add_argument('--glmsingle-deriv', default='glmsingle',
                        help='GLMsingle derivatives folder to read betas from '
                             '(default: glmsingle). Use glmsingle_s6mm for smoothed betas.')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU (disable MPS/CUDA). Useful for local debugging.')
    parser.add_argument('--voxel-chunk-size', type=int, default=5000,
                        help='Number of voxels per gradient-descent chunk (default: 5000). '
                             'Reduce if GPU runs out of memory; increase for speed on large GPUs.')
    args = parser.parse_args()

    main(
        args.subject,
        sessions=args.sessions,
        bids_folder=args.bids_folder,
        gm_threshold=args.gm_threshold,
        gd_iterations=args.gd_iterations,
        glmsingle_deriv=args.glmsingle_deriv,
        voxel_chunk_size=args.voxel_chunk_size,
    )
