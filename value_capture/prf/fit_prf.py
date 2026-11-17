#!/usr/bin/env python3
"""
Fit a 2-D Gaussian PRF model to GLMsingle single-trial betas.

Each GLMsingle beta is treated as a single "frame".  The corresponding
stimulus is the visual input of that trial's target phase, reconstructed as closely as
possible (``--paradigm bar_items``, default since 2026-11-17):

  - the flickering bar (bar_position × bar_orientation) inside its aperture (radius 3.17°);
  - all 8 search-array items: grey 1.5 × 0.375° rectangles at 4° eccentricity
    (angles i·45°, i = 0..7, from experiment/stimuli.py), the target with
    ``target_orientation`` and the 7 others (incl. the colour distractor) orthogonal to it.

The items sit OUTSIDE the bar aperture, so the grid extends to ±5° (it was ±3.2°, bar only,
which forced every PRF near the items onto the aperture edge). Colour (distractor vs grey) and
the small dots inside the items are not in this luminance paradigm; target / distractor
effects belong in gain fields (as in retsupp). The whole display, fixation included, was
shifted down by 1.1° on screen; coordinates here are relative to fixation, so no shift.

No HRF is modelled because GLMsingle already handles the haemodynamic convolution.

Geometry is derived from experiment/settings/default.yml so it stays in
sync with the experiment code, following the same logic as
ValueCaptureSession.__init__ in session.py:

  radius_bar_aperture = eccentricity_stimulus - size_stimuli / 1.8
  fov_size            = radius_bar_aperture * 2
  bar_width           = fov_size / n_bar_positions

Stimulus grid: ±5° at 0.1°/pixel (100 × 100); images are 4× supersampled, so a pixel holds
the fraction covered.

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
    os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # force JAX to CPU (disables Metal/CUDA)
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
PIXEL_SIZE_DEG = 0.1   # degrees per pixel (item short side is 0.375°)
GRID_RADIUS = 5.0      # half-width of the square grid (items reach 4.75°)
SUPERSAMPLE = 4

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
        item_ecc=float(e),
        item_long=float(s),
        item_short=float(s / 4.0),        # visual.Rect(width=size, height=size / 4)
        radius=float(radius),
        fov_size=float(fov_size),
        bar_width=float(bar_width),
        bar_positions=bar_positions,
        n_pos=n,
    )


GEOM = _load_geometry()
GRID_RES = int(round(2 * GRID_RADIUS / PIXEL_SIZE_DEG))   # 100


# ── stimulus helpers ──────────────────────────────────────────────────────────

def _axis(n=GRID_RES):
    """Pixel-centre coordinates (deg) along one axis."""
    step = 2 * GRID_RADIUS / n
    return (np.arange(n, dtype=np.float32) + .5) * step - GRID_RADIUS


def make_grid_coordinates():
    """Return (GRID_RES², 2) DataFrame of pixel-centre (x, y) in visual degrees.

    Pixel ordering matches the images' .ravel(): x increases left → right,
    y increases bottom → top (row index = y).
    """
    xx, yy = np.meshgrid(_axis(), _axis())
    return pd.DataFrame({'x': xx.ravel(), 'y': yy.ravel()})


_FINE = np.meshgrid(_axis(GRID_RES * SUPERSAMPLE), _axis(GRID_RES * SUPERSAMPLE))


def _downsample(fine):
    """Block-average the supersampled image to GRID_RES (fraction of each pixel covered)."""
    k = SUPERSAMPLE
    return fine.reshape(GRID_RES, k, GRID_RES, k).mean((1, 3)).astype(np.float32)


def make_bar_image(position, orientation):
    """Bar image (GRID_RES, GRID_RES) inside the bar aperture.

    orientation 'horizontal': bar spans x, positioned on y; 'vertical': spans y, positioned on x.
    """
    xx, yy = _FINE
    aperture = (xx ** 2 + yy ** 2) <= GEOM['radius'] ** 2
    hw = GEOM['bar_width'] / 2
    bar = (np.abs(yy - position) <= hw) if orientation == 'horizontal' else (np.abs(xx - position) <= hw)
    return _downsample((bar & aperture).astype(np.float32))


ITEM_XY = np.array([(GEOM['item_ecc'] * np.cos(np.radians(i * 45.)),
                     GEOM['item_ecc'] * np.sin(np.radians(i * 45.))) for i in range(8)])


def make_items_image(target_location, target_orientation):
    """The 8 search-array rectangles of one trial (experiment/stimuli.py TargetStimulusArray).

    Item ``target_location`` has ``target_orientation`` (0 = long axis horizontal, 90 = vertical);
    all others have the orthogonal orientation.
    """
    xx, yy = _FINE
    img = np.zeros_like(xx, dtype=bool)
    other = 90. if float(target_orientation) == 0. else 0.
    for i, (cx, cy) in enumerate(ITEM_XY):
        ori = float(target_orientation) if i == int(target_location) else other
        hx, hy = ((GEOM['item_long'] / 2, GEOM['item_short'] / 2) if ori == 0.
                  else (GEOM['item_short'] / 2, GEOM['item_long'] / 2))
        img |= (np.abs(xx - cx) <= hx) & (np.abs(yy - cy) <= hy)
    return _downsample(img.astype(np.float32))


def build_paradigm(trials, paradigm='bar_items'):
    """Return (n_trials, GRID_RES²) float32 stimulus matrix, one row per GLMsingle beta.

    paradigm 'bar': bar only (the pre-2026-11-17 model, but on the ±5° grid);
    'bar_items': bar + the 8 search items (needs target_location / target_orientation).
    """
    imgs = []
    for _, row in trials.iterrows():
        img = make_bar_image(row.bar_position, row.bar_orientation)
        if paradigm == 'bar_items':
            img = np.maximum(img, make_items_image(row.target_location, row.target_orientation))
        imgs.append(img.ravel())
    return np.array(imgs, dtype=np.float32)


def add_item_info(sub, trials):
    """Add target_location / target_orientation from the raw events (matched on session, run, trial_nr)."""
    rows = []
    for (ses, run), _ in trials.groupby(['session', 'run']):
        ev = sub.get_onsets(int(ses), int(run))
        ev = ev[ev['event_type'] == 'target'][['trial_nr', 'target_location', 'target_orientation']]
        rows.append(ev.drop_duplicates('trial_nr').assign(session=ses, run=run))
    info = pd.concat(rows)
    out = trials.merge(info, on=['session', 'run', 'trial_nr'], how='left', validate='one_to_one')
    assert len(out) == len(trials) and out[['target_location', 'target_orientation']].notna().all().all(), \
        'target_location / target_orientation missing for some trials'
    return out


def save_param_nifti(arr, mask_img, path):
    """Unmask a per-voxel parameter and write it as float32.

    The unmasked image inherits the mask's dtype (the GM mask is boolean -> uint8), which quantises every
    parameter to 256 values with an auto-picked slope. Force float32 and an identity scaling.
    """
    img = masking.unmask(np.asarray(arr, dtype=np.float32), mask_img)
    img.set_data_dtype(np.float32)
    img.header.set_slope_inter(slope=1, inter=0)
    img.to_filename(str(path))


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
         voxel_chunk_size=5000, learning_rate=0.01, paradigm='bar_items', train_sessions=None,
         model_kind='gauss', min_init_r2=.02):

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
    if paradigm == 'bar_items':
        trials = add_item_info(sub, trials)
    keep_rows = None
    if train_sessions:
        # Cross-validation: fit on the trials of these sessions only (betas from the joint
        # all-session GLMsingle fit; its HRF/denoising choices saw all runs, a mild leak).
        keep_rows = trials['session'].isin(train_sessions).to_numpy()
        trials = trials[keep_rows].reset_index(drop=True)
        print(f'  Training on sessions {train_sessions}: {keep_rows.sum()} trials')

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
    if keep_rows is not None:
        betas_2d = betas_2d[keep_rows]
    n_trials, n_voxels = betas_2d.shape
    print(f'  Betas: {n_trials} trials × {n_voxels} voxels')

    data_df = pd.DataFrame(betas_2d.astype(np.float32))
    data_df.index.name = 'frame'

    # ── build stimulus paradigm ───────────────────────────────────────────────
    grid_coords = make_grid_coordinates()
    paradigm_arr = build_paradigm(trials, paradigm)   # (n_trials, GRID_RES²)
    print(f'  Paradigm: {paradigm}, mean coverage per trial {paradigm_arr.sum(1).mean() * PIXEL_SIZE_DEG ** 2:.2f} deg²')
    paradigm_df = pd.DataFrame(paradigm_arr)
    paradigm_df.index.name = 'frame'

    # ── PRF model (no HRF — betas are already in amplitude space) ────────────
    from braincoder.models import (DifferenceOfGaussiansPRF2D, DivisiveNormalizationGaussianPRF2D,
                                   GaussianPRF2D)
    from braincoder.optimize import ParameterFitter
    prf_deriv = f'prf_{glmsingle_deriv}' + ('_items' if paradigm == 'bar_items' else '')
    if train_sessions:
        prf_deriv += '_train-ses' + ''.join(str(t) for t in train_sessions)
    gauss_deriv = prf_deriv
    if model_kind != 'gauss':
        prf_deriv += f'_{model_kind}'
    MODEL = {'gauss': GaussianPRF2D, 'dog': DifferenceOfGaussiansPRF2D,
             'dn': DivisiveNormalizationGaussianPRF2D}[model_kind]
    fit_idx = np.arange(data_df.shape[1])

    if model_kind == 'gauss':
        model = GaussianPRF2D(
            data=data_df,
            paradigm=paradigm_df,
            grid_coordinates=grid_coords,
        )
        par_fitter = ParameterFitter(model=model, data=data_df, paradigm=paradigm_df)

        # ── coarse grid search (correlation cost → amplitude/baseline-free) ───────
        r = GRID_RADIUS * 0.9          # include the item ring (4°), not only the bar aperture
        x_grid = np.linspace(-r, r, 13, dtype=np.float32)
        y_grid = np.linspace(-r, r, 13, dtype=np.float32)
        sd_grid = np.geomspace(0.1, 3.0, 10).astype(np.float32)

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
    else:
        # Start from this subject's Gaussian fit (same paradigm and training data); fit only voxels with
        # Gaussian R² >= min_init_r2, the rest stay NaN.
        gfn = (Path(bids_folder) / 'derivatives' / gauss_deriv / f'sub-{subject}' / ses_label / 'func'
               / f'sub-{subject}_{ses_label}_task-valuecapture_space-T1w_desc-prf_params.tsv')
        g = pd.read_csv(gfn, sep='\t', index_col=0)
        assert len(g) == data_df.shape[1], (len(g), data_df.shape[1])
        fit_idx = np.flatnonzero(g.r2.to_numpy() >= min_init_r2)
        print(f'  {model_kind}: init from {gauss_deriv}, {len(fit_idx)} voxels with Gaussian R² >= {min_init_r2}')
        g = g.iloc[fit_idx]
        if model_kind == 'dog':
            pars_ols = pd.DataFrame({'x': g.x, 'y': g.y, 'sd': g.sd, 'baseline': g.baseline,
                                     'amplitude': g.amplitude.clip(lower=1e-3), 'srf_amplitude': .1,
                                     'srf_size': 2.})
        else:
            pars_ols = pd.DataFrame({'x': g.x, 'y': g.y, 'sd': g.sd, 'rf_amplitude': g.amplitude,
                                     'srf_amplitude': .1, 'srf_size': 2.,
                                     'neural_baseline': g.baseline.clip(lower=.01), 'surround_baseline': 1.})
        pars_ols = pars_ols.astype(np.float32).reset_index(drop=True)
        data_df = data_df.iloc[:, fit_idx]
        data_df.columns = range(len(fit_idx))
        par_fitter = ParameterFitter(model=MODEL(data=data_df, paradigm=paradigm_df, grid_coordinates=grid_coords),
                                     data=data_df, paradigm=paradigm_df)

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
        chunk_model = MODEL(
            data=chunk_data,
            paradigm=paradigm_df,
            grid_coordinates=grid_coords,
        )
        chunk_fitter = ParameterFitter(
            model=chunk_model, data=chunk_data, paradigm=paradigm_df)
        pars_chunks.append(
            chunk_fitter.fit(init_pars=chunk_init, max_n_iterations=gd_iterations,
                             learning_rate=learning_rate))
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
    if model_kind != 'gauss':      # back to all mask voxels, NaN where not fitted
        full = pd.DataFrame(np.nan, index=range(n_voxels), columns=pars_gd.columns, dtype=np.float32)
        full.iloc[fit_idx] = pars_gd.to_numpy()
        pars_gd = full
        r2_full = pd.Series(np.nan, index=range(n_voxels), dtype=np.float32)
        r2_full.iloc[fit_idx] = r2.to_numpy()
        r2 = r2_full
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
        save_param_nifti(arr, mask_img, out_dir / (fn_base.format(desc=desc) + '_pe.nii.gz'))

    # Individual parameter NIfTIs
    for col in pars_gd.columns:
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
    parser.add_argument('--train-sessions', type=int, nargs='+', default=None,
                        help='Fit on the trials of these sessions only (cross-validation); output '
                             'prf_<deriv>[_items]_train-ses<N>.')
    parser.add_argument('--model', choices=['gauss', 'dog', 'dn'], default='gauss',
                        help='gauss (grid search + GD); dog / dn start from the Gaussian fit of the same '
                             'paradigm and write prf_<deriv>[_items][_train-sesN]_<model>.')
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
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='Adam learning rate for gradient descent (default: 0.01).')
    parser.add_argument('--paradigm', choices=['bar_items', 'bar'], default='bar_items',
                        help='bar_items (default): bar + the 8 search items, output prf_<deriv>_items; '
                             'bar: bar only, output prf_<deriv>.')
    args = parser.parse_args()

    main(
        args.subject,
        sessions=args.sessions,
        bids_folder=args.bids_folder,
        gm_threshold=args.gm_threshold,
        gd_iterations=args.gd_iterations,
        glmsingle_deriv=args.glmsingle_deriv,
        voxel_chunk_size=args.voxel_chunk_size,
        learning_rate=args.learning_rate,
        paradigm=args.paradigm,
        train_sessions=args.train_sessions,
        model_kind=args.model,
    )
