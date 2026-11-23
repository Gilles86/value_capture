"""Pycortex renders of one subject's PRF maps on the inflated surface (value_capture, 7T).

Polar angle, eccentricity, PRF size and R² of the event-related bar PRF fitted to GLMsingle
single-trial betas (``derivatives/prf_glmsingle``, sampled to fsnative by
``prf/sample_prf_to_surface.py``), with neuropythy's visual-area borders
(``surf/{lh,rh}.inferred_varea``, from ``prf/register_retinotopy.py``). Colour and borders
are baked per vertex with curvature (``VertexRGB``); colour fades in with R².

Adapted from retsupp ``visualize/pycortex/render_fig1_prf.py``. The viewer's
``surface.<subject>.<control>`` paths split on dots, so ``valuecapture.sub-04`` cannot be
driven: the script links an alias without dots in the pycortex database and removes it after.
``save_3d_views``-style capture needs a browser, so this opens the viewer in headless Chrome.
Runs in pycortex2::

    ~/mambaforge/envs/pycortex2/bin/python \\
        value_capture/visualize/pycortex/render_prf_inflated.py 4 --views pole medial ventral
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
import webbrowser
from pathlib import Path

import cortex
import matplotlib as mpl
import nibabel as nib
import numpy as np
from nibabel.freesurfer import read_morph_data
from scipy import sparse

BIDS = Path('/data/ds-valuecapture')
FS = BIDS / 'derivatives' / 'fmriprep' / 'sourcedata' / 'freesurfer'
CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
R2_RAMP = (.05, .20)         # R² -> opacity of the colour over the curvature
ECC_MAX_SHOWN = 5.0          # stimulus grid radius (bar aperture 3.17 deg, search items at 4 deg)
CURV = dict(brightness=.55, contrast=.17, smooth=20)
PANELS = {'angle': ('hsv', 0, 1), 'ecc': ('nipy_spectral', 0, 5),
          'sd': ('nipy_spectral', 0, 2), 'r2': ('inferno', 0, .6)}
PRF_DIR = 'prf_glmsingle_items'   # bar + search items (2026-11-17); 'prf_glmsingle' = bar only
# 'unfold' runs fiducial -> inflated -> flat when the subject has a flatmap (0 / .5 / 1) and
# fiducial -> inflated (0 / 1) when it has none; main() sets it (the flatmaps came 2026-11-18).
BASE_VIEW = {'camera.target': [0, 0, 0], 'camera.radius': 190, 'surface.{subject}.unfold': 1,
             'surface.{subject}.specularity': 0,
             'surface.{subject}.left': True, 'surface.{subject}.right': True}
VIEWS = {
    'back': {'camera.azimuth': 180, 'camera.altitude': 90,
             'surface.{subject}.pivot': 0, 'surface.{subject}.shift': 0},
    'medial': {'camera.azimuth': 0, 'camera.altitude': 90,
               'surface.{subject}.pivot': 180, 'surface.{subject}.shift': 10},
    'pole': {'camera.azimuth': 180, 'camera.altitude': 90, 'camera.radius': 190,
             'camera.target': [0, 0, -6],
             'surface.{subject}.pivot': 0, 'surface.{subject}.shift': 0},
    # whole occipital pole in frame (the retsupp 'pole' crops the top on these 7T brains)
    'pole2': {'camera.azimuth': 180, 'camera.altitude': 100, 'camera.radius': 215,
              'camera.target': [0, 0, 18],
              'surface.{subject}.pivot': 0, 'surface.{subject}.shift': 0},
    'dorsal': {'camera.azimuth': 180, 'camera.altitude': 35, 'camera.radius': 225,
               'camera.target': [0, -10, 0],
               'surface.{subject}.pivot': 0, 'surface.{subject}.shift': 0},
    'ventral': {'camera.azimuth': 180, 'camera.altitude': 145, 'camera.radius': 165,
                'camera.target': [0, -75, 0],
                'surface.{subject}.pivot': 0, 'surface.{subject}.shift': 0},
    'occipital': {'camera.azimuth': 180, 'camera.altitude': 118, 'camera.radius': 175,
                  'camera.target': [0, -40, -3],
                  'surface.{subject}.pivot': 0, 'surface.{subject}.shift': 0},
    'fef_left': {'camera.azimuth': 75, 'camera.altitude': 35, 'camera.radius': 150,
                 'camera.target': [-25, 25, 20],
                 'surface.{subject}.pivot': 0, 'surface.{subject}.shift': 0},
    'fef_right': {'camera.azimuth': 285, 'camera.altitude': 35, 'camera.radius': 150,
                  'camera.target': [25, 25, 20],
                  'surface.{subject}.pivot': 0, 'surface.{subject}.shift': 0},
    'lateral': {'camera.azimuth': 180, 'camera.altitude': 90,
                'surface.{subject}.pivot': 180, 'surface.{subject}.shift': 10},
    # Whole hemispheres, one at a time (other hemisphere hidden), for the whole-brain
    # supplement (figS_prf_whole_brain.py): azimuth 90 faces the left, 270 the right side.
    'lateral_lh': {'camera.azimuth': 90, 'camera.altitude': 90, 'camera.radius': 260,
                   'surface.{subject}.pivot': 0, 'surface.{subject}.shift': 0,
                   'surface.{subject}.right': False},
    'medial_lh': {'camera.azimuth': 270, 'camera.altitude': 90, 'camera.radius': 260,
                  'surface.{subject}.pivot': 0, 'surface.{subject}.shift': 0,
                  'surface.{subject}.right': False},
    'medial_rh': {'camera.azimuth': 90, 'camera.altitude': 90, 'camera.radius': 260,
                  'surface.{subject}.pivot': 0, 'surface.{subject}.shift': 0,
                  'surface.{subject}.left': False},
    'lateral_rh': {'camera.azimuth': 270, 'camera.altitude': 90, 'camera.radius': 260,
                   'surface.{subject}.pivot': 0, 'surface.{subject}.shift': 0,
                   'surface.{subject}.left': False},
    'ventral_whole': {'camera.azimuth': 180, 'camera.altitude': 178, 'camera.radius': 280,
                      'surface.{subject}.pivot': 0, 'surface.{subject}.shift': 0},
    # pole2 pulled back: pole2 crops sub-12's (large) occipital lobe at the bottom
    'posterior': {'camera.azimuth': 180, 'camera.altitude': 100, 'camera.radius': 245,
                  'camera.target': [0, 0, 12],
                  'surface.{subject}.pivot': 0, 'surface.{subject}.shift': 0},
    'dorsal_whole': {'camera.azimuth': 180, 'camera.altitude': 2, 'camera.radius': 280,
                     'surface.{subject}.pivot': 0, 'surface.{subject}.shift': 0},
}
IMAGE_SIZE = (2400, 1800)
# Visual areas of the neuropythy atlas, grouped as in the analyses.
# Colours: paper_style.ROI_COLORS (Fig. 2 only: early blues, dorsal greens, occipitotemporal warm hues).
AREA_COLORS = {'V1': ((1,), '#08519C'), 'V2': ((2,), '#3182BD'), 'V3': ((3,), '#7FB8E0'),
               'hV4': ((4,), '#C2185B'), 'VO': ((5, 6), '#E8590C'), 'LO': ((7, 8), '#F2A60D'),
               'TO': ((9, 10), '#B04FB5'), 'V3AB': ((11, 12), '#1A7F3C'),
               'IPS': ((101,), '#7BC77E'), 'FEF': ((102,), '#1F2933')}
# Wang 2015 labels (prf/slurm_jobs/wang_atlas.sh) added where neuropythy has no area, as in retsupp:
# IPS0-5 -> IPS, FEF -> FEF. Codes as in AREA_COLORS.
WANG = {101: (18, 19, 20, 21, 22, 23), 102: (25,)}


def theta_value(theta, hemi):
    """Visual-field angle -> hsv index, as ``visualize_prf._theta_color_value``."""
    if hemi == 'R':
        v = np.mod(theta, 2 * np.pi)
        return np.clip((v - .25 * np.pi) / (1.5 * np.pi), 0, 1)
    v = np.mod(theta + np.pi, 2 * np.pi)
    return 1. - np.clip((v - .25 * np.pi) / (1.5 * np.pi), 0, 1)


GII_PAR = {'r2': 'R2', 'ecc': 'ecc', 'sd': 'sd', 'x': 'x', 'y': 'y'}


def gii(subject, par, hemi):
    if par == 'theta':
        return np.arctan2(gii(subject, 'y', hemi), gii(subject, 'x', hemi))
    fn = (BIDS / 'derivatives' / PRF_DIR / f'sub-{subject:02d}' / 'ses-all' / 'func'
          / f'sub-{subject:02d}_ses-all_task-valuecapture_hemi-{hemi}_space-fsnative'
            f'_desc-prf{GII_PAR[par]}_pe.func.gii')
    return nib.load(fn).darrays[0].data.astype(np.float64)


def read_areas(subject, h):
    surf = FS / f'sub-{subject:02d}_ses-1' / 'surf'
    va = np.round(read_morph_data(surf / f'{h}.inferred_varea')).astype(int)
    wfn = surf / f'{h}.wang15_mplbl.mgz'
    if wfn.exists():
        wang = np.round(np.asarray(nib.load(wfn).dataobj).squeeze()).astype(int)
        tri = nib.freesurfer.read_geometry(surf / f'{h}.inflated')[1]
        for code, labels in WANG.items():
            va[(va == 0) & despeckle(np.isin(wang, labels), tri)] = code
    return va


def despeckle(mask, tri, n_iter=8, threshold=.3):
    """For display only: the atlas projection leaves one-vertex islands and holes."""
    n = len(mask)
    i = np.concatenate([tri[:, 0], tri[:, 1], tri[:, 2], np.arange(n)])
    j = np.concatenate([tri[:, 1], tri[:, 2], tri[:, 0], np.arange(n)])
    a = sparse.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    a = ((a + a.T) > 0).astype(float)
    a = sparse.diags(1. / np.asarray(a.sum(1)).ravel()) @ a
    m = mask.astype(float)
    for _ in range(n_iter):
        m = a @ m
    return m > threshold


def read_varea(subject):
    return np.concatenate([read_areas(subject, h) for h in ('lh', 'rh')])


def area_rgb(subject, cx_subject, border):
    va = read_varea(subject)
    curv = cortex.db.get_surfinfo(cx_subject, smooth=CURV['smooth']).data
    out = np.tile(((curv > 0) * CURV['contrast'] + CURV['brightness'])[:, None], (1, 3))
    for labels, col in AREA_COLORS.values():
        m = np.isin(va, labels)
        out[m] = .15 * out[m] + .85 * np.array(mpl.colors.to_rgb(col))
    out[border] = 1.
    return (out * 255).astype(np.uint8)


def border_mask(subject, cx_subject):
    """Vertices on the edge between visual areas (or between an area and the rest)."""
    out = []
    surfs = cortex.db.get_surf(cx_subject, 'inflated')
    for h, (_, tri) in zip(('lh', 'rh'), surfs):
        va = read_areas(subject, h)
        t = va[tri]
        mixed = ~((t[:, 0] == t[:, 1]) & (t[:, 1] == t[:, 2]))
        m = np.zeros(len(va), bool)
        m[np.unique(tri[mixed])] = True
        out.append(m)
    return np.concatenate(out)


def vertex_rgb(subject, cx_subject, par, border):
    cmap, lo, hi = PANELS[par]
    values, r2s, eccs = [], [], []
    for hemi in 'LR':
        r2, ecc = gii(subject, 'r2', hemi), gii(subject, 'ecc', hemi)
        if par == 'angle':
            v = theta_value(gii(subject, 'theta', hemi), hemi)
        elif par == 'r2':
            v = r2 / hi
        else:
            v = (gii(subject, par, hemi) - lo) / (hi - lo)
        values.append(v)
        r2s.append(r2)
        eccs.append(ecc)
    v, r2, ecc = (np.nan_to_num(np.concatenate(x)) for x in (values, r2s, eccs))
    rgb = mpl.colormaps[cmap](np.clip(v, 0, 1))[:, :3]
    alpha = np.clip((r2 - R2_RAMP[0]) / (R2_RAMP[1] - R2_RAMP[0]), 0, 1)
    if par != 'r2':
        alpha = alpha * (ecc <= ECC_MAX_SHOWN)
    curv = cortex.db.get_surfinfo(cx_subject, smooth=CURV['smooth']).data
    grey = (curv > 0) * CURV['contrast'] + CURV['brightness']
    out = alpha[:, None] * rgb + (1 - alpha[:, None]) * grey[:, None]
    out[border] = 1.
    return (out * 255).astype(np.uint8)


def headless_chrome(url):
    url = 'http://localhost:' + url.rsplit(':', 1)[1]
    profile = tempfile.mkdtemp()
    return subprocess.Popen(
        [CHROME, '--headless=new', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
         '--ignore-gpu-blocklist', f'--user-data-dir={profile}',
         f'--window-size={IMAGE_SIZE[0]},{IMAGE_SIZE[1]}', url],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def render(ds, cx_subject, out_dir, stem, views, wait):
    procs = []
    webbrowser.open = lambda url, *a, **k: procs.append(headless_chrome(url)) or True
    handle = cortex.webgl.show(ds, open_browser=True, autoclose=False)
    try:   # headless Chrome must not outlive a failed render
        time.sleep(wait)
        for view in views:
            fn = (out_dir / f'{stem}_{view}.png').resolve()
            if fn.exists():
                print('exists', fn, flush=True)
                continue
            # Until the surface has loaded the viewer ignores the view, silently.
            want = VIEWS[view]['camera.azimuth']
            t0 = time.time()
            while True:
                try:
                    handle._set_view(**{**BASE_VIEW, **VIEWS[view]})
                except (TypeError, IndexError, KeyError):   # UI not up yet
                    pass
                time.sleep(4)
                try:
                    got = handle.ui.get('camera.azimuth')[0]
                    if abs((got - want + 180) % 360 - 180) < 1e-3:   # 0 may read back as 360
                        break
                except (TypeError, IndexError, KeyError):   # UI not up yet
                    pass
                if time.time() - t0 > 180:
                    raise TimeoutError(f'{stem} {view}: viewer did not take the view')
            time.sleep(4)
            handle.getImage(str(fn), IMAGE_SIZE)
            t0 = time.time()
            while not fn.exists():
                if time.time() - t0 > 300:
                    raise TimeoutError(fn)
                time.sleep(1)
            time.sleep(2)
            print('wrote', fn, flush=True)
    finally:
        handle.server.stop()
        for p in procs:
            p.terminate()


def write_settings(out_dir):
    """Store the colour settings next to the renders (render_settings.json), so composing scripts draw
    keys that match the baked colours; refuse to mix renders made with different settings in one folder."""
    fn = out_dir / 'render_settings.json'
    new = {'prf_dir': PRF_DIR, 'R2_RAMP': list(R2_RAMP), 'PANELS': {k: list(v) for k, v in PANELS.items()},
           'ECC_MAX_SHOWN': ECC_MAX_SHOWN, 'CURV': CURV}
    if fn.exists():
        old = json.loads(fn.read_text())
        if old != new:
            raise ValueError(f'{fn} holds other colour settings ({old}); use another --out-dir')
    else:
        fn.write_text(json.dumps(new, indent=1))


def main(subject, out_dir, views, pars, wait, r2_ramp=R2_RAMP, prf_dir=PRF_DIR, r2_max=None):
    global R2_RAMP, PRF_DIR
    R2_RAMP, PRF_DIR = tuple(r2_ramp), prf_dir
    if r2_max is not None:
        PANELS['r2'] = (PANELS['r2'][0], 0, r2_max)
    store = Path(cortex.database.default_filestore)
    cx_subject = f'valuecapturesub{subject:02d}'
    alias = store / cx_subject
    if alias.exists() or alias.is_symlink():
        raise FileExistsError(alias)
    alias.symlink_to(store / f'valuecapture.sub-{subject:02d}')
    try:
        cortex.db.reload_subjects()
        has_flat = (store / cx_subject / 'surfaces' / 'flat_lh.gii').exists()
        BASE_VIEW['surface.{subject}.unfold'] = .5 if has_flat else 1
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_settings(out_dir)
        border = border_mask(subject, cx_subject)
        for par in pars:
            if all((out_dir / f'sub-{subject:02d}_{par}_{v}.png').exists() for v in views):
                continue
            rgb = (area_rgb(subject, cx_subject, border) if par == 'areas'
                   else vertex_rgb(subject, cx_subject, par, border))
            ds = cortex.Dataset(**{par: cortex.VertexRGB(rgb[:, 0], rgb[:, 1], rgb[:, 2],
                                                         cx_subject)})
            render(ds, cx_subject, out_dir, f'sub-{subject:02d}_{par}', views, wait)
    finally:
        alias.unlink()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('subject', type=int)
    ap.add_argument('--out-dir', default='notes/figures/prf_inflated')
    ap.add_argument('--views', nargs='+', default=['pole'])
    ap.add_argument('--pars', nargs='+', default=['areas'] + list(PANELS))
    ap.add_argument('--r2-ramp', nargs=2, type=float, default=R2_RAMP,
                    help='R² at which the colour starts to show and is fully opaque')
    ap.add_argument('--r2-max', type=float, default=None,
                    help=f'Top of the R² colour map (default {PANELS["r2"][2]})')
    ap.add_argument('--prf-dir', default=PRF_DIR, help='Fit folder under derivatives')
    ap.add_argument('--wait', type=float, default=45, help='Seconds for the viewer to load')
    a = ap.parse_args()
    main(a.subject, a.out_dir, a.views, a.pars, a.wait, a.r2_ramp, a.prf_dir, a.r2_max)
