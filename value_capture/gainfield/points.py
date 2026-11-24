"""The points display as a stimulus: its image, its delay after the search display, and the expected leak.

The points (``+8``, ``+850``, ``+0`` + ``Incorrect!``/``Too late!``) appear at fixation for 1 s, 0.5-4 s after the
1.75 s search display (experiment/trial.py, phase 4; height = 1.5 x size_fixation, default font). Their response leaks
into the search beta even when GLMsingle models them, in proportion to the overlap of the two HRF-convolved events
(``check_points_leak.py``: leak ~ overlap in 12/12 subjects, plus more characters = more leak).
"""
from functools import lru_cache

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from nilearn.glm.first_level import compute_regressor

SEARCH_DUR, POINTS_DUR = 1.75, 1.0
TEXT_HEIGHT = .25 * 1.5          # size_fixation (deg) x 1.5, trial.py
LINE_SPACING = 1.2
# Foveal patch that holds the points layer (deg); 'Incorrect!' is the widest string, ~2 deg wide on 2 lines
PATCH = dict(x=1.5, y=.9)


def overlap(delay, dt=.05):
    """Regression weight of a points regressor (onset ``delay`` s after the search onset) on the search regressor."""
    t = np.arange(0, 40, dt)
    s = compute_regressor([[0.], [SEARCH_DUR], [1.]], 'spm', t)[0][:, 0]
    p = compute_regressor([[delay], [POINTS_DUR], [1.]], 'spm', t)[0][:, 0]
    return float(s @ p / (s @ s))


def points_text(row):
    if pd.isna(row.correct) or row.correct in ('', 'nan'):
        return '+0\nToo late!'
    if not (row.correct is True or str(row.correct) in ('True', '1', '1.0')):
        return '+0\nIncorrect!'
    return f'+{int(row.earned_points)}'


@lru_cache(maxsize=None)
def _render(text, grid_key, supersample=4):
    """Fraction of each pixel covered by ink (pixel size and centres from ``grid_key``)."""
    xs, ys, step = grid_key
    k = supersample
    off = (np.arange(k) + .5) / k * step - step / 2
    fx = (np.asarray(xs)[:, None] + off[None]).ravel()
    fy = (np.asarray(ys)[:, None] + off[None]).ravel()
    FX, FY = np.meshgrid(fx, fy)
    pts = np.c_[FX.ravel(), FY.ravel()]
    ink = np.zeros(len(pts), bool)
    lines = text.split('\n')
    prop = FontProperties(family='DejaVu Sans')
    for i, line in enumerate(lines):
        tp = TextPath((0, 0), line, size=TEXT_HEIGHT, prop=prop)
        ext = tp.get_extents()
        y0 = (len(lines) - 1) / 2 * TEXT_HEIGHT * LINE_SPACING - i * TEXT_HEIGHT * LINE_SPACING
        dx, dy = -(ext.x0 + ext.x1) / 2, y0 - (ext.y0 + ext.y1) / 2
        verts = tp.vertices + [dx, dy]
        path = MplPath(verts, tp.codes)
        for poly in path.to_polygons():
            ink ^= MplPath(poly).contains_points(pts)          # even-odd: holes in 0, 6, 8, 9
    img = ink.reshape(len(ys) * k, len(xs) * k).reshape(len(ys), k, len(xs), k).mean((1, 3))
    return img.ravel().astype(np.float32)


def patch_coordinates(grid):
    """The pixels of ``grid`` (DataFrame x, y) inside the foveal points patch."""
    m = (grid.x.abs() <= PATCH['x']) & (grid.y.abs() <= PATCH['y'])
    return grid[m].reset_index(drop=True)


def points_images(trials, patch):
    """(trials, patch pixels): the points display of each trial scaled by its overlap, 0 without points."""
    xs, ys = np.sort(patch.x.unique()), np.sort(patch.y.unique())
    step = float(np.diff(xs).mean())
    key = (tuple(np.round(xs, 6)), tuple(np.round(ys, 6)), round(step, 6))
    out = np.zeros((len(trials), len(patch)), np.float32)
    for i, row in enumerate(trials.itertuples()):
        if row.show_feedback:
            out[i] = _render(points_text(row), key) * overlap(row.delay)
    return out


def add_points_info(sub, trials):
    """Merge delay (feedback - search onset), correct and earned_points into ``trials`` (session, run, trial_nr)."""
    ev = []
    for (s, r), _ in trials.groupby(['session', 'run']):
        o = sub.get_onsets(int(s), int(r))
        t = o[o.event_type == 'target'].drop_duplicates('trial_nr').set_index('trial_nr').onset
        f = o[o.event_type == 'feedback'].drop_duplicates('trial_nr').set_index('trial_nr').onset
        ev.append(pd.DataFrame({'session': int(s), 'run': int(r), 'delay': f - t}).rename_axis('trial_nr').reset_index())
    beh = sub.get_behavioral_data(exclude_outlier_rts=False).reset_index()
    beh[['session', 'run']] = beh[['session', 'run']].astype(int)
    out = (trials.drop(columns=['delay', 'correct', 'earned_points'], errors='ignore')
           .merge(pd.concat(ev), on=['session', 'run', 'trial_nr'], how='left', validate='one_to_one')
           .merge(beh[['session', 'run', 'trial_nr', 'correct', 'earned_points']], on=['session', 'run', 'trial_nr'],
                  how='left', validate='one_to_one'))
    assert out.delay[out.show_feedback.astype(bool)].notna().all()
    return out


def ink_map(trials, grid):
    """Mean ink of the points texts shown to this subject on ``grid`` (n_pixels,), 0 outside the points patch.
    ``trials`` needs show_feedback, correct and earned_points (``add_points_info``)."""
    patch = patch_coordinates(grid)
    xs, ys = np.sort(patch.x.unique()), np.sort(patch.y.unique())
    key = (tuple(np.round(xs, 6)), tuple(np.round(ys, 6)), round(float(np.diff(xs).mean()), 6))
    shown = trials[trials.show_feedback.astype(bool)]
    img = np.mean([_render(points_text(r), key) for r in shown.itertuples()], 0)
    out = np.zeros(len(grid), np.float32)
    inside = (grid.x.abs() <= PATCH['x']) & (grid.y.abs() <= PATCH['y'])
    out[np.flatnonzero(inside.to_numpy())] = img
    return out


def text_overlap(ink, grid, x, y, sd):
    """Fraction of each Gaussian pRF (x, y, sd arrays) that falls on the points ink (``ink_map``).
    Voxels below ~.003 carry no timing-dependent points leak (check_points_leak_overlap.py)."""
    gx, gy = grid.x.to_numpy()[None], grid.y.to_numpy()[None]
    out = np.empty(len(x), np.float32)
    for i in range(0, len(x), 2000):
        sl = slice(i, i + 2000)
        rf = np.exp(-((gx - np.asarray(x)[sl, None]) ** 2 + (gy - np.asarray(y)[sl, None]) ** 2)
                    / (2 * np.asarray(sd)[sl, None] ** 2))
        out[sl] = (rf * ink[None]).sum(1) / rf.sum(1)
    return out
