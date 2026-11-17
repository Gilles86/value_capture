"""Compose the pycortex PRF renders (render_prf_inflated.py) into figures with colour keys.

1. Per subject: rows = views, columns = visual areas, polar angle, eccentricity, PRF size, R².
2. Cohort sheets: one panel per subject for one parameter and view (e.g. polar angle, pole).

Renders are cropped to the brain (black background kept, as in the viewer). Colour ranges are
read from render_prf_inflated.py so the keys always match the baked colours. Runs anywhere with
matplotlib (no pycortex import)::

    python value_capture/visualize/pycortex/compose_prf_renders.py \\
        --render-dir notes/figures/prf_inflated --subjects 4 5 6 7 8 9 10 11 12 13 14 15
"""
import argparse
import ast
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
mpl.rcParams.update({
    'font.family': 'Helvetica', 'font.size': 7, 'axes.labelsize': 7, 'pdf.fonttype': 42,
    'savefig.facecolor': 'black', 'text.color': 'w', 'axes.edgecolor': 'w',
    'xtick.color': 'w', 'ytick.color': 'w', 'xtick.major.size': 2, 'xtick.major.width': .6,
    'axes.linewidth': .6})


def render_constants():
    """PANELS, R2_RAMP, ECC_MAX_SHOWN and AREA_COLORS from the render script, without importing cortex."""
    tree = ast.parse((HERE / 'render_prf_inflated.py').read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and \
                node.targets[0].id in ('PANELS', 'R2_RAMP', 'ECC_MAX_SHOWN', 'AREA_COLORS'):
            out[node.targets[0].id] = ast.literal_eval(node.value)
    return out


C = render_constants()
TITLES = {'areas': 'Visual areas', 'angle': 'Polar angle', 'ecc': 'Eccentricity',
          'sd': 'PRF size (σ)', 'r2': 'Variance explained'}
VIEW_LABELS = {'occipital': 'Posterior-ventral', 'pole2': 'Posterior', 'ventral': 'Ventral',
               'pole': 'Posterior', 'medial': 'Medial'}
UNITS = {'ecc': '°', 'sd': '°', 'r2': 'R²'}


def load_cropped(fn, box):
    im = plt.imread(fn)[..., :3]
    y0, y1, x0, x1 = box
    return im[y0:y1, x0:x1]


def crop_box(files, margin=12):
    """One bounding box of non-black pixels over a set of renders, so panels share a scale."""
    box = None
    for f in files:
        if not f.exists():
            continue
        m = plt.imread(f)[..., :3].max(-1) > .03
        ys, xs = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
        b = np.array([ys[0], ys[-1], xs[0], xs[-1]])
        box = b if box is None else np.array([min(box[0], b[0]), max(box[1], b[1]),
                                              min(box[2], b[2]), max(box[3], b[3])])
    h, w = plt.imread(next(f for f in files if f.exists())).shape[:2]
    return (max(box[0] - margin, 0), min(box[1] + margin, h), max(box[2] - margin, 0),
            min(box[3] + margin, w))


def key(ax, par):
    """Colour key under a column: polar-angle wheel, area swatches, or a colorbar."""
    ax.set_axis_off()
    if par == 'angle':
        # Right hemisphere codes the left visual field and vice versa; hue as theta_value().
        sub = ax.inset_axes([.3, -.15, .4, 1.2], projection='polar')
        th = np.linspace(0, 2 * np.pi, 721)
        r = np.linspace(.45, 1, 2)
        T, R = np.meshgrid(th, r)
        # Left visual field is coded by the right hemisphere (theta_value 'R'), right field by 'L'.
        v = np.where(np.cos(T) < 0,
                     np.clip((np.mod(T, 2 * np.pi) - .25 * np.pi) / (1.5 * np.pi), 0, 1),
                     1 - np.clip((np.mod(T + np.pi, 2 * np.pi) - .25 * np.pi) / (1.5 * np.pi), 0, 1))
        sub.pcolormesh(T, R, v, cmap='hsv', vmin=0, vmax=1, shading='auto')
        sub.set_axis_off()
        sub.set_ylim(0, 1)
        sub.text(np.pi / 2, 1.35, 'Upper', ha='center', va='center', fontsize=5.5)
        sub.text(-np.pi / 2, 1.4, 'Lower', ha='center', va='center', fontsize=5.5)
        sub.text(np.pi, 1.6, 'Left', ha='center', va='center', fontsize=5.5)
        sub.text(0, 1.6, 'Right', ha='center', va='center', fontsize=5.5)
        return
    if par == 'areas':
        names = list(C['AREA_COLORS'])
        for i, n in enumerate(names):
            col, row = i % 3, i // 3
            x, y = .06 + col * .32, .8 - row * .3
            ax.add_patch(plt.Rectangle((x, y - .09), .07, .18, color=C['AREA_COLORS'][n][1],
                                       transform=ax.transAxes, clip_on=False))
            ax.text(x + .1, y, n, transform=ax.transAxes, va='center', fontsize=6)
        return
    cmap, lo, hi = C['PANELS'][par]
    cb = ax.inset_axes([.12, .55, .76, .16])
    cb.imshow(np.linspace(0, 1, 256)[None], cmap=cmap, aspect='auto', extent=(lo, hi, 0, 1))
    cb.set_yticks([])
    cb.set_xticks([lo, (lo + hi) / 2, hi])
    cb.tick_params(labelsize=6, pad=1.5)
    for s in cb.spines.values():
        s.set_visible(False)
    cb.set_xlabel(UNITS.get(par, ''), fontsize=6, labelpad=1)


def subject_figure(render_dir, subject, pars, views, out):
    files = [render_dir / f'sub-{subject:02d}_{p}_{v}.png' for p in pars for v in views]
    boxes = {v: crop_box([render_dir / f'sub-{subject:02d}_{p}_{v}.png' for p in pars]) for v in views}
    asp = {v: (b[1] - b[0]) / (b[3] - b[2]) for v, b in boxes.items()}
    col_w = 1.45
    heights = [col_w * asp[v] for v in views] + [.45]
    fig, axes = plt.subplots(len(views) + 1, len(pars), figsize=(col_w * len(pars), sum(heights) + .25),
                             gridspec_kw=dict(height_ratios=heights, hspace=.02, wspace=.02),
                             facecolor='black')
    axes = np.atleast_2d(axes)
    for j, p in enumerate(pars):
        for i, v in enumerate(views):
            ax = axes[i, j]
            ax.set_axis_off()
            fn = render_dir / f'sub-{subject:02d}_{p}_{v}.png'
            if fn.exists():
                ax.imshow(load_cropped(fn, boxes[v]))
            if i == 0:
                ax.set_title(TITLES[p], fontsize=7.5, color='w', pad=3)
        key(axes[-1, j], p)
    for i, v in enumerate(views):
        axes[i, 0].text(-.04, .5, VIEW_LABELS.get(v, v.capitalize()), transform=axes[i, 0].transAxes, rotation=90,
                        ha='right', va='center', fontsize=7)
    axes[0, 0].text(-.12, 1.13, f'sub-{subject:02d}', transform=axes[0, 0].transAxes, fontsize=8.5,
                    weight='bold', va='bottom')
    note = (f'Colour fades in from R² = {C["R2_RAMP"][0]:.2f} to {C["R2_RAMP"][1]:.2f}; '
            f'maps other than R² only within {C["ECC_MAX_SHOWN"]:.1f}° (stimulus aperture). '
            'White lines: neuropythy area borders.')
    fig.text(.01, .002, note, fontsize=5.5, color='.75', ha='left', va='bottom')
    fig.savefig(out, dpi=300, bbox_inches='tight', pad_inches=.04)
    plt.close(fig)
    print('wrote', out)


def cohort_sheet(render_dir, subjects, par, view, out, ncol=4):
    files = [render_dir / f'sub-{s:02d}_{par}_{view}.png' for s in subjects]
    box = crop_box(files)
    asp = (box[1] - box[0]) / (box[3] - box[2])
    nrow = int(np.ceil(len(subjects) / ncol))
    w = 1.75
    fig, axes = plt.subplots(nrow + 1, ncol, figsize=(w * ncol, w * asp * nrow + .55),
                             gridspec_kw=dict(height_ratios=[w * asp] * nrow + [.45],
                                              hspace=.06, wspace=.02), facecolor='black')
    for ax in axes.ravel():
        ax.set_axis_off()
    for k, (s, f) in enumerate(zip(subjects, files)):
        ax = axes[k // ncol, k % ncol]
        if f.exists():
            ax.imshow(load_cropped(f, box))
        ax.text(.03, .97, f'sub-{s:02d}', transform=ax.transAxes, va='top', fontsize=6.5)
    key(axes[-1, 0], par)
    fig.suptitle(f'{TITLES[par]} ({VIEW_LABELS.get(view, view).lower()} view)', x=.01, ha='left', fontsize=8, color='w', y=.995)
    fig.savefig(out, dpi=300, bbox_inches='tight', pad_inches=.04)
    plt.close(fig)
    print('wrote', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--render-dir', default='notes/figures/prf_inflated')
    ap.add_argument('--subjects', nargs='+', type=int, default=list(range(4, 16)))
    ap.add_argument('--pars', nargs='+', default=['areas', 'angle', 'ecc', 'sd', 'r2'])
    ap.add_argument('--views', nargs='+', default=['occipital', 'pole2', 'ventral'])
    a = ap.parse_args()
    rd = Path(a.render_dir)
    for s in a.subjects:
        subject_figure(rd, s, a.pars, a.views, rd / f'sub-{s:02d}_prf_overview.png')
    for p in a.pars:
        for v in a.views:
            cohort_sheet(rd, a.subjects, p, v, rd / f'cohort_{p}_{v}.png')
