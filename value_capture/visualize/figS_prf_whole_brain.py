"""Supplementary figure: whole-brain PRF R² maps of all 12 subjects on the inflated cortex.

Shows where well-fitted PRFs lie on the whole cortex and that they sit inside the retinotopic
visual areas (neuropythy inferred_varea + Wang 2015 IPS0-5/FEF; white lines = area borders).
Top row: one subject's visual areas (colour key below it); then one row per subject: R² of the
bar + search-items PRF fit (derivatives/prf_glmsingle_items) in six views: left lateral, left
medial, right medial, right lateral, ventral and posterior. Colour (inferno, 0 to .15) fades in
over the curvature from R² = .01 to .06. The colour bar reads the ramp and range from
render_settings.json, which render_prf_inflated.py writes next to the renders, so the key always
matches the baked colours.

Regenerate:

1. Renders (pycortex2 env, headless Chrome, one subject at a time, ~2 min each)::

    V="lateral_lh medial_lh medial_rh lateral_rh ventral_whole posterior"
    ~/mambaforge/envs/pycortex2/bin/python value_capture/visualize/pycortex/render_prf_inflated.py 4 \\
        --views $V --pars areas --out-dir notes/figures/prf_inflated_wholebrain
    for s in 4 5 6 7 8 9 10 11 12 13 14 15; do
        ~/mambaforge/envs/pycortex2/bin/python value_capture/visualize/pycortex/render_prf_inflated.py $s \\
            --views $V --pars r2 --r2-ramp .01 .06 --r2-max .15 \\
            --out-dir notes/figures/prf_inflated_wholebrain
    done

   (In zsh, word-split $V with ${=V} or run the loop in bash.)

2. This figure (any env with matplotlib + Pillow)::

    ~/mambaforge/bin/python value_capture/visualize/figS_prf_whole_brain.py

Writes notes/figures/figS_prf_whole_brain.{pdf,png}; legend in notes/figures/figS_prf_whole_brain.md.
"""
import argparse
import ast
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RENDER_SCRIPT = ROOT / 'value_capture' / 'visualize' / 'pycortex' / 'render_prf_inflated.py'
mpl.rcParams.update({
    'font.family': 'Helvetica', 'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'xtick.direction': 'out', 'xtick.major.size': 2,
    'xtick.major.width': .6, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300})
FG, MUTED = '#1F2933', '#7B8794'
MM = 1 / 25.4
VIEWS = [('lateral_lh', 'Left lateral'), ('medial_lh', 'Left medial'), ('medial_rh', 'Right medial'),
         ('lateral_rh', 'Right lateral'), ('ventral_whole', 'Ventral'), ('posterior', 'Posterior')]
# Two-hemisphere views: L / R marked in the column header.
LR_VIEWS = ('ventral_whole', 'posterior')


def render_constants():
    """AREA_COLORS (and default PANELS, R2_RAMP, CURV) from the render script, without importing cortex.
    The R² colour settings actually used are read per render folder (render_settings())."""
    out = {}
    for node in ast.parse(RENDER_SCRIPT.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and \
                node.targets[0].id in ('PANELS', 'R2_RAMP', 'CURV', 'AREA_COLORS'):
            v = node.value
            if isinstance(v, ast.Call) and getattr(v.func, 'id', '') == 'dict':   # CURV = dict(...)
                out[node.targets[0].id] = {k.arg: ast.literal_eval(k.value) for k in v.keywords}
            else:
                out[node.targets[0].id] = ast.literal_eval(v)
    return out


C = render_constants()


def render_settings(render_dir):
    """Colour settings the renders were made with (render_prf_inflated.write_settings)."""
    fn = render_dir / 'render_settings.json'
    if not fn.exists():
        raise FileNotFoundError(f'{fn}: re-render with the current render_prf_inflated.py')
    st = json.loads(fn.read_text())
    C.update(R2_RAMP=tuple(st['R2_RAMP']), CURV=st['CURV'], PANELS={k: tuple(v) for k, v in st['PANELS'].items()})
    return st


def alpha_box(path):
    a = np.asarray(Image.open(path).convert('RGBA'))[..., 3] > 0
    ys, xs = np.flatnonzero(a.any(1)), np.flatnonzero(a.any(0))
    return np.array([ys[0], ys[-1], xs[0], xs[-1]])


def view_sizes(render_dir, stems, margin=6):
    """Per view, the largest tight crop (h, w in pixels) over all subjects: every image of a column is
    drawn at one scale (pixels per mm), centred in its cell, so brain sizes stay comparable."""
    out = {}
    for v, _ in VIEWS:
        bs = np.array([alpha_box(p) for s in stems if (p := render_dir / f'{s}_{v}.png').exists()])
        out[v] = (int((bs[:, 1] - bs[:, 0]).max()) + 2 * margin, int((bs[:, 3] - bs[:, 2]).max()) + 2 * margin)
    return out


def show(ax, path, size):
    """Tight crop of one render, centred on a white canvas of the column's common size."""
    ax.set_axis_off()
    H, W = size
    canvas = np.ones((H, W, 3))
    if not path.exists():
        ax.imshow(canvas)
        ax.text(.5, .5, 'Missing', ha='center', va='center', transform=ax.transAxes, color=MUTED)
        return
    im = Image.open(path).convert('RGBA')
    y0, y1, x0, x1 = alpha_box(path)
    bg = Image.new('RGBA', im.size, 'white')
    bg.alpha_composite(im)
    crop = np.asarray(bg.convert('RGB'))[y0:y1 + 1, x0:x1 + 1] / 255.
    h, w = crop.shape[:2]
    oy, ox = (H - h) // 2, (W - w) // 2
    canvas[oy:oy + h, ox:ox + w] = crop
    ax.imshow(canvas, interpolation='antialiased')


def r2_key(ax):
    """R² colour bar exactly as baked: inferno over the curvature grey, fading in over R2_RAMP."""
    cmap, lo, hi = C['PANELS']['r2']
    r2 = np.linspace(lo, hi, 600)
    a = np.clip((r2 - C['R2_RAMP'][0]) / (C['R2_RAMP'][1] - C['R2_RAMP'][0]), 0, 1)[:, None]
    grey = C['CURV']['brightness'] + C['CURV']['contrast'] / 2
    rgb = a * mpl.colormaps[cmap](r2 / hi)[:, :3] + (1 - a) * grey
    ax.imshow(rgb[None], aspect='auto', extent=(lo, hi, 0, 1))
    ax.set_yticks([])
    ticks = [lo, C['R2_RAMP'][0], C['R2_RAMP'][1], hi]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{t:g}'.replace('0.', '.') for t in ticks])
    ax.tick_params(labelsize=6, pad=1.5, colors=FG)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title('Variance explained (R²)', fontsize=7, color=FG, pad=3)


def area_key(ax):
    ax.set_axis_off()
    names = list(C['AREA_COLORS'])
    step = 1 / len(names)
    for i, n in enumerate(names):
        x = i * step
        ax.add_patch(plt.Rectangle((x, .3), step * .28, .4, color=C['AREA_COLORS'][n][1],
                                   transform=ax.transAxes, clip_on=False, lw=0))
        ax.text(x + step * .33, .5, n, transform=ax.transAxes, va='center', fontsize=6.5, color=FG)


def main(render_dir, subjects, example, out, width_mm=180):
    render_settings(render_dir)
    stems = [f'sub-{s:02d}_r2' for s in subjects] + [f'sub-{example:02d}_areas']
    sizes = view_sizes(render_dir, stems)
    hmax = max(h for h, _ in sizes.values())
    # One row height; each column's width follows from its largest render at the shared scale
    asp = np.array([sizes[v][1] / hmax for v, _ in VIEWS])
    label_w = 9 * MM                       # subject labels at the row start
    W = width_mm * MM
    row_h = (W - label_w) / asp.sum()      # every row spans the full width
    head_h, key_h, gap = 4 * MM, 7 * MM, 1.8 * MM
    heights = [head_h, row_h, key_h, gap] + [row_h] * len(subjects) + [key_h + 4 * MM]
    H = sum(heights)
    fig = plt.figure(figsize=(W, H), facecolor='white')
    ys = H - np.cumsum(heights)            # bottom of each band (inches)

    def at(x, y, w, h):
        return fig.add_axes([x / W, y / H, w / W, h / H])

    xs = label_w + np.concatenate([[0], np.cumsum(asp * row_h)[:-1]])
    # Column titles
    for x, (v, title), a in zip(xs, VIEWS, asp):
        fig.text((x + a * row_h / 2) / W, (ys[0] + head_h * .2) / H, title, ha='center', va='bottom',
                 fontsize=7, color=FG)
        if v in LR_VIEWS:      # camera geometry: right hemisphere on the image right
            for dx, t, ha in ((.08, 'L', 'left'), (.92, 'R', 'right')):
                fig.text((x + a * row_h * dx) / W, (ys[0] + head_h * .2) / H, t, ha=ha, va='bottom',
                         fontsize=6.5, color=MUTED)

    def row(stem, y, label):
        for x, (v, _), a in zip(xs, VIEWS, asp):
            ax = at(x, y, a * row_h, row_h)
            ax.set_position([x / W, y / H, a * row_h / W, row_h / H])
            show(ax, render_dir / f'{stem}_{v}.png', (hmax, sizes[v][1]))
        fig.text(.5 * MM / W, (y + row_h / 2) / H, label, ha='left', va='center', fontsize=7, color=FG)

    row(f'sub-{example:02d}_areas', ys[1], f'Areas\nsub-{example:02d}')
    area_key(at(label_w + 2 * MM, ys[2] + 1 * MM, W - label_w - 4 * MM, key_h - 2 * MM))
    for i, s in enumerate(subjects):
        row(f'sub-{s:02d}_r2', ys[4 + i], f'sub-{s:02d}')
    r2_key(at(W * .3, ys[-1] + 4 * MM, W * .4, 1.6 * MM))
    for ext in ('pdf', 'png'):
        fig.savefig(out.with_suffix('.' + ext), facecolor='white')
    print('wrote', out.with_suffix('.pdf'), f'({W / MM:.0f} x {H / MM:.0f} mm)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--render-dir', default=ROOT / 'notes' / 'figures' / 'prf_inflated_wholebrain', type=Path)
    ap.add_argument('--subjects', nargs='+', type=int, default=list(range(4, 16)))
    ap.add_argument('--example', type=int, default=4, help='Subject whose visual areas form the top row')
    ap.add_argument('--out', default=ROOT / 'notes' / 'figures' / 'figS_prf_whole_brain.pdf', type=Path)
    a = ap.parse_args()
    main(a.render_dir, a.subjects, a.example, a.out)
