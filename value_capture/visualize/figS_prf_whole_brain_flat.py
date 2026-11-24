"""Supplementary figure: PRF R² of all 12 subjects on their whole-cortex flatmaps.

Companion of figS_prf_whole_brain.py (inflated views). R² of the bar + search-items PRF fit
(derivatives/prf_glmsingle_items) on each subject's flattened cortex (both hemispheres; autoflatten
patches imported with visualize/pycortex/import_flatmaps.py), over the curvature, with the visual-area
borders in white (neuropythy inferred_varea + Wang 2015 IPS0-5/FEF). Colours, curvature and borders are
baked per vertex exactly as for the inflated renders (render_prf_inflated.vertex_rgb / area_rgb) and
sampled onto the flatmap with cortex.quickflat (nearest vertex per pixel; no browser).

Two steps, both in the pycortex2 env, one subject at a time::

    # 1. Flatmap images -> notes/figures/prf_flat_wholebrain/ (+ render_settings.json)
    ~/mambaforge/envs/pycortex2/bin/python value_capture/visualize/figS_prf_whole_brain_flat.py render \\
        --r2-ramp .005 .03 --r2-max .1
    # 2. The figure -> notes/figures/figS_prf_whole_brain_flat.{pdf,png}
    ~/mambaforge/envs/pycortex2/bin/python value_capture/visualize/figS_prf_whole_brain_flat.py compose

The colour bar reads the ramp and range from render_settings.json, so it matches the baked colours.
Legend: notes/figures/figS_prf_whole_brain_flat.md.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RENDER_DIR = ROOT / 'notes' / 'figures' / 'prf_flat_wholebrain'
OUT = ROOT / 'notes' / 'figures' / 'figS_prf_whole_brain_flat.pdf'
SUBJECTS = list(range(4, 16))
EXAMPLE = 4
HEIGHT = 1024                     # flatmap image height (px); quickflat caches the pixel map per height
mpl.rcParams.update({
    'font.family': 'Helvetica', 'font.size': 7, 'axes.titlesize': 7, 'xtick.labelsize': 6.5,
    'xtick.direction': 'out', 'xtick.major.size': 2, 'xtick.major.width': .6,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300})
FG, MUTED = '#1F2933', '#7B8794'
MM = 1 / 25.4


def rpi():
    sys.path.insert(0, str(HERE / 'pycortex'))
    import render_prf_inflated
    return render_prf_inflated


# ── step 1: flatmap images ────────────────────────────────────────────────────

def render(subjects, out_dir, r2_ramp, r2_max, example=EXAMPLE, height=HEIGHT):
    import cortex
    from cortex.quickflat.utils import make_flatmap_image
    r = rpi()
    r.R2_RAMP = tuple(r2_ramp)
    r.PANELS['r2'] = (r.PANELS['r2'][0], 0, r2_max)
    out_dir.mkdir(parents=True, exist_ok=True)
    r.write_settings(out_dir)
    store = Path(cortex.database.default_filestore)
    for s in subjects:
        cx = f'valuecapture.sub-{s:02d}'
        if not (store / cx / 'surfaces' / 'flat_rh.gii').exists():
            print(f'{cx}: no flatmap, skipped', flush=True)
            continue
        border = r.border_mask(s, cx)
        pars = ['r2', 'areas'] if s == example else ['r2']
        for par in pars:
            fn = out_dir / f'sub-{s:02d}_{par}_flat.png'
            if fn.exists():
                print('exists', fn)
                continue
            rgb = r.area_rgb(s, cx, border) if par == 'areas' else r.vertex_rgb(s, cx, par, border)
            vx = cortex.VertexRGB(rgb[:, 0], rgb[:, 1], rgb[:, 2], cx)
            img, extents = make_flatmap_image(vx, height=height)
            Image.fromarray(img).save(fn)
            np.save(fn.with_suffix('.extents.npy'), np.asarray(extents))
            print('wrote', fn, flush=True)
        if s == example:
            area_label_positions(r, s, cx, out_dir / f'sub-{s:02d}_area_labels.json')


def area_label_positions(r, s, cx, fn):
    """Flat (x, y) of one vertex per area and hemisphere, near the area's median position, for labels."""
    import cortex
    pts, _ = cortex.db.get_surf(cx, 'flat', merge=True, nudge=True)
    va = r.read_varea(s)
    n_lh = len(r.read_areas(s, 'lh'))
    out = []
    for name, (codes, _) in r.AREA_COLORS.items():
        for h, sl in (('lh', slice(0, n_lh)), ('rh', slice(n_lh, None))):
            idx = np.flatnonzero(np.isin(va[sl], codes)) + sl.indices(len(va))[0]
            if len(idx) == 0:
                continue
            xy = pts[idx, :2]
            med = np.median(xy, 0)
            out.append({'area': name, 'hemi': h, 'xy': xy[np.argmin(((xy - med) ** 2).sum(1))].tolist()})
    fn.write_text(json.dumps(out, indent=1))


# ── step 2: figure ────────────────────────────────────────────────────────────

def rpi_area_colors():
    """AREA_COLORS from the render script without importing cortex (compose step needs no pycortex)."""
    import ast
    for node in ast.parse((HERE / 'pycortex' / 'render_prf_inflated.py').read_text()).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') == 'AREA_COLORS':
            return ast.literal_eval(node.value)


def load_settings(render_dir):
    st = json.loads((render_dir / 'render_settings.json').read_text())
    return st


def flat_image(path):
    """RGBA flatmap on white, cropped to the cortex."""
    im = Image.open(path).convert('RGBA')
    bg = Image.new('RGBA', im.size, 'white')
    bg.alpha_composite(im)
    a = np.asarray(im)[..., 3] > 0
    ys, xs = np.flatnonzero(a.any(1)), np.flatnonzero(a.any(0))
    return np.asarray(bg.convert('RGB'))[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1], (ys[0], ys[-1], xs[0], xs[-1]), a.shape


def show_flat(ax, path):
    ax.set_axis_off()
    if not path.exists():
        ax.text(.5, .5, 'Missing', ha='center', va='center', transform=ax.transAxes, color=MUTED)
        return None
    img, box, shape = flat_image(path)
    ax.imshow(img, interpolation='antialiased')
    return box, shape


def r2_key(ax, st):
    cmap, lo, hi = st['PANELS']['r2']
    ramp, curv = st['R2_RAMP'], st['CURV']
    r2 = np.linspace(lo, hi, 600)
    a = np.clip((r2 - ramp[0]) / (ramp[1] - ramp[0]), 0, 1)[:, None]
    grey = curv['brightness'] + curv['contrast'] / 2
    rgb = a * mpl.colormaps[cmap](r2 / hi)[:, :3] + (1 - a) * grey
    ax.imshow(rgb[None], aspect='auto', extent=(lo, hi, 0, 1))
    ax.set_yticks([])
    ticks = [lo, ramp[0], ramp[1], hi]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{t:g}'.replace('0.', '.') for t in ticks])
    ax.tick_params(labelsize=6, pad=1.5, colors=FG)
    if (ramp[0] - lo) / (hi - lo) < .1:          # 0 and the ramp start sit close: push the labels apart
        labels = ax.get_xticklabels()
        labels[0].set_horizontalalignment('right')
        labels[1].set_horizontalalignment('left')
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title('Variance explained (R²)', fontsize=7, color=FG, pad=3)


def label_areas(ax, render_dir, box, shape, example):
    """Direct area labels on the example flatmap (flat coordinates -> cropped image pixels)."""
    fn = render_dir / f'sub-{example:02d}_area_labels.json'
    ext = np.load(render_dir / f'sub-{example:02d}_areas_flat.extents.npy')   # xmin, xmax, ymin, ymax
    if not fn.exists():
        return
    from matplotlib import patheffects
    colors = {k: v[1] for k, v in rpi_area_colors().items()}
    H, W = shape
    for d in json.loads(fn.read_text()):
        x, y = d['xy']
        px = (x - ext[0]) / (ext[1] - ext[0]) * (W - 1) - box[2]
        py = (ext[3] - y) / (ext[3] - ext[2]) * (H - 1) - box[0]
        rgb = np.array(mpl.colors.to_rgb(colors[d['area']]))
        dark = rgb @ [.299, .587, .114] < .45
        kw = dict(ha='center', va='center', fontsize=5.5, weight='bold')
        if d['area'] == 'FEF':          # a few mm across: label beside it
            ax.annotate('FEF', (px, py), xytext=(0, 7), textcoords='offset points', color=FG,
                        arrowprops=dict(arrowstyle='-', lw=.5, color=FG, shrinkA=0, shrinkB=2), **kw)
            continue
        ax.text(px, py, d['area'], color='white' if dark else FG,
                path_effects=[patheffects.withStroke(linewidth=1.2, foreground=colors[d['area']])], **kw)


def compose(render_dir, subjects, example, out, width_mm=180, ncol=3):
    st = load_settings(render_dir)
    r2_files = [render_dir / f'sub-{s:02d}_r2_flat.png' for s in subjects]
    asps = []
    for f in r2_files:
        if f.exists():
            img = flat_image(f)[0]
            asps.append(img.shape[1] / img.shape[0])
    asp = float(np.median(asps))                         # width / height of a flatmap panel
    W = width_mm * MM
    gap_x, gap_y, title_h = 2 * MM, 1.5 * MM, 3.5 * MM
    cell_w = (W - (ncol - 1) * gap_x) / ncol
    cell_h = cell_w / asp
    nrow = int(np.ceil(len(subjects) / ncol))
    top_w = 2 * cell_w + gap_x                           # example areas map spans two columns
    top_h = top_w / asp
    top_title_h = 2 * title_h                            # title line + hemisphere line
    H = top_title_h + top_h + 3 * MM + nrow * (title_h + cell_h) + (nrow - 1) * gap_y + 1 * MM
    fig = plt.figure(figsize=(W, H), facecolor='white')

    def at(x, y_top, w, h):            # y measured from the top (inches)
        return fig.add_axes([x / W, (H - y_top - h) / H, w / W, h / H])

    # Example subject: visual areas, labelled
    ax = at(0, top_title_h, top_w, top_h)
    res = show_flat(ax, render_dir / f'sub-{example:02d}_areas_flat.png')
    if res:
        label_areas(ax, render_dir, *res, example)
    fig.text(0, (H - title_h * .3) / H, f'Visual areas (sub-{example:02d})', va='top', fontsize=7, color=FG,
             weight='bold')
    for x, t in ((.25, 'Left hemisphere'), (.75, 'Right hemisphere')):
        fig.text(x * top_w / W, (H - title_h * 1.2) / H, t, ha='center', va='top', fontsize=6.5, color=MUTED)
    # Keys in the third column
    xk = top_w + gap_x
    r2_key(at(xk + 4 * MM, top_title_h + top_h * .3, cell_w - 8 * MM, 1.8 * MM), st)
    fig.text((xk + cell_w / 2) / W, (H - top_title_h - top_h * .62) / H,
             'White lines: visual-area borders\nGrey: curvature (light gyri, dark sulci)',
             ha='center', va='top', fontsize=6, color=FG, linespacing=1.4)
    # Subjects
    y0 = top_title_h + top_h + 3 * MM
    for k, (s, f) in enumerate(zip(subjects, r2_files)):
        i, j = divmod(k, ncol)
        y = y0 + i * (title_h + cell_h + gap_y)
        ax = at(j * (cell_w + gap_x), y + title_h, cell_w, cell_h)
        show_flat(ax, f)
        fig.text(j * (cell_w + gap_x) / W, (H - y - title_h * .25) / H, f'sub-{s:02d}', va='top', fontsize=7,
                 color=FG)
    for ext in ('pdf', 'png'):
        fig.savefig(out.with_suffix('.' + ext), facecolor='white')
    print('wrote', out, f'({W / MM:.0f} x {H / MM:.0f} mm)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['render', 'compose'])
    ap.add_argument('--render-dir', default=RENDER_DIR, type=Path)
    ap.add_argument('--subjects', nargs='+', type=int, default=SUBJECTS)
    ap.add_argument('--example', type=int, default=EXAMPLE)
    ap.add_argument('--r2-ramp', nargs=2, type=float, default=(.005, .03))
    ap.add_argument('--r2-max', type=float, default=.1)
    ap.add_argument('--out', default=OUT, type=Path)
    a = ap.parse_args()
    if a.step == 'render':
        render(a.subjects, a.render_dir, a.r2_ramp, a.r2_max, a.example)
    else:
        compose(a.render_dir, a.subjects, a.example, a.out)
