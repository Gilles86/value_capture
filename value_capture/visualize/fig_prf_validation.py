"""PRF validation figures (after retsupp Fig. 2), for the bar + search-items PRF fits.

Main figure:
  a, One subject's visual areas, polar angle and eccentricity on the inflated occipital cortex.
  b, Share of PRF centres in the contralateral visual field per ROI (subject dots).
  c, PRF size against eccentricity per ROI.
  d, R² gained by putting the search items into the stimulus model (bar + items vs bar only).
  e, Visual-field coverage: PRF centres of all subjects per pathway, with the search items and the bar aperture.

Supplement (one page): every subject, one row each: visual areas, polar angle, eccentricity, R² and the
PRF centres in the visual field.

Inputs: notes/data/prf_surface_vertices.tsv (prf/collect_prf_surface.py, on the cluster) and the pycortex renders
in notes/figures/prf_inflated_items (visualize/pycortex/render_prf_inflated.py --prf-dir prf_glmsingle_items).

    python value_capture/visualize/fig_prf_validation.py --subject 7
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
mpl.rcParams.update({
    'font.family': 'Helvetica', 'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.linewidth': .8, 'axes.spines.top': False,
    'axes.spines.right': False, 'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.major.size': 3,
    'ytick.major.size': 3, 'xtick.major.width': .8, 'ytick.major.width': .8, 'lines.linewidth': 1.2,
    'pdf.fonttype': 42, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': .03})
FG, MUTED = '#1F2933', '#9AA5B1'
ROI_ORDER = ['V1', 'V2', 'V3', 'hV4', 'VO', 'LO', 'TO', 'V3AB']
ROI_COLORS = {'V1': '#08519C', 'V2': '#3182BD', 'V3': '#7FB8E0', 'hV4': '#C2185B', 'VO': '#E8590C',
              'LO': '#F2A60D', 'TO': '#B04FB5', 'V3AB': '#1A7F3C'}     # = render_prf_inflated.AREA_COLORS
ROI_X = dict(zip(ROI_ORDER, [0, 1, 2, 3.4, 4.4, 5.4, 6.4, 7.8]))       # gaps between early / ventral+lateral / dorsal
PATHWAYS = {'Early (V1–V3)': ('V1', 'V2', 'V3'), 'Ventral (hV4, VO)': ('hV4', 'VO'),
            'Lateral (LO, TO)': ('LO', 'TO'), 'Dorsal (V3AB)': ('V3AB',)}
R2_MIN = .1
APERTURE = 3.17                       # bar aperture radius (deg)
ITEM_ECC, ITEM_LONG, ITEM_SHORT = 4.0, 1.5, .375
ECC_MAX = 5.0                         # stimulus grid radius
MAPS = [('areas', 'Visual areas'), ('angle', 'Polar angle'), ('ecc', 'Eccentricity'), ('r2', 'R²')]


def offset(ax):
    for s in ('left', 'bottom'):
        ax.spines[s].set_position(('outward', 3))


def letter(ax, s, x=-.18, y=1.1):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, weight='bold', va='bottom', color=FG)


def roi_axis(ax):
    ax.set_xticks([ROI_X[r] for r in ROI_ORDER])
    ax.set_xticklabels(ROI_ORDER, rotation=0)
    for t, r in zip(ax.get_xticklabels(), ROI_ORDER):
        t.set_color(ROI_COLORS[r])
    ax.set_xlim(-.6, ROI_X['V3AB'] + .6)


def load(tsv):
    d = pd.read_csv(tsv, sep='\t', dtype={'subject': str})
    d['in_grid'] = d.ecc < ECC_MAX * .98
    return d


def good(d):
    return d[(d.r2 >= R2_MIN) & d.in_grid]


# ── renders ───────────────────────────────────────────────────────────────────

def crop_box(paths, margin=6):
    box = None
    for p in paths:
        if not p.exists():
            continue
        a = np.asarray(Image.open(p).convert('RGBA'))[..., 3] > 0
        ys, xs = np.flatnonzero(a.any(1)), np.flatnonzero(a.any(0))
        b = [ys[0], ys[-1], xs[0], xs[-1]]
        box = b if box is None else [min(box[0], b[0]), max(box[1], b[1]), min(box[2], b[2]), max(box[3], b[3])]
    return box[0] - margin, box[1] + margin, box[2] - margin, box[3] + margin


def show(ax, path, box):
    ax.set_axis_off()
    if not path.exists():
        ax.text(.5, .5, 'missing', ha='center', va='center', transform=ax.transAxes, color=MUTED)
        return
    im = Image.open(path).convert('RGBA')
    bg = Image.new('RGBA', im.size, 'white')
    bg.alpha_composite(im)
    y0, y1, x0, x1 = box
    ax.imshow(np.asarray(bg.convert('RGB'))[y0:y1, x0:x1], interpolation='antialiased')


def angle_key(ax):
    """The visual field coloured as mapped on the contralateral hemisphere (render_prf_inflated.theta_value)."""
    g = np.linspace(-1, 1, 201)
    x, y = np.meshgrid(g, g)
    th = np.arctan2(y, x)
    v = np.where(x < 0, np.clip((np.mod(th, 2 * np.pi) - .25 * np.pi) / (1.5 * np.pi), 0, 1),
                 1 - np.clip((np.mod(th + np.pi, 2 * np.pi) - .25 * np.pi) / (1.5 * np.pi), 0, 1))
    rgba = mpl.colormaps['hsv'](v)
    rgba[..., 3] = (np.hypot(x, y) <= 1).astype(float)
    ax.imshow(rgba, extent=(-1, 1, -1, 1), origin='lower')
    ax.plot([-1, 1], [0, 0], color='w', lw=.4)
    ax.plot([0, 0], [-1, 1], color='w', lw=.4)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect('equal')
    ax.set_axis_off()


def colorbar(ax, cmap, lo, hi, label, ticks):
    cb = mpl.colorbar.ColorbarBase(ax, cmap=mpl.colormaps[cmap], norm=mpl.colors.Normalize(lo, hi),
                                   orientation='horizontal')
    cb.set_ticks(ticks)
    cb.ax.tick_params(labelsize=6, length=2, width=.6, pad=1)
    cb.outline.set_visible(False)
    ax.set_xlabel(label, fontsize=6, labelpad=1)


def map_keys(axes, top=True):
    """Keys under (or over) a row of map axes: areas, angle wheel, eccentricity and R² bars."""
    names = ROI_ORDER
    for k, roi in enumerate(names):
        axes[0].text(.03 + (k % 4) * .245, -.04 - (k // 4) * .13, roi, transform=axes[0].transAxes,
                     color=ROI_COLORS[roi], fontsize=6.3, weight='bold', va='top')
    angle_key(axes[1].inset_axes([.36, -.36, .28, .32]))
    colorbar(axes[2].inset_axes([.15, -.1, .7, .06]), 'nipy_spectral', 0, 5, 'Eccentricity (°)', [0, 2.5, 5])
    if len(axes) > 3:
        colorbar(axes[3].inset_axes([.15, -.1, .7, .06]), 'inferno', 0, .6, 'R²', [0, .3, .6])


# ── analysis panels ───────────────────────────────────────────────────────────

def contralateral(d):
    g = good(d)
    contra = np.where(g.hemi == 'L', g.x > 0, g.x < 0)
    return g.assign(contra=contra).groupby(['subject', 'roi']).agg(p=('contra', 'mean'), n=('contra', 'size'))


def panel_contra(ax, d):
    c = contralateral(d).reset_index()
    c = c[c.n >= 10]
    for i, roi in enumerate(ROI_ORDER):
        p = 100 * c[c.roi == roi].p.to_numpy()
        rng = np.random.default_rng(i)
        ax.scatter(ROI_X[roi] + rng.uniform(-.2, .2, len(p)), p, s=6, color=ROI_COLORS[roi], alpha=.65, lw=0)
        if len(p):
            ax.plot([ROI_X[roi] - .3, ROI_X[roi] + .3], [np.median(p)] * 2, color=FG, lw=1.4,
                    solid_capstyle='butt')
        ax.text(ROI_X[roi], 31, f'{len(p)}', ha='center', va='bottom', fontsize=5.8, color='.45')
    ax.axhline(50, color='.7', lw=.6, ls='--', zorder=0)
    ax.text(ROI_X['V3'], 51.5, 'Chance', fontsize=6, color='.45', ha='center', va='bottom')
    ax.set_yticks([50, 75, 100])
    ax.set_ylim(30, 102)
    ax.set_ylabel('Contralateral (%)')
    offset(ax)
    roi_axis(ax)


def size_by_ecc(d, step=.5):
    g = good(d).assign(eb=lambda x: (np.floor(x.ecc / step) + .5) * step)
    t = g.groupby(['subject', 'roi', 'eb']).agg(sd=('sd', 'median'), n=('sd', 'size')).reset_index()
    return t[t.n >= 5]


def panel_size(ax, d, rois=('V1', 'V2', 'V3', 'hV4', 'LO', 'V3AB')):
    t = size_by_ecc(d)
    ends = []
    for roi in rois:
        g = t[t.roi == roi].pivot(index='subject', columns='eb', values='sd')
        g = g.loc[:, g.notna().sum() >= 6]
        if g.empty:
            continue
        m, se = g.mean(), g.sem()
        ax.fill_between(m.index, m - se, m + se, color=ROI_COLORS[roi], alpha=.2, lw=0)
        ax.plot(m.index, m.values, color=ROI_COLORS[roi], lw=1.3)
        ends.append([m.values[-1], m.index[-1], roi])
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + .13)
    for y, x, roi in ends:
        ax.text(x + .08, y, roi, color=ROI_COLORS[roi], fontsize=6, va='center')
    ax.axvline(APERTURE, color='.75', lw=.6, ls=':', zorder=0)
    ax.text(APERTURE, ax.get_ylim()[1], 'Bar aperture', fontsize=5.8, color='.45', ha='center', va='bottom')
    ax.set_xlim(0, 5)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xlabel('Eccentricity (°)')
    ax.set_ylabel('PRF size σ (°)')
    offset(ax)


def r2_gain(d):
    g = d[((d.r2 >= R2_MIN) | (d.r2_bar >= R2_MIN))]
    return g.groupby(['subject', 'roi']).agg(items=('r2', 'median'), bar=('r2_bar', 'median'),
                                             n=('r2', 'size')).reset_index().query('n >= 10')


def panel_gain(ax, d):
    t = r2_gain(d)
    t['gain'] = t['items'] - t['bar']
    for i, roi in enumerate(ROI_ORDER):
        p = t[t.roi == roi].gain.to_numpy()
        rng = np.random.default_rng(10 + i)
        ax.scatter(ROI_X[roi] + rng.uniform(-.2, .2, len(p)), p, s=6, color=ROI_COLORS[roi], alpha=.65, lw=0)
        if len(p):
            ax.plot([ROI_X[roi] - .3, ROI_X[roi] + .3], [np.median(p)] * 2, color=FG, lw=1.4,
                    solid_capstyle='butt')
    ax.axhline(0, color='.7', lw=.6, ls='--', zorder=0)
    ax.set_ylabel('R² gained by the items\n(median over vertices)')
    offset(ax)
    roi_axis(ax)


def panel_r2(ax, d):
    """Median R² per ROI and subject (vertices with R² >= R2_MIN), when there is no second fit to compare."""
    t = good(d).groupby(['subject', 'roi']).agg(r2=('r2', 'median'), n=('r2', 'size')).reset_index().query('n >= 10')
    for i, roi in enumerate(ROI_ORDER):
        p = t[t.roi == roi].r2.to_numpy()
        rng = np.random.default_rng(20 + i)
        ax.scatter(ROI_X[roi] + rng.uniform(-.2, .2, len(p)), p, s=6, color=ROI_COLORS[roi], alpha=.65, lw=0)
        if len(p):
            ax.plot([ROI_X[roi] - .3, ROI_X[roi] + .3], [np.median(p)] * 2, color=FG, lw=1.4,
                    solid_capstyle='butt')
    ax.set_ylim(0, None)
    ax.set_ylabel(f'Median R² (vertices ≥ {R2_MIN:.1f})')
    offset(ax)
    roi_axis(ax)


def draw_display(ax, lw=.6, color=FG):
    """The 8 search items (drawn with orientations of one example trial) and the bar aperture."""
    ax.add_patch(Circle((0, 0), APERTURE, fill=False, ec=color, lw=lw * .8, ls=(0, (2, 1.5))))
    for i in range(8):
        a = np.radians(i * 45.)
        cx, cy = ITEM_ECC * np.cos(a), ITEM_ECC * np.sin(a)
        w, h = (ITEM_LONG, ITEM_SHORT) if i % 2 else (ITEM_SHORT, ITEM_LONG)
        ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False, ec=color, lw=lw))
    ax.plot(0, 0, '+', color=color, ms=3, mew=.6)


def panel_coverage(fig, gs, d):
    sub = gs.subgridspec(1, len(PATHWAYS), wspace=.08)
    g = good(d)
    bins = np.linspace(-5, 5, 51)
    axes = []
    for j, (name, rois) in enumerate(PATHWAYS.items()):
        ax = fig.add_subplot(sub[j])
        x = g[g.roi.isin(rois)]
        # density per subject, then averaged, so one subject with many vertices does not dominate
        hs = [np.histogram2d(s.x, s.y, bins=[bins, bins])[0] / max(len(s), 1) for _, s in x.groupby('subject')]
        h = np.mean(hs, 0).T if hs else np.zeros((50, 50))
        ax.imshow(h, origin='lower', extent=(-5, 5, -5, 5), cmap='Greys', vmin=0, vmax=np.percentile(h, 99.5) or 1)
        draw_display(ax, color='#D62728')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        ax.set_xticks([-4, 0, 4])
        ax.set_yticks([-4, 0, 4])
        ax.set_title(f'{name}', fontsize=6.5, color=FG, pad=2)
        ax.set_xlabel('x (°)', labelpad=1)
        if j == 0:
            ax.set_ylabel('y (°)', labelpad=1)
        else:
            ax.tick_params(labelleft=False)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(.5)
        axes.append(ax)
    return axes[0]


# ── figures ───────────────────────────────────────────────────────────────────

def main_figure(d, render_dir, subject, view, out):
    fig = plt.figure(figsize=(7.2, 7.4))
    top = fig.add_gridspec(1, 3, left=.02, right=.98, top=.97, bottom=.69, wspace=.03)
    paths = [render_dir / f'sub-{subject:02d}_{p}_{view}.png' for p, _ in MAPS[:3]]
    box = crop_box(paths)
    axes = []
    for j, (p, title) in enumerate(MAPS[:3]):
        ax = fig.add_subplot(top[j])
        show(ax, paths[j], box)
        ax.set_title(title, color=FG, pad=2)
        axes.append(ax)
    axes[0].text(.07, 1.05, f'sub-{subject:02d}', transform=axes[0].transAxes, fontsize=6.5, color='.45', va='bottom')
    map_keys(axes)
    letter(axes[0], 'a', x=-.02, y=1.04)

    mid = fig.add_gridspec(1, 3, width_ratios=[1, .75, 1], left=.08, right=.98, top=.535, bottom=.375, wspace=.42)
    ax = fig.add_subplot(mid[0])
    panel_contra(ax, d)
    ax.set_title('PRFs sit in the contralateral field', loc='left', color=FG, pad=10)
    letter(ax, 'b', x=-.22, y=1.1)
    ax = fig.add_subplot(mid[1])
    panel_size(ax, d)
    ax.set_title('PRFs grow with eccentricity', loc='left', color=FG, pad=10)
    letter(ax, 'c', x=-.3, y=1.1)
    ax = fig.add_subplot(mid[2])
    if np.allclose(d.r2, d.r2_bar):      # bar-only fits: nothing to compare yet
        panel_r2(ax, d)
        ax.set_title('Variance explained per area', loc='left', color=FG, pad=10)
    else:
        panel_gain(ax, d)
        ax.set_title('The search items explain extra variance', loc='left', color=FG, pad=10)
    letter(ax, 'd', x=-.22, y=1.1)

    low = fig.add_gridspec(1, 1, left=.08, right=.98, top=.24, bottom=.04)
    first = panel_coverage(fig, low[0], d)
    first.text(0, 1.2, 'PRF centres cover the bar aperture and the search items (red)', transform=first.transAxes,
               fontsize=7, color=FG)
    letter(first, 'e', x=-.3, y=1.18)
    fig.savefig(out.with_suffix('.pdf'))
    fig.savefig(out.with_suffix('.png'), dpi=300)
    print('wrote', out.with_suffix('.pdf'))


def supplement(d, render_dir, subjects, view, out):
    """Every subject on one page: maps + PRF centres in the visual field."""
    n = len(subjects)
    fig = plt.figure(figsize=(7.2, 9.6))
    gs = fig.add_gridspec(n + 1, 5, width_ratios=[1, 1, 1, 1, .62], height_ratios=[1] * n + [.45],
                          left=.05, right=.99, top=.975, bottom=.05, hspace=.04, wspace=.03)
    box = crop_box([render_dir / f'sub-{s:02d}_{p}_{view}.png' for s in subjects for p, _ in MAPS])
    g = good(d)
    for i, s in enumerate(subjects):
        row = []
        for j, (p, title) in enumerate(MAPS):
            ax = fig.add_subplot(gs[i, j])
            show(ax, render_dir / f'sub-{s:02d}_{p}_{view}.png', box)
            if i == 0:
                ax.set_title(title, color=FG, pad=2)
            row.append(ax)
        x = g[g.subject == f'{s:02d}']
        row[0].text(-.04, .5, f'sub-{s:02d}\n{len(x)} vertices', transform=row[0].transAxes, rotation=90,
                    ha='right', va='center', fontsize=6, color=FG, linespacing=1.1)
        ax = fig.add_subplot(gs[i, 4])
        ax.scatter(x.x, x.y, s=.3, c=x.roi.map(ROI_COLORS), lw=0, alpha=.5, rasterized=True)
        draw_display(ax, lw=.35, color=FG)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        ax.set_axis_off()
        if i == 0:
            ax.set_title('PRF centres', color=FG, pad=2)
    keys = [fig.add_subplot(gs[n, j]) for j in range(4)]
    for k in keys:
        k.set_axis_off()
    for k in keys:
        k.set_ylim(0, 1)
    map_keys(keys)
    fig.text(.05, .003, f'Colour fades in from R² = .05 to .20; maps other than R² within {ECC_MAX:.0f}°. '
             f'PRF centres: vertices with R² ≥ {R2_MIN:.1f}, coloured by visual area; '
             'outline: bar aperture and search items.', fontsize=5.5, color='.4')
    fig.savefig(out.with_suffix('.pdf'), dpi=300)
    fig.savefig(out.with_suffix('.png'), dpi=200)
    print('wrote', out.with_suffix('.pdf'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tsv', default=str(ROOT / 'notes/data/prf_surface_vertices.tsv'))
    ap.add_argument('--render-dir', default=str(ROOT / 'notes/figures/prf_inflated_items'))
    ap.add_argument('--subject', type=int, default=7)
    ap.add_argument('--subjects', nargs='+', type=int, default=list(range(4, 16)))
    ap.add_argument('--view', default='occipital')
    ap.add_argument('--supp-view', default='occipital')
    ap.add_argument('--out-dir', default=str(ROOT / 'notes/figures'))
    ap.add_argument('--suffix', default='', help='appended to the output names, e.g. _baronly')
    a = ap.parse_args()
    d = load(a.tsv)
    rd = Path(a.render_dir)
    main_figure(d, rd, a.subject, a.view, Path(a.out_dir) / f'prf_validation{a.suffix}')
    supplement(d, rd, a.subjects, a.supp_view, Path(a.out_dir) / f'prf_validation_subjects{a.suffix}')
    s = good(d).groupby(['subject', 'roi']).size().unstack().reindex(columns=ROI_ORDER)
    print('vertices with R² >= .1 per subject x ROI\n', s.fillna(0).astype(int))
    t = r2_gain(d).assign(gain=lambda x: x['items'] - x['bar'])
    print('median R² gain by ROI\n', t.groupby('roi').gain.median().reindex(ROI_ORDER).round(4))
