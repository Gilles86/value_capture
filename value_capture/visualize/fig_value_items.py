"""Figure: target, low- and high-value distractor in the measured responses, the gain fields and the decoded drive.

Style of retsupp Figs. 3, 5 and 6. Primary pipeline throughout: glmsingle_fbevents betas, only trials without the
points display, DoG pRFs. Per area: dots = subjects (triangles beyond the axis), points and bars = median and
bootstrapped 95% CI; numbers above = Wilcoxon signed-rank p against 0 per series; brackets = p of high vs low value
(one-sided Wilcoxon, high > low). All p uncorrected.
Right of the divider: each pathway pooled, per subject the mean over its areas (Dorsal = V3AB only).

a  Single-trial response at the item's location: voxels with a pRF (fitted on the other session) within 2 deg of
   the item, the item as target / low / high-value distractor minus the same location holding a neutral item with
   the bar at the same position (% signal change). `gainfield/item_location_responses.py`.
b  Gain-field amplitude per effect (one field per effect with its own width; pRFs fitted jointly).
c  Gain-field width (FWHM) per effect.
d  Decoded intensity of the item, vs. the same item when neutral at the same bar position (outline decoder).
e  Decoded intensity of the bar within 2 deg of the item, vs. the far part (>= 3.5 deg) of the same bar on the same
   trial.

    python value_capture/visualize/fig_value_items.py
"""
import glob
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from value_capture.visualize.fig_value_gain import (DOUBLE, FG, HIGH, LOW, MUTED, ROI_ORDER, ROI_X, TARGET, load,
                                                     median_ci, offset_spines, panel_fits, roi_axis)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'notes' / 'data'
OUT = ROOT / 'notes' / 'figures'
TERMS = [('target', TARGET, -.27), ('low', LOW, 0.), ('high', HIGH, .27)]
# Pathways pooled per subject (mean over its areas), tested as well; drawn right of a divider
POOLED = [('Early', ('V1', 'V2', 'V3')), ('Dorsal', ('V3AB',)), ('Occ.-temp.', ('hV4', 'LO'))]
XPOS = dict(ROI_X)
_x0 = max(ROI_X.values()) + 1.9
for _i, (_n, _) in enumerate(POOLED):
    XPOS[_n] = _x0 + 1.05 * _i
ORDER = ROI_ORDER + [n for n, _ in POOLED]


def with_pooled(s, log=False):
    """Add one row per subject and pathway: the mean over the pathway's areas (geometric mean if log)."""
    out = [s]
    for name, rois in POOLED:
        sub = s[s.index.get_level_values('roi').isin(rois)]
        if not len(sub):
            continue
        g = (np.exp(np.log(sub).groupby(level='subject').mean()) if log else sub.groupby(level='subject').mean())
        out.append(g.assign(roi=name).set_index('roi', append=True))
    return pd.concat(out)


def pooled_axis(ax, n_per, pad_pt=3):
    import matplotlib.transforms as mtrans
    from value_capture.visualize.fig_value_gain import ROI_GROUPS
    xs = [XPOS[r] for r in ORDER]
    ax.set_xticks(xs, [r if n_per.get(r, 0) >= 11 else f'{r}\n(n = {n_per.get(r, 0)})' for r in ORDER],
                  fontsize=6.5)
    for t, r in zip(ax.get_xticklabels(), ORDER):
        if r in dict(POOLED):
            t.set_fontweight('bold')
    ax.set_xlim(xs[0] - .6, xs[-1] + .6)
    ax.axvline(_x0 - .95, color=MUTED, lw=.6, zorder=0)
    fig = ax.figure
    fig.canvas.draw()
    inv = ax.transAxes.inverted()
    bottom = min(inv.transform((0, t.get_window_extent().y0))[1] for t in ax.get_xticklabels() if t.get_text())
    tr = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
    groups = list(ROI_GROUPS) + [('Pathway, pooled', [n for n, _ in POOLED])]
    for name, rois in groups:
        x0, x1 = XPOS[rois[0]] - .38, XPOS[rois[-1]] + .38
        ax.plot([x0, x1], [bottom] * 2, transform=mtrans.offset_copy(tr, fig=fig, y=-pad_pt, units='points'),
                color=MUTED, lw=.7, clip_on=False, solid_capstyle='butt')
        ax.text((x0 + x1) / 2, bottom, name, transform=mtrans.offset_copy(tr, fig=fig, y=-pad_pt - 1.5, units='points'),
                ha='center', va='top', fontsize=6, color='.4', clip_on=False)


def p_short(p):
    return '<.001' if p < .001 else (f'{p:.3f}' if p < .01 else f'{p:.2f}').replace('0.', '.')


def wil(v, alternative='two-sided'):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return stats.wilcoxon(v, alternative=alternative).pvalue if len(v) >= 5 and np.any(v != 0) else np.nan


def auto_ylim(s, log=False, q=(.03, .97), pad=.08):
    """Limits that hold the bulk of the subjects; the rest are drawn as triangles at the edge."""
    v = s[['target', 'low', 'high']].to_numpy().ravel()
    v = v[np.isfinite(v)]
    if log:
        a, b = np.log10(np.quantile(v, q))
        return 10 ** (a - pad * (b - a)), 10 ** (b + pad * (b - a))
    a, b = np.quantile(v, q)
    a, b = min(a, 0.), max(b, 0.)
    return a - pad * (b - a), b + pad * (b - a)


def panel_triplet(ax, s, ylabel, ylim=None, log=False, test=True, ref=0.):
    """s: one row per subject x roi, columns target / low / high. test=False: medians printed, no tests."""
    s = with_pooled(s.dropna(how='all'), log)
    lo, hi = ylim or auto_ylim(s, log)
    test_zero = test
    span = (np.log10(hi) - np.log10(lo)) if log else hi - lo

    def above(frac):
        return 10 ** (np.log10(hi) + frac * span) if log else hi + frac * span
    for k, (term, col, dx) in enumerate(TERMS):
        for i, roi in enumerate(ORDER):
            if roi not in s.index.get_level_values('roi'):
                continue
            v = s.xs(roi, level='roi')[term].dropna().to_numpy()
            x = XPOS[roi] + dx
            xx = x + np.random.default_rng(10 * k + i).uniform(-.07, .07, len(v))
            ok = (v >= lo) & (v <= hi)
            ax.scatter(xx[ok], v[ok], s=6, color=col, alpha=.35, lw=0, zorder=1)
            ax.scatter(xx[v > hi], np.full((v > hi).sum(), hi), s=9, marker='^', color=col, alpha=.6, lw=0,
                       clip_on=False)
            ax.scatter(xx[v < lo], np.full((v < lo).sum(), lo), s=9, marker='v', color=col, alpha=.6, lw=0,
                       clip_on=False)
            m, a, b = median_ci(v, seed=k + 10 * i)
            ax.errorbar(x, m, yerr=[[m - a], [b - m]], fmt='o', ms=4.5, color=col, mec='w', mew=.6, lw=1.4,
                        capsize=0, zorder=4)
            if not test:
                ax.text(x, above(.02 + .075 * k), f'{m:.1f}', ha='center', va='bottom', fontsize=5.3, color=col,
                        clip_on=False)
            if test_zero:
                p = wil(v - ref)
                ax.text(x, above(.02 + .075 * k), p_short(p), ha='center', va='bottom', fontsize=5.3,
                        color=col if p < .05 else MUTED, clip_on=False)
    for roi in (ORDER if test else []):                   # high vs low bracket
        if roi not in s.index.get_level_values('roi'):
            continue
        g = s.xs(roi, level='roi')
        p = wil(g.high - g.low, 'greater')
        y = above(.3 if test_zero else .07)
        x0, x1 = XPOS[roi], XPOS[roi] + .27
        yb = above(.27 if test_zero else .04)
        ax.plot([x0, x0, x1, x1], [yb, y, y, yb], color=HIGH if p < .1 else FG, lw=.6, clip_on=False)
        ax.text((x0 + x1) / 2, y, p_short(p), ha='center', va='bottom', fontsize=5.5,
                color=HIGH if p < .1 else FG, clip_on=False)
    if log:
        ax.set_yscale('log')
    else:
        ax.axhline(ref, color='.6', lw=.8, ls='--', zorder=0)
    ax.set_ylim(lo, hi)
    ax.set_ylabel(ylabel)
    offset_spines(ax)


def items_betas(points='exclude'):
    f = sorted(glob.glob(str(DATA / 'item_location_responses' / '*.tsv')))
    t = pd.concat([pd.read_csv(x, sep='\t', dtype={'subject': str}) for x in f]).query("points == @points")
    # trial-weighted mean over sessions and locations, in % signal change
    t['w'] = t['diff'] * t.n_trials
    g = t.groupby(['subject', 'roi', 'role'])
    s = (g.w.sum() / g.n_trials.sum()).unstack('role')
    return s.rename(columns={'distractor_low': 'low', 'distractor_high': 'high'})[['target', 'low', 'high']]


def gain_widths(folder='gain_fields_dog_fbevents_free_exclude_widths'):
    if not glob.glob(str(DATA / folder / '*.tsv')):
        raise FileNotFoundError(DATA / folder)
    t = pd.concat([pd.read_csv(f, sep='\t', dtype={'subject': str}) for f in glob.glob(str(DATA / folder / '*.tsv'))])
    t = t[t.scope == 'all'].set_index(['subject', 'roi'])
    amp = t[[f'field_gain_{k}' for k in ('target', 'distractor_low', 'distractor_high')]] * 100
    amp.columns = ['target', 'low', 'high']
    fwhm = t[[f'field_sigma_{k}' for k in ('target', 'distractor_low', 'distractor_high')]] * 2.355
    fwhm.columns = ['target', 'low', 'high']
    return amp, fwhm


def decoded(variant='fbevents_dog_none'):
    d = DATA / 'decoded_gain_variants' / variant
    ig = pd.read_csv(d / 'item_gain.tsv', sep='\t', dtype={'subject': str}).query("decoder == 'outline'")
    ig['w'] = ig.pct * ig.n_trials
    g = ig.groupby(['subject', 'roi', 'role'])
    items = (g.w.sum() / g.n_trials.sum()).unstack('role').rename(
        columns={'distractor_low': 'low', 'distractor_high': 'high'})[['target', 'low', 'high']]
    bd = pd.read_csv(d / 'bar_distance.tsv', sep='\t', dtype={'subject': str}).query(
        "decoder == 'outline' and distance <= 2")
    bd['w'] = bd.pct_within * bd.n_trials
    bd['term'] = np.where(bd.aligned_to == 'target', 'target', bd.value)
    g = bd.groupby(['subject', 'roi', 'term'])
    bars = (g.w.sum() / g.n_trials.sum()).unstack('term')[['target', 'low', 'high']]
    return items, bars


def heading(fig, x, y, letter, parts, size=7.5):
    fig.text(x, y, letter, fontsize=8, fontweight='bold', va='bottom')
    r = fig.canvas.get_renderer()
    x += .02
    for s, c in parts:
        tx = fig.text(x, y, s, fontsize=size, color=c, va='bottom')
        x += tx.get_window_extent(r).width / fig.bbox.width

RING, APERTURE, ITEM = 4., 3.17, 1   # item ring and bar aperture (deg); the marked item is the upper-right one
CHARCOAL = '#3E4C59'


def icon(fig, rect, kind, color=CHARCOAL):
    """Small outline of the display showing what a panel measures (after retsupp's stimulus_icon)."""
    from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
    a = fig.add_axes(rect)
    a.set_xlim(-6.4, 6.4)
    a.set_ylim(-6.4, 6.4)
    a.set_aspect('equal')
    a.set_axis_off()
    xy = [(RING * np.cos(np.radians(45 * i)), RING * np.sin(np.radians(45 * i))) for i in range(8)]
    ix, iy = xy[ITEM]
    if kind == 'decoded':
        a.add_patch(Circle((0, 0), 6.2, facecolor='k', lw=0, zorder=0))
        g = np.linspace(-5.4, 5.4, 120)
        X, Y = np.meshgrid(g, g)
        img = np.exp(-((X - ix) ** 2 + (Y - iy) ** 2) / (2 * .7 ** 2))
        img += .35 * sum(np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * .6 ** 2)) for k, (x, y) in enumerate(xy) if k != ITEM)
        im = a.imshow(img, extent=(-5.4, 5.4, -5.4, 5.4), origin='lower', cmap='magma', vmin=0, vmax=1.1, zorder=1)
        im.set_clip_path(Circle((0, 0), 6.2, transform=a.transData))
        a.add_patch(Circle((ix, iy), 1.3, fill=False, ec=color, lw=1., ls=(0, (2, 1.2)), zorder=3))
        return a
    if kind in ('field', 'width'):
        g = np.linspace(-5.4, 5.4, 120)
        X, Y = np.meshgrid(g, g)
        f = np.exp(-((X - ix) ** 2 + (Y - iy) ** 2) / (2 * 1.1 ** 2))
        cm = mpl.colors.LinearSegmentedColormap.from_list('f', ['#FFFFFF00', color])
        a.imshow(f, extent=(-5.4, 5.4, -5.4, 5.4), origin='lower', cmap=cm, vmin=0, vmax=1.4, zorder=0)
    if kind == 'bar':
        bx = ix - .4
        h = np.sqrt(APERTURE ** 2 - bx ** 2)
        a.add_patch(Rectangle((bx - .3, -h), .6, 2 * h, facecolor='#D5DAE0', lw=0, zorder=1))
        lo = max(-h, iy - 2.)
        a.add_patch(Rectangle((bx - .3, lo), .6, h - lo, facecolor=color, lw=0, zorder=1.5))
    a.add_patch(Circle((0, 0), APERTURE, fill=False, ec=MUTED, lw=.5, zorder=2))
    for k, (x, y) in enumerate(xy):
        vert = k % 2 == 0
        w, hh = (.5, 1.3) if vert else (1.3, .5)
        a.add_patch(Rectangle((x - w / 2, y - hh / 2), w, hh, facecolor=MUTED if k != ITEM else
                              (color if kind == 'key' else CHARCOAL), lw=0, zorder=3))
    if kind == 'item':
        a.add_patch(Circle((ix, iy), 2., facecolor=color, alpha=.15, ec=color, lw=.7, ls=(0, (2, 1.2)), zorder=2.5))
    if kind == 'width':
        r = 1.1 * 1.1775
        a.add_patch(FancyArrowPatch((ix - r, iy - 1.9), (ix + r, iy - 1.9), arrowstyle='<|-|>', mutation_scale=4,
                                    color=FG, lw=.6, zorder=4))
    a.plot(0, 0, '+', color=FG, ms=3, mew=.6, zorder=4)
    return a


def heading_icon(fig, x, y, letter, parts, kind, size=.06):
    """Panel heading followed by its icon."""
    heading(fig, x, y, letter, parts)
    r = fig.canvas.get_renderer()
    end = max(t.get_window_extent(r).x1 for t in fig.texts[-len(parts):]) / fig.bbox.width
    h = size
    w = h * fig.bbox.height / fig.bbox.width
    icon(fig, [end + .01, y - h / 2 + .006, w, h], kind)


def section(fig, y, text):
    fig.text(.012, y, text, fontsize=7.5, color=FG, va='bottom')
    fig.add_artist(plt.Line2D([.012, .99], [y - .005] * 2, color=MUTED, lw=.6, transform=fig.transFigure))


VARIANTS = {'exclude': dict(decoded='fbevents_dog_none', title='only trials without the points display'),
            'subtract': dict(decoded='fbevents_dog_regress', title='all trials, points response subtracted')}


PANELS = [  # key, data, y label, icon, heading, extra panel_triplet kwargs
    ('a', 'betas', 'Response vs neutral\n(% signal)', 'item',
     [('Measured response where the item is', FG), ('  (pRF within 2° of the item; vs the same location '
                                                     'when neutral, same bar position)', MUTED)], {}),
    ('b', 'amp', 'Gain-field\namplitude (%)', 'field',
     [('Gain-field amplitude', FG), ('  (one field per effect, pRFs fitted jointly)', MUTED)], {}),
    ('c', 'fwhm', 'Gain-field width,\nFWHM (deg)', 'width',
     [('Gain-field width', FG), ('  (numbers: medians; no tests)', MUTED)], dict(log=True, test=False)),
    ('d', 'items', 'Decoded item vs\nneutral item (%)', 'decoded',
     [('Decoded intensity of the item', FG), ('  (vs the same item when neutral, same bar position)', MUTED)], {}),
    ('e', 'bars', 'Decoded bar near item\nvs far part (%)', 'bar',
     [('Decoded bar within 2° of the item', FG), ('  (vs the far part of the same bar, same trial)', MUTED)], {}),
]


def main(points='exclude'):
    amp, fwhm = gain_widths(f'gain_fields_dog_fbevents_free_{points}_widths')
    items, bars = decoded(VARIANTS[points]['decoded'])
    data = dict(betas=items_betas(points), amp=amp, fwhm=fwhm, items=items, bars=bars)
    fig = plt.figure(figsize=(DOUBLE, 13.5))
    X, W, H, step = .1, .76, .085, .19
    tops = [.9 - i * step for i in range(len(PANELS))]
    r = fig.canvas.get_renderer()
    fig.text(.012, .995, f"DoG pRFs; {VARIANTS[points]['title']}", ha='left', va='top', fontsize=6.5, color=MUTED)
    fig.text(.99, .995, 'Numbers above: p vs 0 per series; bracket: high > low, one-sided. Right of the divider: '
             'pathways pooled per subject', ha='right', va='top', fontsize=6, color=MUTED)
    x = .012
    for lab, c in [('Target', TARGET), ('Low-value distractor', LOW), ('High-value distractor', HIGH)]:
        h = .026
        w = h * fig.bbox.height / fig.bbox.width
        icon(fig, [x, .962, w, h], 'key', color=c)
        tx = fig.text(x + w + .004, .972, lab, color=c, fontsize=7.5, va='center')
        x = tx.get_window_extent(r).x1 / fig.bbox.width + .03
    for (k, key, ylabel, kind, head, kw), top in zip(PANELS, tops):
        ax = fig.add_axes([X, top - H, W, H])
        d = data[key]
        panel_triplet(ax, d, ylabel, **kw)
        n = with_pooled(d.dropna(how='all')).reset_index().groupby('roi').subject.nunique().to_dict()
        pooled_axis(ax, n)
        heading(fig, .012, top + .045, k, head)
        h = .075
        icon(fig, [.865, top - H / 2 - h / 2, h * fig.bbox.height / fig.bbox.width, h], kind)
    OUT.mkdir(exist_ok=True)
    name = 'fig_value_items' + ('' if points == 'exclude' else f'_{points}')
    fig.savefig(OUT / f'{name}.pdf', dpi=300)
    fig.savefig(OUT / f'{name}.png', dpi=200)
    print(OUT / f'{name}.pdf')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--points', choices=list(VARIANTS), default='exclude')
    main(ap.parse_args().points)
