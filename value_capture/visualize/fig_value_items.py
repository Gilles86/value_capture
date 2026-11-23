"""Figure: target, low- and high-value distractor in the measured responses, the gain fields and the decoded drive.

Style of retsupp Figs. 3, 5 and 6. Primary pipeline throughout: glmsingle_fbevents betas, only trials without the
points display, DoG pRFs. Per area: dots = subjects (triangles beyond the axis), points and bars = median and
bootstrapped 95% CI; numbers above = Wilcoxon signed-rank p against 0 per series; brackets = p of high vs low value
(one-sided Wilcoxon, high > low). All p uncorrected.

a  Single-trial response at the item's location: voxels with a pRF (fitted on the other session) within 2 deg of
   the item, the item as target / low / high-value distractor minus the same location holding a neutral item with
   the bar at the same position (% signal change). `gainfield/item_location_responses.py`.
b  Gain-field amplitude per effect (one field per effect with its own width; pRFs fitted jointly).
c  Gain-field width (FWHM) per effect.
d  Decoded intensity of the item, vs. the same item when neutral at the same bar position (outline decoder).
e  Decoded intensity of the bar within 2 deg of the item, vs. the far part (>= 3.5 deg) of the same bar on the same
   trial.
f  hV4 gain-field high - low value in the six ways to fit it.

    python value_capture/visualize/fig_value_items.py
"""
import glob
from pathlib import Path

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


def p_short(p):
    return '<.001' if p < .001 else (f'{p:.3f}' if p < .01 else f'{p:.2f}').replace('0.', '.')


def wil(v, alternative='two-sided'):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return stats.wilcoxon(v, alternative=alternative).pvalue if len(v) >= 5 and np.any(v != 0) else np.nan


def panel_triplet(ax, s, ylim, ylabel, log=False, test_zero=True, ref=0.):
    s = s.dropna(how='all')
    """s: one row per subject x roi, columns target / low / high."""
    lo, hi = ylim
    span = (np.log10(hi) - np.log10(lo)) if log else hi - lo

    def above(frac):
        return 10 ** (np.log10(hi) + frac * span) if log else hi + frac * span
    for k, (term, col, dx) in enumerate(TERMS):
        for i, roi in enumerate(ROI_ORDER):
            if roi not in s.index.get_level_values('roi'):
                continue
            v = s.xs(roi, level='roi')[term].dropna().to_numpy()
            x = ROI_X[roi] + dx
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
            if test_zero:
                p = wil(v - ref)
                ax.text(x, above(.02 + .075 * k), p_short(p), ha='center', va='bottom', fontsize=5.3,
                        color=col if p < .05 else MUTED, clip_on=False)
    for roi in ROI_ORDER:                                     # high vs low bracket
        if roi not in s.index.get_level_values('roi'):
            continue
        g = s.xs(roi, level='roi')
        p = wil(g.high - g.low, 'greater')
        y = above(.3 if test_zero else .07)
        x0, x1 = ROI_X[roi], ROI_X[roi] + .27
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


def items_betas():
    f = sorted(glob.glob(str(DATA / 'item_location_responses' / '*.tsv')))
    t = pd.concat([pd.read_csv(x, sep='\t', dtype={'subject': str}) for x in f]).query("points == 'exclude'")
    # trial-weighted mean over sessions and locations, in % signal change
    t['w'] = t['diff'] * t.n_trials
    g = t.groupby(['subject', 'roi', 'role'])
    s = (g.w.sum() / g.n_trials.sum()).unstack('role')
    return s.rename(columns={'distractor_low': 'low', 'distractor_high': 'high'})[['target', 'low', 'high']]


def gain_widths(folder='gain_fields_dog_fbevents_free_exclude_widths'):
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


def section(fig, y, text):
    fig.text(.012, y, text, fontsize=7.5, color=FG, va='bottom')
    fig.add_artist(plt.Line2D([.012, .99], [y - .005] * 2, color=MUTED, lw=.6, transform=fig.transFigure))


def main():
    betas = items_betas()
    amp, fwhm = gain_widths()
    items, bars = decoded()
    fig = plt.figure(figsize=(DOUBLE, 9.2))
    W, H = .37, .145
    X0, X1 = .1, .605
    rows = [.74, .415, .09]
    ax = {k: fig.add_axes([x, y, W, H]) for k, (x, y) in
          dict(a=(X0, rows[0]), b=(X1, rows[0]), c=(X0, rows[1]), d=(X1, rows[1]), e=(X0, rows[2])).items()}
    ax['f'] = fig.add_axes([X1 + .17, rows[2], .12, H])
    data = dict(a=betas, b=amp, c=fwhm, d=items, e=bars)
    panel_triplet(ax['a'], betas, (-.4, .8), 'Response vs neutral\n(% signal)')
    panel_triplet(ax['b'], amp, (-80, 300), 'Gain-field\namplitude (%)')
    panel_triplet(ax['c'], fwhm, (.3, 30), 'Gain-field width,\nFWHM (deg)', log=True, test_zero=False)
    panel_triplet(ax['d'], items, (-25, 45), 'Decoded item vs\nneutral item (%)')
    panel_triplet(ax['e'], bars, (-40, 50), 'Decoded bar near item\nvs far part (%)')
    for k in 'abcde':
        n = data[k].dropna(how='all').reset_index().groupby('roi').subject.nunique().to_dict()
        roi_axis(ax[k], n)
    panel_fits(ax['f'], load())
    ax['f'].set_xlabel('hV4 gain field,\nhigh − low value (%)', fontsize=7)

    def top(r):
        return rows[r] + H
    section(fig, top(0) + .085, 'Measured: the single-trial response where the item is, and the gain field fitted to it')
    r = fig.canvas.get_renderer()
    x = .6
    for lab, c in [('Target', TARGET), ('Low value', LOW), ('High value', HIGH)]:
        tx = fig.text(x, top(0) + .087, '● ' + lab, color=c, fontsize=7, va='bottom')
        x += tx.get_window_extent(r).width / fig.bbox.width + .025
    heading(fig, .012, top(0) + .055, 'a', [('Response at the item location (', FG), ('p', FG),
                                           (' vs 0 per series; bracket: high > low, one-sided)', MUTED)])
    heading(fig, X1 - .07, top(0) + .055, 'b', [('Gain-field amplitude per effect', FG)])
    section(fig, top(1) + .085, 'Gain-field width, and the stimulus drive decoded from the same betas')
    heading(fig, .012, top(1) + .055, 'c', [('Gain-field width per effect (', FG), ('bracket: high > low, one-sided', MUTED),
                                           (')', FG)])
    heading(fig, X1 - .07, top(1) + .055, 'd', [('Decoded intensity of the item', FG)])
    section(fig, top(2) + .085, 'Decoded bar near the item, and hV4 across analysis choices')
    heading(fig, .012, top(2) + .055, 'e', [('Decoded bar within 2° of the item', FG)])
    heading(fig, X1 - .07, top(2) + .055, 'f', [('hV4: ', FG), ('high', HIGH), (' vs ', FG), ('low', LOW),
                                                (' in the six gain-field fits', FG)])
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / 'fig_value_items.pdf', dpi=300)
    fig.savefig(OUT / 'fig_value_items.png', dpi=200)
    print(OUT / 'fig_value_items.pdf')


if __name__ == '__main__':
    main()
