"""Figure: gain at the target and at the low- and high-value distractor per visual area (style of retsupp Fig. 5).

Gain fields fitted jointly with the pRFs on the primary betas (glmsingle_fbevents), all sessions together
(`gainfield/fit_gain_fields.py --free-prf`). Gain in %: +50% = the stimulus at that item drives the voxels 1.5x as
much as at a neutral item.

a, gain at the target, the low- and the high-value distractor per area (primary fit: DoG pRFs, points trials
   excluded). Dots, subjects (triangles beyond the axis); points and bars, median and bootstrapped 95% CI;
   p, Wilcoxon signed-rank against 0 per area, uncorrected.
b, high - low value per area, same fit and statistics.
c, hV4 high - low in each of the six ways to fit it (Gaussian / DoG pRFs x points trials excluded, subtracted, or
   modelled as part of the stimulus) and averaged per subject over the six.

    python value_capture/visualize/fig_value_gain.py
"""
import glob
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'notes' / 'data'
OUT = ROOT / 'notes' / 'figures'
mpl.rcParams.update({
    'font.family': 'Helvetica', 'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': .8,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.labelpad': 3,
    'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': .8, 'ytick.major.width': .8, 'lines.linewidth': 1.2,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.pad_inches': .02})

DOUBLE = 180 / 25.4
FG, MUTED = '#1F2933', '#9AA5B1'
TARGET, LOW, HIGH = '#F77F00', '#009E73', '#7B3294'
ROI_GROUPS = [('Early', ('V1', 'V2', 'V3')), ('Dorsal', ('V3AB',)), ('Occipitotemporal', ('hV4', 'LO'))]
GROUP_GAP = 1.
ROI_ORDER = [r for _, g in ROI_GROUPS for r in g]
ROI_X, _x = {}, 0.
for _, _g in ROI_GROUPS:
    for _r in _g:
        ROI_X[_r] = _x
        _x += 1
    _x += GROUP_GAP
FITS = [('dog', 'exclude'), ('dog', 'subtract'), ('dog', 'model'),
        ('gauss', 'exclude'), ('gauss', 'subtract'), ('gauss', 'model')]
PRIMARY = ('dog', 'exclude')
POINTS = {'exclude': 'points trials excluded', 'subtract': 'points subtracted', 'model': 'points modelled'}


def load():
    out = []
    for prf, points in FITS:
        d = DATA / f"gain_fields{'_dog' if prf == 'dog' else ''}_fbevents_free_{points}"
        t = pd.concat([pd.read_csv(f, sep='\t', dtype={'subject': str}) for f in glob.glob(f'{d}/*.tsv')])
        out.append(t[t.scope == 'all'].assign(prf=prf, points=points))
    t = pd.concat(out).rename(columns={'field_gain_target': 'target', 'field_gain_distractor_low': 'low',
                                       'field_gain_distractor_high': 'high'})
    for c in ('target', 'low', 'high'):
        t[c] *= 100
    t['high_low'] = t.high - t.low
    return t


def median_ci(x, n_boot=5000, seed=0):
    x = np.asarray(x)[np.isfinite(x)]
    boots = np.median(np.random.default_rng(seed).choice(x, (n_boot, len(x))), axis=1)
    return np.median(x), *np.percentile(boots, [2.5, 97.5])


def ptext(p):
    if p < .001:
        return 'p < .001'
    return (f'p = {p:.3f}' if p < .01 else f'p = {p:.2f}').replace('0.', '.')


def offset_spines(ax, by=3):
    ax.spines['left'].set_position(('outward', by))
    ax.spines['bottom'].set_position(('outward', by))


def roi_axis(ax, n_per_roi, pad_pt=3):
    xs = [ROI_X[r] for r in ROI_ORDER]
    ax.set_xticks(xs, [r if n_per_roi[r] >= 11 else f'{r}\n(n = {n_per_roi[r]})' for r in ROI_ORDER], fontsize=7)
    ax.set_xlim(xs[0] - .6, xs[-1] + .6)
    fig = ax.figure
    fig.canvas.draw()
    inv = ax.transAxes.inverted()
    bottom = min(inv.transform((0, t.get_window_extent().y0))[1] for t in ax.get_xticklabels() if t.get_text())
    tr = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
    for name, rois in ROI_GROUPS:
        x0, x1 = ROI_X[rois[0]] - .38, ROI_X[rois[-1]] + .38
        ax.plot([x0, x1], [bottom] * 2, transform=mtrans.offset_copy(tr, fig=fig, y=-pad_pt, units='points'),
                color=MUTED, lw=.7, clip_on=False, solid_capstyle='butt')
        ax.text((x0 + x1) / 2, bottom, name, transform=mtrans.offset_copy(tr, fig=fig, y=-pad_pt - 1.5, units='points'),
                ha='center', va='top', fontsize=6.5, color='.4', clip_on=False)


def dots(ax, x, v, col, lo, hi, seed, jitter=.07, s=6):
    v = np.asarray(v, float)
    xx = x + np.random.default_rng(seed).uniform(-jitter, jitter, len(v))
    ok = (v >= lo) & (v <= hi)
    ax.scatter(xx[ok], v[ok], s=s, color=col, alpha=.35, lw=0, zorder=1)
    ax.scatter(xx[v > hi], np.full((v > hi).sum(), hi), s=9, marker='^', color=col, alpha=.6, lw=0, clip_on=False)
    ax.scatter(xx[v < lo], np.full((v < lo).sum(), lo), s=9, marker='v', color=col, alpha=.6, lw=0, clip_on=False)
    m, a, b = median_ci(v, seed=seed)
    ax.errorbar(x, m, yerr=[[m - a], [b - m]], fmt='o', ms=5, color=col, mec='w', mew=.7, lw=1.5, capsize=0, zorder=4)


def panel_gains(ax, d, ylim=(-60, 220)):
    lo, hi = ylim
    terms = [('target', TARGET, -.27), ('low', LOW, 0.), ('high', HIGH, .27)]
    for k, (term, col, dx) in enumerate(terms):
        for i, roi in enumerate(ROI_ORDER):
            v = d[d.roi == roi][term].dropna()
            dots(ax, ROI_X[roi] + dx, v, col, lo, hi, seed=10 * k + i)
            p = stats.wilcoxon(v).pvalue
            ax.text(ROI_X[roi] + dx, hi + .04 * (hi - lo), ptext(p)[4:].replace('< ', '<'), ha='center',
                    va='bottom', fontsize=5.5, color=col if p < .05 else MUTED, clip_on=False)
    ax.text(ROI_X[ROI_ORDER[0]] - .6, hi + .04 * (hi - lo), 'p:', fontsize=5.5, color=FG, ha='right', va='bottom',
            clip_on=False)
    ax.axhline(0, color='.6', lw=.8, ls='--', zorder=0)
    ax.set_ylim(lo, hi)
    ax.set_ylabel('Gain at item (%)')
    offset_spines(ax)


def panel_value(ax, d, ylim=(-120, 160)):
    lo, hi = ylim
    for i, roi in enumerate(ROI_ORDER):
        v = d[d.roi == roi].high_low.dropna()
        dots(ax, ROI_X[roi], v, HIGH, lo, hi, seed=40 + i, jitter=.15, s=8)
        p = stats.wilcoxon(v).pvalue
        ax.text(ROI_X[roi], hi + .04 * (hi - lo), f'{ptext(p)}\n{(v > 0).sum()}/{len(v)} > 0', ha='center',
                va='bottom', fontsize=6, color=HIGH if p < .1 else FG, clip_on=False)
    ax.axhline(0, color='.6', lw=.8, ls='--', zorder=0)
    ax.set_ylim(lo, hi)
    ax.set_ylabel('High − low value (%)')
    offset_spines(ax)


def panel_fits(ax, t):
    d = t[t.roi == 'hV4']
    rows = [(f"{'DoG' if prf == 'dog' else 'Gaussian'}, {POINTS[points]}",
             d[(d.prf == prf) & (d.points == points)].set_index('subject').high_low) for prf, points in FITS]
    rows.append(('Average of the six', d.groupby('subject').high_low.mean()))
    y = np.arange(len(rows))[::-1].astype(float)
    y[-1] -= .6
    for (lab, v), yy in zip(rows, y):
        last = lab.startswith('Average')
        m, a, b = median_ci(v)
        ax.plot([a, b], [yy, yy], color=HIGH, lw=1.6, solid_capstyle='butt')
        ax.plot(m, yy, 'D' if last else 'o', ms=5.5 if last else 4.5, color=HIGH, mec='w', mew=.6)
        ax.text(1.04, yy, f'{(v > 0).sum()}/{len(v)}   {ptext(stats.wilcoxon(v).pvalue)}',
                transform=ax.get_yaxis_transform(), va='center', fontsize=6, color=FG)
    ax.text(1.04, y[0] + 1, '> 0', transform=ax.get_yaxis_transform(), va='center', fontsize=6, color=MUTED)
    ax.axvline(0, color='.6', lw=.8, ls='--', zorder=0)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=6.5)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_position(('outward', 3))
    ax.set_ylim(y[-1] - .7, y[0] + 1.4)
    ax.set_xlabel('hV4, high − low value (%)')


def heading(fig, x, y, letter, parts):
    fig.text(x, y, letter, fontsize=8, fontweight='bold', va='bottom')
    r = fig.canvas.get_renderer()
    x += .022
    for s, c in parts:
        tx = fig.text(x, y, s, fontsize=8, color=c, va='bottom')
        x += tx.get_window_extent(r).width / fig.bbox.width


def main():
    t = load()
    d = t[(t.prf == PRIMARY[0]) & (t.points == PRIMARY[1])]
    n_per_roi = d.groupby('roi').subject.nunique().to_dict()
    fig = plt.figure(figsize=(DOUBLE, 5.6))
    ax_a = fig.add_axes([.075, .64, .9, .27])
    ax_b = fig.add_axes([.075, .12, .42, .26])
    ax_c = fig.add_axes([.775, .12, .13, .26])
    panel_gains(ax_a, d)
    roi_axis(ax_a, n_per_roi)
    panel_value(ax_b, d)
    roi_axis(ax_b, n_per_roi)
    panel_fits(ax_c, t)
    hv4 = d[d.roi == 'hV4']
    heading(fig, .01, .955, 'a', [('Gain at the ', FG), ('target', TARGET), (', the ', FG), ('low-value', LOW),
                                  (' and the ', FG), ('high-value', HIGH), (' distractor', FG)])
    heading(fig, .01, .455, 'b', [('More gain for the ', FG), ('high-value', HIGH),
                                (f' distractor in hV4 ({(hv4.high_low > 0).sum()} of {len(hv4)} subjects)', FG)])
    heading(fig, .54, .455, 'c', [('hV4: ', FG), ('high', HIGH), (' > ', FG), ('low', LOW),
                                (' in all six ways to fit it', FG)])
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / 'fig_value_gain.pdf', dpi=300)
    fig.savefig(OUT / 'fig_value_gain.png', dpi=200)
    print(OUT / 'fig_value_gain.pdf')


if __name__ == '__main__':
    main()
