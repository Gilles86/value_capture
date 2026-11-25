"""Figure: distractor value and colour in every ROI, Bayesian group models with a noise level per participant.

Rows (all: trials without the points display):
a  Gain field at the distractor, high - low (%; pRF-mapped areas only: VO, TO and IPS have almost no voxels with
   pRF R2 >= .1). Model: per-run gains, ``gainfield/hierarchical_value_gain.py``.
b  Mean response, high - low (% signal), colour as a covariate. ``decode/bayes_value_rois.py``.
c  Decoding across sessions (colours swapped), AUC - .5 per test run: a value code that survives the colour swap.
d  Mean response, green - orange (% signal): the colour term of the model in b.

Dots: participants (mean over runs), shaded by precision (1 / SE over runs): pale = noisy. Bars: posterior mean and
95% HDI of the per-ROI hierarchical model (Student-t, residual scale per participant); purple = HDI above 0, green =
below 0. Right: pathway means and the overall mean of the SE-weighted meta-analysis over ROIs (retsupp model).

    python value_capture/visualize/fig_value_all_rois_bayes.py
"""
import glob
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import numpy as np
import pandas as pd

from value_capture.decode.bayes_value_rois import GROUPS
from value_capture.visualize.fig_value_gain import DOUBLE, FG, HIGH, MUTED, offset_spines

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'notes' / 'data'
OUT = ROOT / 'notes' / 'figures'
GF = 'gain_fields_dog_fbevents_free_exclude_widths'
LOW = '#009E73'
GREEN, ORANGE = '#00ab78', '#d56f2c'
LABEL = {'fusiform': 'Fusiform', 'parahippocampal': 'Parahipp.', 'entorhinal': 'Entorhinal', 'hippocampus': 'Hippoc.',
         'amygdala': 'Amygdala', 'caudate': 'Caudate', 'putamen': 'Putamen', 'accumbens': 'Accumbens',
         'medialOFC': 'Medial OFC', 'lateralOFC': 'Lateral OFC'}
SHORT = {'Early': 'Early', 'Dorsal': 'Dorsal', 'Occipito-temporal': 'Occ.-temp.', 'Ventral temporal / MTL': 'MTL',
         'Striatum / OFC': 'Str./OFC', 'Overall': 'Overall'}
XPOS, _x = {}, 0.
for _rr in GROUPS.values():
    for _r in _rr:
        XPOS[_r] = _x
        _x += 1
    _x += .6
GX = [_x + .4 + i * .85 for i in range(len(GROUPS) + 1)]


def rd(name):
    f = DATA / name
    return pd.read_csv(f, sep='\t', dtype={'subject': str}) if f.exists() else pd.DataFrame()


def p_text(p):
    return 'P > .999' if p > .999 else 'P < .001' if p < .001 else f'P = {p:.3f}'.replace('0.', '.')


def row(ax, dots, post, meta, ylabel, title, pos_col=HIGH, neg_col=LOW, q=(.02, .98)):
    """dots: DataFrame subject, roi, y, se; post: roi -> (mean, lo, hi, p_gt0); meta: summary rows (variant normal)."""
    v = dots.y.to_numpy()
    lo, hi = np.quantile(v, q)
    lo, hi = min(lo, 0), max(hi, 0)
    lo, hi = lo - .08 * (hi - lo), hi + .08 * (hi - lo)
    rng = np.random.default_rng(3)
    for roi, x in XPOS.items():
        d = dots[dots.roi == roi]
        if len(d):
            prec = 1 / d.se.to_numpy()
            a = .15 + .85 * (prec / prec.max()) ** .7
            xs = x - .14 + rng.uniform(-.1, .1, len(d))
            for xx, yy, aa in zip(xs, d.y.to_numpy(), a):
                c = mpl.colors.to_rgba(FG, aa)
                if lo <= yy <= hi:
                    ax.scatter(xx, yy, s=7, color=c, lw=0, zorder=2)
                else:
                    ax.scatter(xx, hi if yy > hi else lo, s=9, marker='^' if yy > hi else 'v', color=c, lw=0,
                               clip_on=False, zorder=2)
        if roi in post:
            m, a_, b_, p = post[roi]
            col = pos_col if a_ > 0 else neg_col if b_ < 0 else FG
            ax.plot([x + .12] * 2, [max(a_, lo), min(b_, hi)], color=col, lw=1.5, solid_capstyle='butt', zorder=3)
            ax.plot(x + .12, m, 'o', ms=3.8, color=col, zorder=4)
            sig = col != FG
            ax.text(x, hi + .03 * (hi - lo), p_text(p).replace('P = ', '').replace('P ', ''), ha='center',
                    va='bottom', fontsize=5, color=col if sig else '.5', fontweight='bold' if sig else 'normal')
    m = meta[meta.variant == 'normal'] if len(meta) else meta
    names = list(GROUPS) + ['Overall']
    for g, x in zip(names, GX):
        r = m[(m.unit == g) | ((g == 'Overall') & (m.level == 'overall'))] if len(m) else m
        if not len(r):
            continue
        r = r.iloc[0]
        col = pos_col if r.hdi_low > 0 else neg_col if r.hdi_high < 0 else FG
        ax.plot([x] * 2, [max(r.hdi_low, lo), min(r.hdi_high, hi)], color=col, lw=1.5, solid_capstyle='butt')
        ax.plot(x, r['mean'], 'D' if g == 'Overall' else 's', ms=3.4, color=col)
    ax.axhline(0, color=MUTED, lw=.7, ls='--', zorder=0)
    ax.set_ylim(lo, hi)
    ax.set_xlim(-.6, GX[-1] + .5)
    offset_spines(ax)
    ax.set_xticks(list(XPOS.values()) + GX, [LABEL.get(r, r) for r in XPOS] + [SHORT[g] for g in names],
                  fontsize=6)
    plt.setp(ax.get_xticklabels(), rotation=40, ha='right', rotation_mode='anchor')
    ax.set_ylabel(ylabel, fontsize=7)
    ax.yaxis.set_label_coords(-.075, .5)
    ax.set_title(title, loc='left', fontsize=7.5, pad=12)


def group_labels(ax):
    tr = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
    for g, rr in GROUPS.items():
        x0, x1 = XPOS[rr[0]] - .35, XPOS[rr[-1]] + .35
        ax.plot([x0, x1], [1.25, 1.25], transform=tr, color=MUTED, lw=.7, clip_on=False)
        ax.text((x0 + x1) / 2, 1.27, g, transform=tr, ha='center', va='bottom', fontsize=6.3, color='.3')
    ax.plot([GX[0] - .35, GX[-1] + .35], [1.25, 1.25], transform=tr, color=MUTED, lw=.7, clip_on=False)
    ax.text((GX[0] + GX[-1]) / 2, 1.27, 'Pathway means', transform=tr, ha='center', va='bottom', fontsize=6.3,
            color='.3')


def post_of(t, **sel):
    for k, v in sel.items():
        t = t[t[k] == v]
    return {r.roi: (r['mean'], r.hdi_low, r.hdi_high, r.p_gt0 if 'p_gt0' in r else r.p_high_gt_low)
            for _, r in t.iterrows()}


def colour_dots():
    t = pd.concat(pd.read_csv(f, sep='\t', dtype={'subject': str})
                  for f in sorted(glob.glob(str(DATA / 'value_rois' / 'sub-*_value_rois_runs.tsv'))))
    t = t[(t.points == 'exclude') & (t.measure == 'univariate_high_minus_low')].copy()
    odd = t.subject.astype(int) % 2 == 1
    t['gh'] = np.where(odd == (t.session.astype(int) == 1), 1, -1)
    rows = []
    for (s, roi), d in t.groupby(['subject', 'roi']):
        a, b = d[d.gh == 1].value, d[d.gh == -1].value
        # y = value + colour * gh / 2 per run, so mean(gh=+1) - mean(gh=-1) = colour (green - orange)
        rows.append(dict(subject=s, roi=roi, y=a.mean() - b.mean(),
                         se=np.sqrt(a.var() / len(a) + b.var() / len(b))))
    return pd.DataFrame(rows)


def main():
    bayes = rd('bayes_value_rois_exclude.tsv')
    gf_meta, gf_data, gf_post = rd(f'meta_value_gain_{GF}.tsv'), rd(f'meta_value_gain_{GF}_data.tsv'), \
        rd(f'hierarchical_value_gain_{GF}.tsv')
    fig, axes = plt.subplots(4, 1, figsize=(DOUBLE, 8.2), gridspec_kw=dict(hspace=1.05))
    row(axes[0], gf_data, post_of(gf_post), gf_meta, 'High − low\ngain field (%)',
        'Gain field at the distractor (pRF-mapped areas)')
    group_labels(axes[0])
    uni = 'univariate_high_minus_low'
    row(axes[1], rd(f'meta_value_rois_{uni}_exclude_data.tsv'), post_of(bayes, measure=uni, effect='value'),
        rd(f'meta_value_rois_{uni}_exclude.tsv'), 'High − low\n(% signal)', 'Mean response, high vs low value')
    auc = 'auc_across'
    row(axes[2], rd(f'meta_value_rois_{auc}_exclude_data.tsv'), post_of(bayes, measure=auc, effect='value'),
        rd(f'meta_value_rois_{auc}_exclude.tsv'), 'AUC − .5 (×100)',
        'Value decoded across sessions (colours swapped)')
    row(axes[3], colour_dots(), post_of(bayes, measure=uni, effect='colour'), pd.DataFrame(),
        'Green − orange\n(% signal)', 'Mean response, green vs orange distractor', pos_col=GREEN, neg_col=ORANGE)
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    for ax, l in zip(axes, 'abcd'):
        bb = inv.transform(ax.get_tightbbox(fig.canvas.get_renderer()))
        fig.text(bb[0, 0] - .005, bb[1, 1], l, fontsize=10, fontweight='bold', va='top', ha='right')
    # key, top right
    ax = axes[0]
    k = dict(transform=ax.transAxes, clip_on=False)
    y0, x0 = 1.62, .62
    for i, (a, lab) in enumerate([(1., 'Participant, precise'), (.2, 'Participant, noisy')]):
        ax.scatter([x0], [y0 - i * .14], s=8, color=mpl.colors.to_rgba(FG, a), lw=0, **k)
        ax.text(x0 + .012, y0 - i * .14, lab, fontsize=6, va='center', transform=ax.transAxes)
    ax.plot([x0 + .17] * 2, [y0 - .07, y0 + .07], color=HIGH, lw=1.5, **k)
    ax.plot(x0 + .17, y0, 'o', ms=3.8, color=HIGH, **k)
    ax.text(x0 + .182, y0, 'Group, 95% HDI above 0', fontsize=6, va='center', transform=ax.transAxes)
    ax.plot([x0 + .17] * 2, [y0 - .21, y0 - .07], color=LOW, lw=1.5, **k)
    ax.plot(x0 + .17, y0 - .14, 'o', ms=3.8, color=LOW, **k)
    ax.text(x0 + .182, y0 - .14, 'Group, 95% HDI below 0', fontsize=6, va='center', transform=ax.transAxes)
    ax.text(0, 1.62, 'Numbers: P(> 0)', transform=ax.transAxes, fontsize=6, color='.4', va='center')
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'fig_value_all_rois_bayes.{ext}', dpi=300, bbox_inches='tight')
    print(OUT / 'fig_value_all_rois_bayes.png')


if __name__ == '__main__':
    main()
