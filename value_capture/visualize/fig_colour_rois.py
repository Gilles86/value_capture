"""Figure: how well can we see the distractor's colour (and value), in every area?

Colour and value are confounded within a session and swapped between sessions, so:

a  Decoding across sessions (train one session, test the other, value labels; shrinkage LDA, AUC). Above .5 = a
   value code that survives the colour swap; below .5 = a colour code (colour AUC = 1 - AUC).
b  Decoding within a session (leave one run out): colour OR value, the upper bound on what an area knows about the
   distractor.
c  Mean response, green - orange distractor (% signal change, both sessions, so value is balanced).
d  Gain-field amplitude at the distractor, green - orange (DoG, pRFs fitted jointly, points trials excluded); visual
   areas only.

Dots = subjects (filled: trials without the points display; the points text is drawn in the distractor's colour, so
those trials are the clean ones). Open circles = all trials, median only. Bars = median and bootstrapped 95% CI.
Numbers = two-sided Wilcoxon p against chance / 0 (trials without points), uncorrected.
`decode/value_rois.py`, `gainfield/fit_gain_fields.py --split-colour`.

    python value_capture/visualize/fig_colour_rois.py
"""
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import numpy as np
import pandas as pd

from value_capture.visualize.fig_value_gain import DOUBLE, FG, HIGH, MUTED, median_ci, offset_spines
from value_capture.visualize.fig_value_items import p_short, wil

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'notes' / 'data'
OUT = ROOT / 'notes' / 'figures'
GREEN, ORANGE = '#00ab78', '#d56f2c'         # the stimulus colours
ALL_C = '#52606D'
GROUPS = [('Early', ['V1', 'V2', 'V3']), ('Dorsal', ['V3AB', 'IPS']), ('Occ.-temp.', ['hV4', 'VO', 'LO', 'TO']),
          ('Ventral temporal / MTL', ['fusiform', 'parahippocampal', 'entorhinal', 'hippocampus', 'amygdala']),
          ('Striatum / OFC', ['caudate', 'putamen', 'accumbens', 'medialOFC', 'lateralOFC'])]
LABEL = {'fusiform': 'Fusiform', 'parahippocampal': 'Parahipp.', 'entorhinal': 'Entorhinal',
         'hippocampus': 'Hippoc.', 'amygdala': 'Amygdala', 'caudate': 'Caudate', 'putamen': 'Putamen',
         'accumbens': 'Accumbens', 'medialOFC': 'Medial OFC', 'lateralOFC': 'Lateral OFC'}
XPOS, _x = {}, 0.
for _g, _rois in GROUPS:
    for _r in _rois:
        XPOS[_r] = _x
        _x += 1
    _x += .6
ORDER = [r for _, rs in GROUPS for r in rs]


def read(kind):
    return pd.concat(pd.read_csv(f, sep='\t', dtype={'subject': str})
                     for f in sorted(glob.glob(str(DATA / 'value_rois' / f'sub-*_value_rois_{kind}.tsv'))))


def load():
    dec = read('decoding').groupby(['points', 'scheme', 'roi', 'subject']).auc.mean()
    uni = read('univariate').groupby(['points', 'roi', 'subject', 'colour']).beta.mean().unstack('colour')
    uni = uni.green - uni.orange
    gf = []
    for f in sorted(glob.glob(str(DATA / 'gain_fields_dog_fbevents_free_exclude_colour' / 'sub-*_gain_fields.tsv'))):
        t = pd.read_csv(f, sep='\t', dtype={'subject': str})
        gf.append(t[t.scope == 'all'])
    gf = pd.concat(gf) if gf else pd.DataFrame()
    if len(gf):
        c = lambda k: gf[f'field_gain_distractor_{k}']                      # noqa: E731
        gf = gf.assign(d=(c('green_low') + c('green_high') - c('orange_low') - c('orange_high')) / 2 * 100) \
            .set_index(['roi', 'subject']).d
    return dec, uni, gf


def series(ax, clean, allt, ref, ylim):
    """clean / allt: Series indexed (roi, subject)."""
    lo, hi = ylim
    for roi in ORDER:
        if roi not in clean.index.get_level_values('roi'):
            continue
        x = XPOS[roi]
        v = clean.xs(roi, level='roi').dropna().to_numpy()
        jit = np.random.default_rng(abs(hash(roi)) % 2 ** 32).uniform(-.13, .13, len(v))
        inside = (v >= lo) & (v <= hi)
        ax.scatter(x - .12 + jit[inside], v[inside], s=5, color=FG, alpha=.45, lw=0, zorder=2)
        for vv, jj in zip(v[~inside], jit[~inside]):
            ax.scatter(x - .12 + jj, hi if vv > hi else lo, s=9, marker='^' if vv > hi else 'v', color=FG,
                       alpha=.45, lw=0, zorder=2, clip_on=False)
        m, a, b = median_ci(v)
        ax.plot([x + .08] * 2, [a, b], color=FG, lw=1.3, solid_capstyle='butt', zorder=3)
        ax.plot(x + .08, m, 'o', ms=3.6, color=FG, zorder=4)
        if allt is not None and roi in allt.index.get_level_values('roi'):
            ax.plot(x + .3, np.median(allt.xs(roi, level='roi').dropna()), 'o', ms=3.4, mfc='white', mec=ALL_C,
                    mew=.9, zorder=4)
        p = wil(v - ref)
        ax.text(x, hi + .04 * (hi - lo), p_short(p), ha='center', va='bottom', fontsize=5.5,
                color=FG if p < .05 else '.45', fontweight='bold' if p < .05 else 'normal', clip_on=False)
    ax.axhline(ref, color=MUTED, lw=.7, ls='--', zorder=0)
    ax.set_ylim(lo, hi)
    ax.set_xlim(-.7, max(XPOS.values()) + .7)
    offset_spines(ax)                       # before the tick labels: moving a spine resets them
    ax.set_xticks([XPOS[r] for r in ORDER], [LABEL.get(r, r) for r in ORDER], fontsize=6.3)
    plt.setp(ax.get_xticklabels(), rotation=35, ha='right', rotation_mode='anchor')


def group_labels(ax):
    tr = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
    for name, rois in GROUPS:
        x0, x1 = XPOS[rois[0]] - .4, XPOS[rois[-1]] + .4
        ax.plot([x0, x1], [1.17, 1.17], transform=tr, color=MUTED, lw=.7, clip_on=False, solid_capstyle='butt')
        ax.text((x0 + x1) / 2, 1.19, name, transform=tr, ha='center', va='bottom', fontsize=6.3, color='.3')


def lim(s, ref, q=(.03, .97), pad=.1):
    v = s.dropna().to_numpy()
    a, b = np.quantile(v, q)
    a, b = min(a, ref), max(b, ref)
    return a - pad * (b - a), b + pad * (b - a)


def main():
    dec, uni, gf = load()
    n_gf = gf.index.get_level_values('subject').nunique() if len(gf) else 0
    rows = 4 if n_gf else 3
    fig, axes = plt.subplots(rows, 1, figsize=(DOUBLE, 1.75 * rows + .5), gridspec_kw=dict(hspace=1.2))
    letters = 'abcd'

    # a: across sessions
    ax = axes[0]
    c, a_ = dec.loc['exclude', 'across'], dec.loc['keep', 'across']
    series(ax, c, a_, .5, lim(c, .5))
    ax.set_ylabel('AUC, high vs low\n(across sessions)')
    tr = mtrans.blended_transform_factory(ax.transAxes, ax.transData)
    lo, hi = ax.get_ylim()
    ax.annotate('', xy=(-.068, hi), xytext=(-.068, .5 + .1 * (hi - .5)), xycoords=tr,
                arrowprops=dict(arrowstyle='-|>', color=HIGH, lw=.9), annotation_clip=False)
    ax.annotate('', xy=(-.068, lo), xytext=(-.068, .5 - .1 * (.5 - lo)), xycoords=tr,
                arrowprops=dict(arrowstyle='-|>', color=GREEN, lw=.9), annotation_clip=False)
    ax.text(-.078, (hi + .5) / 2, 'Value', transform=tr, color=HIGH, ha='right', va='center', fontsize=6.5, rotation=0)
    ax.text(-.078, (lo + .5) / 2, 'Colour', transform=tr, color=GREEN, ha='right', va='center', fontsize=6.5, rotation=0)
    ax.yaxis.set_label_coords(-.15, .5)
    ax.set_title('Trained on one session, tested on the other (colours swapped)',
                 loc='left', fontsize=7.5, pad=30)
    group_labels(ax)

    # b: within session
    ax = axes[1]
    c, a_ = dec.loc['exclude', 'within'], dec.loc['keep', 'within']
    series(ax, c, a_, .5, lim(c, .5))
    ax.set_ylabel('AUC, high vs low\n(within session)')
    ax.yaxis.set_label_coords(-.15, .5)
    ax.set_title('Within a session: colour or value (any information about the distractor)', loc='left',
                 fontsize=7.5, pad=14)

    # c: univariate
    ax = axes[2]
    c, a_ = uni.loc['exclude'] * 1, uni.loc['keep'] * 1
    series(ax, c, a_, 0., lim(c, 0.))
    ax.set_ylabel('Green − orange\n(% signal)')
    ax.yaxis.set_label_coords(-.15, .5)
    ax.set_title('Mean response to a green vs an orange distractor', loc='left', fontsize=7.5, pad=14)

    if n_gf:
        ax = axes[3]
        series(ax, gf, None, 0., lim(gf, 0.))
        ax.set_ylabel('Green − orange\ngain (%)')
        ax.yaxis.set_label_coords(-.15, .5)
        prov = '' if n_gf >= 12 else f'  [PROVISIONAL: {n_gf} of 12 subjects]'
        ax.set_title('Gain field at the distractor, green vs orange (visual areas)' + prov, loc='left',
                     fontsize=7.5, pad=14)
    for ax, l in zip(axes, letters):
        ax.text(-.19, 1.12 if ax is not axes[0] else 1.32, l, transform=ax.transAxes, fontsize=10,
                fontweight='bold', va='bottom')

    # key, top right of panel a
    ax = axes[0]
    k = dict(transform=ax.transAxes, clip_on=False)
    y0 = 1.5
    ax.scatter([.80], [y0], s=6, color=FG, alpha=.45, lw=0, **k)
    ax.plot([.81], [y0], 'o', ms=3.6, color=FG, **k)
    ax.plot([.81, .81], [y0 - .07, y0 + .07], color=FG, lw=1.3, **k)
    ax.text(.825, y0, 'Trials without points: subjects, median, 95% CI', fontsize=6, va='center',
            transform=ax.transAxes)
    ax.plot([.81], [y0 - .14], 'o', ms=3.4, mfc='white', mec=ALL_C, mew=.9, **k)
    ax.text(.825, y0 - .14, 'All trials: median', fontsize=6, va='center', transform=ax.transAxes)
    OUT.mkdir(exist_ok=True, parents=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'fig_colour_rois.{ext}', dpi=300, bbox_inches='tight')
    print(OUT / 'fig_colour_rois.pdf')


if __name__ == '__main__':
    main()
