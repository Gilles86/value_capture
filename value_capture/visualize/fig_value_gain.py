"""Figure: hV4 amplifies the colour-singleton distractor, more so when it signals high value.

Gain fields fitted jointly with the PRFs on the primary betas (glmsingle_fbevents), all sessions together
(`gainfield/fit_gain_fields.py --free-prf`). A gain of 0.5 means the response to the stimulus at that item's
location is 1.5x that at a neutral item.

a, gain at the target, the low-value and the high-value distractor along the visual hierarchy (primary fit: DoG
PRFs, points trials excluded; mean +- SEM over subjects).
b, hV4, every subject: gain at the low- and the high-value distractor (primary fit).
c, hV4 high - low value in each of the six ways to fit it (Gaussian / DoG PRFs x points subtracted, excluded or
modelled as part of the stimulus), and averaged per subject over the six (mean, 95% bootstrap CI; exact
sign-flip p; subjects with high > low).

    python value_capture/visualize/fig_value_gain.py
"""
import glob
import itertools
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'notes' / 'data'
mpl.rcParams.update({
    'font.family': 'Helvetica', 'font.size': 7, 'axes.labelsize': 7.5, 'axes.titlesize': 7.5,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': .8, 'axes.spines.top': False,
    'axes.spines.right': False, 'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.major.size': 3,
    'ytick.major.size': 3, 'xtick.major.width': .8, 'ytick.major.width': .8, 'pdf.fonttype': 42,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': .02})
FG, MUTED = '#1F2933', '#9AA5B1'
COL = {'target': '#F77F00', 'low': '#009E73', 'high': '#7B3294'}
ROIS = ['V1', 'V2', 'V3', 'hV4', 'LO']
FITS = [('dog', 'exclude'), ('dog', 'subtract'), ('dog', 'model'),
        ('gauss', 'exclude'), ('gauss', 'subtract'), ('gauss', 'model')]
PRIMARY = ('dog', 'exclude')
FIT_LABEL = {'exclude': 'points excluded', 'subtract': 'points subtracted', 'model': 'points modelled'}


def signflip(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    s = np.array(list(itertools.product([-1, 1], repeat=len(x))))
    return (np.abs((s * x).mean(1)) >= abs(x.mean()) - 1e-12).mean()


def boot_ci(x, n=20000, seed=0):
    x = np.asarray(x, float)
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return np.percentile(m, [2.5, 97.5])


def load():
    out = []
    for prf, points in FITS:
        d = DATA / f"gain_fields{'_dog' if prf == 'dog' else ''}_fbevents_free_{points}"
        t = pd.concat([pd.read_csv(f, sep='\t', dtype={'subject': str}) for f in glob.glob(f'{d}/*.tsv')])
        out.append(t[t.scope == 'all'].assign(prf=prf, points=points))
    t = pd.concat(out)
    t = t.rename(columns={'field_gain_target': 'target', 'field_gain_distractor_low': 'low',
                          'field_gain_distractor_high': 'high'})
    t['high_low'] = t.high - t.low
    return t[['prf', 'points', 'subject', 'roi', 'n_voxels', 'target', 'low', 'high', 'high_low']]


def panel_profile(ax, t):
    d = t[(t.prf == PRIMARY[0]) & (t.points == PRIMARY[1]) & t.roi.isin(ROIS)]
    x = np.arange(len(ROIS))
    for k, (c, dx) in enumerate([('target', -.2), ('low', 0), ('high', .2)]):
        g = d.groupby('roi')[c]
        m, se = g.mean().reindex(ROIS), (g.std() / np.sqrt(g.count())).reindex(ROIS)
        ax.plot(x + dx, m, '-', color=COL[c], lw=1, alpha=.5, zorder=2)
        ax.errorbar(x + dx, m, se, fmt='o', ms=3.6, color=COL[c], mec='w', mew=.4, elinewidth=1, capsize=0,
                    zorder=3)
    ax.axhline(0, color=MUTED, lw=.6, ls=(0, (2, 2)), zorder=1)
    ax.axvspan(2.55, 3.45, color='#EEF0F2', lw=0, zorder=0)
    ax.set_xticks(x, ROIS)
    ax.set_xlim(-.5, len(ROIS) - .5)
    ax.set_ylabel('Gain at item (vs. neutral item)')
    for i, (lab, c) in enumerate([('Target', 'target'), ('Low-value distractor', 'low'),
                                  ('High-value distractor', 'high')]):
        ax.plot(.04, .95 - i * .085, 'o', ms=3.6, color=COL[c], mec='w', mew=.4, transform=ax.transAxes,
                clip_on=False)
        ax.text(.08, .95 - i * .085, lab, color=COL[c], fontsize=6.5, va='center', transform=ax.transAxes)
    ax.text(3, ax.get_ylim()[0] + .03, 'Colour\narea', ha='center', va='bottom', fontsize=6.5, color=FG)


def panel_subjects(ax, t):
    d = t[(t.prf == PRIMARY[0]) & (t.points == PRIMARY[1]) & (t.roi == 'hV4')].set_index('subject')
    up = d.high > d.low
    for s, r in d.iterrows():
        ax.plot([0, 1], [r.low, r.high], '-', color=COL['high'] if up[s] else MUTED, lw=.8,
                alpha=.8 if up[s] else .6, zorder=1)
    rng = np.random.default_rng(1)
    for i, c in enumerate(['low', 'high']):
        ax.scatter(np.full(len(d), i) + rng.uniform(-.04, .04, len(d)), d[c], s=9, color=COL[c],
                   edgecolor='w', lw=.3, zorder=2)
        ax.errorbar(i + (-.22 if i == 0 else .22), d[c].mean(), d[c].std() / np.sqrt(len(d)), fmt='o', ms=4.5,
                    color=COL[c], elinewidth=1.2, capsize=0, zorder=3)
    ax.set_xticks([0, 1], ['Low\nvalue', 'High\nvalue'])
    for lab, c in zip(ax.get_xticklabels(), ['low', 'high']):
        lab.set_color(COL[c])
    ax.set_xlim(-.45, 1.45)
    ax.axhline(0, color=MUTED, lw=.6, ls=(0, (2, 2)), zorder=0)
    ax.set_ylabel('hV4 gain at distractor')
    hl = d.high - d.low
    ax.set_title(f'{up.sum()} of {len(d)} subjects:\nhigh > low', fontsize=7, color=FG, pad=6)


def panel_multiverse(ax, t):
    d = t[t.roi == 'hV4']
    rows = []
    for prf, points in FITS:
        v = d[(d.prf == prf) & (d.points == points)].set_index('subject').high_low
        rows.append((f"{'DoG' if prf == 'dog' else 'Gaussian'}, {FIT_LABEL[points]}", v))
    avg = d.groupby('subject').high_low.mean()
    rows.append(('Average over the six', avg))
    y = np.arange(len(rows))[::-1].astype(float)
    y[-1] -= .5
    for (lab, v), yy in zip(rows, y):
        last = lab.startswith('Average')
        lo, hi = boot_ci(v)
        col = COL['high']
        ax.plot([lo, hi], [yy, yy], color=col, lw=1.6 if last else 1.1, solid_capstyle='butt')
        ax.plot(v.mean(), yy, 'D' if last else 'o', ms=5 if last else 3.8, color=col, mec='w', mew=.4)
        ax.text(1.02, yy, f'{(v > 0).sum()}/{len(v)}   p = {signflip(v):.2f}'.replace('0.', '.'),
                transform=ax.get_yaxis_transform(), va='center', fontsize=6.5, color=FG)
    ax.axvline(0, color=MUTED, lw=.6, ls=(0, (2, 2)))
    ax.set_yticks(y, [r[0] for r in rows])
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('hV4 gain, high − low value')
    ax.text(1.02, y[0] + .9, 'High > low', transform=ax.get_yaxis_transform(), fontsize=6.5, color=MUTED,
            va='center')
    ax.set_ylim(y[-1] - .7, y[0] + 1.2)


def main():
    t = load()
    fig = plt.figure(figsize=(7.25, 2.6))
    axes = [fig.add_axes(r) for r in ([.07, .17, .3, .68], [.45, .17, .13, .68], [.83, .17, .12, .68])]
    panel_profile(axes[0], t)
    panel_subjects(axes[1], t)
    panel_multiverse(axes[2], t)
    for ax in axes[:2]:
        sns.despine(ax=ax, offset=4, trim=False)
    sns.despine(ax=axes[2], offset=4, left=True)
    for ax, l in zip(axes, 'abc'):
        ax.text({'a': -.2, 'b': -.5, 'c': -1.95}[l], 1.12, l, transform=ax.transAxes, fontsize=9, fontweight='bold', ha='right', va='bottom')
    out = ROOT / 'notes' / 'figures' / 'fig_value_gain.pdf'
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix('.png'), dpi=200)
    print(out)


if __name__ == '__main__':
    main()
