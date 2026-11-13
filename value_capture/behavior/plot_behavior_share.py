"""Shareable behaviour figure (main study): RT capture and accuracy by distractor value.

    python value_capture/behavior/plot_behavior_share.py
Reads notes/data/behavior_trials_main.tsv (from fit_logrt_bambi.py export).
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
mpl.rcParams.update({
    'font.family': 'Helvetica', 'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': .8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.major.size': 3,
    'ytick.major.size': 3, 'xtick.major.width': .8, 'ytick.major.width': .8,
    'pdf.fonttype': 42, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': .02})

ORDER = ['none', 'low', 'high']
LABELS = {'none': 'No\ndistractor', 'low': 'Low-value\ndistractor', 'high': 'High-value\ndistractor'}
PAL = {'none': '#9C9C9C', 'low': '#3B5BA5', 'high': '#C44E52'}

df = pd.read_csv(ROOT / 'notes/data/behavior_trials_main.tsv', sep='\t', dtype={'subject': str})
df['correct'] = df['correct'].map({True: 1., False: 0., 'True': 1., 'False': 0.})  # n/a = no response
n = df.subject.nunique()
rt = (df[df.correct == 1].groupby(['subject', 'cond']).rt.mean() * 1000).unstack()[ORDER]
acc = (df.groupby(['subject', 'cond']).correct.mean() * 100).unstack()[ORDER]
cap = rt[['low', 'high']].sub(rt['none'], axis=0)


def ptxt(a, b, alternative='greater'):
    # One-sided: directional predictions (distractors slow RT / cost accuracy,
    # high value more than low; e.g. Anderson et al. 2011, Le Pelley et al. 2016)
    p = stats.ttest_rel(a, b, alternative=alternative).pvalue
    return 'p < .001' if p < .001 else f'p = {p:.3f}'.replace('0.', '.')


def panel(ax, w, ylabel, cols):
    x = np.arange(len(cols))
    for _, row in w[cols].iterrows():
        ax.plot(x, row.values, color='0.82', lw=.6, zorder=1)
    m, se = w[cols].mean(), w[cols].sem()
    for i, c in enumerate(cols):
        ax.errorbar(i, m[c], se[c], fmt='o', color=PAL[c], ms=5, lw=1.2, capsize=0, zorder=3,
                    mec='k', mew=.6)
    ax.set_xticks(x, [LABELS[c] for c in cols])
    ax.set_xlim(-.4, len(cols) - .6)
    ax.set_ylabel(ylabel)


def bracket(ax, i, j, y, txt, dy):
    ax.plot([i, i, j, j], [y - dy, y, y, y - dy], color='0.2', lw=.7)
    ax.text((i + j) / 2, y + dy * .4, txt, ha='center', va='bottom', fontsize=6.5)


fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), constrained_layout=True,
                         gridspec_kw=dict(width_ratios=[1.2, .9, 1.2]))
ax = axes[0]
panel(ax, rt, 'RT, correct trials (ms)', ORDER)
ax.set_ylim(650, 1200)
bracket(ax, 0, 1, 1110, ptxt(rt.low, rt.none), 12)
bracket(ax, 0, 2, 1165, ptxt(rt.high, rt.none), 12)

ax = axes[1]
panel(ax, cap, 'Capture: RT − no-distractor RT (ms)', ['low', 'high'])
ax.axhline(0, color='0.7', lw=.6, ls='--', zorder=0)
ax.set_ylim(-50, 125)
bracket(ax, 0, 1, 108, f'Value: {ptxt(cap.high, cap.low)}', 5)

ax = axes[2]
panel(ax, acc, 'Accuracy (% correct)', ORDER)
ax.set_ylim(86, 100.5)
bracket(ax, 1, 2, 98.8, ptxt(acc.high, acc.low, 'less'), .35)

for a, l in zip(axes, 'abc'):
    a.text(-.22, 1.04, l, transform=a.transAxes, fontsize=8, weight='bold', ha='right', va='bottom')
fig.suptitle(f'Colour singletons capture attention; high value costs accuracy  (N = {n}; one-sided paired t-tests)',
             fontsize=8, x=.02, ha='left')
sns.despine(fig=fig, offset=5, trim=True)
out = ROOT / 'notes/figures/behavior_share'
fig.savefig(f'{out}.pdf'); fig.savefig(f'{out}.png', dpi=300)
print(out, '\n', rt.mean().round(1).to_dict(), cap.mean().round(1).to_dict(), acc.mean().round(2).to_dict())
