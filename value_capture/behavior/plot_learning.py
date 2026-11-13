"""Learning of value-driven capture across runs (main study).

Per run (1-8) in each session: capture (distractor − no-distractor RT) for low and high value,
and the value effect (high − low). Mean ± SEM across subjects; the colour→value mapping reverses
at session 2, so a learned value effect should dip early in session 2 and rebuild.

    python value_capture/behavior/plot_learning.py
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
mpl.rcParams.update({
    'font.family': 'Helvetica', 'font.size': 7, 'axes.labelsize': 7.5, 'axes.linewidth': .8,
    'axes.spines.top': False, 'axes.spines.right': False, 'xtick.direction': 'out',
    'ytick.direction': 'out', 'xtick.major.size': 3, 'ytick.major.size': 3, 'pdf.fonttype': 42,
    'savefig.bbox': 'tight', 'savefig.pad_inches': .03})
FG, MUTED = '#1F2933', '#9AA5B1'
COL = {'low': '#4A7FB5', 'high': '#B2182B', 'value': '#6A3D9A'}

df = pd.read_csv(ROOT / 'notes/data/behavior_trials_main.tsv', sep='\t', dtype={'subject': str})
df['correct'] = df['correct'].map({True: 1., False: 0., 'True': 1., 'False': 0.})
c = df[df.correct == 1]
BIN = 2   # runs per bin (40 trials/run is noisy per subject)
c = c.assign(rbin=(c.run - 1) // BIN)
rt = (c.groupby(['subject', 'session', 'rbin', 'cond']).rt.mean() * 1000).unstack()
d = pd.DataFrame({'low': rt.low - rt.none, 'high': rt.high - rt.none, 'value': rt.high - rt.low}).reset_index()
d['x'] = d.rbin + (d.session - 1) * (8 // BIN + .8)
n = d.subject.nunique()

fig, axes = plt.subplots(1, 2, figsize=(7.09, 2.2), gridspec_kw=dict(wspace=.5))
for ax, cols, ylab in ((axes[0], ['low', 'high'], 'Capture: RT − no distractor (ms)'),
                       (axes[1], ['value'], 'Value effect: high − low (ms)')):
    for k, col in enumerate(cols):
        g = d.groupby('x')[col]
        m, se = g.mean(), g.sem()
        off = (k - (len(cols) - 1) / 2) * .12
        for ses in (1, 2):
            xs = m.index[(m.index >= (ses - 1) * (8 // BIN + .8)) & (m.index < ses * (8 // BIN + .8) - .5)]
            ax.plot(xs + off, m[xs], color=COL[col], lw=1.2, zorder=3)
        ax.errorbar(m.index + off, m, se, fmt='o', color=COL[col], ms=4.5, mec='w', mew=.6, lw=1.1, capsize=0, zorder=4)
        lab = {'low': 'Low value', 'high': 'High value', 'value': None}[col]
        if lab:
            ax.text(m.index[-1] + .3, m.iloc[-1] + (6 if col == 'high' else -6), lab, color=COL[col], fontsize=6.5, va='center')
    ax.axhline(0, color='.75', lw=.6, ls='--', zorder=0)
    ax.axvline(8 // BIN - .1, color=MUTED, lw=.6, zorder=0)
    ticks = [b + (s - 1) * (8 // BIN + .8) for s in (1, 2) for b in range(8 // BIN)]
    ax.set_xticks(ticks, [f'{b * BIN + 1}–{b * BIN + BIN}' for s in (1, 2) for b in range(8 // BIN)])
    ax.set_xlabel('Runs', labelpad=2)
    ax.set_ylabel(ylab)
    for s in ('left', 'bottom'):
        ax.spines[s].set_position(('outward', 4))
    yl = ax.get_ylim()
    for s, xm in ((1, 1.5), (2, 1.5 + 8 // BIN + .8)):
        ax.text(xm, yl[1], f'Session {s}', ha='center', va='bottom', fontsize=7, color=FG)
    if len(cols) == 2:
        ax.text(8 // BIN - .1, yl[0], ' Colour–value\n mapping swaps', fontsize=6, color='.45', va='bottom')
# value effect: early (first bin) vs rest of session 2
v = d.pivot_table(index='subject', columns=['session', 'rbin'], values='value')
early, late = v[(2, 0)], v[2].iloc[:, 1:].mean(1)
p = stats.ttest_rel(late, early, alternative='greater').pvalue
axes[1].text(.98, .03, f'Session 2, runs 3–8 vs 1–2:\none-sided p = {p:.2f}'.replace('0.', '.'),
             transform=axes[1].transAxes, ha='right', fontsize=6.3, color=FG)
for a, l in zip(axes, 'ab'):
    a.text(-.18, 1.1, l, transform=a.transAxes, fontsize=8, weight='bold')
out = ROOT / 'notes/figures/behavior_learning'
fig.savefig(f'{out}.pdf')
fig.savefig(f'{out}.png', dpi=300)
print(out, f'n = {n}')
print(d.groupby(['session', 'rbin'])[['low', 'high', 'value']].mean().round(1))
