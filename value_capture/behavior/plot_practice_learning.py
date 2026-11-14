"""Is the colour→value mapping learned during practice, or present from the first trials?

Practice per session: run -1 (psychophysics room, 40 trials) then run -2 (scanner, during
shimming, 40 trials), feedback on every trial. Session 2 reverses the mapping; the value effect is
always coded by the CURRENT mapping (high − low), so carry-over from session 1 would make it
negative early in session 2. Instructions show the mapping explicitly before each session.

a, Capture (distractor − absent RT) per block: practice runs, then scanner runs in pairs.
b, Value effect (high − low RT) per block.
c, Within practice, by 20-trial bin (80 practice trials): value effect, session 1 and 2.
d, Mixed model on log RT within practice, per session: value effect at the first practice trial
   and its change over the 80 trials (random intercept + value slope per subject).

Correct trials, RT ≥ 0.2 s, raw data (no outlier removal) for practice and scanner runs alike.
Mean ± SEM across subjects (n = 12, main study).

    python value_capture/behavior/plot_practice_learning.py
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
mpl.rcParams.update({
    'font.family': 'Helvetica', 'font.size': 7, 'axes.labelsize': 7.5, 'axes.linewidth': .8,
    'axes.spines.top': False, 'axes.spines.right': False, 'xtick.direction': 'out',
    'ytick.direction': 'out', 'xtick.major.size': 3, 'ytick.major.size': 3, 'pdf.fonttype': 42,
    'savefig.bbox': 'tight', 'savefig.pad_inches': .03})
FG, MUTED = '#1F2933', '#9AA5B1'
COL = {'low': '#4A7FB5', 'high': '#B2182B', 'value': '#6A3D9A', 1: '#9AA5B1', 2: '#6A3D9A'}
SHADE = '#EEF0F2'


def prep(path):
    d = pd.read_csv(ROOT / path, sep='\t', dtype={'subject': str})
    d['correct'] = d['correct'].map({True: 1., False: 0., 'True': 1., 'False': 0., 1.: 1., 0.: 0.})
    return d


prac, main = prep('notes/data/behavior_trials_practice.tsv'), prep('notes/data/behavior_trials_main_raw.tsv')
prac['block'] = prac.run.map({-1: 0, -2: 1})                 # P1 (lab), P2 (scanner shim)
prac['ptrial'] = prac.block * 40 + prac.trial_nr              # 1..80 within practice
main['block'] = 2 + (main.run - 1) // 2                        # runs 1-2 .. 7-8 -> 2..5
ok = lambda d: d[(d.correct == 1) & (d.rt >= .2)]  # noqa: E731
allc = pd.concat([ok(prac), ok(main)])
rt = (allc.groupby(['subject', 'session', 'block', 'cond']).rt.mean() * 1000).unstack()
d = pd.DataFrame({'low': rt.low - rt.none, 'high': rt.high - rt.none, 'value': rt.high - rt.low}).reset_index()
NB, GAP = 6, 1.0
d['x'] = d.block + (d.session - 1) * (NB + GAP)
LABS = ['P1', 'P2', '1–2', '3–4', '5–6', '7–8']

W, H = 7.09, 4.3
fig = plt.figure(figsize=(W, H))
at = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])  # noqa: E731


def timeline(ax, cols, ylab):
    for s in (1, 2):
        x0 = (s - 1) * (NB + GAP)
        ax.axvspan(x0 - .45, x0 + 1.45, color=SHADE, lw=0, zorder=0)
    for k, col in enumerate(cols):
        g = d.groupby('x')[col]
        m, se = g.mean(), g.sem()
        off = (k - (len(cols) - 1) / 2) * .14
        for s in (1, 2):
            xs = m.index[(m.index >= (s - 1) * (NB + GAP)) & (m.index < s * (NB + GAP) - GAP)]
            ax.plot(xs + off, m[xs], color=COL[col], lw=1.1, zorder=3)
        ax.errorbar(m.index + off, m, se, fmt='o', color=COL[col], ms=4, mec='w', mew=.6, lw=1, capsize=0, zorder=4)
    ax.axhline(0, color='.6', lw=.6, ls='--', zorder=1)
    ax.set_xticks([b + (s - 1) * (NB + GAP) for s in (1, 2) for b in range(NB)], LABS * 2)
    ax.set_xlim(-.7, 2 * NB + GAP - .3)
    ax.set_ylabel(ylab)
    ax.set_xlabel('Practice runs  ·  scanner runs', labelpad=2)
    yl = ax.get_ylim()
    for s in (1, 2):
        x0 = (s - 1) * (NB + GAP)
        ax.text(x0 + 2.5, yl[1], f'Session {s}' + ('  (mapping reversed)' if s == 2 else ''),
                ha='center', va='bottom', fontsize=7, color=FG)
        ax.text(x0 + .5, yl[0], 'Practice', ha='center', va='bottom', fontsize=6, color='.45')


ax = at(.55, 2.55, 2.75, 1.35)
timeline(ax, ['low', 'high'], 'Capture (ms)\ndistractor − absent')
for col, lab in (('high', 'High value'), ('low', 'Low value')):
    m = d[d.x == d.x.max()][col].mean()
    ax.text(2 * NB + GAP - .45, m + (5 if col == 'high' else -5), lab, color=COL[col], fontsize=6.3, va='center', ha='left')
ax.text(-.2, 1.12, 'a', transform=ax.transAxes, fontsize=8, weight='bold')

ax = at(4.3, 2.55, 2.75, 1.35)
timeline(ax, ['value'], 'Value effect (ms)\nhigh − low')
ax.text(-.2, 1.12, 'b', transform=ax.transAxes, fontsize=8, weight='bold')

# c: within practice, 10-trial bins
pc = ok(prac).assign(bin=lambda x: (x.ptrial - 1) // 20)
r = (pc.groupby(['subject', 'session', 'bin', 'cond']).rt.mean() * 1000).unstack()
v = (r.high - r.low).rename('value').reset_index()
ax = at(.55, .45, 2.6, 1.4)
ax.axvspan(1.5, 3.5, color=SHADE, lw=0, zorder=0)
for k, s in enumerate((1, 2)):
    g = v[v.session == s].groupby('bin').value
    m, se = g.mean(), g.sem()
    off = (k - .5) * .15
    ax.plot(m.index + off, m, color=COL[s], lw=1.1, zorder=3)
    ax.errorbar(m.index + off, m, se, fmt='o', color=COL[s], ms=4, mec='w', mew=.6, lw=1, capsize=0, zorder=4)
    ax.text(3.3, m.iloc[-1], f'Session {s}', color=COL[s], fontsize=6.3, va='center')
ax.axhline(0, color='.6', lw=.6, ls='--', zorder=1)
ax.set_xticks(range(4), [f'{20 * b + 1}–{20 * b + 20}' for b in range(4)])
ax.set_xlim(-.4, 4.1)
ax.set_xlabel('Practice trial', labelpad=1)
ax.set_ylabel('Value effect (ms)\nhigh − low')
yl = ax.get_ylim()
ax.text(.5, yl[1], 'Lab (P1)', ha='center', va='top', fontsize=6, color='.45')
ax.text(2.5, yl[1], 'Scanner (P2)', ha='center', va='top', fontsize=6, color='.45')
ax.text(-.2, 1.12, 'c', transform=ax.transAxes, fontsize=8, weight='bold')

# d: mixed model on log RT within practice
pm_ = ok(prac).copy()
pm_['logrt'] = np.log(pm_.rt)
pm_['t'] = (pm_.ptrial - 1) / 79                                  # 0 = first, 1 = last practice trial
pm_ = pm_[pm_.cond != 'none'].assign(value=lambda x: np.where(x.cond == 'high', .5, -.5))
rows = []
for s in (1, 2):
    dd = pm_[pm_.session == s]
    f = smf.mixedlm('logrt ~ value * t', dd, groups='subject', re_formula='~value').fit(reml=True)
    ci = f.conf_int()
    for term, lab in (('value', 'Value effect at\nfirst trial'), ('value:t', 'Change over\n80 trials')):
        rows.append(dict(session=s, term=lab, est=100 * f.params[term], lo=100 * ci.loc[term, 0],
                         hi=100 * ci.loc[term, 1], p=f.pvalues[term]))
    vt = dd.assign(late=dd.t > .5)
    print(f'session {s}: value {f.params["value"]:+.3f} p={f.pvalues["value"]:.3f}; '
          f'value:t {f.params["value:t"]:+.3f} p={f.pvalues["value:t"]:.3f}')
mm = pd.DataFrame(rows)
ax = at(4.35, .45, 2.2, 1.4)
for i, term in enumerate(mm.term.unique()):
    for k, s in enumerate((1, 2)):
        rr = mm[(mm.term == term) & (mm.session == s)].iloc[0]
        y = -(i + k * .3)
        ax.plot([rr.lo, rr.hi], [y, y], color=COL[s], lw=1.4, solid_capstyle='butt')
        ax.plot(rr.est, y, 'o', color=COL[s], ms=4.5, mec='w', mew=.5, zorder=3)
        ax.text(max(rr.hi, 0) + 1, y, f'p = {rr.p:.2f}'.replace('0.', '.'), fontsize=6, color=COL[s], va='center')
ax.axvline(0, color='.6', lw=.6, ls='--', zorder=0)
ax.set_yticks([-.15, -1.15], mm.term.unique())
ax.tick_params(axis='y', length=0)
ax.spines['left'].set_visible(False)
ax.set_ylim(-1.6, .5)
ax.set_xlabel('Effect on RT (%)\nhigh − low, current mapping', labelpad=2)
for k, s in enumerate((1, 2)):
    ax.text(1.0, 1.05 - k * .1, f'Session {s}', transform=ax.transAxes, color=COL[s], fontsize=6.3, ha='right')
ax.text(-.55, 1.12, 'd', transform=ax.transAxes, fontsize=8, weight='bold')

for a_ in fig.axes:
    for s in ('left', 'bottom'):
        if a_.spines[s].get_visible():
            a_.spines[s].set_position(('outward', 3))
out = ROOT / 'notes/figures/behavior_practice_learning'
fig.savefig(f'{out}.pdf')
fig.savefig(f'{out}.png', dpi=300)
print(out)
print(d.groupby(['session', 'block'])[['low', 'high', 'value']].agg(['mean', 'sem']).round(1))
print(mm.round(3))
