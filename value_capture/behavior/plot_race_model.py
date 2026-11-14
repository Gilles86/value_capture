"""Figure of the racing-diffusion fit (fit_race_model.py): drift costs, value effect, PPC.

a, Group-level change in the correct-response drift for low- and high-value distractors
   (vs no distractor), all runs and runs 3-8: posterior mean and 95% HDI.
b, The value effect (high − low) per subject, sorted; dashed: group mean.
c, d, Posterior predictive check: observed mean correct RT / accuracy (dots) vs the model
   (mean and 95% interval over 200 simulated datasets, within-subject then across).

    python value_capture/behavior/plot_race_model.py
"""
import sys
from pathlib import Path

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from fit_race_model import load, simulate  # noqa: E402

mpl.rcParams.update({
    'font.family': 'Helvetica', 'font.size': 7, 'axes.labelsize': 7.5, 'axes.linewidth': .8,
    'axes.spines.top': False, 'axes.spines.right': False, 'xtick.direction': 'out',
    'ytick.direction': 'out', 'xtick.major.size': 3, 'ytick.major.size': 3, 'pdf.fonttype': 42,
    'savefig.bbox': 'tight', 'savefig.pad_inches': .03})
FG, MUTED = '#1F2933', '#9AA5B1'
COL = {'none': MUTED, 'low': '#4A7FB5', 'high': '#B2182B', 'value': '#6A3D9A'}
ORDER = ['none', 'low', 'high']
LAB = {'none': 'Absent', 'low': 'Low', 'high': 'High'}
pct = lambda x: 100 * (np.exp(x) - 1)  # noqa: E731  log-drift -> % change

fits = {'All runs': az.from_netcdf(ROOT / 'notes/data/race_runs1-8.nc').posterior,
        'Runs 3–8': az.from_netcdf(ROOT / 'notes/data/race_runs3-8.nc').posterior}

W, H = 7.09, 2.25
fig = plt.figure(figsize=(W, H))
at = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])  # noqa: E731

# a: group forest
ax = at(.75, .45, 1.55, 1.45)
rows = [('beta_low_mu', 'Low value', COL['low']), ('beta_high_mu', 'High value', COL['high']),
        ('value_effect', 'High − low', COL['value'])]
for i, (k, lab, c) in enumerate(rows):
    for j, (fname, post) in enumerate(fits.items()):
        d = post[k].values.ravel()
        h = az.hdi(d, hdi_prob=.95)
        y = -(i * 1.0 + j * .32)
        ax.plot(pct(h), [y, y], color=c, lw=1.4, solid_capstyle='butt')
        ax.plot(pct(d.mean()), y, 'o', color=c if j == 0 else 'w', mec=c, mew=1.1, ms=5, zorder=3)
        if k == 'value_effect':
            ax.text(4, y, f'P = {(d < 0).mean():.3f}'.replace('0.', '.'), fontsize=6, color=FG, va='center')
ax.axvline(0, color='.6', lw=.6, ls='--', zorder=0)
ax.set_yticks([-.16, -1.16, -2.16], [r[1] for r in rows])
ax.tick_params(axis='y', length=0)
ax.spines['left'].set_visible(False)
ax.set_xlim(-20, 12)
ax.set_xticks([-20, -10, 0, 10])
ax.set_xlabel('Change in target drift (%)')
ax.spines['bottom'].set_position(('outward', 4))
# glyph key: filled = all runs, open = runs 3-8
for j, lab in enumerate(fits):
    ax.plot(.62, .99 - j * .09, 'o', transform=ax.transAxes, color=FG if j == 0 else 'w', mec=FG, mew=1, ms=4,
            clip_on=False)
    ax.text(.67, .99 - j * .09, lab, transform=ax.transAxes, fontsize=6.3, va='center')
ax.text(-.62, 1.08, 'a', transform=ax.transAxes, fontsize=8, weight='bold')

# b: per-subject value effect, sorted (all runs)
post = fits['All runs']
ve = (post['beta_high'].values - post['beta_low'].values).reshape(-1, post['beta_high'].shape[-1]).T  # subject x samples
m = pct(ve.mean(1))
h = np.array([az.hdi(v, hdi_prob=.95) for v in ve])
order = np.argsort(m)
ax = at(2.85, .45, 1.2, 1.45)
for r, sidx in enumerate(order):
    ax.plot(pct(h[sidx]), [r, r], color=COL['value'], lw=1, alpha=.7)
    ax.plot(m[sidx], r, 'o', color=COL['value'], ms=3.5, mec='w', mew=.5, zorder=3)
gm = pct(post['value_effect'].values.mean())
ax.axvline(gm, color=COL['value'], lw=.8, ls='--')
ax.axvline(0, color='.6', lw=.6, zorder=0)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.set_xlabel('Value effect on drift (%)')
ax.spines['bottom'].set_position(('outward', 4))
n_neg = int((m < 0).sum())
ax.set_title(f'{n_neg}/{len(m)} subjects below 0', fontsize=6.5, color=FG, loc='left')
ax.text(-.12, 1.08, 'b', transform=ax.transAxes, fontsize=8, weight='bold')

# c, d: PPC (all runs)
df = load(1)
sim = simulate(df, post, n_draws=200)
obs = df.assign(rt=np.where(df.correct == 1, df.rt, np.nan))
for kpanel, (col, ylab, scale, x0, letter) in enumerate(
        [('rt', 'Correct RT (ms)', 1000, 4.65, 'c'), ('correct', 'Accuracy (%)', 100, 6.2, 'd')]):
    ax = at(x0, .45, .95, 1.45)
    o = obs.groupby(['subject', 'cond'])[col].mean().unstack()[ORDER] * scale
    p = sim.groupby(['draw', 'subject', 'cond'])[col].mean().groupby(['draw', 'cond']).mean().unstack()[ORDER] * scale
    lo, hi, mu = p.quantile(.025), p.quantile(.975), p.mean()
    x = np.arange(3)
    ax.fill_between(x, lo, hi, color=MUTED, alpha=.35, lw=0)
    ax.plot(x, mu, color=FG, lw=1.1)
    for i, c in enumerate(ORDER):
        ax.plot(i, o[c].mean(), 'o', color=COL[c], ms=5.5, mec='w', mew=.7, zorder=4)
    ax.set_xticks(x, [LAB[c] for c in ORDER])
    ax.set_xlim(-.4, 2.4)
    ax.set_ylabel(ylab)
    ax.set_xlabel('Distractor value', labelpad=2)
    for s in ('left', 'bottom'):
        ax.spines[s].set_position(('outward', 4))
    if kpanel == 0:
        ax.add_patch(plt.Rectangle((.04, .91), .14, .07, transform=ax.transAxes, color=MUTED, alpha=.35, lw=0))
        ax.plot([.04, .18], [.945, .945], transform=ax.transAxes, color=FG, lw=1.1)
        ax.text(.22, .945, 'Model', transform=ax.transAxes, fontsize=6.3, va='center')
        ax.plot(.11, .83, 'o', transform=ax.transAxes, color=FG, ms=4, mec='w')
        ax.text(.22, .83, 'Data', transform=ax.transAxes, fontsize=6.3, va='center')
    ax.text(-.38, 1.08, letter, transform=ax.transAxes, fontsize=8, weight='bold')

out = ROOT / 'notes/figures/race_model'
fig.savefig(f'{out}.pdf')
fig.savefig(f'{out}.png', dpi=300)
print(out)
