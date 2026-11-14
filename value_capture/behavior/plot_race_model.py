"""Figure of the racing-diffusion fits (fit_race_models.py), built around the 'minimal' model.

a, Group effects on the target drift (% change): distractor present vs absent, and high vs low
   value. Minimal model on scanner runs (filled) and with practice runs (open); for the value
   effect also the drift-only model with per-subject value slopes (grey), whose interval is wider.
b, Posterior predictive check of the minimal model: correct-RT quantiles per condition
   (data dots, model line + 95% band; quantiles per subject, then averaged, per draw).
c, Same for accuracy.
d, PSIS-LOO: ELPD of every variant relative to the best (± SE of the difference).

    python value_capture/behavior/plot_race_model.py
"""
import sys
from pathlib import Path

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from fit_race_model import load  # noqa: E402

mpl.rcParams.update({
    'font.family': 'Helvetica', 'font.size': 7, 'axes.labelsize': 7.5, 'axes.linewidth': .8,
    'axes.spines.top': False, 'axes.spines.right': False, 'xtick.direction': 'out',
    'ytick.direction': 'out', 'xtick.major.size': 3, 'ytick.major.size': 3, 'pdf.fonttype': 42,
    'savefig.bbox': 'tight', 'savefig.pad_inches': .03})
FG, MUTED = '#1F2933', '#9AA5B1'
COL = {'none': '#7B8794', 'low': '#4A7FB5', 'high': '#B2182B', 'value': '#6A3D9A', 'dist': '#3E4C59'}
ORDER = ['none', 'low', 'high']
LAB = {'none': 'Absent', 'low': 'Low', 'high': 'High'}
Q = [.1, .3, .5, .7, .9]
pct = lambda x: 100 * (np.exp(x) - 1)  # noqa: E731  log-drift -> % change
nc = lambda tag: az.from_netcdf(ROOT / f'notes/data/race_{tag}.nc')  # noqa: E731

post_min, post_pr, post_drift = nc('minimal').posterior, nc('minimal_wpractice').posterior, nc('drift').posterior
drift_dist = .5 * (post_drift['log_vc_low_mu'] + post_drift['log_vc_high_mu'])

W, H = 7.09, 2.0
fig = plt.figure(figsize=(W, H))
at = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])  # noqa: E731
Y0, AH = .42, 1.3


def letter(ax, s, dx=-.3):
    ax.text(dx, 1.1, s, transform=ax.transAxes, fontsize=8, weight='bold')


# a: group effects
ax = at(.85, Y0, 1.55, AH)
rows = [('Distractor\npresent', COL['dist'], [(post_min['beta_dist_mu'], 'f'), (post_pr['beta_dist_mu'], 'o'),
                                              (drift_dist, 'g')]),
        ('High − low\nvalue', COL['value'], [(post_min['beta_value'], 'f'), (post_pr['beta_value'], 'o'),
                                             (post_drift['log_vc_value'], 'g')])]
for i, (lab, c, fits) in enumerate(rows):
    for j, (d, style) in enumerate(fits):
        d = d.values.ravel()
        h = pct(az.hdi(d, hdi_prob=.95))
        y = -(i + j * .25)
        cc = MUTED if style == 'g' else c
        ax.plot(h, [y, y], color=cc, lw=1.4, solid_capstyle='butt')
        ax.plot(pct(d.mean()), y, 'o', color='w' if style == 'o' else cc, mec=cc, mew=1.1, ms=4.5, zorder=3)
        if i == 1:
            ax.text(h[1] + 1.8, y, f'P = {(d < 0).mean():.3f}'.replace('0.', '.'), fontsize=6, color=cc, va='center')
ax.axvline(0, color='.6', lw=.6, ls='--', zorder=0)
ax.set_yticks([-.25, -1.25], [r[0] for r in rows])
ax.tick_params(axis='y', length=0)
ax.spines['left'].set_visible(False)
ax.set_xlim(-14, 8)
ax.set_xticks([-10, -5, 0, 5])
ax.set_xlabel('Change in target drift (%)')
ax.set_ylim(-1.75, .75)
for j, (lab, style) in enumerate((('Minimal', 'f'), ('Minimal + practice', 'o'), ('Per-subject slopes', 'g'))):
    cc = MUTED if style == 'g' else FG
    yy = 1.13 - j * .085
    ax.plot([.02, .1], [yy, yy], transform=ax.transAxes, color=cc, lw=1.4, solid_capstyle='butt', clip_on=False)
    ax.plot(.06, yy, 'o', transform=ax.transAxes, color='w' if style == 'o' else cc, mec=cc, mew=1, ms=3.8,
            clip_on=False, zorder=4)
    ax.text(.14, yy, lab, transform=ax.transAxes, fontsize=6.2, va='center', color=cc)
letter(ax, 'a', -.55)

# b, c: PPC of the minimal model
df = load(1)
sim = pd.read_parquet(ROOT / 'notes/data/race_minimal_ppc.parquet')


def quantiles(d, by):
    c = d[d.correct == 1]
    return c.groupby(by + ['subject', 'cond']).rt.quantile(Q).unstack().groupby(by + ['cond']).mean() * 1000


obs_q = quantiles(df.assign(k=0), ['k']).droplevel(0)
pq = quantiles(sim, ['draw'])
ax = at(3.0, Y0, 1.45, AH)
for i, c in enumerate(ORDER):
    band = pq.xs(c, level='cond')
    ax.fill_between(Q, band.quantile(.025), band.quantile(.975), color=COL[c], alpha=.18, lw=0)
    ax.plot(Q, band.mean(), color=COL[c], lw=1)
    ax.plot(np.array(Q) + (i - 1) * .018, obs_q.loc[c], 'o', color=COL[c], ms=3.4, mec='w', mew=.5, zorder=3)
for j, c in enumerate(ORDER[::-1]):
    ax.text(.05, .95 - j * .1, LAB[c], transform=ax.transAxes, color=COL[c], fontsize=6.3, va='top')
ax.set_xticks(Q, [f'{q:.1f}' for q in Q])
ax.set_xlabel('RT quantile')
ax.set_ylabel('Correct RT (ms)')
letter(ax, 'b', -.42)

obs_acc = df.groupby(['subject', 'cond']).correct.mean().unstack()[ORDER].mean() * 100
pa = sim.groupby(['draw', 'subject', 'cond']).correct.mean().groupby(['draw', 'cond']).mean().unstack()[ORDER] * 100
ax = at(4.95, Y0, .85, AH)
x = np.arange(3)
ax.fill_between(x, pa.quantile(.025), pa.quantile(.975), color=MUTED, alpha=.35, lw=0)
ax.plot(x, pa.mean(), color=FG, lw=1)
for i, c in enumerate(ORDER):
    ax.plot(i, obs_acc[c], 'o', color=COL[c], ms=4.5, mec='w', mew=.6, zorder=3)
ax.set_xticks(x, [LAB[c] for c in ORDER])
ax.set_xlim(-.4, 2.4)
ax.set_ylabel('Accuracy (%)')
ax.set_xlabel('Distractor')
# glyph key: model line + band, data dot
ax.add_patch(plt.Rectangle((.55, .9), .14, .07, transform=ax.transAxes, color=MUTED, alpha=.35, lw=0))
ax.plot([.55, .69], [.935, .935], transform=ax.transAxes, color=FG, lw=1)
ax.text(.73, .935, 'Model', transform=ax.transAxes, fontsize=6.2, va='center')
ax.plot(.62, .8, 'o', transform=ax.transAxes, color=FG, ms=3.8, mec='w')
ax.text(.73, .8, 'Data', transform=ax.transAxes, fontsize=6.2, va='center')
letter(ax, 'c', -.6)

# d: LOO
MODELS = {'drift': 'Per-subject\nslopes', 'minimal': 'Minimal', 'advantage_ta': 'Advantage\n+ a, t0',
          'advantage': 'Advantage'}
cmp = az.compare({m: nc(m) for m in MODELS})
ax = at(6.35, Y0, .7, AH)
for r, m in enumerate(MODELS):
    d, se = cmp.loc[m, 'elpd_diff'], cmp.loc[m, 'dse']
    cc = COL['value'] if m == 'minimal' else FG
    ax.plot([-d - se, -d + se], [-r, -r], color=cc, lw=1.2, solid_capstyle='butt')
    ax.plot(-d, -r, 'o', color=cc, ms=4, mec='w', mew=.5, zorder=3)
ax.axvline(0, color='.6', lw=.6, ls='--', zorder=0)
ax.set_yticks(-np.arange(len(MODELS)), list(MODELS.values()), fontsize=6.2)
ax.yaxis.tick_right()
ax.tick_params(axis='y', length=0)
ax.spines['left'].set_visible(False)
ax.set_ylim(-len(MODELS) + .4, .6)
ax.set_xlabel('ΔELPD vs best')
letter(ax, 'd', -.25)

for a_ in fig.axes:
    for s in ('left', 'bottom'):
        if a_.spines[s].get_visible():
            a_.spines[s].set_position(('outward', 3))
out = ROOT / 'notes/figures/race_model'
fig.savefig(f'{out}.pdf')
fig.savefig(f'{out}.png', dpi=300)
print(out)
print(cmp[['elpd_loo', 'elpd_diff', 'dse']])
