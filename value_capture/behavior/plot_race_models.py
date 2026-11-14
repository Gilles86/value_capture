"""Compare race-model variants: quantile PPC (correct-RT quantiles, accuracy) and value effects.

Rows: models (drift / advantage / advantage_ta). Columns: correct-RT quantiles per condition
(data dots, model line + 95% band), accuracy, and the group value effect (high − low) on every
condition-dependent parameter. Quantiles: per subject, then averaged over subjects (per draw).

    python value_capture/behavior/plot_race_models.py
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
from fit_race_models import COND_PARS  # noqa: E402

mpl.rcParams.update({
    'font.family': 'Helvetica', 'font.size': 7, 'axes.labelsize': 7.5, 'axes.linewidth': .8,
    'axes.spines.top': False, 'axes.spines.right': False, 'xtick.direction': 'out',
    'ytick.direction': 'out', 'xtick.major.size': 3, 'ytick.major.size': 3, 'pdf.fonttype': 42,
    'savefig.bbox': 'tight', 'savefig.pad_inches': .03})
FG, MUTED = '#1F2933', '#9AA5B1'
COL = {'none': '#7B8794', 'low': '#4A7FB5', 'high': '#B2182B'}
ORDER = ['none', 'low', 'high']
LAB = {'none': 'Absent', 'low': 'Low', 'high': 'High'}
Q = [.1, .3, .5, .7, .9]
MODELS = {'drift': 'Drift only', 'advantage': 'Advantage', 'advantage_ta': 'Advantage + a, t0'}
PNAME = {'log_vc': 'Target drift', 'log_v0': 'Urgency v0', 'q': 'Advantage', 'log_a': 'Threshold a',
         't0_raw': 'Non-decision t0'}


def quantiles(d, by):
    c = d[d.correct == 1]
    return c.groupby(by + ['subject', 'cond']).rt.quantile(Q).unstack().groupby(by + ['cond']).mean() * 1000


df = load(1)
obs_q = quantiles(df.assign(k=0), ['k']).droplevel(0)
obs_acc = df.groupby(['subject', 'cond']).correct.mean().unstack().mean() * 100
loo = {m: az.loo(az.from_netcdf(ROOT / f'notes/data/race_{m}.nc')) for m in MODELS}
best = max(loo, key=lambda m: loo[m].elpd_loo)

fig, axes = plt.subplots(len(MODELS), 3, figsize=(7.09, 1.75 * len(MODELS)),
                         gridspec_kw=dict(width_ratios=[1.5, .8, 1.4], wspace=.55, hspace=.75))
for r, (m, title) in enumerate(MODELS.items()):
    sim = pd.read_parquet(ROOT / f'notes/data/race_{m}_ppc.parquet')
    pq = quantiles(sim, ['draw'])
    pa = sim.groupby(['draw', 'subject', 'cond']).correct.mean().groupby(['draw', 'cond']).mean().unstack() * 100
    ax = axes[r, 0]
    for i, c in enumerate(ORDER):
        x = np.array(Q) + (i - 1) * .018
        band = pq.xs(c, level='cond')
        ax.fill_between(Q, band.quantile(.025), band.quantile(.975), color=COL[c], alpha=.18, lw=0)
        ax.plot(Q, band.mean(), color=COL[c], lw=1)
        ax.plot(x, obs_q.loc[c], 'o', color=COL[c], ms=3.6, mec='w', mew=.5, zorder=3)
        if r == 0:
            ax.text(.92, obs_q.loc[c].iloc[-1], f' {LAB[c]}', color=COL[c], fontsize=6.3, va='center')
    ax.set_xticks(Q, [f'{q:.1f}' for q in Q])
    ax.set_xlabel('RT quantile', labelpad=1)
    ax.set_ylabel('Correct RT (ms)')
    d_elpd = loo[m].elpd_loo - loo[best].elpd_loo
    ax.set_title(f'{title}   ΔELPD = {d_elpd:.0f}' + ('' if m != best else ' (best)'), loc='left', fontsize=7.5)
    ax = axes[r, 1]
    x = np.arange(3)
    ax.fill_between(x, pa[ORDER].quantile(.025), pa[ORDER].quantile(.975), color=MUTED, alpha=.35, lw=0)
    ax.plot(x, pa[ORDER].mean(), color=FG, lw=1)
    for i, c in enumerate(ORDER):
        ax.plot(i, obs_acc[c], 'o', color=COL[c], ms=4.5, mec='w', mew=.6, zorder=3)
    ax.set_xticks(x, [LAB[c] for c in ORDER])
    ax.set_xlim(-.4, 2.4)
    ax.set_ylabel('Accuracy (%)')
    ax = axes[r, 2]
    post = az.from_netcdf(ROOT / f'notes/data/race_{m}.nc').posterior
    pars = COND_PARS[m]
    for j, k in enumerate(pars):
        for jj, (c, colr) in enumerate((('low', COL['low']), ('high', COL['high']), ('value', '#6A3D9A'))):
            nm = f'{k}_{c}_mu' if c != 'value' else f'{k}_value'
            d = post[nm].values.ravel()
            h = az.hdi(d, hdi_prob=.95)
            y = -(j + jj * .25)
            ax.plot(h, [y, y], color=colr, lw=1.3, solid_capstyle='butt')
            ax.plot(d.mean(), y, 'o', color=colr, ms=3.5, mec='w', mew=.4, zorder=3)
    ax.axvline(0, color='.6', lw=.6, ls='--', zorder=0)
    ax.set_yticks([-(j + .25) for j in range(len(pars))], [PNAME[k] for k in pars])
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.set_ylim(-(len(pars) - 1) - .7, .35)
    ax.set_xlabel('Effect (log / logit scale)', labelpad=1)
    if r == 0:
        for jj, (lab, colr) in enumerate((('Low − absent', COL['low']), ('High − absent', COL['high']),
                                          ('High − low', '#6A3D9A'))):
            ax.text(1.02, 1 - jj * .14, lab, transform=ax.transAxes, color=colr, fontsize=6.3, va='top')
for ax in axes.ravel():
    for s in ('left', 'bottom'):
        if ax.spines[s].get_visible():
            ax.spines[s].set_position(('outward', 3))
out = ROOT / 'notes/figures/race_models_comparison'
fig.savefig(f'{out}.pdf')
fig.savefig(f'{out}.png', dpi=300)
print(out)
for m in MODELS:
    print(m, round(loo[m].elpd_loo, 1), round(loo[m].se, 1), 'p_loo', round(loo[m].p_loo, 1))
