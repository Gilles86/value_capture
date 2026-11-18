"""Across subjects: does the high - low value difference in decoded gain go with the value effect on drift rate?

Neural: high - low value at the distractor (decoded item intensity, or the aligned map inside the distractor ring).
Behaviour: log drift of the correct accumulator, high - low value, per subject, from the hierarchical race model 'drift'
(behavior/fit_race_models.py). Negative = a high-value distractor slows evidence accumulation more.

r of the posterior-mean drift effects with a permutation p (two-sided), and the r over the posterior draws of the drift
effects, which carries their uncertainty. Log RT (correct trials) is the model-free counterpart.

    python value_capture/decode/correlate_gain_behaviour.py
"""
import argparse
from pathlib import Path

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from value_capture.decode.plot_decoded_gain import FG, HIGH, fmt_p, group_maps, ring_mean

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'notes/data'
SETS = {'DoG, points subtracted': 'decoded_gain_dog_regress', 'DoG, trials without points': 'decoded_gain_dog_nofeedback',
        'Gaussian, points subtracted': 'decoded_gain_regress'}
EARLY = ['V1', 'V2', 'V3']


def neural_effects(folder):
    """Subject-wise high - low value: early areas, V2, and the map at the distractor (V1-V3)."""
    i = pd.read_csv(DATA / folder / 'item_gain.tsv', sep='\t', dtype={'subject': str}).query("decoder == 'outline'")

    def items(rois):
        s = i[i.roi.isin(rois)].groupby(['subject', 'role']).pct.mean().unstack()
        return s.distractor_high - s.distractor_low
    z = np.load(DATA / folder / 'aligned_maps.npz')
    subjects = sorted({k.split('|')[0] for k in z.files if k.split('|')[3] == 'free' and k.split('|')[2] in EARLY})
    m = group_maps(DATA / folder)
    ring = pd.Series(ring_mean(m['high'] - m['low']) * 1000, index=subjects)
    return pd.DataFrame({'Early areas (V1–V3), item': items(EARLY), 'V2, item': items(['V2']), 'Map at the distractor': ring})


def drift_effects():
    p = az.from_netcdf(DATA / 'race_drift.nc').posterior
    d = (p.log_vc_high - p.log_vc_low).values.reshape(-1, p.log_vc_high.shape[-1])    # draws x subjects
    b = pd.read_csv(DATA / 'behavior_trials_main_raw.tsv', sep='\t', dtype={'subject': str})
    return sorted(b.subject.str.zfill(2).unique()), d


def logrt_effects():
    b = pd.read_csv(DATA / 'behavior_trials_main.tsv', sep='\t', dtype={'subject': str})
    b['subject'] = b.subject.str.zfill(2)
    b = b[b.correct.astype(str).eq('True') & b.rt.notna() & (b.rt > .2) & b.cond.isin(['low', 'high'])]
    m = np.log(b.rt).groupby([b.subject, b.cond]).mean().unstack()
    return m.high - m.low


def perm_p(x, y, n=100000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = (x - x.mean()) / x.std(), (y - y.mean()) / y.std()
    obs = np.mean(x * y) * len(x) / (len(x) - 1)
    null = np.array([np.mean(x * rng.permutation(y)) for _ in range(n)]) * len(x) / (len(x) - 1)
    return float((np.abs(null) >= abs(obs) - 1e-12).mean())


def main(out):
    subjects, draws = drift_effects()
    drift = pd.Series(100 * (np.exp(draws.mean(0)) - 1), index=subjects)
    draws_s = pd.DataFrame(draws, columns=subjects)
    rt = logrt_effects()
    rows = []
    for label, folder in SETS.items():
        n = neural_effects(folder)
        for measure in n.columns:
            x = n[measure].dropna()
            common = x.index.intersection(drift.index)
            a, b = x[common].to_numpy(), drift[common].to_numpy()
            per_draw = np.array([np.corrcoef(a, d)[0, 1] for d in draws_s[common].to_numpy()[::4]])
            rows.append(dict(neural=label, measure=measure, n=len(common), r_drift=np.corrcoef(a, b)[0, 1], p_perm=perm_p(a, b),
                             spearman=stats.spearmanr(a, b)[0], r_draws_mean=per_draw.mean(),
                             r_draws_lo=np.percentile(per_draw, 2.5), r_draws_hi=np.percentile(per_draw, 97.5),
                             r_logrt=np.corrcoef(a, rt[common])[0, 1], p_logrt=perm_p(a, rt[common].to_numpy())))
    tab = pd.DataFrame(rows)
    tab.to_csv(DATA / 'decoded_gain/subjectwise_value_gain_vs_drift.tsv', sep='\t', index=False)
    print(tab.round(3).to_string(index=False))
    print('drift effect (high - low, %):', drift.round(1).to_dict(), 'r(drift, log RT) = %.2f' % np.corrcoef(drift, rt[drift.index])[0, 1])

    mpl.rcParams.update({'font.family': 'Helvetica', 'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    n = neural_effects(SETS['DoG, points subtracted'])
    units = ['High − low decoded intensity (%)', 'High − low decoded intensity (%)', 'High − low in the map (×10⁻³)']
    sd = 100 * draws_s.std()
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.6), sharey=True, gridspec_kw=dict(wspace=.12, left=.08, right=.985, top=.74, bottom=.17))
    fig.text(.01, .97, f'Does the neural value effect track the behavioural value effect? DoG decoding, points subtracted, n = {len(n)}',
             fontsize=9, va='top')
    for ax, measure, unit in zip(axes, n.columns, units):
        x = n[measure].dropna()
        y = drift[x.index]
        ax.errorbar(x, y, yerr=sd[x.index], fmt='o', color=HIGH, ms=4, lw=.7, mew=0, alpha=.85)
        slope, icpt = np.polyfit(x, y, 1)
        xx = np.array([x.min(), x.max()])
        ax.plot(xx, icpt + slope * xx, color=FG, lw=1)
        ax.axhline(0, color='.35', lw=.8, ls=(0, (3, 2)))
        ax.axvline(0, color='.35', lw=.8, ls=(0, (3, 2)))
        r = tab[(tab.neural == 'DoG, points subtracted') & (tab.measure == measure)].iloc[0]
        ax.set_title(f'{measure}\nr = {r.r_drift:.2f}, {fmt_p(r.p_perm)}', pad=4)
        ax.set_xlabel(unit)
    axes[0].set_ylabel('Drift rate, high − low value (%)')
    fig.savefig(f'{out}.pdf'), fig.savefig(f'{out}.png', dpi=200)
    print(f'{out}.pdf')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(ROOT / 'notes/figures/decoded_gain_vs_drift'))
    main(ap.parse_args().out)
