"""Hierarchical Bayesian log-RT (and accuracy) models for the value-capture task.

Main study only (sub-04..15; pilots had a different 3-rank design).

    log(rt) ~ condition * session_c + colour + (condition + session_c | subject)

condition: none / low / high value distractor (reference = none).
colour (green/orange distractor) is orthogonal to value within subject because
the colour→value mapping swaps between sessions, so value effects are not colour
effects. Correct trials only for RT.

Usage (two steps: data export in value_capture env, fit in an env with bambi):
    python -m value_capture.behavior.fit_logrt_bambi export --bids-folder /data/ds-valuecapture
    python value_capture/behavior/fit_logrt_bambi.py fit
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TRIALS = Path(__file__).resolve().parents[2] / 'notes' / 'data' / 'behavior_trials_main.tsv'
OUTDIR = Path(__file__).resolve().parents[2] / 'notes' / 'data'
MAIN = [f'{i:02d}' for i in range(4, 16)]


def export(bids_folder):
    from value_capture.utils.data import get_all_behavioral_data
    df = get_all_behavioral_data(bids_folder=bids_folder, subjects=MAIN).reset_index()
    df['cond'] = np.where(~df['distractor_present'].astype(bool), 'none',
                          np.where(df['value_rank'] == 2, 'high', 'low'))
    # condition 0: rank0 = green, rank2 = orange; condition 1: the reverse
    green = (df['value_rank'] == 0) == (df['value_condition'] == 0)
    df['colour'] = np.where(df['cond'] == 'none', 'none', np.where(green, 'green', 'orange'))
    df.to_csv(TRIALS, sep='\t', index=False)
    print(f'{len(df)} trials → {TRIALS}')


def fit(draws, tune):
    import arviz as az
    import bambi as bmb
    df = pd.read_csv(TRIALS, sep='\t', dtype={'subject': str})
    df['correct'] = df['correct'].map({True: 1., False: 0., 'True': 1., 'False': 0.})  # n/a = no response
    df['session_c'] = df['session'] - 1.5
    df['cond'] = pd.Categorical(df['cond'], ['none', 'low', 'high'])
    # Colour regressor: +0.5 orange, -0.5 green, 0 no distractor
    df['orange'] = df['colour'].map({'orange': .5, 'green': -.5, 'none': 0.})
    rt = df[(df['correct'] == 1) & df['rt'].notna() & (df['rt'] > .2)].copy()
    rt['log_rt'] = np.log(rt['rt'])

    results = {}
    m = bmb.Model('log_rt ~ cond * session_c + orange + (cond + session_c | subject)', rt)
    results['logrt'] = m.fit(draws=draws, tune=tune, chains=4, target_accept=.95,
                             cores=4, random_seed=1)
    acc = df[df['correct'].notna()].copy()
    acc['correct'] = acc['correct'].astype(int)
    m_acc = bmb.Model('correct ~ cond * session_c + orange + (cond | subject)', acc,
                      family='bernoulli')
    results['acc'] = m_acc.fit(draws=draws, tune=tune, chains=4, target_accept=.95,
                               cores=4, random_seed=1)

    rows = []
    for name, idata in results.items():
        post = idata.posterior
        az.to_netcdf(idata, OUTDIR / f'bambi_{name}.nc')
        summ = az.summary(idata, var_names=['~_sigma', '~|'], filter_vars='like', hdi_prob=.95)
        print(f'\n=== {name} ===\n{summ}')
        hi, lo = post['cond'].sel(cond_dim='high'), post['cond'].sel(cond_dim='low')
        diff = (hi - lo).values.ravel()
        for label, d in [('low - none', lo.values.ravel()), ('high - none', hi.values.ravel()),
                         ('high - low', diff)]:
            h = az.hdi(d, hdi_prob=.95)
            rows.append(dict(model=name, contrast=label, mean=d.mean(), hdi_lo=h[0], hdi_hi=h[1],
                             p_gt_0=(d > 0).mean()))
        # Value effect per session (interaction)
        inter = post['cond:session_c'].sel({'cond:session_c_dim': 'high'}).values.ravel() - \
            post['cond:session_c'].sel({'cond:session_c_dim': 'low'}).values.ravel()
        for ses, sc in [(1, -.5), (2, .5)]:
            d = diff + sc * inter
            h = az.hdi(d, hdi_prob=.95)
            rows.append(dict(model=name, contrast=f'high - low, session {ses}', mean=d.mean(),
                             hdi_lo=h[0], hdi_hi=h[1], p_gt_0=(d > 0).mean()))
        print('max r_hat', float(az.rhat(idata).to_array().max()),
              'divergences', int(idata.sample_stats['diverging'].sum()))
    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR / 'bambi_contrasts.tsv', sep='\t', index=False)
    print('\n', out.round(4).to_string(index=False))
    print('\nlog-RT contrasts ×1000 ≈ % RT change ×10 (e.g. 0.03 ≈ +3% ≈ +25 ms at 850 ms)')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=['export', 'fit'])
    p.add_argument('--bids-folder', default='/data/ds-valuecapture')
    p.add_argument('--draws', type=int, default=1000)
    p.add_argument('--tune', type=int, default=1000)
    a = p.parse_args()
    export(a.bids_folder) if a.step == 'export' else fit(a.draws, a.tune)
