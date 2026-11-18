"""Does the decoded gain on a trial predict that trial's response time? Hierarchical log-RT models (bambi), per ROI.

    log_rt ~ cond + session_c + orange + neutral_z + target_z + dist_low_z + dist_high_z
             + (cond + target_z + dist_low_z + dist_high_z | subject)

target_z, dist_low_z and dist_high_z are the trial's decoded gain at the target and at a low- or high-value distractor
(decode/aggregate_decoded.py: matched on item location and bar position), z-scored within subject × session × ROI.
Distractor gains are 0 on trials without that distractor. neutral_z is the gain at the neutral items of the same trial:
slow trials evoke larger responses everywhere, so without it any gain would correlate with RT. Correct trials only.

    python value_capture/behavior/fit_logrt_gains.py --gains notes/data/decoded_gain/trial_gains.tsv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRIALS = ROOT / 'notes/data/behavior_trials_main.tsv'
TERMS = ['target_z', 'dist_low_z', 'dist_high_z', 'neutral_z']
FORMULA = ('log_rt ~ cond + session_c + orange + neutral_z + target_z + dist_low_z + dist_high_z '
           '+ (cond + target_z + dist_low_z + dist_high_z | subject)')


def zscore(x):
    """Robust z-score, clipped at ±3: single-trial decoded gains have extreme tails."""
    mad = 1.4826 * (x - x.median()).abs().median()
    return ((x - x.median()) / mad).clip(-3, 3)


def add_gain_regressors(df):
    """z-scored gains (within subject × session × ROI) and their split by distractor value."""
    keys = ['subject', 'session', 'roi']
    df = df.copy()
    df['target_z'] = df.groupby(keys).target_gain.transform(zscore)
    df['neutral_z'] = df.groupby(keys).neutral_gain.transform(zscore)
    dist = df.groupby(keys).distractor_gain.transform(zscore)
    df['dist_low_z'] = dist.where(df.cond == 'low', 0.)
    df['dist_high_z'] = dist.where(df.cond == 'high', 0.)
    return df


def load(gains):
    g = pd.read_csv(gains, sep='\t', dtype={'subject': str})
    b = pd.read_csv(TRIALS, sep='\t', dtype={'subject': str})
    b['subject'] = b.subject.str.zfill(2)
    df = b.merge(g, on=['subject', 'session', 'run', 'trial_nr'], how='inner', validate='one_to_many')
    df = df[df.correct.astype(str).eq('True') & df.rt.notna() & (df.rt > .2)].copy()
    df['log_rt'] = np.log(df.rt)
    df['session_c'] = df.session - 1.5
    df['orange'] = df.colour.fillna('none').map({'orange': .5, 'green': -.5, 'none': 0.})
    df['cond'] = pd.Categorical(df.cond, ['none', 'low', 'high'])
    df = add_gain_regressors(df)
    return df.dropna(subset=['target_z', 'neutral_z', 'dist_low_z', 'dist_high_z'])


def main(gains, rois, draws, tune, out):
    import arviz as az
    import bambi as bmb
    df = load(gains)
    rows = []
    for roi in rois:
        d = df[df.roi == roi].copy()
        d['subject'] = d.subject.astype(str)
        print(f'{roi}: {d.subject.nunique()} subjects, {len(d)} trials', flush=True)
        idata = bmb.Model(FORMULA, d).fit(draws=draws, tune=tune, chains=4, cores=4, target_accept=.95, random_seed=1,
                                          progressbar=False)
        post = idata.posterior
        draws_ = {t: post[t].values.ravel() for t in TERMS}
        draws_['high_minus_low'] = draws_['dist_high_z'] - draws_['dist_low_z']
        rhat = float(az.rhat(idata, var_names=TERMS).to_array().max())
        div = int(idata.sample_stats.diverging.sum())
        for term, x in draws_.items():
            lo, hi = az.hdi(x, hdi_prob=.95)
            rows.append(dict(roi=roi, term=term, mean=x.mean(), hdi_lo=lo, hdi_hi=hi, p_gt_0=(x > 0).mean(),
                             pct_rt_per_sd=100 * (np.exp(x.mean()) - 1), n_subjects=d.subject.nunique(),
                             n_trials=len(d), max_rhat=rhat, divergences=div))
        pd.DataFrame(rows).to_csv(out, sep='\t', index=False)
        print(pd.DataFrame(rows).query('roi == @roi').round(4).to_string(index=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gains', default=str(ROOT / 'notes/data/decoded_gain/trial_gains.tsv'))
    ap.add_argument('--rois', nargs='+', default=['V1', 'V2', 'V3', 'hV4', 'LO'])
    ap.add_argument('--draws', type=int, default=1000)
    ap.add_argument('--tune', type=int, default=1000)
    ap.add_argument('--out', default=str(ROOT / 'notes/data/decoded_gain/logrt_gain_effects.tsv'))
    a = ap.parse_args()
    main(a.gains, a.rois, a.draws, a.tune, a.out)
