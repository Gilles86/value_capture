"""A very simple hierarchical race (racing-diffusion / Wald) model for the search task.

Two accumulators race to a common threshold a: one for the correct response (drift v_c), one
for the error (drift v_e), diffusion SD fixed to 1, plus non-decision time t0. Trial likelihood
(Tillman, Van Zandt & Logan 2020):
    responded, winner w at t:  f_w(t − t0) · S_loser(t − t0)
    no response by deadline D: S_c(D − t0) · S_e(D − t0)        (D = 1.75 s; censored)
with the Wald first-passage density f and survivor S.

Distractor effects act on the correct drift (log scale, per-subject random effects):
    log v_c = β0 + β_low·low + β_high·high + β_ses·session_c
so β_high − β_low is the value effect on evidence accumulation (as a % change in drift).
v_e, a, t0 vary by subject only.

    python value_capture/behavior/fit_race_model.py               # all runs
    python value_capture/behavior/fit_race_model.py --min-run 3   # after learning (runs 3-8)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEADLINE = 1.75


def load(min_run):
    df = pd.read_csv(ROOT / 'notes/data/behavior_trials_main_raw.tsv', sep='\t', dtype={'subject': str})
    df['correct'] = df['correct'].map({True: 1., False: 0., 'True': 1., 'False': 0.})
    df = df[df.run >= min_run].copy()
    df['responded'] = df.rt.notna() & df.correct.notna()
    df = df[~df.responded | (df.rt > .2)]
    df['subj_idx'] = pd.Categorical(df.subject).codes
    df['low'] = (df.cond == 'low').astype(float)
    df['high'] = (df.cond == 'high').astype(float)
    df['session_c'] = df.session - 1.5
    return df.reset_index(drop=True)


def build(df):
    import pymc as pm
    import pytensor.tensor as pt

    S = df.subj_idx.nunique()
    s = df.subj_idx.values
    resp = df.responded.values
    corr = np.nan_to_num(df.correct.values).astype(bool)
    t = np.where(resp, df.rt.values, DEADLINE)
    min_rt = df[df.responded].groupby('subj_idx').rt.min().reindex(range(S)).values

    def log_phi(x):
        return pm.math.log(.5 * pt.erfc(-x / np.sqrt(2)))

    def wald_logpdf(tt, v, a):
        return pt.log(a) - .5 * pt.log(2 * np.pi * tt ** 3) - (a - v * tt) ** 2 / (2 * tt)

    def wald_logsf(tt, v, a):
        sq = pt.sqrt(tt)
        logF = pt.logaddexp(log_phi((v * tt - a) / sq), 2 * v * a + log_phi(-(v * tt + a) / sq))
        return pt.log1mexp(pt.minimum(logF, -1e-10))

    def hier(name, mu, sd_mu, sd_sigma=.3):
        m = pm.Normal(f'{name}_mu', mu, sd_mu)
        sg = pm.HalfNormal(f'{name}_sd', sd_sigma)
        z = pm.Normal(f'{name}_z', 0, 1, shape=S)
        return pm.Deterministic(name, m + sg * z), m

    with pm.Model() as model:
        b0, _ = hier('log_vc', np.log(2.5), .5)
        b_low, _ = hier('beta_low', 0., .3, .15)
        b_high, _ = hier('beta_high', 0., .3, .15)
        b_ses = pm.Normal('beta_session', 0., .3)
        lve, _ = hier('log_ve', np.log(.8), .7)
        la, _ = hier('log_a', np.log(1.5), .5)
        t0r, _ = hier('t0_raw', 0., 1., .5)
        t0 = pm.Deterministic('t0', .9 * min_rt * pm.math.sigmoid(t0r))
        pm.Deterministic('value_effect', model['beta_high_mu'] - model['beta_low_mu'])
        vc = pt.exp(b0[s] + b_low[s] * df.low.values + b_high[s] * df.high.values + b_ses * df.session_c.values)
        ve = pt.exp(lve[s])
        a = pt.exp(la[s])
        tt = pt.maximum(t - t0[s], 1e-3)
        lpdf_c, lsf_c = wald_logpdf(tt, vc, a), wald_logsf(tt, vc, a)
        lpdf_e, lsf_e = wald_logpdf(tt, ve, a), wald_logsf(tt, ve, a)
        ll = pt.switch(resp, pt.switch(corr, lpdf_c + lsf_e, lpdf_e + lsf_c), lsf_c + lsf_e)
        pm.Potential('ll', ll.sum())
    return model


def simulate(df, post, n_draws=200, seed=0):
    """Posterior predictive: simulate each trial from posterior draws; return mean RT/accuracy per cond."""
    rng = np.random.default_rng(seed)
    p = post.stack(sample=('chain', 'draw'))
    idx = rng.choice(p.sizes['sample'], n_draws, replace=False)
    s = df.subj_idx.values
    out = []
    for i in idx:
        g = lambda k: p[k].values[..., i]  # noqa: E731
        vc = np.exp(g('log_vc')[s] + g('beta_low')[s] * df.low + g('beta_high')[s] * df.high
                    + float(g('beta_session')) * df.session_c).values
        ve, a, t0 = np.exp(g('log_ve'))[s], np.exp(g('log_a'))[s], g('t0')[s]
        fc = rng.wald(a / vc, a ** 2)          # Wald(mean a/v, shape a^2) = first passage, sigma 1
        fe = rng.wald(a / ve, a ** 2)
        rt = np.minimum(fc, fe) + t0
        ok = rt <= DEADLINE
        out.append(pd.DataFrame({'subject': df.subject, 'cond': df.cond, 'draw': i,
                                 'rt': np.where(ok & (fc < fe), rt, np.nan),
                                 'correct': np.where(ok, (fc < fe).astype(float), np.nan)}))
    return pd.concat(out)


def main(min_run, draws, tune):
    import arviz as az
    import pymc as pm
    df = load(min_run)
    print(f'{len(df)} trials, {df.subject.nunique()} subjects, runs >= {min_run}, '
          f'{(~df.responded).sum()} no-response trials')
    model = build(df)
    with model:
        idata = pm.sample(draws=draws, tune=tune, chains=4, target_accept=.95, nuts_sampler='numpyro',
                          random_seed=1)
    tag = f'race_runs{min_run}-8'
    az.to_netcdf(idata, ROOT / f'notes/data/{tag}.nc')
    names = ['log_vc_mu', 'beta_low_mu', 'beta_high_mu', 'value_effect', 'beta_session', 'log_ve_mu', 'log_a_mu',
             'beta_low_sd', 'beta_high_sd']
    summ = az.summary(idata, var_names=names, hdi_prob=.95)
    print(summ[['mean', 'sd', 'hdi_2.5%', 'hdi_97.5%', 'r_hat', 'ess_bulk']].round(3))
    print('divergences', int(idata.sample_stats.diverging.sum()))
    post = idata.posterior
    for k in ('beta_low_mu', 'beta_high_mu', 'value_effect'):
        d = post[k].values.ravel()
        h = az.hdi(d, hdi_prob=.95)
        print(f'{k:14s} drift change {100 * (np.exp(d.mean()) - 1):+.1f}%  95% HDI [{100 * (np.exp(h[0]) - 1):+.1f}, '
              f'{100 * (np.exp(h[1]) - 1):+.1f}]%  P(<0) = {(d < 0).mean():.3f}')
    print(f't0 mean {post["t0"].mean().item():.3f} s')
    sim = simulate(df, post)
    obs = df.assign(rt=np.where(df.correct == 1, df.rt, np.nan)).groupby('cond')[['rt', 'correct']].mean()
    pred = sim.groupby(['draw', 'cond'])[['rt', 'correct']].mean().groupby('cond').agg(['mean', lambda x: x.quantile(.025), lambda x: x.quantile(.975)])
    print('\nPPC (observed vs predicted mean correct RT / accuracy):')
    print(pd.concat([obs, pred], axis=1).round(3))
    pd.concat([obs.assign(kind='observed'), sim.groupby('cond')[['rt', 'correct']].mean().assign(kind='predicted')]).to_csv(
        ROOT / f'notes/data/{tag}_ppc.tsv', sep='\t')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-run', type=int, default=1)
    ap.add_argument('--draws', type=int, default=1000)
    ap.add_argument('--tune', type=int, default=1000)
    a = ap.parse_args()
    main(a.min_run, a.draws, a.tune)
