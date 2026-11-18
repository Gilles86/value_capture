"""Hierarchical racing-diffusion (Wald race) models for the search task, three variants.

Two accumulators (correct / error), common threshold a, diffusion SD 1, non-decision time t0,
deadline 1.75 s (non-responses censored). Variants differ in which parameters the distractor
condition (low / high value vs none; per-subject random slopes) can change:

  drift      : log v_c ~ cond + session                      (v_e, a, t0 per subject)
  advantage  : v_c = v0 + Δ, v_e = v0 − Δ                     (urgency v0, advantage Δ = v0·logistic(q))
               log v0 ~ cond ;  q ~ cond + session            (a, t0 per subject)
  advantage_ta : advantage + log a ~ cond + t0 ~ cond          (everything can move)
  neural     : minimal + the trial's decoded gains on log v_c   (target, low- and high-value distractor, neutral items;
               decode/aggregate_decoded.py, robust z within subject x session x area, averaged over --gain-rois)

Value effect = high − low on each condition-dependent parameter.
Saves the posterior, pointwise log-likelihood (PSIS-LOO), and a quantile PPC.

    python value_capture/behavior/fit_race_models.py advantage_ta
    python value_capture/behavior/fit_race_models.py neural --gain-rois V1 V2 V3 --gain-label early
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, log_ndtr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from fit_logrt_gains import add_gain_regressors  # noqa: E402
from fit_race_model import DEADLINE, load  # noqa: E402

COND_PARS = {'drift': ['log_vc'], 'advantage': ['log_v0', 'q'],
             'advantage_ta': ['log_v0', 'q', 'log_a', 't0_raw'], 'minimal': [], 'neural': []}
GAIN_TERMS = {'beta_gain_target': 'target_z', 'beta_gain_low': 'dist_low_z', 'beta_gain_high': 'dist_high_z',
              'beta_gain_neutral': 'neutral_z'}
# 'minimal': the fewest free parameters that still describe the data —
#   log v_c = b0[subj] + b_dist[subj]*present + b_value*(+.5 high / -.5 low) + session + practice
#   v_e one group value (errors are rare: per-subject error drift is barely identified),
#   value effect group-level only (no per-subject value slopes), a and t0 per subject.
INIT = {'log_vc': np.log(2.5), 'log_ve': np.log(.8), 'log_v0': np.log(1.5), 'q': .5,
        'log_a': np.log(1.5), 't0_raw': 0.}
SESSION_PAR = {'drift': 'log_vc', 'advantage': 'q', 'advantage_ta': 'q', 'minimal': 'log_vc', 'neural': 'log_vc'}


def params_for(model):
    return ['log_vc', 'log_ve', 'log_a', 't0_raw'] if model in ('drift', 'minimal', 'neural') else ['log_v0', 'q', 'log_a', 't0_raw']


def trialwise(get, model, df, xp=np):
    """Trial-wise v_c, v_e, a, t0 from parameter getter get(name) -> per-subject array(s)."""
    s, low, high, ses = df.subj_idx.values, df.low.values, df.high.values, df.session_c.values
    lin = {}
    for k in params_for(model):
        x = get(k)[s]
        if k in COND_PARS[model]:
            x = x + get(f'{k}_low')[s] * low + get(f'{k}_high')[s] * high
        if model in ('minimal', 'neural') and k == 'log_vc':
            x = x + get('beta_dist')[s] * (low + high) + get('beta_value') * .5 * (high - low)
        if model == 'neural' and k == 'log_vc':
            for name, col in GAIN_TERMS.items():
                x = x + get(name)[s] * df[col].values
        if k == SESSION_PAR[model]:
            x = x + get('beta_session') * ses
            if 'practice' in df:
                x = x + get('beta_practice') * df.practice.values
        lin[k] = x
    exp = xp.exp
    sig = (lambda z: 1 / (1 + xp.exp(-z)))
    if model in ('drift', 'minimal', 'neural'):
        vc, ve = exp(lin['log_vc']), exp(lin['log_ve'])
    else:
        v0 = exp(lin['log_v0'])
        d = v0 * sig(lin['q'])
        vc, ve = v0 + d, v0 - d + 1e-3
    a = exp(lin['log_a'])
    t0 = get('t0_max')[s] * sig(lin['t0_raw'])
    return vc, ve, a, t0


def build(df, model):
    import pymc as pm
    import pytensor.tensor as pt
    S = df.subj_idx.nunique()
    resp = df.responded.values
    corr = np.nan_to_num(df.correct.values).astype(bool)
    t = np.where(resp, df.rt.values, DEADLINE)
    t0_max = .9 * df[df.responded].groupby('subj_idx').rt.min().reindex(range(S)).values

    def log_phi(x):
        return pm.math.log(.5 * pt.erfc(-x / np.sqrt(2)))

    def logpdf(tt, v, a):
        return pt.log(a) - .5 * pt.log(2 * np.pi * tt ** 3) - (a - v * tt) ** 2 / (2 * tt)

    def logsf(tt, v, a):
        sq = pt.sqrt(tt)
        logF = pt.logaddexp(log_phi((v * tt - a) / sq), 2 * v * a + log_phi(-(v * tt + a) / sq))
        return pt.log1mexp(pt.minimum(logF, -1e-10))

    with pm.Model() as m:
        P = {'t0_max': pt.as_tensor(t0_max)}

        def hier(name, mu, sd_mu, sd_sigma):
            g = pm.Normal(f'{name}_mu', mu, sd_mu)
            sg = pm.HalfNormal(f'{name}_sd', sd_sigma)
            P[name] = pm.Deterministic(name, g + sg * pm.Normal(f'{name}_z', 0, 1, shape=S))

        for k in params_for(model):
            if model in ('minimal', 'neural') and k == 'log_ve':   # one group-level error drift
                g = pm.Normal('log_ve_mu', INIT[k], .7)
                P[k] = pm.Deterministic(k, g * pt.ones(S))
                continue
            hier(k, INIT[k], .7 if k != 't0_raw' else 1., .3 if k != 't0_raw' else .5)
        if model in ('minimal', 'neural'):
            hier('beta_dist', 0., .3, .15)
            P['beta_value'] = pm.Normal('beta_value', 0., .3)
        if model == 'neural':
            for name in GAIN_TERMS:
                hier(name, 0., .2, .1)
        if model == 'minimal':
            if k in COND_PARS[model]:
                for c in ('low', 'high'):
                    hier(f'{k}_{c}', 0., .3, .15)
                pm.Deterministic(f'{k}_value', m[f'{k}_high_mu'] - m[f'{k}_low_mu'])
        P['beta_session'] = pm.Normal('beta_session', 0., .3)
        if 'practice' in df:
            P['beta_practice'] = pm.Normal('beta_practice', 0., .3)
        vc, ve, a, t0 = trialwise(lambda k: P[k], model, df, xp=pt)
        tt = pt.maximum(t - t0, 1e-3)
        ll = pt.switch(resp, pt.switch(corr, logpdf(tt, vc, a) + logsf(tt, ve, a),
                                       logpdf(tt, ve, a) + logsf(tt, vc, a)),
                       logsf(tt, vc, a) + logsf(tt, ve, a))
        pm.Potential('ll', ll.sum())
    return m, t0_max


def np_loglik(df, model, get):
    vc, ve, a, t0 = trialwise(get, model, df)
    resp = df.responded.values
    corr = np.nan_to_num(df.correct.values).astype(bool)
    t = np.where(resp, df.rt.values, DEADLINE)
    tt = np.maximum(t - t0, 1e-3)

    def lpdf(v):
        return np.log(a) - .5 * np.log(2 * np.pi * tt ** 3) - (a - v * tt) ** 2 / (2 * tt)

    def lsf(v):
        sq = np.sqrt(tt)
        logF = np.logaddexp(log_ndtr((v * tt - a) / sq), 2 * v * a + log_ndtr(-(v * tt + a) / sq))
        return np.log(-np.expm1(np.minimum(logF, -1e-10)))
    return np.where(resp, np.where(corr, lpdf(vc) + lsf(ve), lpdf(ve) + lsf(vc)), lsf(vc) + lsf(ve))


def add_gains(df, gains, rois):
    """Trial-wise decoded gains (robust z within subject x session x area), averaged over the areas."""
    g = pd.read_csv(gains, sep='\t', dtype={'subject': str})
    g = g[g.roi.isin(rois)]
    assert len(g), f'no decoded gains for areas {rois}'
    keys = ['subject', 'session', 'run', 'trial_nr']
    d = df.assign(subject=df.subject.astype(str).str.zfill(2))
    z = add_gain_regressors(g.merge(d[keys + ['cond']], on=keys, how='inner', validate='many_to_one'))
    z = z.groupby(keys, as_index=False)[list(GAIN_TERMS.values())].mean().dropna()
    out = d.merge(z, on=keys, how='inner', validate='one_to_one').reset_index(drop=True)
    out['subj_idx'] = pd.Categorical(out.subject).codes
    return out


def main(model, draws, tune, post_only=False, practice=False, gains=None, gain_rois=None, gain_label='early'):
    import arviz as az
    import pymc as pm
    import xarray as xr
    df = load(1, practice=practice)
    if model == 'neural':
        df = add_gains(df, gains, gain_rois)
    tag = model + ('_wpractice' if practice else '') + (f'_{gain_label}' if model == 'neural' else '')
    print(f'{len(df)} trials ({int(df.practice.sum())} practice), {int((~df.responded).sum())} no response')
    m, t0_max = build(df, model)
    out_nc = ROOT / f'notes/data/race_{tag}.nc'
    if post_only:
        idata = az.from_netcdf(out_nc)
    else:
        # PyMC's jitter of ±1 on every parameter leaves this model's chains stuck (every transition diverges), and it is
        # also applied to initvals unless switched off: start the chains close to the initial point instead
        rng = np.random.default_rng(1)
        ip = m.initial_point()
        starts = [{k: v + rng.uniform(-.05, .05, np.shape(v)) for k, v in ip.items()} for _ in range(4)]
        with m:
            idata = pm.sample(draws=draws, tune=tune, chains=4, target_accept=.95, nuts_sampler='numpyro',
                              random_seed=1, initvals=starts, nuts_sampler_kwargs=dict(jitter=False))
    post = idata.posterior
    flat = {k: post[k].values.reshape(-1, *post[k].shape[2:]) for k in post.data_vars}
    n = flat['beta_session'].shape[0]
    if 'log_likelihood' not in idata.groups():
        ll = np.stack([np_loglik(df, model, lambda k, i=i: t0_max if k == 't0_max' else flat[k][i])
                       for i in range(n)])
        idata.add_groups(log_likelihood=xr.Dataset({'y': (('chain', 'draw', 'trial'),
                                                           ll.reshape(post.sizes['chain'], post.sizes['draw'], -1))}))
        az.to_netcdf(idata, out_nc)
    summ = az.summary(idata, var_names=[v for v in post.data_vars if not v.endswith('_z')])
    print(f'=== {tag}: divergences {int(idata.sample_stats.diverging.sum())}, max r_hat {summ.r_hat.max():.3f}, '
          f'min ess_bulk {summ.ess_bulk.min():.0f}')
    print(az.loo(idata))
    if model == 'neural':
        rows = []
        draws_ = {k: post[f'{k}_mu'].values.ravel() for k in GAIN_TERMS}
        draws_['beta_gain_high_minus_low'] = draws_['beta_gain_high'] - draws_['beta_gain_low']
        draws_['beta_value'] = post['beta_value'].values.ravel()
        for nm, d in draws_.items():
            h = az.hdi(d, hdi_prob=.95)
            rows.append(dict(areas=gain_label, term=nm, mean=d.mean(), hdi_lo=h[0], hdi_hi=h[1], p_gt_0=(d > 0).mean(),
                             pct_drift_per_sd=100 * (np.exp(d.mean()) - 1), n_subjects=df.subj_idx.nunique(), n_trials=len(df)))
        res = pd.DataFrame(rows)
        res.to_csv(ROOT / f'notes/data/decoded_gain/race_gain_effects_{gain_label}.tsv', sep='\t', index=False)
        print(res.round(4).to_string(index=False))
    if model in ('minimal', 'neural'):
        for nm in ('beta_dist_mu', 'beta_value', 'beta_session') + (('beta_practice',) if practice else ()):
            d = post[nm].values.ravel()
            h = az.hdi(d, hdi_prob=.95)
            print(f'{nm:20s} {d.mean():+.3f} [{h[0]:+.3f}, {h[1]:+.3f}]  drift {100 * (np.exp(d.mean()) - 1):+.1f}%  '
                  f'P(<0) = {(d < 0).mean():.3f}')
    for k in COND_PARS[model]:
        for c in ('low', 'high', 'value'):
            nm = f'{k}_{c}_mu' if c != 'value' else f'{k}_value'
            d = post[nm].values.ravel()
            h = az.hdi(d, hdi_prob=.95)
            print(f'{nm:20s} {d.mean():+.3f} [{h[0]:+.3f}, {h[1]:+.3f}]  P(>0) = {(d > 0).mean():.3f}')

    # Quantile PPC: simulate 200 datasets
    rng = np.random.default_rng(0)
    rows = []
    for i in rng.choice(n, 200, replace=False):
        vc, ve, a, t0 = trialwise(lambda k: t0_max if k == 't0_max' else flat[k][i], model, df)
        fc, fe = rng.wald(a / vc, a ** 2), rng.wald(a / ve, a ** 2)
        rt = np.minimum(fc, fe) + t0
        ok = rt <= DEADLINE
        rows.append(pd.DataFrame({'draw': i, 'subject': df.subject, 'cond': df.cond, 'practice': df.practice,
                                  'rt': np.where(ok, rt, np.nan), 'correct': np.where(ok, (fc < fe) * 1., np.nan)}))
    pd.concat(rows).to_parquet(ROOT / f'notes/data/race_{tag}_ppc.parquet')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('model', choices=list(COND_PARS))
    ap.add_argument('--draws', type=int, default=1000)
    ap.add_argument('--tune', type=int, default=1000)
    ap.add_argument('--post-only', action='store_true', help='reuse the saved posterior')
    ap.add_argument('--practice', action='store_true', help='also fit the practice runs (practice offset on drift)')
    ap.add_argument('--gains', default=str(ROOT / 'notes/data/decoded_gain/trial_gains.tsv'))
    ap.add_argument('--gain-rois', nargs='+', default=['V1', 'V2', 'V3'])
    ap.add_argument('--gain-label', default='early')
    a = ap.parse_args()
    main(a.model, a.draws, a.tune, a.post_only, a.practice, a.gains, a.gain_rois, a.gain_label)
