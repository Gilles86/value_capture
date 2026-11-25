"""Hierarchical Bayesian meta-analysis of the distractor value effect on the gain field, all visual areas jointly.

Same model as retsupp's ``trialwise_meta_bayes.py``: each subject x area contributes one estimate with its own
standard error, so a subject whose runs scatter widely counts for less than one whose runs agree::

    y[s, a] ~ Normal(theta[s, a], sqrt(se[s, a]^2 + tau^2))
    theta[s, a] = m[group(a)] + v[a] + w[s]

y: high - low distractor gain (%) of the subject's all-runs fit (scope ``all``). se: the SD of the per-run high - low
gains (``--per-run`` rows; pRFs and widths fixed, gains re-estimated per run) divided by sqrt(n runs). m: group means
(fixed; Early V1-V3, Dorsal V3AB IPS, Occipito-temporal hV4 LO TO VO; areas without fits drop out), v: area within group, w: subject, tau: subject x area
heterogeneity beyond the sampling error. Overall effect = mean of the area means. Non-centred, HalfNormal SDs scaled
to sd(y). Variants: ``normal`` (primary), ``t`` (Student-t likelihood), ``nose`` (no SE weighting).

    python value_capture/gainfield/meta_value_gain.py --folder notes/data/gain_fields_dog_fbevents_free_exclude_widths
Output: <folder>/../meta_value_gain_<folder name>.tsv (+ _diagnostics.tsv)
"""
import argparse
import glob
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

GROUPS = {'Early': ['V1', 'V2', 'V3'], 'Dorsal': ['V3AB', 'IPS'], 'Occipito-temporal': ['hV4', 'LO', 'TO', 'VO']}
AREAS = [a for g in GROUPS.values() for a in g]
GROUP_OF = {a: g for g, aa in GROUPS.items() for a in aa}


def table(folder, lo='low', hi='high', min_subjects=5):
    files = [f for f in sorted(glob.glob(str(Path(folder) / 'sub-*_gain_fields*.tsv'))) if Path(f).stat().st_size > 1]
    t = pd.concat(pd.read_csv(f, sep='\t', dtype={'subject': str}) for f in files)
    t = t[t.roi.isin(AREAS)].drop_duplicates(['subject', 'roi', 'scope'])
    t['d'] = (t[f'field_gain_distractor_{hi}'] - t[f'field_gain_distractor_{lo}']) * 100
    y = t[t.scope == 'all'].set_index(['subject', 'roi']).d.rename('y')
    runs = t[t.scope.str.startswith('run-')].groupby(['subject', 'roi']).d
    se = (runs.std() / np.sqrt(runs.count())).rename('se')
    t = pd.concat([y, se, runs.count().rename('n_runs')], axis=1).dropna().reset_index()
    n = t.groupby('roi').subject.transform('nunique')
    return t[n >= min_subjects].reset_index(drop=True)          # areas fitted in too few subjects drop out


def fit(t, likelihood='normal', use_se=True, seed=0, draws=2000, tune=2000, group_def=None):
    group_def = group_def or GROUPS
    group_of = {a: g for g, aa in group_def.items() for a in aa}
    areas = [a for aa in group_def.values() for a in aa if a in set(t.roi)]
    groups = [g for g in group_def if any(group_of[a] == g for a in areas)]
    subs = sorted(t.subject.unique())
    ai = t.roi.map({a: i for i, a in enumerate(areas)}).values
    gi_area = np.array([groups.index(group_of[a]) for a in areas])
    si = t.subject.map({s: i for i, s in enumerate(subs)}).values
    y, se = t.y.values, (t.se.values if use_se else np.zeros(len(t)))
    sy = float(np.std(y))
    with pm.Model(coords=dict(area=areas, group=groups, subject=subs)) as m:
        gm = pm.Normal('group_mean', 0, 2 * sy, dims='group')
        sd_a = pm.HalfNormal('sd_area', sy)
        sd_s = pm.HalfNormal('sd_subject', sy)
        tau = pm.HalfNormal('tau', sy)
        v = pm.Deterministic('v', sd_a * pm.Normal('v_z', 0, 1, dims='area'), dims='area')
        w = pm.Deterministic('w', sd_s * pm.Normal('w_z', 0, 1, dims='subject'), dims='subject')
        am = pm.Deterministic('area_mean', gm[gi_area] + v, dims='area')
        pm.Deterministic('mu', am.mean())
        sigma = pm.math.sqrt(se ** 2 + tau ** 2)
        if likelihood == 't':
            pm.StudentT('y', nu=pm.Gamma('nu', 2, .1), mu=am[ai] + w[si], sigma=sigma, observed=y)
        else:
            pm.Normal('y', mu=am[ai] + w[si], sigma=sigma, observed=y)
        idata = pm.sample(draws=draws, tune=tune, chains=4, target_accept=.99, random_seed=seed, progressbar=False)
    return idata, areas, groups


def summarize(idata, areas, groups, variant):
    post, rows = idata.posterior, []

    def add(level, unit, x):
        x = np.asarray(x).ravel()
        lo, hi = az.hdi(x, hdi_prob=.95)
        rows.append(dict(variant=variant, level=level, unit=unit, mean=x.mean(), hdi_low=lo, hdi_high=hi,
                         p_gt0=(x > 0).mean()))
    add('overall', 'all', post['mu'])
    for g in groups:
        add('group', g, post['group_mean'].sel(group=g))
    for a in areas:
        add('area', a, post['area_mean'].sel(area=a))
    s = az.summary(idata, var_names=['mu', 'sd_area', 'sd_subject', 'tau', 'group_mean', 'area_mean'], kind='diagnostics')
    return rows, dict(variant=variant, divergences=int(idata.sample_stats.diverging.sum()),
                      rhat_max=float(s.r_hat.max()), ess_bulk_min=float(s.ess_bulk.min()))


def main(folder, variants, lo='low', hi='high'):
    t = table(folder, lo, hi)
    rows, diags = [], []
    for variant in variants:
        idata, areas, groups = fit(t, likelihood='t' if variant == 't' else 'normal', use_se=variant != 'nose')
        r, d = summarize(idata, areas, groups, variant)
        print(pd.DataFrame(r).round(3).to_string(index=False), d, flush=True)
        rows += r
        diags.append(d)
    tag = '' if (lo, hi) == ('low', 'high') else f'_{hi}_minus_{lo}'
    out = Path(folder).parent / f'meta_value_gain_{Path(folder).name}{tag}.tsv'
    pd.DataFrame(rows).round(4).to_csv(out, sep='\t', index=False)
    pd.DataFrame(diags).to_csv(out.with_name(out.stem + '_diagnostics.tsv'), sep='\t', index=False)
    t.round(4).to_csv(out.with_name(out.stem + '_data.tsv'), sep='\t', index=False)
    print(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--folder', required=True)
    ap.add_argument('--variants', nargs='+', default=['normal', 't', 'nose'])
    ap.add_argument('--levels', nargs=2, default=['low', 'high'], metavar=('LO', 'HI'),
                    help='Contrast HI - LO of the field_gain_distractor_<level> columns (e.g. low medium)')
    a = ap.parse_args()
    main(a.folder, a.variants, *a.levels)
