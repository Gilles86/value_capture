"""Hierarchical Bayesian estimate of the distractor value effect on the gain field (high - low), per area.

Input: the per-run gains of a gain-field fit (``fit_gain_fields.py --per-run``: rows with scope ``run-<ses>-<run>``;
pRFs and field widths fixed to the subject's all-trials fit, gains re-estimated per run). One row per subject x
session x run x distractor value, gain in %. Model (bambi, Student-t likelihood for the noisy per-run fits):

    gain ~ high_minus_low + (high_minus_low | subject) + (1 | subject:session:run)

high_minus_low = +.5 / -.5, so its coefficient is the population high - low effect; the subject-level slopes shrink towards it, so a subject with a
few wild runs no longer dominates as it does in a test on 12 independent per-subject fits. Reported: posterior mean,
95% HDI and P(high > low).

    python value_capture/gainfield/hierarchical_value_gain.py --folder notes/data/gain_fields_dog_fbevents_free_exclude_widths
Output: <folder>/../hierarchical_value_gain_<folder name>.tsv (+ .nc per area with --save-idata)
"""
import argparse
import glob
from pathlib import Path

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd


def load(folder, rois):
    files = [f for f in sorted(glob.glob(str(Path(folder) / 'sub-*_gain_fields*.tsv'))) if Path(f).stat().st_size > 1]
    t = pd.concat(pd.read_csv(f, sep='\t', dtype={'subject': str}) for f in files)
    t = t[t.scope.str.startswith('run-') & t.roi.isin(rois)].drop_duplicates(['subject', 'roi', 'scope'])
    cols = {'field_gain_distractor_low': 'low', 'field_gain_distractor_high': 'high'}
    long = t.melt(id_vars=['subject', 'roi', 'session', 'run'], value_vars=list(cols), var_name='value',
                  value_name='gain')
    long['value'] = long.value.map(cols)
    long['high_minus_low'] = np.where(long.value == 'high', .5, -.5)   # explicit coding: slope = high - low
    long['gain'] *= 100
    long['run_id'] = long.subject + '_' + long.session.astype(int).astype(str) + '_' + long.run.astype(int).astype(str)
    return long.dropna(subset=['gain'])


def fit(d, draws, tune, seed):
    m = bmb.Model('gain ~ high_minus_low + (high_minus_low | subject) + (1 | run_id)', d, family='t')
    idata = m.fit(draws=draws, tune=tune, chains=4, target_accept=.95, random_seed=seed, progressbar=False)
    return m, idata


def main(folder, rois, draws, tune, save_idata, seed=0):
    d_all = load(folder, rois)
    rows = []
    for roi in rois:
        d = d_all[d_all.roi == roi]
        if d.subject.nunique() < 5:
            print(roi, 'skipped:', d.subject.nunique(), 'subjects')
            continue
        m, idata = fit(d, draws, tune, seed)
        post = idata.posterior['high_minus_low'].values.ravel()
        hdi = az.hdi(post, hdi_prob=.95)
        summ = az.summary(idata, var_names=['high_minus_low', 'Intercept'], kind='diagnostics')
        rows.append(dict(roi=roi, n_subjects=d.subject.nunique(), n_runs=d.run_id.nunique(), mean=post.mean(),
                         hdi_low=hdi[0], hdi_high=hdi[1], p_high_gt_low=(post > 0).mean(),
                         rhat_max=float(summ.r_hat.max()), ess_bulk_min=float(summ.ess_bulk.min()),
                         divergences=int(idata.sample_stats.diverging.sum())))
        print(pd.DataFrame(rows[-1:]).round(3).to_string(index=False), flush=True)
        if save_idata:
            idata.to_netcdf(Path(folder).parent / f'hierarchical_value_gain_{Path(folder).name}_{roi}.nc')
    out = Path(folder).parent / f'hierarchical_value_gain_{Path(folder).name}.tsv'
    pd.DataFrame(rows).round(4).to_csv(out, sep='\t', index=False)
    print(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--folder', required=True)
    ap.add_argument('--rois', nargs='+', default=['V1', 'V2', 'V3', 'hV4', 'LO', 'V3AB'])
    ap.add_argument('--draws', type=int, default=2000)
    ap.add_argument('--tune', type=int, default=2000)
    ap.add_argument('--save-idata', action='store_true')
    a = ap.parse_args()
    main(a.folder, a.rois, a.draws, a.tune, a.save_idata)
