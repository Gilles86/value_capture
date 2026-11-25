"""Hierarchical Bayesian estimate of the distractor value effect on the gain field (high - low), per area.

Input: the per-run gains of a gain-field fit (``fit_gain_fields.py --per-run``: rows with scope ``run-<ses>-<run>``;
pRFs and field widths fixed to the subject's all-trials fit, gains re-estimated per run). One row per subject x
session x run x distractor value, gain in %. Model (bambi, Student-t likelihood for the noisy per-run fits):

    gain ~ high_minus_low + (high_minus_low | subject) + (1 | subject:session:run)
    log sigma ~ 1 + (1 | subject)          (--noise subject, the default; --noise shared: one sigma)

high_minus_low = +.5 / -.5, so its coefficient is the population high - low effect; the subject-level slopes shrink towards it, so a subject with a
few wild runs no longer dominates as it does in a test on 12 independent per-subject fits. Reported: posterior mean,
95% HDI and P(high > low).

    python value_capture/gainfield/hierarchical_value_gain.py --folder notes/data/gain_fields_dog_fbevents_free_exclude_widths
Output: <folder>/../hierarchical_value_gain_<folder name>.tsv (+ .nc per area with --save-idata)

Other contrasts (--contrast medium_low / high_medium, for fits with a medium value level): the same model on those
two levels; the coefficient keeps the name high_minus_low (= the second level minus the first) and the output file
gets the contrast name appended. --per-subject also writes each subject's slope (population + subject deviation).
--min-subjects (default 5) is the smallest number of subjects for which an area is fitted.
"""
import argparse
import glob
from pathlib import Path

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd


CONTRASTS = {'high_low': ('low', 'high'), 'medium_low': ('low', 'medium'), 'high_medium': ('medium', 'high')}


def load(folder, rois, contrast='high_low'):
    lo, hi = CONTRASTS[contrast]
    files = [f for f in sorted(glob.glob(str(Path(folder) / 'sub-*_gain_fields*.tsv'))) if Path(f).stat().st_size > 1]
    t = pd.concat(pd.read_csv(f, sep='\t', dtype={'subject': str}) for f in files)
    t = t[t.scope.str.startswith('run-') & t.roi.isin(rois)].drop_duplicates(['subject', 'roi', 'scope'])
    cols = {f'field_gain_distractor_{lo}': lo, f'field_gain_distractor_{hi}': hi}
    long = t.melt(id_vars=['subject', 'roi', 'session', 'run'], value_vars=list(cols), var_name='value',
                  value_name='gain')
    long['value'] = long.value.map(cols)
    long['high_minus_low'] = np.where(long.value == hi, .5, -.5)   # explicit coding: slope = high - low (or hi - lo)
    long['gain'] *= 100
    long['run_id'] = long.subject + '_' + long.session.astype(int).astype(str) + '_' + long.run.astype(int).astype(str)
    return long.dropna(subset=['gain'])


FORMULA = 'gain ~ high_minus_low + (high_minus_low | subject) + (1 | run_id)'


def fit(d, draws, tune, seed, noise='subject'):
    # noise='subject': every subject has its own residual scale (log sigma ~ 1 + (1 | subject)). The run-to-run spread
    # of the gains differs about 15-fold between subjects (hV4: SD 50% to 875%), so one shared sigma lets the noisy
    # subjects' runs count as much as the clean subjects' runs.
    f = bmb.Formula(FORMULA, 'sigma ~ 1 + (1 | subject)') if noise == 'subject' else FORMULA
    m = bmb.Model(f, d, family='t')
    idata = m.fit(draws=draws, tune=tune, chains=4, target_accept=.95, random_seed=seed, progressbar=False)
    return m, idata


def main(folder, rois, draws, tune, save_idata, seed=0, contrast='high_low', min_subjects=5, per_subject=False,
         noise='subject'):
    d_all = load(folder, rois, contrast)
    tag = ('' if contrast == 'high_low' else f'_{contrast}') + ('' if noise == 'subject' else '_sharednoise')
    rows, subj_rows = [], []
    for roi in rois:
        d = d_all[d_all.roi == roi]
        if d.subject.nunique() < min_subjects:
            print(roi, 'skipped:', d.subject.nunique(), 'subjects')
            continue
        m, idata = fit(d, draws, tune, seed, noise)
        post = idata.posterior['high_minus_low'].values.ravel()
        hdi = az.hdi(post, hdi_prob=.95)
        summ = az.summary(idata, var_names=['high_minus_low', 'Intercept'], kind='diagnostics')
        rows.append(dict(roi=roi, n_subjects=d.subject.nunique(), n_runs=d.run_id.nunique(), mean=post.mean(),
                         hdi_low=hdi[0], hdi_high=hdi[1], p_high_gt_low=(post > 0).mean(),
                         rhat_max=float(summ.r_hat.max()), ess_bulk_min=float(summ.ess_bulk.min()),
                         divergences=int(idata.sample_stats.diverging.sum())))
        print(pd.DataFrame(rows[-1:]).round(3).to_string(index=False), flush=True)
        if per_subject:
            dev = idata.posterior['high_minus_low|subject']
            sdim = [c for c in dev.dims if c not in ('chain', 'draw')][0]
            for s in dev[sdim].values:
                v = (idata.posterior['high_minus_low'] + dev.sel({sdim: s})).values.ravel()
                h = az.hdi(v, hdi_prob=.95)
                subj_rows.append(dict(roi=roi, subject=s, mean=v.mean(), hdi_low=h[0], hdi_high=h[1],
                                      p_high_gt_low=(v > 0).mean()))
        if save_idata:
            idata.to_netcdf(Path(folder).parent / f'hierarchical_value_gain_{Path(folder).name}{tag}_{roi}.nc')
    out = Path(folder).parent / f'hierarchical_value_gain_{Path(folder).name}{tag}.tsv'
    pd.DataFrame(rows).round(4).to_csv(out, sep='\t', index=False)
    print(out)
    if per_subject:
        out_s = out.with_name(out.stem + '_subjects.tsv')
        pd.DataFrame(subj_rows).round(4).to_csv(out_s, sep='\t', index=False)
        print(out_s)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--folder', required=True)
    ap.add_argument('--rois', nargs='+', default=['V1', 'V2', 'V3', 'V3AB', 'IPS', 'hV4', 'LO', 'TO', 'VO'])
    ap.add_argument('--draws', type=int, default=2000)
    ap.add_argument('--tune', type=int, default=2000)
    ap.add_argument('--save-idata', action='store_true')
    ap.add_argument('--contrast', choices=list(CONTRASTS), default='high_low')
    ap.add_argument('--min-subjects', type=int, default=5)
    ap.add_argument('--per-subject', action='store_true')
    ap.add_argument('--noise', choices=['subject', 'shared'], default='subject',
                    help='Residual scale per subject (default) or one shared scale (output tag _sharednoise)')
    a = ap.parse_args()
    main(a.folder, a.rois, a.draws, a.tune, a.save_idata, contrast=a.contrast, min_subjects=a.min_subjects,
         per_subject=a.per_subject, noise=a.noise)
