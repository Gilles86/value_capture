"""Bayesian group models of distractor value (and colour) in every ROI, with a noise level per subject.

Input: the per-run table of ``decode/value_rois.py`` (``sub-XX_value_rois_runs.tsv``). Per ROI and measure:

- ``univariate``: high - low ROI-mean beta (% signal) per run. Within a session colour and value are confounded,
  so the model carries the colour as a covariate that flips sign between sessions (``green_high`` = +.5 when green is
  the high-value colour, -.5 when orange is)::

      y ~ 1 + green_high + (1 + green_high | subject),   log sigma ~ 1 + (1 | subject),   Student-t

  Intercept = value effect (high - low, colour balanced); green_high = colour effect (green - orange).
- ``auc_across``: AUC - .5 (x 100) of the other-session decoder on each run: > 0 a value code that survives the
  colour swap. ``y ~ 1 + (1 | subject)``, same noise model.

Plus the SE-weighted meta-analysis over all ROIs (``gainfield/meta_value_gain.py``, retsupp model): per subject x
ROI the mean over runs and its SE, with pathway means.

    python value_capture/decode/bayes_value_rois.py --points exclude
Output: notes/data/bayes_value_rois_<points>.tsv, notes/data/meta_value_rois_<measure>_<points>.tsv
"""
import argparse
import glob
from pathlib import Path

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

from value_capture.gainfield.meta_value_gain import fit as meta_fit, summarize as meta_summarize

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'notes' / 'data'
GROUPS = {'Early': ['V1', 'V2', 'V3'], 'Dorsal': ['V3AB', 'IPS'], 'Occipito-temporal': ['hV4', 'VO', 'LO', 'TO'],
          'Ventral temporal / MTL': ['fusiform', 'parahippocampal', 'entorhinal', 'hippocampus', 'amygdala'],
          'Striatum / OFC': ['caudate', 'putamen', 'accumbens', 'medialOFC', 'lateralOFC']}
ROIS = [r for rr in GROUPS.values() for r in rr]


def load(points):
    t = pd.concat(pd.read_csv(f, sep='\t', dtype={'subject': str})
                  for f in sorted(glob.glob(str(DATA / 'value_rois' / 'sub-*_value_rois_runs.tsv'))))
    t = t[t.points == points].copy()
    odd = t.subject.astype(int) % 2 == 1
    t['green_high'] = np.where(odd == (t.session.astype(int) == 1), .5, -.5)     # CLAUDE.md counterbalancing
    t['y'] = np.where(t.measure.str.startswith('auc'), (t.value - .5) * 100, t.value)
    return t


def fit_roi(d, measure, seed=0):
    f = 'y ~ 1 + green_high + (1 + green_high | subject)' if measure == 'univariate_high_minus_low' \
        else 'y ~ 1 + (1 | subject)'
    m = bmb.Model(bmb.Formula(f, 'sigma ~ 1 + (1 | subject)'), d, family='t')
    idata = m.fit(draws=2000, tune=2000, chains=4, target_accept=.95, random_seed=seed, progressbar=False)
    out = []
    for term, label in [('Intercept', 'value'), ('green_high', 'colour')]:
        if term not in idata.posterior:
            continue
        v = idata.posterior[term].values.ravel()
        h = az.hdi(v, hdi_prob=.95)
        out.append(dict(effect=label, mean=v.mean(), hdi_low=h[0], hdi_high=h[1], p_gt0=(v > 0).mean()))
    s = az.summary(idata, var_names=[k for k in ('Intercept', 'green_high') if k in idata.posterior],
                   kind='diagnostics')
    for o in out:
        o.update(rhat_max=float(s.r_hat.max()), divergences=int(idata.sample_stats.diverging.sum()))
    return out


def main(points, measures):
    t = load(points)
    rows = []
    for measure in measures:
        tm = t[t.measure == measure]
        for roi in ROIS:
            d = tm[tm.roi == roi]
            if d.subject.nunique() < 5:
                continue
            for r in fit_roi(d, measure):
                rows.append(dict(points=points, measure=measure, roi=roi, n_subjects=d.subject.nunique(), **r))
                print(pd.DataFrame(rows[-1:]).round(3).to_string(index=False, header=len(rows) == 1), flush=True)
        # meta-analysis over all ROIs: per subject x ROI mean over runs and its SE
        g = tm.groupby(['subject', 'roi']).y
        mt = pd.concat([g.mean().rename('y'), (g.std() / np.sqrt(g.count())).rename('se')], axis=1).reset_index()
        mrows = []
        for variant in ('normal', 't', 'nose'):
            idata, areas, groups = meta_fit(mt, likelihood='t' if variant == 't' else 'normal',
                                            use_se=variant != 'nose', group_def=GROUPS)
            r, d = meta_summarize(idata, areas, groups, variant)
            mrows += [dict(x, **{k: v for k, v in d.items() if k != 'variant'}) for x in r]
        pd.DataFrame(mrows).round(4).to_csv(DATA / f'meta_value_rois_{measure}_{points}.tsv', sep='\t', index=False)
        mt.round(4).to_csv(DATA / f'meta_value_rois_{measure}_{points}_data.tsv', sep='\t', index=False)
    out = DATA / f'bayes_value_rois_{points}.tsv'
    pd.DataFrame(rows).round(4).to_csv(out, sep='\t', index=False)
    print(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--points', choices=['exclude', 'keep'], default='exclude')
    ap.add_argument('--measures', nargs='+', default=['univariate_high_minus_low', 'auc_across'])
    a = ap.parse_args()
    main(a.points, a.measures)
