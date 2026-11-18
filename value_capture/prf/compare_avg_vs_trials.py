"""Experiment: does fitting PRFs to condition-averaged betas differ from fitting all single-trial betas?

Both fits use the bar-only paradigm on the ±5° grid (fit_prf.py --paradigm bar --out-suffix _grid5, with and
without --average-trials). Expectation: the same parameters (with identical predictions within a stimulus image,
least squares on the image means minimises the same between-image term; the within-image scatter does not depend
on the parameters), but a much higher R² on the averaged data, because averaging removes the within-image noise
from the denominator too. So R² thresholds select very different voxel sets.

Per subject × ROI (neuropythy volume labels, V1-V3AB; 'other' = rest of the grey-matter mask), over voxels with
R² ≥ .02 in either fit:
  - R² of each fit on its own data (single trials; image means)
  - single-trial R² of the averaged fit's parameters (predict each trial from its image)
  - agreement of x, y, sd, amplitude, baseline (Pearson r; median absolute difference)
  - voxel counts passing R² ≥ .1 under each R²

    python value_capture/prf/compare_avg_vs_trials.py --subjects 04 05 ... --out notes/data/prf_avg_vs_trials.tsv
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault('KERAS_BACKEND', 'jax')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from nilearn import masking  # noqa: E402

from value_capture.prf import fit_prf as fp  # noqa: E402
from value_capture.utils.data import Subject  # noqa: E402

ROIS = {1: 'V1', 2: 'V2', 3: 'V3', 4: 'hV4', 5: 'VO', 6: 'VO', 7: 'LO', 8: 'LO', 9: 'TO', 10: 'TO',
        11: 'V3AB', 12: 'V3AB'}
PARS = ['x', 'y', 'sd', 'baseline', 'amplitude']


def r2(y, yhat):
    return 1 - ((y - yhat) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


def main(subjects, bids, deriv, out, r2_keep=.02):
    from braincoder.models import GaussianPRF2D
    rows, vox_rows = [], []
    for s in subjects:
        sub = Subject(s, bids_folder=bids)
        sessions = sub.get_sessions()
        d = Path(bids) / 'derivatives'
        tsv = lambda name: pd.read_csv(d / name / f'sub-{s}' / 'ses-all' / 'func' /  # noqa: E731
                                       f'sub-{s}_ses-all_task-valuecapture_space-T1w_desc-prf_params.tsv',
                                       sep='\t', index_col=0)
        st, av = tsv(deriv), tsv(deriv + '_avg')
        keep = np.flatnonzero((st.r2.to_numpy() >= r2_keep) | (av.r2.to_numpy() >= r2_keep))
        mask = fp._get_gm_mask(sub, sessions[0], bids, .1)
        Y = masking.apply_mask(sub.get_single_trial_estimates(sessions, glmsingle_deriv='glmsingle'),
                               mask).astype(np.float32)[:, keep]
        varea = np.round(masking.apply_mask(sub.get_retinotopic_atlas(bold_space=True, session=sessions[0]),
                                            mask)).astype(int)[keep]
        trials = sub.get_trial_metadata(sessions, glmsingle_deriv='glmsingle').reset_index(drop=True)
        P = fp.build_paradigm(trials, 'bar')
        imgs, grp = np.unique(P, axis=0, return_inverse=True)
        grp = grp.ravel()
        grid = fp.make_grid_coordinates()
        par_df = pd.DataFrame(imgs, index=pd.Index(range(len(imgs)), name='frame'))

        def predict(pars):
            p = pars.iloc[keep][PARS].reset_index(drop=True).astype(np.float32)
            m = GaussianPRF2D(grid_coordinates=grid)
            return np.asarray(m.predict(paradigm=par_df, parameters=p)).astype(np.float32)   # (n_images, n_vox)

        pred_st, pred_av = predict(st), predict(av)
        means = np.stack([Y[grp == g].mean(0) for g in range(len(imgs))])
        res = pd.DataFrame({
            'roi': [ROIS.get(v, 'other') for v in varea],
            'r2_trials_fit_trials': r2(Y, pred_st[grp]),          # single-trial fit, on single trials
            'r2_trials_fit_avg': r2(Y, pred_av[grp]),             # averaged fit, on single trials
            'r2_avg_fit_avg': r2(means, pred_av),                 # averaged fit, on image means
            'r2_avg_fit_trials': r2(means, pred_st),              # single-trial fit, on image means
        })
        for p_ in PARS:
            res[f'{p_}_trials'] = st[p_].to_numpy()[keep]
            res[f'{p_}_avg'] = av[p_].to_numpy()[keep]
        res['subject'] = s
        vox_rows.append(res)
        for roi, g in res.groupby('roi'):
            sel = g.r2_trials_fit_trials >= .1
            row = {'subject': s, 'roi': roi, 'n': len(g),
                   'n_r2trials_ge_0.1': int((g.r2_trials_fit_trials >= .1).sum()),
                   'n_r2avg_ge_0.1': int((g.r2_avg_fit_avg >= .1).sum()),
                   'med_r2_trials': g.r2_trials_fit_trials.median(), 'med_r2_avg': g.r2_avg_fit_avg.median(),
                   'med_r2_trials_from_avgfit': g.r2_trials_fit_avg.median(),
                   'med_dr2_trials': (g.r2_trials_fit_trials - g.r2_trials_fit_avg).median()}
            for p_ in PARS:
                a_, b_ = g.loc[sel, f'{p_}_trials'], g.loc[sel, f'{p_}_avg']
                row[f'r_{p_}'] = a_.corr(b_) if sel.sum() > 5 else np.nan
                row[f'mad_{p_}'] = (a_ - b_).abs().median() if sel.sum() > 5 else np.nan
            rows.append(row)
        print(s, 'done', flush=True)
    t = pd.DataFrame(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    t.round(4).to_csv(out, sep='\t', index=False)
    pd.concat(vox_rows).round(5).to_csv(str(out).replace('.tsv', '_voxels.tsv'), sep='\t', index=False)
    pd.set_option('display.width', 220)
    print(t.groupby('roi')[[c for c in t.columns if c not in ('subject', 'roi')]].median().round(3))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--subjects', nargs='+', default=[f'{i:02d}' for i in range(4, 16)])
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-valuecapture')
    ap.add_argument('--deriv', default='prf_glmsingle_grid5')
    ap.add_argument('--out', default='notes/data/prf_avg_vs_trials.tsv')
    a = ap.parse_args()
    main(a.subjects, a.bids_folder, a.deriv, a.out)
