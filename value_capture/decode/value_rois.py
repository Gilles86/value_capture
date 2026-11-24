"""Distractor value and colour outside the pRF-mapped cortex: medial temporal lobe, striatum, OFC (+ visual controls).

Single-trial search betas (``glmsingle_fbevents``, T1w, unsmoothed) of distractor-present trials, in anatomical ROIs
from the subject's FreeSurfer aparc+aseg (entorhinal, parahippocampal, fusiform, hippocampus, amygdala, caudate,
putamen, accumbens, medial / lateral OFC) and, as controls, all visual areas from the
neuropythy atlas
(V1-V3, hV4, VO, LO, TO, V3AB) and Wang IPS0-5.

Colour and value are confounded within a session and reversed between sessions (CLAUDE.md), which separates them:

- Univariate: ROI-mean beta per trial, averaged per session x colour x value cell (``*_univariate.tsv``). High - low
  over both sessions is value; green - orange over both sessions is colour.
- Decoding high vs low value (shrinkage LDA, voxels z-scored within run; AUC; ``*_decoding.tsv``):
  ``within``  leave-one-run-out inside a session: value or colour (they are the same split there);
  ``across``  train on one session, test on the other with the value labels: AUC > .5 = a value code that survives
              the colour swap, AUC < .5 = a colour code (1 - AUC is the colour AUC).

Points handling: ``keep`` all trials (the points are their own events in glmsingle_fbevents), ``exclude`` only
trials without the points display.

    python value_capture/decode/value_rois.py 04
Output: <bids>/derivatives/value_rois/sub-04_value_rois_{univariate,decoding}.tsv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import image, masking
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

from value_capture.gainfield.fit_gain_fields import distractor_colour, trial_table
from value_capture.utils.data import Subject

ASEG_ROIS = {'entorhinal': (1006, 2006), 'parahippocampal': (1016, 2016), 'fusiform': (1007, 2007),
             'hippocampus': (17, 53), 'amygdala': (18, 54), 'caudate': (11, 50), 'putamen': (12, 51),
             'accumbens': (26, 58), 'medialOFC': (1014, 2014), 'lateralOFC': (1012, 2012)}
VAREA_ROIS = {'V1': (1,), 'V2': (2,), 'V3': (3,), 'hV4': (4,), 'VO': (5, 6), 'LO': (7, 8), 'TO': (9, 10),
              'V3AB': (11, 12), 'IPS': (101,)}


def zscore_within(X, groups):
    X = X.copy()
    for g in np.unique(groups):
        i = groups == g
        X[i] = (X[i] - X[i].mean(0)) / (X[i].std(0) + 1e-6)
    return X


def auc(Xtr, ytr, Xte, yte):
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(Xtr, ytr)
    return roc_auc_score(yte, clf.decision_function(Xte))


def main(subject, bids, glmsingle_deriv, out_dir):
    sub = Subject(subject, bids_folder=bids)
    sessions = sub.get_sessions()
    assert len(sessions) == 2, sessions
    trials = trial_table(sub, sessions, glmsingle_deriv)
    mask = sub.get_brain_mask(int(sessions[0]))
    betas = sub.get_single_trial_estimates(sessions, glmsingle_deriv=glmsingle_deriv)
    assert betas.shape[:3] == mask.shape[:3], (betas.shape, mask.shape)
    assert betas.shape[3] == len(trials)
    fs = Path(sub.freesurfer_subjects_dir) / sub.freesurfer_subject_id / 'mri'
    aseg = np.round(masking.apply_mask(image.resample_to_img(image.load_img(fs / 'aparc+aseg.mgz'), mask,
                                                             interpolation='nearest', force_resample=True,
                                                             copy_header=True), mask))
    varea = np.round(masking.apply_mask(sub.get_retinotopic_atlas(bold_space=True, session=int(sessions[0]),
                                                             include_wang=True), mask))
    rois = {k: np.isin(aseg, v) for k, v in ASEG_ROIS.items()} | {k: np.isin(varea, v) for k, v in VAREA_ROIS.items()}
    keep_vox = np.flatnonzero(np.any(list(rois.values()), 0))
    Y_all = masking.apply_mask(betas, mask)[:, keep_vox].astype(np.float32)
    rois = {k: v[keep_vox] for k, v in rois.items()}

    pres = trials.distractor_present.to_numpy(bool)
    tr = trials[pres].reset_index(drop=True)
    Y_all = Y_all[pres]
    tr['value'] = np.where(tr.value_rank == 2, 'high', 'low')
    tr['colour'] = [distractor_colour(t) for _, t in tr.iterrows()]
    y = (tr.value == 'high').to_numpy(int)
    ses = tr.session.to_numpy(int)
    run = tr.run.to_numpy(int)
    uni, dec = [], []
    for points in ('keep', 'exclude'):
        use = np.ones(len(tr), bool) if points == 'keep' else ~tr.show_feedback.to_numpy(bool)
        for roi, m in rois.items():
            Y = Y_all[:, m]
            good = np.isfinite(Y).all(0) & (Y.std(0) > 0)
            Y = Y[:, good]
            nvox = Y.shape[1]
            if nvox < 10:
                continue
            ym = Y.mean(1)
            cells = pd.DataFrame(dict(session=ses, colour=tr.colour, value=tr.value, beta=ym))[use]
            for (s, c, v), g in cells.groupby(['session', 'colour', 'value']):
                uni.append(dict(subject=subject, points=points, roi=roi, n_voxels=nvox, session=s, colour=c,
                                value=v, n_trials=len(g), beta=g.beta.mean()))
            X = zscore_within(Y, ses * 100 + run)
            for s in sessions:
                s = int(s)
                inses = use & (ses == s)
                scores = []
                for r in np.unique(run[inses]):
                    te, trn = inses & (run == r), inses & (run != r)
                    if len(np.unique(y[te])) < 2:
                        continue
                    scores.append(auc(X[trn], y[trn], X[te], y[te]))
                dec.append(dict(subject=subject, points=points, roi=roi, n_voxels=nvox, scheme='within',
                                test_session=s, auc=np.mean(scores)))
                trn, te = use & (ses != s), inses
                dec.append(dict(subject=subject, points=points, roi=roi, n_voxels=nvox, scheme='across',
                                test_session=s, auc=auc(X[trn], y[trn], X[te], y[te])))
            print(points, roi, nvox, {d['scheme']: round(d['auc'], 3) for d in dec[-2:]}, flush=True)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(uni).round(6).to_csv(out_dir / f'sub-{subject}_value_rois_univariate.tsv', sep='\t', index=False)
    pd.DataFrame(dec).round(6).to_csv(out_dir / f'sub-{subject}_value_rois_decoding.tsv', sep='\t', index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject')
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-valuecapture')
    ap.add_argument('--glmsingle-deriv', default='glmsingle_fbevents')
    ap.add_argument('--out-dir', default='/shares/zne.uzh/gdehol/ds-valuecapture/derivatives/value_rois')
    a = ap.parse_args()
    main(a.subject, a.bids_folder, a.glmsingle_deriv, a.out_dir)
