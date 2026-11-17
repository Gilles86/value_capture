#!/usr/bin/env python3
"""Group (second-level) inference with permutations: whole-brain cluster and TFCE FWE, and reward ROIs.

For each first-level contrast of fit_nilearn_glm_reward.py (MNI152NLin2009cAsym, 2 mm, 6 mm smoothing):
  - nilearn non_parametric_inference, one-sample sign flipping (n = 12 → 4096 distinct flips; 10000 draws),
    two-sided, cluster-forming threshold p < .001 (voxel t), FWE-corrected p for voxel max-t, cluster size,
    cluster mass and TFCE. Maps saved as -log10 p.
  - The same restricted to an a-priori reward mask (bilateral nucleus accumbens, caudate, putamen and the
    vmPFC = Harvard-Oxford frontal medial + subcallosal cortex): small-volume FWE.
  - ROI means per subject (effect size averaged over the ROI) → one-sample t-test and exact sign-flip p.
  - Cluster tables (peak MNI coordinates, size, mass/TFCE FWE p) for clusters with FWE p < .05.

ROIs come from the Harvard-Oxford max-probability atlases (thr 25, 2 mm, FSL MNI152); fMRIPrep's
MNI152NLin2009cAsym differs from it by ~1-2 mm, small next to 6 mm smoothing.

    python value_capture/glm/fit_second_level_perm.py --bids-folder /shares/zne.uzh/gdehol/ds-valuecapture --n-jobs 16
Output: <bids>/derivatives/nilearn_glm_reward/group/  (maps, clusters_*.tsv, roi_stats.tsv, roi_subject_means.tsv)
"""
import argparse
import itertools
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets, image, masking
from nilearn.glm.second_level import non_parametric_inference
from nilearn.reporting import get_clusters_table
from scipy import stats

CONTRASTS = ['outcome_logreward', 'outcome_logreward_cued', 'outcome_logreward_uncued', 'outcome_rpe', 'cue_value',
             'capture', 'feedback_vs_omit', 'omission_value']
SPACE = 'space-MNI152NLin2009cAsym_res-2'


def reward_rois(ref):
    sub = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
    cort = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    sub_img, cort_img = (image.resample_to_img(a.maps, ref, interpolation='nearest', force_resample=True,
                                               copy_header=True) for a in (sub, cort))
    s, c = np.asarray(sub_img.dataobj), np.asarray(cort_img.dataobj)
    lab = lambda labels, names: [labels.index(n) for n in names]  # noqa: E731
    defs = {'accumbens': (s, lab(sub.labels, ['Left Accumbens', 'Right Accumbens'])),
            'caudate': (s, lab(sub.labels, ['Left Caudate', 'Right Caudate'])),
            'putamen': (s, lab(sub.labels, ['Left Putamen', 'Right Putamen'])),
            'vmPFC': (c, lab(cort.labels, ['Frontal Medial Cortex', 'Subcallosal Cortex']))}
    return {k: nib.Nifti1Image(np.isin(arr, idx).astype(np.uint8), ref.affine) for k, (arr, idx) in defs.items()}


def sign_flip_p(x):
    """Exact two-sided sign-flip p for the mean of x (n ≤ 16)."""
    x = np.asarray(x, float)
    flips = np.array(list(itertools.product([1, -1], repeat=len(x))))
    null = (flips * x).mean(1)
    return float((np.abs(null) >= abs(x.mean()) - 1e-12).mean())


def main(bids, subjects, n_perm, n_jobs, contrasts):
    root = Path(bids) / 'derivatives' / 'nilearn_glm_reward'
    out = root / 'group'
    out.mkdir(parents=True, exist_ok=True)
    roi_rows, subj_rows = [], []
    rois = None
    for con in contrasts:
        imgs = [root / f'sub-{s}' / 'func' / f'sub-{s}_task-valuecapture_{SPACE}_contrast-{con}_stat-effect_statmap.nii.gz'
                for s in subjects]
        imgs = [str(i) for i in imgs if i.exists()]
        n = len(imgs)
        if n < 6:
            print(f'{con}: only {n} subjects, skipped')
            continue
        data = image.load_img(imgs)
        arr = np.asarray(data.dataobj)
        mask = image.new_img_like(data.slicer[..., 0], (np.isfinite(arr) & (arr != 0)).all(-1).astype(np.uint8))
        if rois is None:
            rois = reward_rois(mask)
            union = image.new_img_like(mask, (sum(np.asarray(r.dataobj) for r in rois.values()) > 0).astype(np.uint8))
            reward_mask = masking.intersect_masks([mask, union], threshold=1, connected=False)
            reward_mask.to_filename(str(out / 'reward_mask.nii.gz'))
        dm = pd.DataFrame({'intercept': np.ones(n)})
        kw = dict(design_matrix=dm, n_perm=n_perm, two_sided_test=True, n_jobs=n_jobs, random_state=0)
        for scope, m in (('wholebrain', mask), ('reward', reward_mask)):
            res = non_parametric_inference(imgs, mask=m, threshold=.001, tfce=True, **kw)
            for k, img in res.items():
                img.to_filename(str(out / f'group_{SPACE}_contrast-{con}_scope-{scope}_stat-{k}.nii.gz'))
            # clusters surviving cluster-mass or TFCE FWE
            t = res['t']
            sig = np.maximum(np.asarray(res['logp_max_mass'].dataobj), np.asarray(res['logp_max_tfce'].dataobj))
            t_sig = image.new_img_like(t, np.where(sig > -np.log10(.05), np.asarray(t.dataobj), 0))
            tab = get_clusters_table(t_sig, stat_threshold=1e-6, cluster_threshold=1, two_sided=True)
            tab.to_csv(out / f'clusters_contrast-{con}_scope-{scope}.tsv', sep='\t', index=False)
            best = {k.replace('logp_max_', ''): float(np.nanmax(np.asarray(res[k].dataobj)))
                    for k in res if k.startswith('logp_max')}
            print(f'{con} [{scope}]: n={n}, max -log10 p_FWE: ' +
                  ', '.join(f'{k} {v:.2f}' for k, v in best.items()) + f'; {(len(tab))} cluster rows', flush=True)
        # ROI means per subject
        for name, roi in rois.items():
            vals = masking.apply_mask(imgs, masking.intersect_masks([mask, roi], threshold=1)).mean(1)
            tt = stats.ttest_1samp(vals, 0)
            roi_rows.append(dict(contrast=con, roi=name, n=n, mean=vals.mean(), sem=stats.sem(vals), t=tt.statistic,
                                 p_t=tt.pvalue, p_signflip=sign_flip_p(vals), n_positive=int((vals > 0).sum())))
            subj_rows += [dict(contrast=con, roi=name, subject=s, effect=v)
                          for s, v in zip([Path(i).name[4:6] for i in imgs], vals)]
    r = pd.DataFrame(roi_rows)
    r.round(5).to_csv(out / 'roi_stats.tsv', sep='\t', index=False)
    pd.DataFrame(subj_rows).round(6).to_csv(out / 'roi_subject_means.tsv', sep='\t', index=False)
    pd.set_option('display.width', 200)
    print(r.round(3).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-valuecapture')
    ap.add_argument('--subjects', nargs='+', default=[f'{i:02d}' for i in range(4, 16)])
    ap.add_argument('--contrasts', nargs='+', default=CONTRASTS)
    ap.add_argument('--n-perm', type=int, default=10000)
    ap.add_argument('--n-jobs', type=int, default=16)
    a = ap.parse_args()
    main(a.bids_folder, a.subjects, a.n_perm, a.n_jobs, a.contrasts)
