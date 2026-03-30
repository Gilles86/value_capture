#!/usr/bin/env python3
"""
Extract voxel-wise PRF parameters per retinotopic ROI and save as a TSV.

Requires:
  - GLMsingle betas (fit_prf.py already run → volumetric PRF maps in T1w space)
  - Benson atlas in subject's FreeSurfer mri/ directory (register_retinotopy.py run)

Output: derivatives/prf_summaries/<prf_deriv>/sub-{id}/sub-{id}_prf_voxels.tsv
  Columns: roi, hemi, voxel, x, y, sd, amplitude, baseline, ecc, angle, R2

Usage:
    python summarize_prf.py 01
    python summarize_prf.py 01 --prf-deriv prf_glmsingle_s6mm --r2-threshold 0.05
    python summarize_prf.py 01 02 03   # multiple subjects
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import image
from nilearn.maskers import NiftiMasker

from value_capture.utils.data import BIDS_FOLDER, Subject


def summarize_prf(subject, prf_deriv='prf_glmsingle', sessions=None,
                  bids_folder=BIDS_FOLDER, r2_threshold=0.0):
    sub = Subject(subject, bids_folder=bids_folder)
    ses_label = sub._ses_label(sessions)

    print(f'sub-{sub.subject_id}  [{prf_deriv}]  r2>{r2_threshold}')

    prf_imgs = sub.get_prf_parameters_volume(prf_deriv=prf_deriv, sessions=sessions)
    labels = sub.get_retinotopic_labels()

    dfs = []
    for idx, roi_name in labels.items():
        for hemi in ['L', 'R']:
            roi_mask = sub.get_retinotopic_roi(
                f'{roi_name}_{hemi}', bold_space=True)
            n_vox = int(np.sum(roi_mask.get_fdata() > 0))
            if n_vox == 0:
                continue

            masker = NiftiMasker(mask_img=roi_mask)
            data = {}
            for param, img in prf_imgs.items():
                data[param] = masker.fit_transform(img).squeeze()

            df = pd.DataFrame(data)
            df['roi'] = roi_name
            df['hemi'] = hemi
            df['voxel'] = np.arange(n_vox)

            if r2_threshold > 0:
                df = df[df['R2'] > r2_threshold]

            dfs.append(df)
            print(f'  {roi_name}_{hemi}: {n_vox} voxels ({len(df)} above R²>{r2_threshold})')

    if not dfs:
        print('  No voxels found — has register_retinotopy.py been run?')
        return

    result = pd.concat(dfs, ignore_index=True)
    result = result.set_index(['roi', 'hemi', 'voxel'])

    out_dir = (Path(bids_folder) / 'derivatives' / 'prf_summaries' /
               prf_deriv / f'sub-{sub.subject_id}')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fn = out_dir / f'sub-{sub.subject_id}_{ses_label}_prf_voxels.tsv'
    result.to_csv(out_fn, sep='\t')
    print(f'  Saved → {out_fn}')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract PRF parameters per retinotopic ROI.')
    parser.add_argument('subjects', nargs='+', help='Subject ID(s) (e.g. 01 02 03)')
    parser.add_argument('--prf-deriv', default='prf_glmsingle')
    parser.add_argument('--sessions', nargs='+', type=int, default=None)
    parser.add_argument('--bids-folder', default=BIDS_FOLDER)
    parser.add_argument('--r2-threshold', type=float, default=0.0,
                        help='Only keep voxels with R² above this (default: 0 = all)')
    args = parser.parse_args()

    for sub in args.subjects:
        summarize_prf(
            sub,
            prf_deriv=args.prf_deriv,
            sessions=args.sessions,
            bids_folder=args.bids_folder,
            r2_threshold=args.r2_threshold,
        )
