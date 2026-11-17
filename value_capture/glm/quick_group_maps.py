#!/usr/bin/env python3
"""Quick look: one-sample group z maps (parametric OLS, uncorrected) for the reward GLM contrasts.

For eyeballing before the permutation inference (fit_second_level_perm.py) finishes. Writes
<bids>/derivatives/nilearn_glm_reward/group_quick/contrast-<name>_stat-z.nii.gz (float32).

    python value_capture/glm/quick_group_maps.py --bids-folder /shares/zne.uzh/gdehol/ds-valuecapture
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.glm.second_level import SecondLevelModel

from value_capture.glm.fit_second_level_perm import CONTRASTS, SPACE

ap = argparse.ArgumentParser()
ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-valuecapture')
ap.add_argument('--subjects', nargs='+', default=[f'{i:02d}' for i in range(4, 16)])
a = ap.parse_args()
root = Path(a.bids_folder) / 'derivatives' / 'nilearn_glm_reward'
out = root / 'group_quick'
out.mkdir(parents=True, exist_ok=True)
for con in CONTRASTS:
    imgs = [root / f'sub-{s}' / 'func' / f'sub-{s}_task-valuecapture_{SPACE}_contrast-{con}_stat-effect_statmap.nii.gz'
            for s in a.subjects]
    imgs = [str(i) for i in imgs if i.exists()]
    if len(imgs) < 6:
        print(con, 'skipped', len(imgs))
        continue
    m = SecondLevelModel().fit(imgs, design_matrix=pd.DataFrame({'intercept': np.ones(len(imgs))}))
    z = m.compute_contrast('intercept', output_type='z_score')
    z.set_data_dtype(np.float32)
    z.to_filename(str(out / f'contrast-{con}_stat-z.nii.gz'))
    print(con, len(imgs), f'max |z| {np.nanmax(np.abs(z.get_fdata())):.2f}', flush=True)
