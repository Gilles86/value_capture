#!/usr/bin/env python3
"""
Register retinotopic areas (V1/V2/V3/...) using surface PRF estimates + neuropythy.

Loads fsnative surface PRF parameters, writes them as MGZ files into the
FreeSurfer subject's surf/ directory, then calls:
    python -m neuropythy register_retinotopy <subject>

Neuropythy writes ``inferred_varea.mgz``, ``inferred_eccen.mgz``, etc. into
the subject's mri/ directory.  These can then be loaded via
Subject.get_retinotopic_atlas().

Usage:
    python register_retinotopy.py 01
    python register_retinotopy.py 01 --prf-deriv prf_glmsingle_s6mm
"""

import argparse
import os
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np

from value_capture.utils.data import BIDS_FOLDER, Subject


def register_retinotopy(subject, prf_deriv='prf_glmsingle', sessions=None,
                        bids_folder=BIDS_FOLDER):
    sub = Subject(subject, bids_folder=bids_folder)
    pars = sub.get_prf_parameters_surface(prf_deriv=prf_deriv, sessions=sessions,
                                          space='fsnative')

    fs_subject = sub.freesurfer_subject_id
    surf_dir = Path(sub.freesurfer_subjects_dir) / fs_subject / 'surf'
    surf_dir.mkdir(parents=True, exist_ok=True)

    # neuropythy polar angle convention:
    #   LH: 0° = upper vertical, 90° = right, increasing clockwise when viewed
    #       from lateral; computed as  -arctan2(y, x) + 90, clipped to [0, 180]
    #   RH: mirror x-axis  →  -arctan2(y, -x) + 90, clipped to [0, 180]
    y_vals = pars['y']
    x_vals = pars['x']
    lh_angle = np.clip(-np.degrees(np.arctan2(y_vals.loc['L'], x_vals.loc['L'])) + 90, 0, 180)
    rh_angle = np.clip(-np.degrees(np.arctan2(y_vals.loc['R'], -x_vals.loc['R'])) + 90, 0, 180)

    param_map = {
        'prf_eccen':  {
            'L': np.sqrt(x_vals.loc['L'] ** 2 + y_vals.loc['L'] ** 2),
            'R': np.sqrt(x_vals.loc['R'] ** 2 + y_vals.loc['R'] ** 2),
        },
        'prf_angle': {'L': lh_angle, 'R': rh_angle},
        'prf_vexpl': {'L': pars['R2'].loc['L'], 'R': pars['R2'].loc['R']},
        'prf_radius': {'L': pars['sd'].loc['L'],  'R': pars['sd'].loc['R']},
    }

    hemi_files = {'lh': {}, 'rh': {}}
    for suffix, hemi_data in param_map.items():
        for hemi, fs_hemi in [('L', 'lh'), ('R', 'rh')]:
            data = np.asarray(hemi_data[hemi], dtype=np.float32)
            data = np.nan_to_num(data)
            arr = data[np.newaxis, np.newaxis, :]  # MGH shape: 1 × 1 × n_vertices
            img = nib.MGHImage(arr, affine=np.eye(4))
            out_fn = surf_dir / f'{fs_hemi}.{suffix}.mgz'
            nib.save(img, str(out_fn))
            hemi_files[fs_hemi][suffix] = str(out_fn)
            print(f'  Wrote {out_fn.name}')

    env = os.environ.copy()
    env['SUBJECTS_DIR'] = sub.freesurfer_subjects_dir

    # neuropythy 0.12.x uses np.complex which was removed in NumPy ≥1.24.
    # Patch it before importing by running via -c instead of -m.
    np_patch = (
        'import numpy as _np; _np.complex = complex; '
        'from neuropythy.__main__ import main; import sys; sys.exit(main(sys.argv[1:]))'
    )
    cmd = [
        'conda', 'run', '-n', 'abstract_values',
        'python', '-W', 'ignore', '-c', np_patch,
        'register_retinotopy', fs_subject,
        '--lh-eccen',  hemi_files['lh']['prf_eccen'],
        '--lh-angle',  hemi_files['lh']['prf_angle'],
        '--lh-weight', hemi_files['lh']['prf_vexpl'],
        '--lh-radius', hemi_files['lh']['prf_radius'],
        '--rh-eccen',  hemi_files['rh']['prf_eccen'],
        '--rh-angle',  hemi_files['rh']['prf_angle'],
        '--rh-weight', hemi_files['rh']['prf_vexpl'],
        '--rh-radius', hemi_files['rh']['prf_radius'],
        '--verbose',
    ]
    print('\nRunning neuropythy register_retinotopy...')
    print(' '.join(cmd))
    subprocess.run(cmd, env=env, check=True)
    print('\nDone. Benson atlas written to:')
    print(f'  {Path(sub.freesurfer_subjects_dir) / fs_subject / "mri"}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Register V1/V2/V3 ROIs via neuropythy using surface PRF estimates.')
    parser.add_argument('subject', help='Subject ID (e.g. 01)')
    parser.add_argument('--prf-deriv', default='prf_glmsingle',
                        help='PRF derivatives folder (default: prf_glmsingle)')
    parser.add_argument('--sessions', nargs='+', type=int, default=None)
    parser.add_argument('--bids-folder', default=BIDS_FOLDER)
    args = parser.parse_args()

    register_retinotopy(
        args.subject,
        prf_deriv=args.prf_deriv,
        sessions=args.sessions,
        bids_folder=args.bids_folder,
    )
