#!/usr/bin/env python3
"""
Sample volumetric PRF parameter maps to the cortical surface.

Uses nilearn's vol_to_surf to project each parameter NIfTI (in T1w space)
onto the fsnative white/pial surfaces produced by fmriprep.  Output .func.gii
files are written alongside the volumetric maps in the same derivatives folder
and can then be transformed to fsaverage if needed.

Usage:
    python sample_prf_to_surface.py 01
    python sample_prf_to_surface.py 01 --prf-deriv prf_glmsingle_s6mm
    python sample_prf_to_surface.py 01 --to-fsaverage   # also write fsaverage .func.gii
"""

import argparse
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn import surface

from value_capture.utils.data import BIDS_FOLDER, Subject


def sample_to_surface(subject, prf_deriv='prf_glmsingle', sessions=None,
                      bids_folder=BIDS_FOLDER, to_fsaverage=False):
    sub = Subject(subject, bids_folder=bids_folder)
    surf_info = sub.get_surf_info()
    ses_label = sub._ses_label(sessions)

    func_dir = Path(bids_folder) / 'derivatives' / prf_deriv / f'sub-{sub.subject_id}' / ses_label / 'func'
    fn_vol = (f'sub-{sub.subject_id}_{ses_label}_task-valuecapture'
              f'_space-T1w_desc-prf{{param}}_pe.nii.gz')
    fn_surf = (f'sub-{sub.subject_id}_{ses_label}_task-valuecapture'
               f'_hemi-{{hemi}}_space-fsnative_desc-prf{{param}}_pe.func.gii')

    prf_imgs = sub.get_prf_parameters_volume(prf_deriv=prf_deriv, sessions=sessions)

    for param, vol_img in prf_imgs.items():
        for hemi in ['L', 'R']:
            data = surface.vol_to_surf(
                vol_img,
                surf_mesh=surf_info[hemi]['outer'],
                inner_mesh=surf_info[hemi]['inner'],
                interpolation='linear',
            ).astype(np.float32)
            data = np.nan_to_num(data)

            gifti_img = nib.gifti.GiftiImage(
                darrays=[nib.gifti.GiftiDataArray(data)]
            )
            out_fn = func_dir / fn_surf.format(hemi=hemi, param=param)
            nib.save(gifti_img, str(out_fn))
            print(f'  Wrote {out_fn.name}')

            if to_fsaverage:
                _transform_to_fsaverage(str(out_fn), hemi, sub, bids_folder)


def _transform_to_fsaverage(in_file, hemi, sub, bids_folder):
    """Use mri_surf2surf to resample from fsnative to fsaverage."""
    fs_hemi = 'lh' if hemi == 'L' else 'rh'
    subjects_dir = sub.freesurfer_subjects_dir
    source_subject = sub.freesurfer_subject_id
    out_file = in_file.replace('space-fsnative', 'space-fsaverage')
    cmd = [
        'mri_surf2surf',
        '--hemi', fs_hemi,
        '--srcsubject', source_subject,
        '--trgsubject', 'fsaverage',
        '--sval', in_file,
        '--tval', out_file,
        '--sd', subjects_dir,
    ]
    print(f'  Resampling to fsaverage: {fs_hemi}')
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample PRF parameter maps to the cortical surface.')
    parser.add_argument('subject', help='Subject ID (e.g. 01)')
    parser.add_argument('--prf-deriv', default='prf_glmsingle',
                        help='PRF derivatives folder name (default: prf_glmsingle)')
    parser.add_argument('--sessions', nargs='+', type=int, default=None,
                        help='Session(s) to use. Default: all (ses-all)')
    parser.add_argument('--bids-folder', default=BIDS_FOLDER)
    parser.add_argument('--to-fsaverage', action='store_true',
                        help='Also write fsaverage-resampled .func.gii files (requires FreeSurfer)')
    args = parser.parse_args()

    sample_to_surface(
        args.subject,
        prf_deriv=args.prf_deriv,
        sessions=args.sessions,
        bids_folder=args.bids_folder,
        to_fsaverage=args.to_fsaverage,
    )
