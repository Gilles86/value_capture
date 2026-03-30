#!/usr/bin/env python3
"""
One-time setup: import value_capture subjects into the pycortex database.

Usage:
    conda run -n pycortex2 python setup_pycortex.py
    conda run -n pycortex2 python setup_pycortex.py --subjects 01 02
"""

import argparse
import cortex

BIDS_FOLDER = '/data/ds-valuecapture'
FS_SUBJECTS_DIR = f'{BIDS_FOLDER}/derivatives/fmriprep/sourcedata/freesurfer'


def import_subject(subject_id, bids_folder=BIDS_FOLDER, force=False):
    sid = f'{int(subject_id):02d}'
    fs_subject = f'sub-{sid}_ses-1'
    cx_subject = f'valuecapture.sub-{sid}'
    fs_dir = f'{bids_folder}/derivatives/fmriprep/sourcedata/freesurfer'

    if cx_subject in cortex.db.subjects and not force:
        print(f'  {cx_subject} already in pycortex database, skipping (use --force to reimport)')
        return

    print(f'  Importing {fs_subject} → {cx_subject} ...')
    cortex.freesurfer.import_subj(
        fs_subject,
        pycortex_subject=cx_subject,
        freesurfer_subject_dir=fs_dir,
    )
    print(f'  Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Import subjects into pycortex database.')
    parser.add_argument('--subjects', nargs='+', default=['01', '02', '03'])
    parser.add_argument('--bids-folder', default=BIDS_FOLDER)
    parser.add_argument('--force', action='store_true', help='Reimport even if already present')
    args = parser.parse_args()

    for sid in args.subjects:
        import_subject(sid, bids_folder=args.bids_folder, force=args.force)
