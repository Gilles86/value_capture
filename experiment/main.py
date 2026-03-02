from psychopy.core import Clock
from psychopy import visual
from exptools2.core import Trial
from psychopy import event
from psychopy import logging
import numpy as np
import os
from utils import get_output_dir_str, get_settings
from session import ValueCaptureSession
import argparse


def main(subject, session, run, settings='default', use_eyetracker=False, force_overwrite=False):
    """
    Start the value_capture experiment.

    subject : int  – participant number
    session : int  – 1 = practice (eye-tracker lab), 2+ = fMRI scanning
    run     : int  – run number within session
    settings: str  – name of settings file in settings/ (without .yml)
    """
    eyetracker_on = False
    calibrate_eyetracker = False

    if session == 1:
        eyetracker_on = use_eyetracker
        calibrate_eyetracker = use_eyetracker

    output_dir, output_str = get_output_dir_str(subject, session, 'val_cap', run)

    if os.path.exists(output_dir) and not force_overwrite:
        raise ValueError(
            f'\n'
            f'================================\n'
            f'=========WATCH OUT!!!===========\n'
            f'Output directory already exists.\n'
            f'====Please check your input.====\n'
            f'================================\n'
        )

    settings_fn, _ = get_settings(settings)
    include_instructions = (run == 1)

    run_session = ValueCaptureSession(
        output_str=output_str,
        subject=subject,
        session=session,
        output_dir=output_dir,
        settings_file=settings_fn,
        run=run,
        eyetracker_on=eyetracker_on,
        calibrate_eyetracker=calibrate_eyetracker,
    )

    run_session.create_trials(include_instructions=include_instructions)
    run_session.run()

    print(f'Total points this run: {run_session.total_points}')
    if eyetracker_on:
        print(f'Eye-movement alerts: {run_session.beep_count} trials')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('subject', type=int, help='Subject number')
    argparser.add_argument('session', type=int, help='Session (1=practice, 2+=scanning)')
    argparser.add_argument('run', type=int, help='Run number')
    argparser.add_argument('--settings', type=str, default='default', help='Settings label')
    argparser.add_argument('--use_eyetracker', action='store_true', help='Enable eyetracker')
    argparser.add_argument('--force_overwrite', action='store_true', help='Overwrite existing output directory (useful for testing)')

    args = argparser.parse_args()
    main(args.subject, args.session, args.run, args.settings, args.use_eyetracker, args.force_overwrite)
