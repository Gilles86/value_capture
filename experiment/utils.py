import os.path as op
import yaml
from psychopy import logging


def get_settings(settings):
    settings_fn = op.join(op.dirname(__file__), 'settings', f'{settings}.yml')
    print(settings_fn)

    with open(settings_fn, 'r') as f:
        settings = yaml.safe_load(f)

    use_eyetracker = 'eyetracker' in settings.keys()

    return settings_fn, use_eyetracker


def get_output_dir_str(subject, session, task, run):
    if not isinstance(subject, int):
        raise ValueError('Subject must be an integer')
    else:
        subject = str(subject).zfill(2)

    if not isinstance(session, int):
        raise ValueError('Session must be an integer')
    else:
        session = str(session).zfill(1)

    if not isinstance(run, int):
        raise ValueError('Run must be an integer')
    else:
        run = str(run).zfill(1)

    output_dir = op.join(op.dirname(__file__), 'logs', f'sub-{subject}')
    logging.warn(f'Writing results to {output_dir}')

    if session:
        output_dir = op.join(output_dir, f'ses-{session}')
        output_str = f'sub-{subject}_ses-{session}_task-{task}'
    else:
        output_str = f'sub-{subject}_task-{task}'

    if run:
        output_dir = op.join(output_dir, f'run-{run}')
        output_str += f'_run-{run}'

    return output_dir, output_str
