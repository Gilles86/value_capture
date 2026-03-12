import os
import os.path as op
from importlib.resources import files
import yaml
import pandas as pd


# Maps numeric location codes to quadrant names
LOCATION_MAPPING = {
    1.0: 'upper_right',
    3.0: 'upper_left',
    5.0: 'lower_left',
    7.0: 'lower_right',
}

# Value color mappings per condition
# condition 0: rank 0=full_green, rank 1=mid_orange, rank 2=full_orange
# condition 1: rank 0=full_orange, rank 1=mid_orange, rank 2=full_green
VALUE_COLORS = {
    0: {0: '#00ab78', 1: '#999253', 2: '#d56f2c'},
    1: {0: '#d56f2c', 1: '#999253', 2: '#00ab78'},
}


def _load_subjects_yml():
    with files('value_capture').joinpath('data', 'subjects.yml').open() as f:
        return yaml.safe_load(f)


def get_all_subject_ids():
    """Return list of all subject IDs defined in subjects.yml."""
    return list(_load_subjects_yml().keys())


def get_all_behavioral_data(bids_folder='/data/ds-valuecapture', subjects=None):
    """Return behavioral data for all subjects concatenated into one DataFrame."""
    if subjects is None:
        subjects = get_all_subject_ids()
    return pd.concat(
        [Subject(s, bids_folder=bids_folder).get_behavioral_data() for s in subjects]
    ).sort_index()


class Subject:

    def __init__(self, subject_id, bids_folder='/data/ds-valuecapture'):
        if isinstance(subject_id, int):
            subject_id = f'{subject_id:02d}'
        self.subject_id = subject_id
        self.bids_folder = bids_folder

    def _behavior_dir(self, session):
        return op.join(self.bids_folder, 'sourcedata', 'behavior',
                       f'sub-{self.subject_id}', f'ses-{session}')

    def _events_path(self, session, run):
        return op.join(self._behavior_dir(session), f'run-{run}',
                       f'sub-{self.subject_id}_ses-{session}_task-val_cap_run-{run}_events.tsv')

    def get_sessions(self):
        """Return the sessions this subject has completed, from subjects.yml."""
        mapping = _load_subjects_yml()
        return sorted(mapping[self.subject_id])

    def get_runs(self, session):
        """Return sorted list of scanning run numbers (positive) for this session."""
        behavior_dir = self._behavior_dir(session)
        runs = []
        for entry in sorted(os.listdir(behavior_dir)):
            if entry.startswith('run-') and not entry.startswith('run--'):
                try:
                    run_nr = int(entry.split('-')[1])
                    if op.exists(self._events_path(session, run_nr)):
                        runs.append(run_nr)
                except ValueError:
                    pass
        return sorted(runs)

    def get_value_condition(self, session):
        """Return value condition (0 or 1) based on subject parity × session."""
        subject_nr = int(self.subject_id)
        if session == 1:
            return 0 if subject_nr % 2 == 0 else 1
        elif session == 2:
            return 1 if subject_nr % 2 == 0 else 0
        else:
            raise ValueError(f'Session must be 1 or 2, got {session}')

    def get_onsets(self, session, run):
        """Load raw events TSV with onset times relative to first scanner pulse."""
        df = pd.read_csv(self._events_path(session, run), sep='\t')
        pulses = df[df['event_type'] == 'pulse']
        if len(pulses) > 0:
            df['onset'] = df['onset'] - pulses['onset'].min()
        return df

    def get_behavioral_data(self, session=None, run=None, exclude_outlier_rts=True):
        """Return per-trial behavioral data as a DataFrame.

        Uses the ``feedback`` event row for each trial (one row per trial),
        which contains RT, accuracy, earned_points and all trial parameters.

        Index levels: subject, session, run, trial_nr
        Key columns: value_rank, distractor_present, target_location,
                     distractor_location, bar_position, bar_orientation,
                     rt, correct, earned_points, show_feedback,
                     value_condition, condition

        Parameters
        ----------
        exclude_outlier_rts : bool
            If True (default), RTs more than 2.5 SD from the mean are set to
            NaN. The number of excluded trials is printed.
        """
        if session is None:
            sessions = self.get_sessions()
            dfs = []
            for ses in sessions:
                try:
                    dfs.append(self.get_behavioral_data(ses, run, exclude_outlier_rts=False))
                except Exception as e:
                    print(f'sub-{self.subject_id} ses-{ses}: {e}')
            fb = pd.concat(dfs).sort_index()

        elif run is None:
            runs = self.get_runs(session)
            dfs = []
            for r in runs:
                try:
                    dfs.append(self.get_behavioral_data(session, r, exclude_outlier_rts=False))
                except Exception as e:
                    print(f'sub-{self.subject_id} ses-{session} run-{r}: {e}')
            fb = pd.concat(dfs).sort_index()

        else:
            df = pd.read_csv(self._events_path(session, run), sep='\t')

            # Feedback rows: one per trial, hold RT and accuracy
            fb = df[df['event_type'] == 'feedback'].copy()
            fb = fb.set_index('trial_nr')

            cols = ['onset', 'value_rank', 'distractor_present', 'target_location',
                    'distractor_location', 'bar_position', 'bar_orientation',
                    'rt', 'correct', 'earned_points', 'show_feedback']
            fb = fb[[c for c in cols if c in fb.columns]].copy()

            fb['value_condition'] = self.get_value_condition(session)

            # Condition label derived from distractor presence and value rank
            rank_labels = {0.0: 'Low Value', 1.0: 'Medium Value', 2.0: 'High Value'}
            fb['condition'] = fb.apply(
                lambda row: 'No Distractor' if not row['distractor_present']
                            else rank_labels.get(row['value_rank']),
                axis=1
            )
            condition_order = ['No Distractor', 'Low Value', 'Medium Value', 'High Value']
            fb['condition'] = pd.Categorical(fb['condition'], categories=condition_order, ordered=True)

            fb.index = pd.MultiIndex.from_arrays(
                [[self.subject_id] * len(fb),
                 [session] * len(fb),
                 [run] * len(fb),
                 fb.index],
                names=['subject', 'session', 'run', 'trial_nr']
            )

        if exclude_outlier_rts:
            rt = fb['rt']
            mean, sd = rt.mean(), rt.std()
            outliers = (rt - mean).abs() > 2.5 * sd
            n = outliers.sum()
            n_total = rt.notna().sum()
            if n > 0:
                print(f'sub-{self.subject_id}: {n}/{n_total} RT(s) excluded as outliers (> 2.5 SD from mean)')
                fb.loc[outliers, 'rt'] = float('nan')

        return fb
