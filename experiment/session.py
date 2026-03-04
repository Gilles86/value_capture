import psychopy
import pygame.mixer as _pgmixer

from psychopy import visual, core, monitors
from trial import (
    InstructionTrial,
    InstructionArrayTrial,
    SingletonTrial,
    SingletonTrial_training,
    BlankTrial,
    DummyWaiterTrial,
    WaitStartTriggerTrial,
    OutroTrial,
    TotalPointsTrial,
)
from pathlib import Path
import yaml
import os.path as op
import numpy as np
from stimuli import (
    PRFBarStimulus,
    FixationStimulus,
    TargetStimulusArray,
    BackgroundCircle,
    VALUE_COLORS_RGB,
    GREY_RGB,
)
from exptools2.core import Session, PylinkEyetrackerSession


# Points multiplier for each value rank (0 = lowest reward, 2 = highest reward)
POINTS_KEY = [1, 10, 100]


class ValueCaptureSession(PylinkEyetrackerSession):

    def __init__(
        self,
        output_str,
        subject=None,
        session=None,
        output_dir=None,
        settings_file=None,
        run=None,
        eyetracker_on=False,
        calibrate_eyetracker=False,
    ):
        super().__init__(
            output_str,
            output_dir=output_dir,
            settings_file=settings_file,
            eyetracker_on=eyetracker_on,
        )
        self.width_deg = 2 * np.degrees(
            np.arctan(self.monitor.getWidth() / self.monitor.getDistance())
        )
        self.pix_per_deg = self.win.size[0] / self.width_deg

        self.mri_trigger = 't'
        self.show_eyetracker_calibration = calibrate_eyetracker
        self.stimulus_shift = self.settings['experiment']['stimulus_shift']

        self.instructions = yaml.safe_load(
            (Path(__file__).parent / 'instructions.yml').read_text()
        )

        self.settings['subject'] = subject
        self.settings['session'] = session
        self.settings['run'] = run

        self.eccentricity_stimuli = self.settings['experiment'].get('eccentricity_stimulus', 5)
        self.size_stimuli = self.settings['experiment'].get('size_stimuli', 1)
        self.radius_bar_aperture = self.eccentricity_stimuli - self.size_stimuli / 1.8

        # Counterbalance value-color mapping by subject × session:
        #   value_condition=0: rank 0 = full_green (lowest), rank 2 = full_orange (highest)
        #   value_condition=1: rank 0 = full_orange (lowest), rank 2 = full_green (highest)
        # Odd subjects:  session 1 → green=high (cond 1), session 2 → orange=high (cond 0)
        # Even subjects: session 1 → orange=high (cond 0), session 2 → green=high (cond 1)
        if session == 1:
            self.value_condition = subject % 2
        elif session == 2:
            self.value_condition = 1 - subject % 2
        else:
            raise ValueError(f'Session must be 1 or 2 (got {session}).')
        self.points_key = POINTS_KEY
        self.total_points = 0

        self.fixation_dot = FixationStimulus(
            self.win,
            size=self.settings['experiment']['size_fixation'],
            position=(0, self.stimulus_shift),
        )

        # PRF bar: discrete random positions, one per trial
        fov_size = self.radius_bar_aperture * 2
        n_bar_pos = self.settings['design'].get('n_bar_positions', 10)
        bar_width = fov_size / n_bar_pos   # tiles the FOV perfectly
        self.bar_positions = np.linspace(
            -fov_size / 2 + bar_width / 2,
             fov_size / 2 - bar_width / 2,
            n_bar_pos,
        )
        self.n_bar_positions = n_bar_pos

        self.prf_bar = PRFBarStimulus(
            self.win,
            session=self,
            fov_size=fov_size,
            bar_width=bar_width,
        )
        self.backgroundcircle = BackgroundCircle(
            self.win,
            session=self,
            fov_size=fov_size,
        )
        self.target_stimuli = TargetStimulusArray(
            self.win,
            eccentricity=self.eccentricity_stimuli,
            stimulus_size=self.size_stimuli,
            stimulus_shift=self.stimulus_shift,
        )

        self.points_stimulus = visual.TextStim(
            self.win,
            text='+0',
            color='white',
            height=self.settings['experiment']['size_fixation'] * 3.0,
            pos=(0, self.stimulus_shift),
        )

        self.rt_clock = core.Clock()

        if not _pgmixer.get_init():
            _pgmixer.init()
        soundfile = str(Path(__file__).parent / 'beep.wav')
        self.beep = _pgmixer.Sound(soundfile)
        self.beep.play()
        core.wait(0.5)
        self.beep.play()
        core.wait(0.02)
        self.beep_count = 0

    def get_distractor_color(self, value_rank):
        """
        Resolve distractor RGB color from value rank and session condition.

        value_rank: 0 (lowest reward), 1 (medium), 2 (highest reward), or None (absent)
        value_condition=0: rank 0=full_green, rank 1=mid_orange, rank 2=full_orange
        value_condition=1: rank 0=full_orange, rank 1=mid_orange, rank 2=full_green

        Returns GREY_RGB for absent trials (value_rank=None).
        """
        if value_rank is None:
            return GREY_RGB
        if self.value_condition == 0:
            return VALUE_COLORS_RGB[value_rank]
        else:
            return VALUE_COLORS_RGB[2 - value_rank]

    def get_bar_schedule(self, n_trials):
        """
        Return a list of (orientation, position) tuples for n_trials.

        Each orientation ('horizontal' / 'vertical') × position combination
        is visited an equal number of times.  The schedule is shuffled in
        blocks so every n_bar_positions × 2 trials contain each position exactly once.

        n_trials must be divisible by 2 * n_bar_positions.
        """
        n = self.n_bar_positions
        assert n_trials % (2 * n) == 0, (
            f'n_trials ({n_trials}) must be divisible by 2 × n_bar_positions ({2*n})'
        )
        one_cycle = (
            [('horizontal', p) for p in self.bar_positions] +
            [('vertical',   p) for p in self.bar_positions]
        )
        schedule = []
        for _ in range(n_trials // (2 * n)):
            block = one_cycle.copy()
            np.random.shuffle(block)
            schedule.extend(block)
        return schedule

    def run(self):
        if self.eyetracker_on and self.show_eyetracker_calibration:
            self.calibrate_eyetracker()

        self.start_experiment()

        if self.eyetracker_on:
            self.start_recording_eyetracker()

        for trial in self.trials:
            trial.run()

        self.close()

    def create_trials(self, include_instructions=True):
        """Create trials for the value_capture experiment."""

        def resolve_image_path(path):
            if path is None:
                return None
            base_dir = op.dirname(op.abspath(__file__))
            abs_path = op.join(base_dir, path)
            return abs_path if op.exists(abs_path) else None

        if include_instructions:
            k_dot   = self.settings['experiment']['keys'][1]  # dot key
            k_nodot = self.settings['experiment']['keys'][0]  # no-dot key
            high_color = self.get_distractor_color(2)

            self.trials = [
                # 1 — introduce the array
                InstructionArrayTrial(
                    self, 0,
                    txt=(
                        'Find the bar pointing\n'
                        'in a UNIQUE direction.\n\n'
                        'All bars are grey.\n'
                        'The target has a\n'
                        'different orientation\n'
                        'from all others.'
                    ),
                    target_loc=3, target_ori=0.0,
                ),
                # 2 — dot present
                InstructionArrayTrial(
                    self, 1,
                    txt=(
                        'Report whether the\n'
                        'unique bar has a small\n'
                        'dot inside it.\n\n'
                        f'Dot present:\n'
                        f"press '{k_dot}'"
                    ),
                    target_loc=3, target_ori=0.0,
                    dot_locs=[3],
                    highlight_loc=3,
                ),
                # 3 — no dot
                InstructionArrayTrial(
                    self, 2,
                    txt=(
                        'If there is NO dot\n'
                        'inside the unique bar:\n\n'
                        f"press '{k_nodot}'"
                    ),
                    target_loc=3, target_ori=0.0,
                    dot_locs=[],
                    highlight_loc=3,
                ),
                # 4 — coloured distractor
                InstructionArrayTrial(
                    self, 3,
                    txt=(
                        'On most trials one bar\n'
                        'has a distinctive\n'
                        'colour.\n\n'
                        'This DISTRACTOR\n'
                        'indicates how valuable\n'
                        'the trial is.\n'
                        'Try to ignore it!'
                    ),
                    target_loc=3, target_ori=0.0,
                    dist_loc=7, dist_color_rgb=high_color,
                    highlight_loc=7,
                ),
                # 5 — value-colour legend
                InstructionArrayTrial(
                    self, 4,
                    txt=(
                        'The distractor colour\n'
                        'shows the reward\n'
                        'multiplier.\n\n'
                        'Respond correctly\n'
                        'and quickly to\n'
                        'maximise points!'
                    ),
                    show_value_legend=True,
                ),
                # 6 — ready
                InstructionArrayTrial(
                    self, 5,
                    txt=(
                        'Ready!\n\n'
                        'Find the grey bar\n'
                        'and report dot or not.\n\n'
                        f"'{k_dot}' = dot\n"
                        f"'{k_nodot}' = no dot\n\n"
                        'Press any button\n'
                        'to start.'
                    ),
                    target_loc=3, target_ori=0.0,
                ),
            ]
        else:
            self.trials = []

        n_trials = self.settings['design']['n_trials']
        possible_iti1 = self.settings['durations']['iti1']
        possible_iti2 = self.settings['durations']['iti2']

        assert n_trials % len(possible_iti1) == 0, 'n_trials must be divisible by len(iti1)'
        assert n_trials % len(possible_iti2) == 0, 'n_trials must be divisible by len(iti2)'

        # Design: 4 equal conditions (25% each), mirroring the online osexp's equal-group structure
        #   - 3 value ranks × n_per_condition distractor-present trials
        #   - n_per_condition distractor-absent trials
        #   - Total: n_trials (must be divisible by 4 and by 2×n_bar_positions)
        assert n_trials % 4 == 0, 'n_trials must be divisible by 4 (3 value ranks + absent, 25% each)'

        n_per_condition = n_trials // 4   # e.g. 15 for n_trials=60

        # Distractor-present: n_per_condition per rank.
        # Balance distractor locations across the 4 positions as evenly as possible.
        distractor_trials = []
        for value_rank in [0, 1, 2]:
            base_reps = n_per_condition // 4
            extra     = n_per_condition  % 4
            d_locs = np.array([1, 3, 5, 7] * base_reps + [1, 3, 5, 7][:extra])
            np.random.shuffle(d_locs)
            for d_loc in d_locs:
                t_loc = np.random.choice([x for x in [1, 3, 5, 7] if x != d_loc])
                distractor_trials.append((int(t_loc), int(d_loc), value_rank, True))

        # Absent: n_per_condition trials, target locations balanced across 4 positions.
        base_reps = n_per_condition // 4
        extra     = n_per_condition  % 4
        t_locs_absent = [1, 3, 5, 7] * base_reps + [1, 3, 5, 7][:extra]
        absent_trials = [(t, None, None, False) for t in t_locs_absent]

        all_trials = distractor_trials + absent_trials
        np.random.shuffle(all_trials)

        # ITIs
        iti1s = np.tile(possible_iti1, n_trials // len(possible_iti1))
        iti2s = np.tile(possible_iti2, n_trials // len(possible_iti2))
        np.random.shuffle(iti1s)
        np.random.shuffle(iti2s)

        # Feedback: exact count derived from feedback_probability, then shuffled
        feedback_p = self.settings['design'].get('feedback_probability', 0.333)
        n_feedback = round(n_trials * feedback_p)
        show_feedbacks = np.array([True] * n_feedback + [False] * (n_trials - n_feedback))
        np.random.shuffle(show_feedbacks)

        # PRF bar schedule: balanced across all positions, independent of task
        bar_schedule = self.get_bar_schedule(n_trials)

        is_training = self.settings['run'] < 0
        TrialClass = SingletonTrial_training if is_training else SingletonTrial

        rest1 = n_trials // 3
        rest2 = (2 * n_trials) // 3

        if is_training:
            # Practice: wait for experimenter key press before starting
            self.trials.append(
                InstructionTrial(
                    session=self,
                    trial_nr=0,
                    txt='Ready to start!\n\nPress any button when you are ready.',
                    bottom_txt='',
                )
            )
            for ix, ((t_loc, d_loc, value_rank, dist_present), (bar_ori, bar_pos)) in enumerate(zip(all_trials, bar_schedule)):
                if ix in (rest1, rest2):
                    self.trials.append(BlankTrial(session=self, trial_nr=ix, duration=10))
                self.trials.append(
                    TrialClass(
                        self,
                        ix + 1,
                        iti1=iti1s[ix],
                        iti2=iti2s[ix],
                        distractor_location=d_loc,
                        target_location=t_loc,
                        value_rank=value_rank,
                        distractor_present=dist_present,
                        bar_position=bar_pos,
                        bar_orientation=bar_ori,
                        show_feedback=bool(show_feedbacks[ix]),
                    )
                )

            self.trials.append(
                OutroTrial(
                    session=self,
                    trial_nr=ix + 2,
                    phase_durations=[self.settings['durations']['blank'], 0.10],
                    phase_names=['outro_dummy_scan', 'end_exp'],
                    draw_each_frame=False,
                )
            )
            self.trials.append(TotalPointsTrial(self, ix + 3))

            run = self.settings['run']
            n_runs = self.settings['design'].get('n_runs', 6)
            entry = self.instructions['fin'] if run == n_runs else self.instructions['break']
            text = entry.format(run=run)
            self.trials.append(InstructionTrial(self, ix + 4, txt=text, image_path=None))

        else:
            # Scanning: wait for MRI trigger
            dummy_trial = DummyWaiterTrial(
                session=self,
                trial_nr=0,
                phase_durations=[np.inf, self.settings['durations']['blank']],
                phase_names=['start_exp', 'intro_dummy_scan'],
                draw_each_frame=False,
            )
            start_trial = WaitStartTriggerTrial(
                session=self,
                trial_nr=0,
                phase_durations=[np.inf],
                draw_each_frame=False,
            )
            self.trials.append(dummy_trial)
            self.trials.append(start_trial)

            for ix, ((t_loc, d_loc, value_rank, dist_present), (bar_ori, bar_pos)) in enumerate(zip(all_trials, bar_schedule)):
                if ix in (rest1, rest2):
                    self.trials.append(BlankTrial(session=self, trial_nr=ix, duration=10))
                self.trials.append(
                    SingletonTrial(
                        self,
                        ix + 1,
                        iti1=iti1s[ix],
                        iti2=iti2s[ix],
                        distractor_location=d_loc,
                        target_location=t_loc,
                        value_rank=value_rank,
                        distractor_present=dist_present,
                        bar_position=bar_pos,
                        bar_orientation=bar_ori,
                        show_feedback=bool(show_feedbacks[ix]),
                    )
                )

            self.trials.append(
                OutroTrial(
                    session=self,
                    trial_nr=ix + 2,
                    phase_durations=[self.settings['durations']['blank'], 0.10],
                    phase_names=['outro_dummy_scan', 'end_exp'],
                    draw_each_frame=False,
                )
            )
            self.trials.append(TotalPointsTrial(self, ix + 3))
