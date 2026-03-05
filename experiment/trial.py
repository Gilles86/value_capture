from exptools2.core import Trial
from psychopy.visual import TextStim, ImageStim
from psychopy import visual
import numpy as np
from collections import deque
from psychopy import core
import os.path as op
from psychopy.core import getTime
from stimuli import GREY_RGB


class InstructionTrial(Trial):

    def __init__(
        self,
        session,
        trial_nr,
        txt,
        bottom_txt=None,
        image_path=None,
        keys=None,
        phase_durations=None,
        phase_names=None,
        **kwargs,
    ):
        self.keys = keys

        if phase_durations is None:
            phase_durations = [0.5, np.inf]

        if phase_names is None:
            phase_names = ['instruction'] * len(phase_durations)

        super().__init__(
            session,
            trial_nr,
            phase_durations=phase_durations,
            phase_names=phase_names,
            **kwargs,
        )

        txt_height = self.session.settings['various'].get('text_height')
        txt_width = self.session.settings['various'].get('text_width')
        txt_color = self.session.settings['various'].get('text_color')

        if image_path:
            self.text = TextStim(
                session.win,
                txt,
                pos=(-2.0, 0.0),
                height=txt_height,
                wrapWidth=txt_width,
                color=txt_color,
            )
        else:
            self.text = TextStim(
                session.win,
                txt,
                pos=(0.0, 0.0),
                height=txt_height,
                wrapWidth=txt_width,
                color=txt_color,
            )

        if bottom_txt is None:
            bottom_txt = 'Press any button to continue'

        self.text2 = TextStim(
            session.win,
            bottom_txt,
            pos=(0.0, -3.0),
            height=txt_height,
            wrapWidth=txt_width,
            color=txt_color,
        )

        self.image = None
        if image_path is not None and op.exists(image_path):
            self.image = ImageStim(
                session.win, image=image_path, pos=(3, 0), size=(5, 5), units='deg'
            )

    def get_events(self):
        events = Trial.get_events(self)

        if self.keys is None:
            if events:
                self.stop_phase()
        else:
            for key, t in events:
                if key in self.keys:
                    self.stop_phase()

    def draw(self):
        self.session.backgroundcircle.draw()
        if self.image:
            self.image.draw()
        self.text.draw()
        self.text2.draw()


class InstructionArrayTrial(Trial):
    """
    Instruction screen that renders the live stimulus array alongside text.

    The array is shown in a fixed demo configuration (target location, orientation,
    optional coloured distractor, optional dot).  A white highlight ring can be drawn
    around any item to direct attention.  For the value-legend slide, pass
    show_value_legend=True and no array is drawn — instead three coloured bars are
    shown with their point multipliers.

    Layout
    ------
    - Array: centred at (0, stimulus_shift) as during the task.
    - Text: left of the array, anchored at x = -9.5.  Safe for the Spinoza
      scanner screen (±10.3 deg) and the laptop debug screen (±15.8 deg).
    - "Press any button": bottom-centre.
    """

    def __init__(
        self,
        session,
        trial_nr,
        txt,
        target_loc=3,
        target_ori=0.0,
        dist_loc=None,
        dist_color_rgb=None,
        dot_locs=None,
        highlight_loc=None,
        show_value_legend=False,
        keys=None,
        **kwargs,
    ):
        super().__init__(
            session,
            trial_nr,
            phase_durations=[0.5, np.inf],
            phase_names=['instruction', 'instruction'],
            **kwargs,
        )

        th = session.settings['various'].get('text_height', 0.5)
        tc = session.settings['various'].get('text_color', 'white')
        self.keys = keys

        self.main_text = visual.TextStim(
            session.win,
            txt,
            pos=(-4.0, 1.5),
            height=th,
            wrapWidth=5.0,
            color=tc,
            alignText='left',
            anchorHoriz='center',
        )
        lowest_item_y = min(pos[1] for pos in session.target_stimuli.positions)
        continue_y = lowest_item_y - session.size_stimuli - 0.3
        if keys is not None and len(keys) == 1:
            continue_txt = f"Press '{keys[0]}' to continue"
        else:
            continue_txt = 'Press any button to continue'
        self.continue_text = visual.TextStim(
            session.win,
            continue_txt,
            pos=(0, continue_y),
            height=th,
            color=tc,
        )

        # --- array config ---
        dot_presence = [False] * 8
        for loc in (dot_locs or []):
            dot_presence[loc] = True

        self._dist_color = (
            dist_color_rgb
            if (dist_loc is not None and dist_color_rgb is not None)
            else GREY_RGB
        )
        self._dist_loc = dist_loc
        self._target_loc = target_loc
        self._target_ori = target_ori
        self._dot_presence = dot_presence

        # --- optional highlight ring ---
        self.highlight_stim = None
        if highlight_loc is not None:
            pos = session.target_stimuli.positions[highlight_loc]
            self.highlight_stim = visual.Circle(
                session.win,
                pos=pos,
                radius=session.size_stimuli * 0.9,
                fillColor=None,
                lineColor='white',
                lineWidth=2,
            )

        # --- value-legend bars (show_value_legend=True) ---
        self.show_value_legend = show_value_legend
        self.legend_bars = []
        self.legend_labels = []
        if show_value_legend:
            entries = [
                (2, '× 100  (high value)'),
                (1, '× 10   (medium value)'),
                (0, '× 1    (low value)'),
            ]
            for i, (rank, label_txt) in enumerate(entries):
                y = 2.5 - i * 2.5
                color = session.get_distractor_color(rank)
                self.legend_bars.append(visual.Rect(
                    session.win,
                    pos=(-1.5, y),
                    width=session.size_stimuli,
                    height=session.size_stimuli / 4,
                    fillColor=color,
                    lineColor=None,
                ))
                self.legend_labels.append(visual.TextStim(
                    session.win,
                    label_txt,
                    pos=(0, y),
                    height=th,
                    color=tc,
                    alignText='left',
                    anchorHoriz='left',
                ))

    def get_events(self):
        events = Trial.get_events(self)
        if self.keys is None:
            if events:
                self.stop_phase()
        else:
            for key, t in events:
                if key in self.keys:
                    self.stop_phase()

    def run(self):
        if not self.show_value_legend:
            self.session.target_stimuli.setup(
                self._dist_color,
                self._target_ori,
                self._dist_loc,
                self._target_loc,
                self._dot_presence,
            )
        super().run()

    def draw(self):
        self.session.backgroundcircle.draw()
        if not self.show_value_legend:
            self.session.target_stimuli.draw()
            self.session.fixation_dot.color = 'white'
            self.session.fixation_dot.draw()
            if self.highlight_stim:
                self.highlight_stim.draw()
        else:
            for bar, lbl in zip(self.legend_bars, self.legend_labels):
                bar.draw()
                lbl.draw()
        self.main_text.draw()
        self.continue_text.draw()


class SingletonTrial(Trial):
    """
    Main trial for the value_capture experiment.

    Phase structure:
      0 - trial_start : brief fixation before cue
      1 - pre_target  : fixation (brief cue period)
      2 - target      : search array + PRF bar (response collected here)
      3 - iti1        : fixation (random, first half of inter-trial interval)
      4 - feedback    : points text or blank fixation (1 s)
      5 - iti2        : fixation (random, second half of inter-trial interval)

    The sweeping PRF bar is drawn ONLY during phase 2 (target).
    """

    def __init__(
        self,
        session,
        trial_nr,
        iti1,
        iti2,
        distractor_location=None,
        target_location=None,
        value_rank=None,
        distractor_present=True,
        target_orientation=None,
        dot_presence=None,
        show_feedback=None,
        bar_position=0.0,
        bar_orientation='horizontal',
        **kwargs,
    ):
        trial_start_duration = session.settings['durations'].get(
            'trial_start', 0.5)
        pre_target_duration = session.settings['durations'].get(
            'trial_wait', 0.5)
        target_duration = session.settings['durations'].get('target', 1.5)
        feedback_duration = session.settings['durations'].get('feedback', 1.0)

        phase_durations = [
            trial_start_duration,
            pre_target_duration,
            target_duration,
            iti1,
            feedback_duration,
            iti2,
        ]
        phase_names = ['trial_start', 'pre_target',
                       'target', 'iti1', 'feedback', 'iti2']

        super().__init__(
            session,
            trial_nr,
            phase_durations=phase_durations,
            phase_names=phase_names,
            **kwargs,
        )

        # distractor_present=False → no coloured singleton on this trial
        self.parameters['distractor_present'] = distractor_present

        # Value rank: 0 = lowest reward, 2 = highest reward (3 levels)
        # None when distractor is absent.
        if value_rank is None and distractor_present:
            self.parameters['value_rank'] = np.random.choice([0, 1, 2])
        else:
            # may be None for absent trials
            self.parameters['value_rank'] = value_rank

        self.parameters['target_orientation'] = (
            np.random.choice(
                [0.0, 90.0]) if target_orientation is None else target_orientation
        )

        # distractor_location is None on absent trials (no coloured item)
        if not distractor_present:
            self.parameters['distractor_location'] = None
        elif distractor_location is None:
            self.parameters['distractor_location'] = np.random.choice([
                                                                      1, 3, 5, 7])
        else:
            self.parameters['distractor_location'] = distractor_location

        if dot_presence is None:
            # 2 dots among the 4 cardinal (odd) positions, 2 among the 4 diagonal (even)
            dot_presence = [False] * 8
            for i in np.random.choice([0, 2, 4, 6], 2, replace=False):
                dot_presence[i] = True
            for i in np.random.choice([1, 3, 5, 7], 2, replace=False):
                dot_presence[i] = True
        # Store as instance attribute for drawing; log as bitmask string (lists break pandas .loc)
        self._dot_presence = dot_presence
        self.parameters['dot_presence'] = ''.join(
            '1' if d else '0' for d in dot_presence)

        if target_location is None:
            # may be None → no exclusion
            exclude = self.parameters['distractor_location']
            self.parameters['target_location'] = np.random.choice(
                [i for i in [1, 3, 5, 7] if i != exclude]
            )
        else:
            self.parameters['target_location'] = target_location

        self.parameters['correct_response'] = self._dot_presence[
            self.parameters['target_location']
        ]

        # Whether to show feedback this trial (33% during scanning, 100% in practice)
        if show_feedback is None:
            feedback_p = session.settings['design'].get(
                'feedback_probability', 0.333)
            self.parameters['show_feedback'] = np.random.random() < feedback_p
        else:
            self.parameters['show_feedback'] = show_feedback

        self.parameters['correct'] = np.nan
        self.parameters['earned_points'] = 0
        # Store RGB tuple as instance attribute; log as string (tuples break pandas .loc)
        self._distractor_color_rgb = session.get_distractor_color(
            self.parameters['value_rank']  # None → returns GREY_RGB
        )
        self.parameters['distractor_color_rgb'] = str(
            self._distractor_color_rgb)

        self.parameters['bar_position'] = bar_position
        self.parameters['bar_orientation'] = bar_orientation

        self.responded = False
        self.stimulus_onset = None

    def draw(self):
        self.session.backgroundcircle.draw()
        self.session.fixation_dot.color = 'white'

        if self.phase == 2:  # target phase: show search array + PRF bar
            if self.stimulus_onset is None:
                self.stimulus_onset = self.session.clock.getTime()
            self.session.target_stimuli.draw()
            self.session.prf_bar.draw()  # PRF bar only during target!

        # feedback phase
        elif self.phase == 4 and self.parameters['show_feedback']:
            stim = self.session.points_stimulus
            stim.color = self._distractor_color_rgb
            stim.height = self.session.settings['experiment']['size_fixation'] * 1.5
            if np.isnan(self.parameters['correct']):
                stim.text = '+0\nToo late!'
            elif not self.parameters['correct']:
                stim.text = '+0\nIncorrect!'
            else:
                stim.text = f'+{self.parameters["earned_points"]}'
            stim.draw()
            return

        self.session.fixation_dot.draw()

    def run(self):
        self.setup_trial_stimuli()
        super().run()

    def setup_trial_stimuli(self):
        self.session.target_stimuli.setup(
            self._distractor_color_rgb,
            self.parameters['target_orientation'],
            self.parameters['distractor_location'],
            self.parameters['target_location'],
            self._dot_presence,
        )
        self.session.prf_bar.set_position(
            self.parameters['bar_position'],
            self.parameters['bar_orientation'],
        )

    def get_events(self):
        events = super().get_events()
        keys = self.session.settings['experiment']['keys']

        if self.phase == 2:
            for key, t in events:
                if (not self.responded) and (key in keys):
                    self.parameters['response'] = key
                    self.parameters['rt'] = t - self.stimulus_onset
                    self.parameters['correct'] = (
                        bool(keys.index(self.parameters['response']))
                        == self.parameters['correct_response']
                    )
                    self.responded = True

                    if self.parameters['correct']:
                        target_duration = self.session.settings['durations'].get('target', 1.5)
                        if self.parameters['distractor_present']:
                            multiplier = self.session.points_key[self.parameters['value_rank']]
                        else:
                            multiplier = 1  # no reward multiplier on absent trials
                        self.parameters['earned_points'] = max(
                            0, round((1 - self.parameters['rt'] / target_duration) * 10 * multiplier)
                        )
                        self.session.total_points += self.parameters['earned_points']


class SingletonTrial_training(SingletonTrial):
    """
    Practice version of SingletonTrial with eyetracker feedback beeps.
    Feedback is shown on 100% of trials.
    """

    def __init__(
        self,
        session,
        trial_nr,
        iti1,
        iti2,
        distractor_location=None,
        target_location=None,
        value_rank=None,
        distractor_present=True,
        target_orientation=None,
        dot_presence=None,
        **kwargs,
    ):
        # always True for training; drop if caller passed it
        kwargs.pop('show_feedback', None)
        super().__init__(
            session,
            trial_nr,
            iti1,
            iti2,
            distractor_location=distractor_location,
            target_location=target_location,
            value_rank=value_rank,
            distractor_present=distractor_present,
            target_orientation=target_orientation,
            dot_presence=dot_presence,
            show_feedback=True,  # always show feedback during practice
            **kwargs,
        )
        self.audio_played = False
        self.trial_frame_count = 0
        self.gaze_x = deque(maxlen=60)
        self.gaze_y = deque(maxlen=60)
        self.gaze_time = deque(maxlen=60)
        self.drift_x = deque(maxlen=60)
        self.drift_y = deque(maxlen=60)
        self.drift_times = deque(maxlen=60)
        self.drift_collecting = False

    def drift_correction_step(self):
        """Incremental gaze collection for online drift correction."""
        now = core.getTime()

        if not self.drift_collecting:
            self.drift_collecting = True
            self.drift_start_time = now
            self.drift_x.clear()
            self.drift_y.clear()
            self.drift_times.clear()
            return

        if now - self.drift_start_time < 0.2:
            el_smp = self.session.tracker.getNewestSample()
            if el_smp is None:
                return
            if el_smp.isLeftSample():
                gaze = el_smp.getLeftEye().getGaze()
            elif el_smp.isRightSample():
                gaze = el_smp.getRightEye().getGaze()
            else:
                return
            if gaze is not None:
                self.drift_x.append(gaze[0])
                self.drift_y.append(gaze[1])
                self.drift_times.append(now)
            return

        if len(self.drift_x) < 5:
            screen_center = np.array(self.session.win.size) / 2
            screen_center[1] -= self.session.pix_stimulus_shift
            self.drift = tuple(screen_center)
        else:
            self.drift = (np.mean(self.drift_x), np.mean(self.drift_y))

        self.drift_collecting = False

    def check_fixation_windowed(self):
        if self.trial_frame_count % 2 != 0:
            return True
        el_smp = self.session.tracker.getNewestSample()
        if el_smp is None:
            return True
        if el_smp.isLeftSample():
            sample = el_smp.getLeftEye().getGaze()
        elif el_smp.isRightSample():
            sample = el_smp.getRightEye().getGaze()
        else:
            return True

        now = core.getTime() * 1000
        self.gaze_x.append(sample[0])
        self.gaze_y.append(sample[1])
        self.gaze_time.append(now)

        times = np.array(self.gaze_time)
        d_times = times - now
        idx = np.where(d_times > -30)[0]

        if idx.size < 2:
            return True

        x_ = np.array(self.gaze_x)[idx] - self.drift[0]
        y_ = np.array(self.gaze_y)[idx] - self.drift[1]
        angles = np.hypot(x_, y_) / self.session.pix_per_deg

        if np.all(angles > self.session.settings['various']['gaze_threshold_deg']):
            return False
        return True

    def draw(self):
        self.trial_frame_count += 1
        self.session.backgroundcircle.draw()

        if self.phase == 1:
            if self.session.eyetracker_on:
                self.drift_correction_step()

        self.session.fixation_dot.color = 'white'

        if self.phase == 2:  # target + PRF bar
            if self.stimulus_onset is None:
                self.stimulus_onset = self.session.clock.getTime()
            self.session.target_stimuli.draw()
            self.session.prf_bar.draw()

            if (
                self.session.eyetracker_on
                and self.session.settings['various']['eyemovements_alert']
            ):
                fix_ok = self.check_fixation_windowed()
                if not fix_ok and not self.audio_played:
                    self.session.beep.play()
                    core.wait(0.03)
                    self.session.beep.stop()
                    self.audio_played = True
                    self.session.beep_count += 1

        elif self.phase == 4:  # feedback (always shown during practice)
            correct = self.parameters['correct']
            pts = self.parameters['earned_points']
            base_h = self.session.settings['experiment']['size_fixation'] * 1.5
            stim = self.session.points_stimulus
            stim.color = self._distractor_color_rgb
            stim.height = base_h
            if np.isnan(correct):
                stim.text = '+0\nToo late!'
            elif not correct:
                stim.text = '+0\nIncorrect!'
            else:
                stim.text = f'+{pts}'
            stim.draw()
            return  # skip fixation so it doesn't overlap the text

        self.session.fixation_dot.draw()


class BlankTrial(Trial):

    def __init__(self, session, trial_nr, duration=None, **kwargs):
        if duration is None:
            duration = session.settings['durations'].get('blank', 1)
        phase_durations = [duration]
        phase_names = ['blank']

        super().__init__(
            session,
            trial_nr,
            phase_durations=phase_durations,
            phase_names=phase_names,
            **kwargs,
        )

        self.parameters['correct'] = np.nan
        self.responded = False
        self.stimulus_onset = None

    def draw(self):
        self.session.backgroundcircle.draw()
        self.session.fixation_dot.color = 'white'
        self.session.fixation_dot.draw()

    def run(self):
        super().run()


class DummyWaiterTrial(Trial):

    def __init__(
        self,
        session,
        trial_nr,
        phase_durations=None,
        phase_names=None,
        draw_each_frame=False,
        **kwargs,
    ):
        super().__init__(
            session,
            trial_nr,
            phase_durations,
            phase_names,
            draw_each_frame=draw_each_frame,
            **kwargs,
        )
        th = session.settings['various'].get('text_height', 0.5)
        self._wait_text = visual.TextStim(
            session.win,
            'Waiting for scanner trigger...\n\n(press t to simulate)',
            pos=(0, 0),
            height=th,
            color='white',
        )

    def draw(self):
        self.session.backgroundcircle.draw()
        self.session.fixation_dot.color = 'white'
        self.session.fixation_dot.draw()
        if self.phase == 0:
            self._wait_text.draw()
        self.session.win.flip()

    def get_events(self):
        events = Trial.get_events(self)
        if events:
            for key, t in events:
                if key == self.session.mri_trigger:
                    print(f'Got trigger: {key}')
                    if self.phase == 0:
                        self.stop_phase()
                        self.session.experiment_start_time = getTime()


class WaitStartTriggerTrial(Trial):
    def __init__(
        self,
        session,
        trial_nr,
        phase_durations=[np.inf],
        phase_names=['waiting_start_trigger'],
        draw_each_frame=False,
    ):
        super().__init__(
            session,
            trial_nr,
            phase_durations,
            phase_names,
            draw_each_frame=draw_each_frame,
        )
        th = session.settings['various'].get('text_height', 0.5)
        self._wait_text = visual.TextStim(
            session.win,
            'Waiting for scanner trigger...\n\n(press t to simulate)',
            pos=(0, 0),
            height=th,
            color='white',
        )

    def draw(self):
        self.session.backgroundcircle.draw()
        self.session.fixation_dot.color = 'white'
        self.session.fixation_dot.draw()
        self._wait_text.draw()
        self.session.win.flip()

    def get_events(self):
        events = Trial.get_events(self)
        if events:
            for key, t in events:
                if key == self.session.mri_trigger:
                    self.stop_phase()
                    self.session.experiment_start_time = getTime()


class OutroTrial(Trial):

    def __init__(
        self,
        session,
        trial_nr,
        phase_durations,
        phase_names,
        draw_each_frame=False,
        **kwargs,
    ):
        super().__init__(
            session,
            trial_nr,
            phase_durations,
            phase_names,
            draw_each_frame=draw_each_frame,
            **kwargs,
        )

    def draw(self):
        self.session.backgroundcircle.draw()
        self.session.fixation_dot.color = 'white'
        self.session.fixation_dot.draw()
        self.session.win.flip()


class TotalPointsTrial(Trial):
    """Shows total points earned this run for 5 seconds."""

    def __init__(self, session, trial_nr, **kwargs):
        super().__init__(
            session,
            trial_nr,
            phase_durations=[3.0],
            phase_names=['total_points'],
            **kwargs,
        )
        th = session.settings['experiment']['size_fixation'] * 3.0
        self._text = visual.TextStim(
            session.win,
            text='',
            color='white',
            height=th,
            pos=(0, session.stimulus_shift),
            wrapWidth=20,
        )

    def draw(self):
        self.session.backgroundcircle.draw()
        self._text.text = (
            f'Total points this run:\n{self.session.total_points}'
        )
        self._text.draw()
        self.session.win.flip()
