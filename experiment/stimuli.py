from psychopy import visual, core, event
from psychopy import plugins
import numpy as np


def hex_to_psychopy_rgb(hex_color):
    """Convert hex color string to PsychoPy RGB tuple (range -1 to 1)."""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (r / 127.5 - 1, g / 127.5 - 1, b / 127.5 - 1)


# Distractor value colors (3 levels)
# Rank 0 = lowest reward, rank 2 = highest reward.
# Condition 0: full_green=lowest, mid_orange=medium, full_orange=highest
# Condition 1: full_orange=lowest, mid_orange=medium, full_green=highest
# The rank-to-color mapping is resolved in session based on value_condition.
FULL_GREEN  = '#00ab78'
MID_ORANGE  = '#999253'   # CIELAB midpoint between green & orange (ΔE≈46.5 to each endpoint; was #b58800 ΔE=75/31)
FULL_ORANGE = '#d56f2c'
GREY        = '#8f8f8f'   # color of target and all non-distractor items

# index 0 = full_green (one end), index 1 = mid_orange, index 2 = full_orange (other end)
VALUE_COLORS_HEX = [FULL_GREEN, MID_ORANGE, FULL_ORANGE]
VALUE_COLORS_RGB = [hex_to_psychopy_rgb(c) for c in VALUE_COLORS_HEX]
GREY_RGB = hex_to_psychopy_rgb(GREY)


class TargetStimulus(object):

    def __init__(self, win, target, pos, size, color, ori):
        self.win = win
        self.target = target
        self._pos = pos
        self._size = size
        self._color = color
        self._ori = ori

        self.rectangle = visual.Rect(
            win=win,
            pos=self._pos,
            width=self._size,
            height=self._size / 4.0,
            fillColor=self._color,
            lineColor=None,
            ori=self._ori,
        )
        self.dot = visual.Circle(
            win=win,
            pos=self._pos,
            radius=self._size / 9,   # diameter = size/4.5 ≈ 89% of bar height (size/4); fits inside
            fillColor=win.color,     # background colour → looks like a hole punched through the bar
            lineColor=None,
        )

    def update(self, target=None, color=None, ori=None, pos=None, size=None):
        if target is not None:
            self.target = target
        if color is not None:
            self.color = color
        if ori is not None:
            self.ori = ori
        if pos is not None:
            self.pos = pos
        if size is not None:
            self.size = size

    def draw(self):
        self.rectangle.draw()
        if self.target:
            self.dot.draw()

    @property
    def ori(self):
        return self._ori

    @ori.setter
    def ori(self, value):
        self._ori = value
        self.rectangle.ori = value

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        self._color = value
        self.rectangle.fillColor = value

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
        self._pos = value
        self.rectangle.pos = value
        self.dot.pos = value

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        self._size = value
        self.rectangle.width = value
        self.rectangle.height = value / 4
        self.dot.radius = value / 9


class TargetStimulusArray(object):

    n_objects = 8

    def __init__(self, win, eccentricity=9, stimulus_size=1, stimulus_shift=0):
        self.win = win
        self.eccentricity = eccentricity
        self.stimulus_size = stimulus_size
        self.stimuli = []
        self.positions = self.get_positions(eccentricity, stimulus_shift)

        for ix in range(self.n_objects):
            self.stimuli.append(
                TargetStimulus(
                    win,
                    target=False,
                    pos=self.positions[ix],
                    size=stimulus_size,
                    color=GREY_RGB,
                    ori=np.pi / 2.0,
                )
            )

    def setup(
        self,
        distractor_color,
        target_orientation,
        distractor_location,
        target_location,
        dot_presence,
    ):
        """
        Configure stimuli for a trial.

        distractor_color: RGB tuple (already resolved from value_rank + condition)
        target_orientation: 0.0 or 90.0 (unique target orientation)
        distractor_location: index 0-7 of the distractor item
        target_location: index 0-7 of the target item
        dot_presence: list of 8 bools, True means this location has a dot
        """
        other_orientation = 90.0 if target_orientation == 0.0 else 0.0

        for ix, stimulus in enumerate(self.stimuli):
            if ix == distractor_location:
                stimulus.update(
                    target=dot_presence[ix],
                    color=distractor_color,
                    ori=other_orientation,
                )
            elif ix == target_location:
                stimulus.update(
                    target=dot_presence[ix],
                    color=GREY_RGB,
                    ori=target_orientation,
                )
            else:
                stimulus.update(
                    target=dot_presence[ix],
                    color=GREY_RGB,
                    ori=other_orientation,
                )

    def get_positions(self, eccentricity, stimulus_shift):
        positions = []
        for i in range(self.n_objects):
            angle = i * 360 / self.n_objects
            x = eccentricity * np.cos(np.radians(angle))
            y = eccentricity * np.sin(np.radians(angle)) + stimulus_shift
            positions.append((x, y))
        return positions

    def draw(self):
        for stimulus in self.stimuli:
            stimulus.draw()


class FixationStimulus:
    def __init__(
        self,
        win,
        position=(0, 0),
        size=0.5,
        color='grey',
        cross_color='black',
        cross_thickness=2,
    ):
        self.win = win
        self.position = position
        self.size = size
        self._color = color

        self.dot = visual.Circle(
            win=win,
            pos=self.position,
            radius=self.size / 2,
            fillColor=self.color,
            lineColor=None,
            edges=128,
        )
        self.minidot = visual.Circle(
            win=win,
            pos=self.position,
            radius=self.size / 4,
            fillColor=self.color,
            lineColor=None,
            edges=128,
        )
        cross_length = self.size
        self.h_line = visual.Line(
            win=win,
            start=(position[0] - cross_length / 2, position[1]),
            end=(position[0] + cross_length / 2, position[1]),
            lineColor=cross_color,
            lineWidth=cross_thickness,
        )
        self.v_line = visual.Line(
            win=win,
            start=(position[0], position[1] - cross_length / 2),
            end=(position[0], position[1] + cross_length / 2),
            lineColor=cross_color,
            lineWidth=cross_thickness,
        )

    def draw(self, cross=True):
        self.dot.draw()
        if cross:
            self.h_line.draw()
            self.v_line.draw()
        self.minidot.draw()

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        self._color = value
        self.dot.color = value


try:
    plugins.loadPlugin('psychopy_visionscience')
except Exception:
    pass


class PRFBarStimulus:
    """
    Flickering checkerboard bar for event-related PRF mapping.

    Unlike a continuous sweep, this bar sits at a fixed position for the
    duration of one trial and simply flickers at 8 Hz.  Call set_position()
    before each trial to place the bar, then call draw() every frame during
    the target phase only.

    Positions are chosen externally (see session.get_bar_schedule) so that
    all positions are visited equally often across a run.
    """

    FLICKER_HZ = 8

    def __init__(self, win, session, fov_size, bar_width):
        self.win = win
        self.session = session
        self.fov_size = fov_size
        self.bar_width = bar_width

        self.flicker_clock = core.Clock()
        self.contrast = 1.0

        self.bar = visual.GratingStim(
            win,
            tex='sqrXsqr',
            mask=None,
            size=(bar_width, fov_size),
            sf=1.5 / bar_width,  # 3 checkers across bar_width (3 checkers = 1.5 cycles)
            phase=0,
            contrast=self.contrast,
            interpolate=False,
            units='deg',
            ori=0,
            pos=(0, session.stimulus_shift),
        )

        self.aperture = visual.Aperture(
            win, size=fov_size, pos=(0, session.stimulus_shift)
        )
        self.aperture.enabled = False

        self.background_circle = visual.Circle(
            win,
            radius=fov_size / 2.0,
            fillColor=None,
            lineColor='darkgray',
            lineWidth=5.0,
            pos=(0, session.stimulus_shift),
        )

    def set_position(self, position, orientation):
        """
        Place the bar before a trial begins.

        position    : position along the axis perpendicular to the bar (deg)
        orientation : 'horizontal' (bar spans X, positioned on Y axis)
                      'vertical'   (bar spans Y, positioned on X axis)
        """
        if orientation == 'horizontal':
            self.bar.pos = (0, position + self.session.stimulus_shift)
            self.bar.ori = 90   # rotate the (width × height) grating into a wide bar
        else:
            self.bar.pos = (position, self.session.stimulus_shift)
            self.bar.ori = 0    # tall bar sweeping horizontally

        # Reset flicker state at the start of each trial
        self.contrast = 1.0
        self.bar.contrast = self.contrast
        self.flicker_clock.reset()

    def flicker(self):
        """Invert contrast at FLICKER_HZ to create an 8 Hz checkerboard flicker."""
        if self.flicker_clock.getTime() >= 1 / (2 * self.FLICKER_HZ):
            self.contrast *= -1
            self.bar.contrast = self.contrast
            self.flicker_clock.reset()

    def draw(self):
        """Draw the flickering bar within the circular aperture."""
        self.flicker()
        self.aperture.enabled = True
        self.bar.draw()
        self.aperture.enabled = False


class BackgroundCircle:
    def __init__(self, win, session, fov_size=20):
        self.background_circle = visual.Circle(
            win,
            radius=fov_size / 2.0,
            fillColor=None,
            lineColor='darkgray',
            lineWidth=5.0,
            pos=(0, session.stimulus_shift),
        )

    def draw(self):
        self.background_circle.draw()
