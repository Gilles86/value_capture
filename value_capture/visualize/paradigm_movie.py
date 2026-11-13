"""Movie of the value-capture paradigm, rendered with matplotlib from the task's geometry.

Geometry and timing follow experiment/settings/default.yml and experiment/stimuli.py:
8 bars (1.5° × 0.375°) on a 4° ring, the target has the odd orientation and may contain a
'hole' (dot); one colour singleton distractor (green / orange = reward level); a checkerboard
PRF bar (3 checks across its width, 10 positions, 8 Hz contrast reversal) in a 3.17° aperture,
shown only during search. ITIs are shortened to 1 s for the movie.

    python value_capture/visualize/paradigm_movie.py   # -> notes/movies/paradigm.{mp4,gif}
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D

ROOT = Path(__file__).resolve().parents[2]
BG, GREY, GREEN, ORANGE = '#646464', '#8f8f8f', '#00ab78', '#d56f2c'
ECC, SIZE, AP_R = 4.0, 1.5, 4.0 - 1.5 / 1.8
BAR_W = 2 * AP_R / 10
BAR_POS = np.linspace(-AP_R + BAR_W / 2, AP_R - BAR_W / 2, 10)
FPS = 32                                  # 8 Hz reversal = every 2 frames
HALF = 6.0                                # half field of view shown (deg)

# trial: target loc, target ori, distractor loc, distractor colour, dots, bar (ori, pos idx), RT, points
TRIALS = [
    dict(t=6, tori=0, d=3, dcol=ORANGE, dots={6, 1, 4}, bar=('vertical', 7), rt=.92, pts='+474', fb=True,
         label='High-value distractor (orange)'),
    dict(t=2, tori=90, d=None, dcol=None, dots={0, 5}, bar=('horizontal', 2), rt=.78, pts=None, fb=False,
         label='No distractor'),
    dict(t=0, tori=0, d=5, dcol=GREEN, dots={0, 2, 3, 7}, bar=('vertical', 3), rt=.85, pts='+5', fb=True,
         label='Low-value distractor (green)'),
]
PHASES = [('Fixation', .5), ('Cue period', .5), ('Search + PRF bar', 1.75), ('Interval', 1.0),
          ('Feedback', 1.0), ('Interval', 1.0)]


def draw_item(ax, x, y, ori, col, hole):
    w, h = (SIZE, SIZE / 4) if ori == 0 else (SIZE / 4, SIZE)
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, color=col, lw=0, zorder=3))
    if hole:
        ax.add_patch(Circle((x, y), SIZE / 9, color=BG, lw=0, zorder=4))


def draw_bar(ax, ori, pos, phase):
    n_len = int(np.ceil(2 * AP_R / (BAR_W / 3)))
    ap = Circle((0, 0), AP_R, transform=ax.transData)
    for i in range(3):
        for j in range(n_len):
            c = 'w' if (i + j + phase) % 2 else 'k'
            u, v = pos - BAR_W / 2 + i * BAR_W / 3, -AP_R + j * BAR_W / 3
            x, y = (u, v) if ori == 'vertical' else (v, u)
            r = Rectangle((x, y), BAR_W / 3, BAR_W / 3, color=c, lw=0, zorder=1)
            ax.add_patch(r)
            r.set_clip_path(ap)


def frame_spec():
    frames = []
    for k, tr in enumerate(TRIALS):
        for name, dur in PHASES:
            for f in range(int(round(dur * FPS))):
                frames.append((k, name, f / FPS))
    return frames


def render(ax, txt, spec, i):
    k, name, t = spec
    tr = TRIALS[k]
    ax.clear()
    ax.set_xlim(-HALF * 16 / 9, HALF * 16 / 9)
    ax.set_ylim(-HALF, HALF)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.add_patch(Rectangle((-20, -10), 40, 20, color=BG, zorder=0))
    if name == 'Search + PRF bar':
        draw_bar(ax, *[tr['bar'][0], BAR_POS[tr['bar'][1]]], (i // 2) % 2)
        for j in range(8):
            a = np.radians(j * 45)
            x, y = ECC * np.cos(a), ECC * np.sin(a)
            col = tr['dcol'] if j == tr['d'] else GREY
            ori = tr['tori'] if j == tr['t'] else (90 if tr['tori'] == 0 else 0)
            draw_item(ax, x, y, ori, col, j in tr['dots'])
        if t >= tr['rt']:
            ax.text(HALF * 16 / 9 - .6, -HALF + .7, 'Response: dot present' if tr['t'] in tr['dots'] else
                    'Response: no dot', color='w', ha='right', fontsize=10)
    if name == 'Feedback' and tr['fb']:
        ax.text(0, 0, tr['pts'], color=tr['dcol'], ha='center', va='center', fontsize=20, weight='bold', zorder=6)
    elif name == 'Feedback':
        ax.text(0, -1.6, '(no feedback on this trial)', color='.8', ha='center', fontsize=9)
    if not (name == 'Feedback' and tr['fb']):
        ax.add_patch(Circle((0, 0), .125, color='k', zorder=5))
    ax.text(-HALF * 16 / 9 + .6, HALF - .8, f'Trial {k + 1}: {tr["label"]}', color='w', fontsize=11, va='top')
    ax.text(-HALF * 16 / 9 + .6, HALF - 2.0, name, color='.85', fontsize=10, va='top')


def main():
    plt.rcParams['font.family'] = 'Helvetica'
    fig = plt.figure(figsize=(8, 4.5), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    frames = frame_spec()
    out = ROOT / 'notes/movies/paradigm'
    for writer, ext in ((FFMpegWriter(fps=FPS, bitrate=4000), 'mp4'), (PillowWriter(fps=16), 'gif')):
        step = 1 if ext == 'mp4' else 2
        with writer.saving(fig, f'{out}.{ext}', dpi=100 if ext == 'mp4' else 60):
            for i in range(0, len(frames), step):
                render(ax, None, frames[i], i)
                writer.grab_frame()
        print(f'{out}.{ext}')


if __name__ == '__main__':
    main()
