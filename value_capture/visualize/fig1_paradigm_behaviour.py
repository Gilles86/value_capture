"""Figure 1 (after retsupp Fig. 1): the task, and that reward value captures attention.

Row 1: a, the display of one trial to scale (grey items, one colour-singleton distractor, the flickering bar in
its aperture); b, the trial timeline, the 20 bar positions and the colour-value mapping per session.
Row 2: c, response time by distractor (absent, low, high value) with within-subject differences beside it;
d, capture (distractor − absent) and the value effect (high − low) per block of runs, practice included, in
both sessions (the mapping reverses in session 2).

Geometry and timing from experiment/settings/default.yml and experiment/stimuli.py.

    python value_capture/visualize/fig1_paradigm_behaviour.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Rectangle

from value_capture.behavior.plot_behavior_share import (COL, FG, MUTED, ORDER, cond_panel, load_behaviour,
                                                        strip)

ROOT = Path(__file__).resolve().parents[2]
SCREEN, GREY = '#646464', '#8f8f8f'
GREEN, ORANGE = '#00ab78', '#d56f2c'
RING, ITEM_LONG, ITEM_SHORT, APERTURE = 4.0, 1.5, .375, 3.17
BAR_POS = np.linspace(-APERTURE * 2 / 2 + .633 / 2, APERTURE - .633 / 2, 10)
BAR_W = 2 * APERTURE / 10


def item_xy(i):
    a = np.radians(i * 45.)
    return np.array([RING * np.cos(a), RING * np.sin(a)])


def draw_item(ax, i, ori, color, dot=False, lw=0, z=4):
    cx, cy = item_xy(i)
    w, h = (ITEM_LONG, ITEM_SHORT) if ori == 0 else (ITEM_SHORT, ITEM_LONG)
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, facecolor=color, lw=lw, zorder=z))
    if dot:
        ax.add_patch(Circle((cx, cy), ITEM_LONG / 9, facecolor='k', lw=0, zorder=z + .1))


def draw_bar(ax, pos, orientation, n_checks=12, z=3):
    """Checkerboard bar clipped to the circular aperture."""
    clip = Circle((0, 0), APERTURE, transform=ax.transData)
    L = 2 * APERTURE
    for k in range(n_checks):
        for j in range(2):
            c = 'k' if (k + j) % 2 else 'w'
            if orientation == 'vertical':
                r = Rectangle((pos - BAR_W / 2 + j * BAR_W / 2, -APERTURE + k * L / n_checks), BAR_W / 2, L / n_checks,
                              facecolor=c, lw=0, zorder=z)
            else:
                r = Rectangle((-APERTURE + k * L / n_checks, pos - BAR_W / 2 + j * BAR_W / 2), L / n_checks, BAR_W / 2,
                              facecolor=c, lw=0, zorder=z)
            ax.add_patch(r)
            r.set_clip_path(clip)


def panel_display(ax, margin=8.6):
    ax.add_patch(Rectangle((-5.6, -5.6), 11.2, 11.2, facecolor=SCREEN, lw=0, zorder=0))
    target, distractor, t_ori = 3, 7, 0
    draw_bar(ax, BAR_POS[3], 'vertical')
    for i in range(8):
        ori = t_ori if i == target else 90
        col = GREEN if i == distractor else GREY
        draw_item(ax, i, ori, col, dot=i in (target, 0, 6))
    ax.add_patch(Circle((0, 0), .125, facecolor='w', lw=0, zorder=5))
    ax.add_patch(Circle((0, 0), APERTURE, fill=False, ec='w', lw=.4, ls=(0, (1.5, 1.5)), zorder=2.5))
    ax.add_patch(Circle((0, 0), RING, fill=False, ec='.8', lw=.4, ls=(0, (1, 2)), zorder=2.4))
    kw = dict(fontsize=7, color=FG, annotation_clip=False,
              arrowprops=dict(arrowstyle='-|>', color=FG, lw=.8, mutation_scale=6, shrinkA=1, shrinkB=2))
    ax.annotate('Target\n(odd orientation:\ndot or no dot?)', xy=item_xy(3) + [-.3, .4], xytext=(-4.2, 7.3),
                ha='center', va='center', **{**kw, 'fontweight': 'bold'})
    ax.annotate('Distractor\n(colour = value)', xy=item_xy(7) + [.3, -.5], xytext=(6.8, -7.4), ha='center',
                va='center', **{**kw, 'fontweight': 'bold'})
    ax.annotate('Bar in 3.17°\naperture', xy=(BAR_POS[3], -2.5), xytext=(-6.6, -7.4), ha='center', va='center', **kw)
    ax.plot([.3, RING - .85], [0, 0], color='w', lw=.7, zorder=6)
    ax.text(RING / 2 - .1, .2, '4°', color='w', fontsize=6, ha='center', va='bottom', zorder=6)
    ax.plot([-5.0, -4.0], [-5.1, -5.1], color='w', lw=1.2, solid_capstyle='butt', zorder=6)
    ax.text(-4.5, -4.95, '1°', color='w', fontsize=6, ha='center', va='bottom', zorder=6)
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-8.3, 8.5)
    ax.set_aspect('equal')
    ax.set_axis_off()


def mini_display(ax, distractor_col=None, bar=None):
    ax.add_patch(Circle((0, 0), 5.4, facecolor=SCREEN, lw=0))
    for i in range(8):
        draw_item(ax, i, 90, distractor_col if (distractor_col and i == 7) else GREY)
    if bar is not None:
        draw_bar(ax, *bar, n_checks=8)
    ax.set_xlim(-5.6, 5.6)
    ax.set_ylim(-5.6, 5.6)
    ax.set_aspect('equal')
    ax.set_axis_off()


def panel_timeline(fig, ax):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    seq = [('Fixation', 1., '1 s'), ('Search + bar', 1.75, '1.75 s'), ('Interval', 1.6, '0.5–4 s'),
           ('Feedback', 1., '1 s'), ('Interval', 1.6, '0.5–4 s')]
    total = sum(d for _, d, _ in seq)
    x0, scale, y, h = .0, .98 / total, .8, .11
    onset = 0.
    for name, dur, lab in seq:
        col, tc = (FG, 'w') if name.startswith('Search') else ('#ECEFF2', FG) if name == 'Interval' else ('#D5DAE0', FG)
        ax.add_patch(Rectangle((x0 + onset * scale, y), dur * scale - .004, h, facecolor=col, lw=0))
        ax.text(x0 + (onset + dur / 2) * scale, y + h / 2, name, ha='center', va='center', fontsize=6.5, color=tc)
        ax.text(x0 + (onset + dur / 2) * scale, y + h + .02, lab, ha='center', va='bottom', fontsize=6.5, color=FG)
        if name == 'Feedback':
            ax.text(x0 + (onset + dur / 2) * scale, y - .03, '+870 points\n(50% of trials)', ha='center', va='top',
                    fontsize=6, color=GREEN)
        onset += dur
    ax.text(x0 + 1.75 * scale / 2 + 1. * scale, y - .03, 'Bar shown only here,\nat 1 of 20 positions',
            ha='center', va='top', fontsize=6, color=FG)

    # 20 bar positions
    bb = ax.get_position()
    aspect = bb.width * fig.get_figwidth() / (bb.height * fig.get_figheight())
    w = .1
    icon = w * aspect
    ax.text(0, .43, '10 vertical + 10 horizontal bar positions', fontsize=6.5, color=FG, va='bottom')
    for k, (p, o) in enumerate([(BAR_POS[1], 'vertical'), (BAR_POS[5], 'vertical'), (BAR_POS[8], 'horizontal'),
                                (BAR_POS[2], 'horizontal')]):
        mini_display(ax.inset_axes([k * .12, .42 - icon - .02, w, icon]), bar=(p, o))

    # value mapping per session
    xs = .55
    ax.text(xs, .43, 'Colour signals points (counterbalanced)', fontsize=6.5, color=FG, va='bottom')
    for s, (hi_col, lo_col) in enumerate(((GREEN, ORANGE), (ORANGE, GREEN))):
        yy = .42 - icon - .02
        x = xs + s * .23
        for dx, c in ((0, hi_col), (.09, lo_col)):
            ins = ax.inset_axes([x + dx, yy, w * .8, icon * .8])
            ins.add_patch(Circle((0, 0), 5.4, facecolor=SCREEN, lw=0))
            ins.add_patch(Rectangle((-.9, -3.2), 1.8, 6.4, facecolor=c, lw=0))
            ins.set_xlim(-5.6, 5.6)
            ins.set_ylim(-5.6, 5.6)
            ins.set_aspect('equal')
            ins.set_axis_off()
        ax.text(x + w * .4, yy - .03, '×100', ha='center', va='top', fontsize=6, color=COL['high'], weight='bold')
        ax.text(x + .09 + w * .4, yy - .03, '×1', ha='center', va='top', fontsize=6, color=COL['low'], weight='bold')
        ax.text(x + .09, yy + icon * .8 + .02, f'Session {s + 1}' + (' (reversed)' if s else ''), ha='center',
                va='bottom', fontsize=6, color='.4')
    ax.text(xs, yy - .16, 'Per session: 2 practice runs (80 trials),\nthen 8 scanner runs of 40 trials',
            fontsize=6, color='.4', va='top')


def panel_blocks(ax, df):
    """Capture and value effect per block (practice P1, P2; scanner runs in pairs), both sessions."""
    prac = pd.read_csv(ROOT / 'notes/data/behavior_trials_practice.tsv', sep='\t', dtype={'subject': str})
    prac['correct'] = prac['correct'].map({True: 1., False: 0., 'True': 1., 'False': 0., 1.: 1., 0.: 0.})
    raw = pd.read_csv(ROOT / 'notes/data/behavior_trials_main_raw.tsv', sep='\t', dtype={'subject': str})
    raw['correct'] = raw['correct'].map({True: 1., False: 0., 'True': 1., 'False': 0., 1.: 1., 0.: 0.})
    prac['block'] = prac.run.map({-1: 0, -2: 1})
    raw['block'] = 2 + (raw.run - 1) // 2
    ok = pd.concat([x[(x.correct == 1) & (x.rt >= .2)] for x in (prac, raw)])
    rt = (ok.groupby(['subject', 'session', 'block', 'cond']).rt.mean() * 1000).unstack()
    d = pd.DataFrame({'capture': (rt.low + rt.high) / 2 - rt.none, 'value': rt.high - rt.low}).reset_index()
    nb, gap = 6, 1.
    d['x'] = d.block + (d.session - 1) * (nb + gap)
    for s in (1, 2):
        x0 = (s - 1) * (nb + gap)
        ax.axvspan(x0 - .45, x0 + 1.45, color='#EEF0F2', lw=0, zorder=0)
    for k, (col, c, lab) in enumerate((('capture', MUTED, 'Capture (distractor − absent)'),
                                       ('value', COL['high'], 'Value (high − low)'))):
        g = d.groupby('x')[col]
        m, se = g.mean(), g.sem()
        off = (k - .5) * .16
        for s in (1, 2):
            xs = m.index[(m.index >= (s - 1) * (nb + gap)) & (m.index < s * (nb + gap) - gap)]
            ax.plot(xs + off, m[xs], color=c, lw=1.1, zorder=3)
        ax.errorbar(m.index + off, m, se, fmt='o', color=c, ms=3.8, mec='w', mew=.5, lw=1, capsize=0, zorder=4)
        ax.text(.99, .97 - k * .1, lab, transform=ax.transAxes, color=c, fontsize=6.3, ha='right', va='top')
    ax.axhline(0, color='.6', lw=.6, ls='--', zorder=1)
    labs = ['P1', 'P2', '1–2', '3–4', '5–6', '7–8']
    ax.set_xticks([b + (s - 1) * (nb + gap) for s in (1, 2) for b in range(nb)], labs * 2, fontsize=6)
    ax.set_xlim(-.7, 2 * nb + gap - .3)
    yl = ax.get_ylim()
    for s in (1, 2):
        x0 = (s - 1) * (nb + gap)
        ax.text(x0 + 2.5, yl[1] + .02 * (yl[1] - yl[0]), f'Session {s}' + (' (mapping reversed)' if s == 2 else ''),
                ha='center', va='bottom', fontsize=6.5, color=FG)
        ax.text(x0 + .5, yl[0], 'Practice', ha='center', va='bottom', fontsize=5.8, color='.45')
    ax.set_ylabel('RT difference (ms)')
    ax.set_xlabel('Runs', labelpad=2)
    for s in ('left', 'bottom'):
        ax.spines[s].set_position(('outward', 3))


def main():
    df = load_behaviour()
    rt = (df[df.correct == 1].groupby(['subject', 'cond']).rt.mean() * 1000).unstack()[ORDER]
    W, H = 7.2, 5.0
    fig = plt.figure(figsize=(W, H))
    at = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])  # noqa: E731

    ax = at(.05, 2.7, 2.35, 2.25)
    panel_display(ax)
    ax.text(0, 1.0, 'a', transform=ax.transAxes, fontsize=8, weight='bold', va='top')
    ax = at(2.75, 2.75, 4.35, 2.1)
    panel_timeline(fig, ax)
    ax.text(-.04, 1.04, 'b', transform=ax.transAxes, fontsize=8, weight='bold', va='top')

    Y0, PH = .45, 1.45
    ax_c = at(.5, Y0, 1.35, PH)
    cond_panel(ax_c, rt, (700, 1050), [700, 800, 900, 1000],
               [('low', 'none', 'greater'), ('high', 'none', 'greater'), ('high', 'low', 'greater')], 'ms')
    ax_c.set_ylabel('Response time (ms)')
    ax_c.text(-.45, 1.5, 'c', transform=ax_c.transAxes, fontsize=8, weight='bold', va='bottom')
    x0, SW, SH = 2.2, 1.3, .2
    for k, (lab, d, c) in enumerate([('Low − absent', rt.low - rt.none, COL['low']),
                                     ('High − absent', rt.high - rt.none, COL['high']),
                                     ('High − low', rt.high - rt.low, COL['high'])]):
        s = at(x0, Y0 + PH - .3 - k * .52, SW, SH)
        strip(s, d, c, lab, (-60, 110), 'slower', 'greater', seed=k)
        s.set_xticks([-50, 0, 50, 100])
        if k < 2:
            s.set_xticklabels([])
        else:
            s.set_xlabel('RT difference (ms)', labelpad=1)
    fig.text((x0 + SW / 2) / W, (Y0 + PH + .15) / H, 'Within subject', fontsize=6.5, color='.4', ha='center')

    ax_d = at(4.25, Y0, 2.85, PH)
    panel_blocks(ax_d, df)
    ax_d.text(-.16, 1.5, 'd', transform=ax_d.transAxes, fontsize=8, weight='bold', va='bottom')

    out = ROOT / 'notes/figures/fig1_paradigm_behaviour'
    fig.savefig(f'{out}.pdf')
    fig.savefig(f'{out}.png', dpi=300)
    print('wrote', out)


if __name__ == '__main__':
    main()
