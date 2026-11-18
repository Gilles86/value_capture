"""Is the high - low value difference at fixation in the decoded maps caused by the points display?

The points appear at fixation 1-2 s after the search display, on about half of the trials, and are larger for high
value. If they leak into the trial's beta, the foveal value effect must be present on trials with points and absent on
trials without. Decoded images (free decoder, V1-V3) minus the no-distractor trials at the same bar position, aligned
to the distractor as in aggregate_decoded.py, split by value and by whether points were shown.

    python value_capture/decode/check_foveal_feedback.py --decoded-root notes/data/decoded_trials_partial
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault('KERAS_BACKEND', 'jax')

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from value_capture.decode.aggregate_decoded import N, align  # noqa: E402
from value_capture.decode.plot_decoded_gain import FG, HIGH, fmt_p, signflip_p  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ROIS = ('V1', 'V2', 'V3')
FOVEA = 1.0     # deg
AXIS = (np.arange(N) + .5) * 10 / N - 5
RADIUS = np.hypot(*np.meshgrid(AXIS, AXIS))


def collect(decoded_root, behaviour):
    beh = pd.read_csv(behaviour, sep='\t', dtype={'subject': str})
    beh['subject'] = beh.subject.str.zfill(2)
    maps = {}
    for f in sorted(Path(decoded_root).glob('sub-*/readout_ses-*_roi-*_dec-free.tsv')):
        roi = f.name.split('_roi-')[1].split('_')[0]
        if roi not in ROIS:
            continue
        subject = f.parent.name[4:]
        ro = pd.read_csv(f, sep='\t').drop(columns=['subject'], errors='ignore').assign(subject=subject)
        ro = ro.merge(beh[['subject', 'session', 'run', 'trial_nr', 'show_feedback']], how='left', validate='one_to_one',
                      on=['subject', 'session', 'run', 'trial_nr'])
        assert ro.show_feedback.notna().all(), f
        dec = np.load(str(f).replace('readout_', 'decoded_').replace('.tsv', '.npz'))['decoded'].astype(np.float32)
        present = ro.distractor_present.astype(bool).to_numpy()
        bars = pd.factorize(ro.bar_orientation + ro.bar_position.round(4).astype(str))[0]
        base = {b: dec[(bars == b) & ~present].mean(0) for b in np.unique(bars[~present])}
        for shown in (True, False):
            for rank, value in ((2., 'high'), (0., 'low')):
                keep = np.flatnonzero(present & (ro.value_rank == rank) & (ro.show_feedback.astype(bool) == shown)
                                      & np.isin(bars, list(base)))
                imgs = [align((dec[k] - base[bars[k]]).reshape(N, N), ro.distractor_location[k], ro.target_location[k])
                        for k in keep]
                maps.setdefault((subject, shown, value), []).append(np.mean(imgs, 0))
    return {k: np.mean(v, 0) for k, v in maps.items()}


def main(decoded_root, behaviour, out):
    maps = collect(decoded_root, behaviour)
    subjects = sorted({k[0] for k in maps})
    diff = {s: np.array([maps[p, s, 'high'] - maps[p, s, 'low'] for p in subjects]) for s in (True, False)}
    fovea = {s: d[:, RADIUS <= FOVEA].mean(1) for s, d in diff.items()}
    tab = pd.DataFrame({'subject': subjects, 'points_shown': fovea[True], 'no_points': fovea[False]})
    tab['shown_minus_none'] = tab.points_shown - tab.no_points
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(f'{out}.tsv', sep='\t', index=False)
    ps = {c: signflip_p(tab[c]) for c in ('points_shown', 'no_points', 'shown_minus_none')}
    print(tab.round(5).to_string(index=False))
    print({k: round(v, 4) for k, v in ps.items()}, tab.drop(columns='subject').mean().round(5).to_dict())

    mpl.rcParams.update({'font.family': 'Helvetica', 'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.5), gridspec_kw=dict(width_ratios=[1, 1, 1.05], wspace=.18, left=.02,
                                                                         right=.985, top=.78, bottom=.12))
    vmax = np.percentile(np.abs(np.r_[diff[True].mean(0).ravel(), diff[False].mean(0).ravel()]), 99.5)
    for ax, s, title in zip(axes, (True, False), ('Points shown', 'No points shown')):
        ax.imshow(diff[s].mean(0), origin='lower', extent=(-5, 5, -5, 5), cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                  interpolation='bilinear')
        ax.add_patch(plt.Circle((0, 0), FOVEA, fill=False, ec=FG, lw=1))
        ax.set_title(f'{title}: high − low value\nInside the circle: {fmt_p(ps["points_shown" if s else "no_points"])}', pad=3)
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    ax = axes[2]
    x = np.array([0, 1])
    for _, r in tab.iterrows():
        ax.plot(x, [r.points_shown, r.no_points], color='.65', lw=.7, marker='o', ms=2.5, zorder=1)
    ax.errorbar(x, [tab.points_shown.mean(), tab.no_points.mean()], [tab.points_shown.sem(), tab.no_points.sem()],
                color=HIGH, marker='o', ms=5, lw=1.5, zorder=3)
    ax.axhline(0, color='.35', lw=.8, ls=(0, (3, 2)))
    ax.set_xticks(x, ['Points shown', 'No points'])
    ax.set_xlim(-.4, 1.4)
    ax.set_ylabel('High − low at fixation')
    ax.set_title(f'Points shown vs none: {fmt_p(ps["shown_minus_none"])}', pad=3)
    sns.despine(ax=ax, offset=3)
    fig.suptitle(f'Is the value effect at fixation caused by the points display? V1–V3, n = {len(subjects)} subjects',
                 x=.01, ha='left', fontsize=9)
    fig.savefig(f'{out}.pdf'), fig.savefig(f'{out}.png', dpi=200)
    print(f'{out}.pdf')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--decoded-root', default=str(ROOT / 'notes/data/decoded_trials_partial'))
    ap.add_argument('--behaviour', default=str(ROOT / 'notes/data/behavior_trials_main.tsv'))
    ap.add_argument('--out', default=str(ROOT / 'notes/figures/decoded_foveal_feedback_check'))
    a = ap.parse_args()
    main(a.decoded_root, a.behaviour, a.out)
