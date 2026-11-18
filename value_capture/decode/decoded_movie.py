"""Movie of the trial-by-trial decoded visual input (decode_trials.py), next to what was on the screen.

One frame per held-out trial, in the order of the session: the true stimulus (bar + 8 search items), the decoder
with every pixel free, and the decoder restricted to the pixels the stimulus covers. Rings mark the target and the
distractor. The rebuilt stimulus is checked against the r(decoded, true) stored by decode_trials.py, so a
misaligned trial order stops the script.

    python value_capture/decode/decoded_movie.py 07 --session 2 --roi V1 \
        --decoded-dir notes/data/decode_test --bids-folder /data/ds-valuecapture
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from value_capture.prf import fit_prf as fp
from value_capture.utils.data import Subject

ROOT = Path(__file__).resolve().parents[2]
BG, FG = '#111318', '#f2f2f2'
TARGET, HIGH, LOW = '#ffffff', '#ff5a36', '#35c9ff'
EXTENT = (-fp.GRID_RADIUS, fp.GRID_RADIUS, -fp.GRID_RADIUS, fp.GRID_RADIUS)


def load(decoded_dir, session, roi, decoder):
    stem = f'ses-{session}_roi-{roi}_dec-{decoder}'
    ro = pd.read_csv(Path(decoded_dir) / f'readout_{stem}.tsv', sep='\t')
    img = np.load(Path(decoded_dir) / f'decoded_{stem}.npz')['decoded'].astype(np.float32)
    assert len(ro) == len(img), (len(ro), img.shape)
    return ro, img


def condition(t):
    if not t.distractor_present:
        return 'No distractor'
    return 'High-value distractor (×100)' if t.value_rank == 2 else 'Low-value distractor (×1)'


def bar_label(t):
    axis = 'y' if t.bar_orientation == 'horizontal' else 'x'
    return f'{t.bar_orientation.capitalize()} bar, {axis} = {t.bar_position:+.1f}°'.replace('-', '−')


def load_session(subject, session, roi, decoded_dir, bids_folder):
    """Trials, true stimulus, both decoders and r(free, true); stops if the rebuilt stimulus is misaligned."""
    ro, free = load(decoded_dir, session, roi, 'free')
    _, outline = load(decoded_dir, session, roi, 'outline')
    sub = Subject(subject, bids_folder=bids_folder)
    trials = fp.add_item_info(sub, ro.drop(columns=['target_location']))
    true = fp.build_paradigm(trials, 'bar_items')
    r = np.array([np.corrcoef(free[k], true[k])[0, 1] for k in range(len(true))])
    assert np.nanmax(np.abs(r - ro.r_true.to_numpy())) < .01, 'rebuilt stimulus does not match the decoded trial order'
    return trials, true, free, outline, r


def bar_average_movie(subject, session, roi, decoded_dir, bids_folder, fps, out):
    """One frame per bar position (20): decoded images averaged over trials, per distractor condition."""
    trials, true, free, outline, _ = load_session(subject, session, roi, decoded_dir, bids_folder)
    present, rank = trials.distractor_present.astype(bool).to_numpy(), trials.value_rank.to_numpy()
    conds = (('All trials', np.ones(len(trials), bool)), ('No distractor', ~present),
             ('Low-value distractor', present & (rank == 0)), ('High-value distractor', present & (rank == 2)))
    bars = sorted({(o, p) for o, p in zip(trials.bar_orientation, trials.bar_position)})
    n = fp.GRID_RES

    def mean_img(a, m):
        return a[m].mean(0).reshape(n, n) if m.any() else np.zeros((n, n))

    frames = []
    for o, p in bars:
        here = (trials.bar_orientation == o).to_numpy() & (trials.bar_position == p).to_numpy()
        frames.append(dict(bar=pd.Series(dict(bar_orientation=o, bar_position=p)), true=mean_img(true, here),
                           n=[int((here & m).sum()) for _, m in conds],
                           free=[mean_img(free, here & m) for _, m in conds],
                           outline=[mean_img(outline, here & m) for _, m in conds]))
    vmax = float(np.percentile(np.concatenate([np.ravel(f[d]) for f in frames for d in ('free', 'outline')]), 99.9))

    mpl.rcParams.update({'font.family': 'Helvetica', 'text.color': FG, 'axes.edgecolor': FG})
    fig = plt.figure(figsize=(13, 5.6), dpi=150, facecolor=BG)
    w, h = .168, .39

    def panel(x, y, vm):
        ax = fig.add_axes([x, y, w, h], facecolor=BG)
        im = ax.imshow(np.zeros((n, n)), origin='lower', extent=EXTENT, cmap='inferno', vmin=0, vmax=vm,
                       interpolation='nearest')
        ax.add_patch(Circle((0, 0), .12, color='#00e5a0', lw=0, zorder=4))
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        return ax, im

    ax0, im_true = panel(.008, .47, 1)
    ax0.set_title('On the screen', color=FG, fontsize=11.5, pad=4)
    xs = [.215 + i * .196 for i in range(4)]
    ims = {d: [panel(x, y, vmax)[1] for x in xs] for d, y in (('free', .47), ('outline', .03))}
    tops = [fig.axes[-8 + i] for i in range(4)]
    fig.text(.203, .665, 'All pixels free', rotation=90, fontsize=11.5, ha='center', va='center')
    fig.text(.203, .225, 'Stimulus pixels only', rotation=90, fontsize=11.5, ha='center', va='center')
    head = fig.text(.008, .955, f'Sub-{subject}, {roi}, session {session}: decoded input averaged per bar position',
                    fontsize=13, va='center')
    score = fig.text(.992, .955, '', fontsize=12, ha='right', va='center')
    cax = fig.add_axes([.02, .2, .145, .025])
    cb = fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(0, vmax), 'inferno'), cax=cax, orientation='horizontal')
    cb.set_ticks([0, vmax]), cb.set_ticklabels(['0', f'{vmax:.2f}'])
    cb.ax.tick_params(colors=FG, labelsize=9.5, length=2), cb.outline.set_visible(False)
    fig.text(.0925, .25, 'Mean decoded intensity', fontsize=10.5, ha='center', va='bottom')

    def draw(k):
        f = frames[k]
        im_true.set_data(f['true'])
        for d in ('free', 'outline'):
            for im, a in zip(ims[d], f[d]):
                im.set_data(a)
        for ax, (name, _), c in zip(tops, conds, f['n']):
            ax.set_title(f'{name} (n = {c})', color=FG, fontsize=11.5, pad=4)
        score.set_text(f'{bar_label(f["bar"])}   |   Position {k + 1} of {len(frames)}')

    out = Path(out) if out else ROOT / f'notes/movies/decoded_sub-{subject}_ses-{session}_roi-{roi}_bar-average.mp4'
    out.parent.mkdir(parents=True, exist_ok=True)
    draw(len(frames) // 4)
    fig.savefig(out.with_suffix('.png'), facecolor=BG)
    writer = FFMpegWriter(fps=fps, codec='libx264', bitrate=6000, extra_args=['-pix_fmt', 'yuv420p'])
    with writer.saving(fig, str(out), dpi=150):
        for k in range(len(frames)):
            draw(k)
            writer.grab_frame(facecolor=BG)
    print(out, f'{len(frames)} bar positions, {len(frames) / fps:.0f} s')


def main(subject, session, roi, decoded_dir, bids_folder, fps, out, sort_bar=False):
    trials, true, free, outline, r = load_session(subject, session, roi, decoded_dir, bids_folder)
    if sort_bar:      # bar orientation, then position across the screen, then session order
        order = np.lexsort((trials.trial_nr, trials.run, trials.bar_position, trials.bar_orientation))
        trials, true, free, outline, r = trials.iloc[order], true[order], free[order], outline[order], r[order]

    n = fp.GRID_RES
    vmax = float(np.percentile(np.r_[free.ravel()[::7], outline.ravel()[::7]], 99.9))
    mpl.rcParams.update({'font.family': 'Helvetica', 'text.color': FG, 'axes.edgecolor': FG})
    fig = plt.figure(figsize=(10.8, 4.6), dpi=150, facecolor=BG)
    titles = ('On the screen', 'Decoded, all pixels free', 'Decoded, stimulus pixels only')
    axes, ims, rings = [], [], []
    for i, title in enumerate(titles):
        ax = fig.add_axes([.012 + i * .33, .13, .31, .72], facecolor=BG)
        im = ax.imshow(np.zeros((n, n)), origin='lower', extent=EXTENT, cmap='inferno', vmin=0,
                       vmax=1 if i == 0 else vmax, interpolation='nearest')
        ax.add_patch(Circle((0, 0), .12, color='#00e5a0', lw=0, zorder=4))
        tr = Circle((0, 0), 1.0, fill=False, ec=TARGET, lw=1.6, zorder=5)
        di = Circle((0, 0), 1.0, fill=False, ec=HIGH, lw=2.0, zorder=5)
        ax.add_patch(tr), ax.add_patch(di)
        ax.set_title(title, color=FG, fontsize=12, pad=5)
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        axes.append(ax), ims.append(im), rings.append((tr, di))
    head = fig.text(.012, .955, '', fontsize=13, weight='bold', va='center')
    score = fig.text(.988, .955, '', fontsize=12, ha='right', va='center')
    fig.legend(handles=[Line2D([], [], marker='o', mfc='none', mec=c, ls='', ms=10, mew=1.8, label=lab)
                        for c, lab in ((TARGET, 'Target'), (HIGH, 'High-value distractor'),
                                       (LOW, 'Low-value distractor'))],
               loc='lower left', bbox_to_anchor=(.008, .005), ncol=3, frameon=False, fontsize=10.5,
               handletextpad=.3, columnspacing=1.5, labelcolor=FG)
    cax = fig.add_axes([.70, .065, .17, .022])
    cb = fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(0, vmax), 'inferno'), cax=cax, orientation='horizontal')
    cb.set_ticks([0, vmax]), cb.set_ticklabels(['0', f'{vmax:.2f}'])
    cb.ax.tick_params(colors=FG, labelsize=9.5, length=2), cb.outline.set_visible(False)
    fig.text(.692, .076, 'Decoded intensity', fontsize=10.5, ha='right', va='center')

    suffix = '_sorted-bar' if sort_bar else ''
    out = Path(out) if out else ROOT / f'notes/movies/decoded_sub-{subject}_ses-{session}_roi-{roi}{suffix}.mp4'
    out.parent.mkdir(parents=True, exist_ok=True)

    def draw(k):
        t = trials.iloc[k]
        for im, a in zip(ims, (true[k], free[k], outline[k])):
            im.set_data(a.reshape(n, n))
        tx, ty = fp.ITEM_XY[int(t.target_location)]
        for tr, di in rings:
            tr.center = (tx, ty)
            di.set_visible(bool(t.distractor_present))
            if t.distractor_present:
                di.center = tuple(fp.ITEM_XY[int(t.distractor_location)])
                di.set_edgecolor(HIGH if t.value_rank == 2 else LOW)
        head.set_text(f'Sub-{subject}, {roi}, session {session}, run {int(t.run)}, trial {int(t.trial_nr)}: {condition(t)}')
        score.set_text(f'{bar_label(t)}   |   r(decoded, true) = {r[k]:.2f}')

    draw(int(np.argsort(r)[len(r) // 2]))
    fig.savefig(out.with_suffix('.png'), facecolor=BG)
    writer = FFMpegWriter(fps=fps, codec='libx264', bitrate=6000, extra_args=['-pix_fmt', 'yuv420p'])
    with writer.saving(fig, str(out), dpi=150):
        for k in range(len(trials)):
            draw(k)
            writer.grab_frame(facecolor=BG)
    print(out, f'{len(trials)} trials, {len(trials) / fps:.0f} s, median r = {np.median(r):.2f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subject')
    ap.add_argument('--session', type=int, required=True)
    ap.add_argument('--roi', default='V1')
    ap.add_argument('--decoded-dir', required=True)
    ap.add_argument('--bids-folder', default='/data/ds-valuecapture')
    ap.add_argument('--fps', type=float, default=3)
    ap.add_argument('--out', default=None)
    ap.add_argument('--sort-bar', action='store_true', help='order trials by bar orientation and position')
    ap.add_argument('--average-bar', action='store_true',
                    help='one frame per bar position, averaged over trials per distractor condition')
    a = ap.parse_args()
    if a.average_bar:
        bar_average_movie(a.subject, a.session, a.roi, a.decoded_dir, a.bids_folder, a.fps, a.out)
    else:
        main(a.subject, a.session, a.roi, a.decoded_dir, a.bids_folder, a.fps, a.out, sort_bar=a.sort_bar)
