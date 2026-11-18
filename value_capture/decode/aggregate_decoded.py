"""Reduce the trial-wise decoded images (decode_trials.py) to gain summaries per subject, session and ROI.

Every comparison is made against physically matching trials, because identical items decode very differently by
location (PRF coverage) and are brighter when the bar is close:

  items: decoded intensity of the diagonal item when it is target or distractor, minus the same item at the same
         bar position when it is neutral, in % of that item's neutral mean.
  bar:   decoded bar intensity minus the no-distractor trials at the same bar position, binned by distance from the
         distractor (or target), in % of the mean decoded bar intensity on no-distractor trials. pct_within subtracts
         the far part of the same bar on the same trial (>= 3.5 deg from that item), which removes trial-wide changes.
  maps:  the same difference over the whole image, rotated so that the distractor is at the upper right and
         reflected so that an adjacent target is at the upper left, averaged per distractor value.

Files older than --since are ignored (earlier outputs of a discarded decoder live in the same folder).

    python value_capture/decode/aggregate_decoded.py --decoded-root <derivatives>/decoded_trials --out-dir notes/data/decoded_gain
"""
import argparse
import os
import re
from datetime import datetime
from pathlib import Path

os.environ.setdefault('KERAS_BACKEND', 'jax')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from value_capture.decode.decode_trials import DIAGONALS, DIST_BINS, item_masks  # noqa: E402
from value_capture.prf import fit_prf as fp  # noqa: E402

N = fp.GRID_RES
CANONICAL = 1             # distractor at 45 deg (upper right)
FAR_DEG = 3.5             # bar pixels at least this far from the item serve as the same-trial reference


def rotate(img, k):
    """Rotate counter-clockwise by k × 90 deg in visual space (row = y, column = x)."""
    return np.rot90(img, k, axes=(1, 0))


def align(img, distractor, target):
    """Distractor to the upper right; a target adjacent to it to the upper left."""
    k = ((CANONICAL - int(distractor)) // 2) % 4
    out = rotate(img, k)
    return out.T if (int(target) + 2 * k) % 8 == 7 else out


def check_alignment():
    g = fp.make_grid_coordinates()
    for d in DIAGONALS:
        for t in DIAGONALS:
            if t == d:
                continue
            blob = lambda i: (np.hypot(g.x - fp.ITEM_XY[i][0], g.y - fp.ITEM_XY[i][1]).to_numpy() < .5).reshape(N, N)  # noqa: E731
            assert (align(blob(d), d, t) == blob(CANONICAL)).all(), (d, t)
            assert any((align(blob(t), d, t) == blob(i)).all() for i in (3, 5)), (d, t)


def remove_points(ro, decoded, shown):
    """Subtract the decoded image of the points display, estimated on the no-distractor trials only.

    On those trials the multiplier is random and nothing else depends on it. Per multiplier: mean image with points minus
    mean image without points, after removing the mean image of each bar position. The map is subtracted from every trial
    with points of that multiplier. Regressing points on all trials would also remove the value effect (points scale with
    value), and fitting the points effect separately per distractor condition would just recover the trials without points.
    """
    present = ro.distractor_present.astype(bool).to_numpy()
    rank = ro.value_rank.to_numpy()
    bars = pd.factorize(ro.bar_orientation + ro.bar_position.round(4).astype(str))[0]
    resid = decoded.astype(np.float32).copy()
    for b in np.unique(bars[~present]):
        m = (bars == b) & ~present
        resid[m] -= decoded[m].mean(0)
    baseline = resid[~present & ~shown].mean(0)
    out = decoded.copy()
    for r in (0., 2.):
        m = ~present & shown & (rank == r)
        if m.any():
            out[shown & (rank == r)] -= resid[m].mean(0) - baseline
    return out


def summarise(ro, decoded, imasks, bar_masks, gx, gy):
    """Item and bar rows and aligned difference maps of one subject × session × ROI × decoder."""
    bars = list(zip(ro.bar_orientation, ro.bar_position.round(4)))
    present = ro.distractor_present.astype(bool).to_numpy()
    target, dist, rank = ro.target_location.to_numpy(), ro.distractor_location.to_numpy(), ro.value_rank.to_numpy()
    bar_ids = pd.factorize(pd.Series(bars))[0]
    labels = {0.: 'distractor_low', 2.: 'distractor_high'}

    # items
    item = np.array([[decoded[k][imasks[i]].mean() for i in DIAGONALS] for k in range(len(ro))])
    is_neutral = np.array([(target != i) & ~(present & (dist == i)) for i in DIAGONALS]).T
    # One denominator per ROI: a single location's neutral mean can be close to 0 where PRF coverage is poor
    norm = item[is_neutral].mean()
    gain = np.full(item.shape, np.nan)            # % of the ROI's neutral intensity, matched on location and bar
    for j in range(len(DIAGONALS)):
        for b in np.unique(bar_ids):
            here = bar_ids == b
            pool = here & is_neutral[:, j]
            n = pool.sum()
            if n == 0:
                continue
            total = item[pool, j].sum()
            gain[here, j] = 100 * (item[here, j] - total / n) / norm
            # a neutral item is compared with the other neutral trials only (leave-one-out)
            gain[pool, j] = (100 * (item[pool, j] - (total - item[pool, j]) / (n - 1)) / norm) if n > 1 else np.nan
    column = {i: j for j, i in enumerate(DIAGONALS)}
    target_gain = gain[np.arange(len(ro)), [column[int(t)] for t in target]]
    dist_gain = np.array([gain[k, column[int(dist[k])]] if present[k] else np.nan for k in range(len(ro))])
    neutral_gain = np.nanmean(np.where(is_neutral, gain, np.nan), 1)
    trials = pd.DataFrame(dict(run=ro.run, trial_nr=ro.trial_nr, target_gain=target_gain, distractor_gain=dist_gain,
                               neutral_gain=neutral_gain))
    item_rows = {'target': target_gain}
    for r, lab in labels.items():
        item_rows[lab] = dist_gain[present & (rank == r)]
    items = [dict(role=k, pct=float(np.nanmean(v)), n_trials=int(np.isfinite(v).sum())) for k, v in item_rows.items()
             if np.isfinite(v).any()]

    # no-distractor baseline image per bar position
    base = {b: decoded[(bar_ids == b) & ~present].mean(0) for b in np.unique(bar_ids) if ((bar_ids == b) & ~present).any()}
    scale = np.mean([decoded[k][bar_masks[bars[k]]].mean() for k in np.flatnonzero(~present)])

    curves, maps = [], {}
    acc = {(who, r, b_): [] for who in ('distractor', 'target') for r in labels for b_ in range(len(DIST_BINS))}
    within = {k: [] for k in acc}
    sums = {r: [] for r in labels}
    for k in np.flatnonzero(present):
        if bar_ids[k] not in base or rank[k] not in labels:
            continue
        diff = decoded[k] - base[bar_ids[k]]
        sums[rank[k]].append(align(diff.reshape(N, N), dist[k], target[k]))
        bar = bar_masks[bars[k]]
        for who, loc in (('distractor', dist[k]), ('target', target[k])):
            cx, cy = fp.ITEM_XY[int(loc)]
            dd = np.hypot(gx - cx, gy - cy)
            far = bar & (dd >= FAR_DEG)
            for b_, (lo, hi) in enumerate(DIST_BINS):
                q = bar & (dd >= lo) & (dd < hi)
                if q.any():
                    acc[who, rank[k], b_].append(100 * diff[q].mean() / scale)
                    if far.any():
                        within[who, rank[k], b_].append(100 * (diff[q].mean() - diff[far].mean()) / scale)
    for (who, r, b_), v in acc.items():
        if v:
            lo, hi = DIST_BINS[b_]
            curves.append(dict(aligned_to=who, value=labels[r].split('_')[1], bin_lo=lo, bin_hi=hi,
                               distance=(lo + hi) / 2, pct=float(np.mean(v)), n_trials=len(v),
                               pct_within=float(np.mean(within[who, r, b_])) if within[who, r, b_] else np.nan))
    for r, lab in labels.items():
        if sums[r]:
            maps[lab.split('_')[1]] = np.mean(sums[r], 0).astype(np.float32)
    return items, curves, maps, trials


def main(decoded_root, out_dir, since, feedback='all', behaviour=None):
    """feedback: 'all' trials, or only those with points shown ('shown') or not shown ('none'), baselines included."""
    check_alignment()
    shown = None
    if feedback != 'all':
        shown = pd.read_csv(behaviour, sep='\t', dtype={'subject': str})
        shown['subject'] = shown.subject.str.zfill(2)
        shown = shown[['subject', 'session', 'run', 'trial_nr', 'show_feedback']]
    since = datetime.fromisoformat(since).timestamp()
    imasks = item_masks()
    g = fp.make_grid_coordinates()
    gx, gy = g.x.to_numpy(), g.y.to_numpy()
    inside = np.hypot(gx, gy) <= fp.GEOM['radius']
    bar_masks = {(o, round(float(p), 4)): (fp.make_bar_image(p, o).ravel() > .5) & inside
                 for o in ('horizontal', 'vertical') for p in fp.GEOM['bar_positions']}
    items, curves, maps, trial_rows = [], [], {}, []
    pat = re.compile(r'sub-(\d+)/readout_ses-(\d)_roi-(\w+)_dec-(\w+)\.tsv$')
    for f in sorted(Path(decoded_root).glob('sub-*/readout_ses-*_roi-*_dec-*.tsv')):
        npz = f.with_name(f.name.replace('readout_', 'decoded_').replace('.tsv', '.npz'))
        if f.stat().st_mtime < since or not npz.exists() or npz.stat().st_mtime < since:
            continue
        sub, ses, roi, dec = pat.search(str(f)).groups()
        ro = pd.read_csv(f, sep='\t')
        decoded = np.load(npz)['decoded'].astype(np.float32)
        assert len(ro) == len(decoded), f
        if shown is not None:
            flag = ro[['session', 'run', 'trial_nr']].assign(subject=sub).merge(
                shown, on=['subject', 'session', 'run', 'trial_nr'], how='left', validate='one_to_one').show_feedback
            assert flag.notna().all(), f
            if feedback == 'regress':
                decoded = remove_points(ro, decoded, flag.astype(bool).to_numpy())
            else:
                keep = flag.astype(bool).to_numpy() == (feedback == 'shown')
                ro, decoded = ro[keep].reset_index(drop=True), decoded[keep]
        i_, c_, m_, t_ = summarise(ro, decoded, imasks, bar_masks, gx, gy)
        key = dict(subject=sub, session=int(ses), roi=roi, decoder=dec)
        if dec == 'outline':
            trial_rows.append(t_.assign(subject=sub, session=int(ses), roi=roi))
        items += [{**key, **r} for r in i_]
        curves += [{**key, **r} for r in c_]
        for value, img in m_.items():
            maps[f'{sub}|{ses}|{roi}|{dec}|{value}'] = img
        print(f.name, sub, flush=True)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(items).to_csv(out_dir / 'item_gain.tsv', sep='\t', index=False)
    pd.DataFrame(curves).to_csv(out_dir / 'bar_distance.tsv', sep='\t', index=False)
    pd.concat(trial_rows).to_csv(out_dir / 'trial_gains.tsv', sep='\t', index=False)
    np.savez_compressed(out_dir / 'aligned_maps.npz', **maps)
    print(out_dir, len(items), len(curves), len(maps))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--decoded-root', default='/shares/zne.uzh/gdehol/ds-valuecapture/derivatives/decoded_trials')
    ap.add_argument('--out-dir', default='notes/data/decoded_gain')
    ap.add_argument('--since', default='2026-11-18T14:05:00', help='ignore decoder outputs older than this')
    ap.add_argument('--feedback', choices=['all', 'shown', 'none', 'regress'], default='all',
                    help='all trials, only those with / without the points display, or all trials with the '
                         'points display subtracted (remove_points)')
    ap.add_argument('--behaviour', default='notes/data/behavior_trials_main.tsv')
    a = ap.parse_args()
    main(a.decoded_root, a.out_dir, a.since, a.feedback, a.behaviour)
