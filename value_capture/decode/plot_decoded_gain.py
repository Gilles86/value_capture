"""Figure of the model-free gain readout: aligned decoded maps, item gain and bar gain around the distractor.

Reads the summaries of aggregate_decoded.py. Conditions are averaged within subject first (over held-out sessions).
p-values: exact two-sided sign-flip tests across subjects, uncorrected.

    python value_capture/decode/plot_decoded_gain.py --in-dir notes/data/decoded_gain --out notes/figures/decoded_gain_prototype
"""
import argparse
import itertools
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patheffects
from matplotlib.patches import Circle, Rectangle

from value_capture.visualize.paradigm_movie import AP_R, BAR_POS, BAR_W, BG, GREY, ORANGE, draw_bar, draw_item

mpl.rcParams.update({'font.family': 'Helvetica', 'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': .8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'xtick.major.size': 3, 'ytick.major.size': 3,
                     'xtick.major.width': .8, 'ytick.major.width': .8, 'pdf.fonttype': 42, 'savefig.dpi': 300})
ROIS = ['V1', 'V2', 'V3', 'hV4', 'LO']     # V3AB has only ~30 voxels with a usable PRF
MAP_ROIS = ['V1', 'V2', 'V3']
PATHWAYS = {'Early': ['V1', 'V2', 'V3'], 'Ventral': ['hV4', 'VO', 'LO', 'TO'], 'Dorsal': ['V3AB', 'IPS']}
HIGH, LOW, TARGET = '#C8102E', '#1F6FB2', '#2B2B2B'
COL = {'distractor_high': HIGH, 'distractor_low': LOW, 'target': TARGET, 'high': HIGH, 'low': LOW}
ROLE = {'target': 'Target', 'distractor_low': 'Low-value distractor', 'distractor_high': 'High-value distractor'}
ITEM_R, ECC = 1.0, 4.0
DIST_EDGES = (0, 1.5, 2.5, 3.5, 5, 8)      # aggregate_decoded / decode_trials DIST_BINS
NEAR, FAR = 2.0, 4.25                      # bin centres: near <= 2 deg, far >= 4.25 deg


def signflip_p(x, n_mc=100000, seed=0):
    """Exact (n <= 16 subjects) two-sided sign-flip p of the mean against 0."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    if len(x) <= 16:
        signs = np.array(list(itertools.product([1, -1], repeat=len(x))))
    else:
        signs = np.random.default_rng(seed).choice([1, -1], (n_mc, len(x)))
    return float((np.abs((signs * x).mean(1)) >= np.abs(x.mean()) - 1e-12).mean())


def fmt_p(p):
    return 'p = n/a' if not np.isfinite(p) else 'p < .001' if p < .001 else f'p = {p:.3f}'.replace('0.', '.', 1)


def group_maps(in_dir):
    """Per-subject aligned maps (mean over V1-V3 and held-out sessions), per distractor value."""
    z = np.load(Path(in_dir) / 'aligned_maps.npz')
    rows = [dict(zip(('subject', 'session', 'roi', 'decoder', 'value'), k.split('|')), key=k) for k in z.files]
    t = pd.DataFrame(rows).query("decoder == 'free' and roi in @MAP_ROIS")
    subjects = sorted(set(t.subject[t.value == 'low']) & set(t.subject[t.value == 'high']))
    return {v: np.array([np.mean([z[k] for k in t.key[(t.value == v) & (t.subject == s)]], 0) for s in subjects])
            for v in ('low', 'high')}


def ring_mean(maps):
    """Mean of each subject's map inside the distractor ring."""
    ax = (np.arange(maps.shape[-1]) + .5) * 10 / maps.shape[-1] - 5
    xx, yy = np.meshgrid(ax, ax)
    d = ECC / np.sqrt(2)
    return maps[:, np.hypot(xx - d, yy - d) <= ITEM_R].mean(1)


def clean_axes(ax):
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def display(ax, distractor=None, bar=True):
    """Miniature search display in the aligned frame (distractor location upper right)."""
    ax.set_xlim(-5, 5), ax.set_ylim(-5, 5), ax.set_aspect('equal')
    ax.add_patch(Rectangle((-5, -5), 10, 10, color=BG, lw=0, zorder=0))
    if bar:
        draw_bar(ax, 'vertical', BAR_POS[6], 0)
    ax.add_patch(Circle((0, 0), AP_R, fill=False, ec='w', lw=.5, zorder=2))
    for i in range(8):
        a = np.radians(i * 45)
        draw_item(ax, ECC * np.cos(a), ECC * np.sin(a), 0 if i == 3 else 90, ORANGE if i == distractor else GREY, False)
    ax.add_patch(Circle((0, 0), .16, color='k', zorder=5))
    clean_axes(ax)


def draw_example(ax):
    display(ax, distractor=1)
    d = ECC / np.sqrt(2)
    ax.text(d, d + 1.0, 'Distractor', color='w', ha='center', va='bottom', fontsize=6.5, zorder=6)
    ax.text(-d, d + .45, 'Target', color='w', ha='center', va='bottom', fontsize=6.5, zorder=6)
    ax.plot([1.4, 1.75], [-AP_R * .9, -4.25], color='w', lw=.6, zorder=6)
    ax.text(2.4, -4.3, 'Bar aperture', color='w', ha='center', va='top', fontsize=6.5, zorder=6)
    ax.set_title('Example display', pad=3)


def equation(fig, spec, left, right, ltitle, rtitle):
    """Two miniature displays with a minus sign between them."""
    g = spec.subgridspec(1, 3, width_ratios=[1, .2, 1], wspace=0)
    a, b = fig.add_subplot(g[0]), fig.add_subplot(g[2])
    left(a), right(b)
    a.set_title(ltitle, fontsize=6.5, pad=2), b.set_title(rtitle, fontsize=6.5, pad=2)
    m = fig.add_subplot(g[1])
    m.axis('off')
    m.text(.5, .5, '−', ha='center', va='center', fontsize=14, transform=m.transAxes)
    return a


def illustrate_items(fig, spec):
    d = ECC / np.sqrt(2)

    def ring(ax, col):
        ax.add_patch(Circle((d, d), 1.25, fill=False, ec=col, lw=1.4, zorder=7))

    def left(ax):
        display(ax, distractor=1), ring(ax, HIGH)

    def right(ax):
        display(ax, distractor=None), ring(ax, 'w')
    return equation(fig, spec, left, right, 'Target / distractor', 'Neutral')


def illustrate_bar(fig, spec):
    d = ECC / np.sqrt(2)
    axis = np.linspace(-5, 5, 400)
    xx, yy = np.meshgrid(axis, axis)
    on_bar = (np.abs(xx - BAR_POS[6]) <= BAR_W / 2) & (np.hypot(xx, yy) <= AP_R)
    bins = np.digitize(np.hypot(xx - d, yy - d), DIST_EDGES).astype(float)
    cmap = mpl.colors.ListedColormap(['#FDE725', '#7AD151', '#22A884', '#2A788E', '#414487'])

    def left(ax):
        display(ax, distractor=1, bar=False)
        ax.imshow(np.ma.masked_where(~on_bar, bins), origin='lower', extent=(-5, 5, -5, 5), cmap=cmap, vmin=.5, vmax=5.5,
                  interpolation='nearest', zorder=3)
        ax.text(BAR_POS[6] - .75, 2.1, 'Near', color='w', ha='right', va='center', fontsize=6, zorder=7)
        ax.text(BAR_POS[6] - .75, -2.3, 'Far', color='w', ha='right', va='center', fontsize=6, zorder=7)
        ax.set_xlim(-5, 5), ax.set_ylim(-5, 5)

    def right(ax):
        display(ax, distractor=None)
    return equation(fig, spec, left, right, 'Bar', 'No distractor')


def draw_map(ax, img, vmax, title, col, p):
    im = ax.imshow(img, origin='lower', extent=(-5, 5, -5, 5), cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='bilinear')
    ax.add_patch(Circle((0, 0), AP_R, fill=False, ec='.3', lw=.7))
    d = ECC / np.sqrt(2)
    ax.add_patch(Circle((d, d), ITEM_R, fill=False, ec=col, lw=1.3))
    for x, y in ((-d, d), (-d, -d)):
        ax.add_patch(Circle((x, y), ITEM_R, fill=False, ec=TARGET, lw=.9, ls=(0, (2, 1.5))))
    ax.plot(0, 0, '+', color='k', ms=4, mew=.8)
    ax.text(d, d + 1.2, 'Distractor', color=col, ha='center', va='bottom', fontsize=6.5)
    ax.text(-d, d + 1.2, 'Target', color=TARGET, ha='center', va='bottom', fontsize=6.5)
    ax.text(-d, -d - 1.2, 'Target', color=TARGET, ha='center', va='top', fontsize=6.5)
    ax.text(d, d - 1.2, fmt_p(p), color=col, ha='center', va='top', fontsize=6.5, weight='bold' if p < .05 else 'normal',
            path_effects=[patheffects.withStroke(linewidth=1.8, foreground='w')], zorder=8)
    ax.set_title(title, color=col, pad=3, fontsize=7.5)
    clean_axes(ax)
    return im


def robust_limits(values, centre_lo, centre_hi, pad=.12):
    """y-limits from the bulk of the subjects and the group means, not from single outliers."""
    lo, hi = np.nanpercentile(values, [5, 95])
    lo, hi = min(lo, np.nanmin(centre_lo), 0), max(hi, np.nanmax(centre_hi), 0)
    r = hi - lo
    return lo - pad * r, hi + pad * r


def panel_items(ax, items, rois):
    s = items.groupby(['subject', 'roi', 'role'], as_index=False).pct.mean()
    order = list(ROLE)
    offs = dict(zip(order, (-.265, 0, .265)))
    g = s.groupby(['roi', 'role']).pct
    centre_lo, centre_hi = g.mean() - g.sem(), g.mean() + g.sem()
    lo, hi = robust_limits(s.pct, centre_lo, centre_hi)
    span = hi - lo
    ymin, ymax = lo - .17 * span, hi + .17 * span       # room for the p-value rows
    inside = s[s.pct.between(lo, hi)]
    sns.stripplot(inside, x='roi', y='pct', hue='role', order=rois, hue_order=order, palette=COL, dodge=True, size=2.6,
                  alpha=.55, jitter=.08, legend=False, ax=ax, zorder=1)
    sns.pointplot(s, x='roi', y='pct', hue='role', order=rois, hue_order=order, palette=COL, dodge=.53, linestyle='none',
                  errorbar=('se', 1), markersize=4, err_kws={'linewidth': 1.2}, legend=False, ax=ax, zorder=3)
    ax.axhline(0, color='.35', lw=.8, ls=(0, (3, 2)))
    ax.set_ylim(ymin, ymax), ax.set_xlim(-.95, len(rois) - .45)
    for i, r in enumerate(rois):
        w = s[s.roi == r].pivot(index='subject', columns='role', values='pct')
        for role in order:
            x = i + offs[role]
            out = s[(s.roi == r) & (s.role == role) & ~s.pct.between(lo, hi)]
            for v in out.pct:                 # subjects beyond the axis stay visible, at its edge
                ax.plot(x, hi if v > hi else lo, '^' if v > hi else 'v', color=COL[role], ms=3, alpha=.7, mew=0, zorder=2)
            p = signflip_p(w[role])
            ax.text(x, lo - .1 * span, fmt_p(p).replace('p = ', ''), color=COL[role], ha='center', va='center', fontsize=5.8,
                    weight='bold' if p < .05 else 'normal')
        p = signflip_p((w.distractor_high - w.distractor_low).dropna())
        y = hi + .05 * span
        ax.plot([i, i, i + .265, i + .265], [y - .025 * span, y, y, y - .025 * span], color='.25', lw=.7)
        ax.text(i + .1325, y + .015 * span, fmt_p(p), ha='center', va='bottom', fontsize=5.8,
                weight='bold' if p < .05 else 'normal')
    ax.text(-.93, lo - .1 * span, 'p vs 0', ha='left', va='center', fontsize=5.8, color='.3')
    ax.set(xlabel='', ylabel='Change (%)')
    sns.despine(ax=ax, offset=4, trim=False)


def panel_bar(fig, spec, bars, rois):
    d = bars.query("aligned_to == 'distractor'").groupby(['subject', 'roi', 'value', 'distance'], as_index=False).pct.mean()
    g = d.groupby(['roi', 'value', 'distance']).pct
    lo, hi = robust_limits(d.pct, (g.mean() - g.sem()).quantile(.1), (g.mean() + g.sem()).quantile(.9), pad=.05)
    span = hi - lo
    sub = spec.subgridspec(1, len(rois), wspace=.1)
    axes = []
    for i, r in enumerate(rois):
        ax = fig.add_subplot(sub[i], sharey=axes[0] if axes else None)
        axes.append(ax)
        q = d[d.roi == r]
        sns.lineplot(q, x='distance', y='pct', hue='value', hue_order=['low', 'high'], palette=COL, errorbar=('se', 1),
                     marker='o', markersize=3, markeredgewidth=0, lw=1.2, legend=False, ax=ax)
        ax.axhline(0, color='.35', lw=.8, ls=(0, (3, 2)))
        ax.set_title(r, pad=2)
        ax.set(xlabel='', ylabel='Change (%)' if i == 0 else '')
        if i:
            ax.tick_params(labelleft=False)
        m = q.groupby(['subject', 'distance']).pct.mean().unstack()
        near = m[[c for c in m.columns if c <= NEAR]].mean(1) - m[[c for c in m.columns if c >= FAR]].mean(1)
        w = q.groupby(['subject', 'value']).pct.mean().unstack()
        p1, p2 = signflip_p(near), signflip_p((w.high - w.low).dropna())
        ax.text(.03, .99, f'Near − far: {fmt_p(p1)}', transform=ax.transAxes, va='top', fontsize=5.6,
                weight='bold' if p1 < .05 else 'normal')
        ax.text(.03, .895, f'High − low: {fmt_p(p2)}', transform=ax.transAxes, va='top', fontsize=5.6,
                weight='bold' if p2 < .05 else 'normal')
        sns.despine(ax=ax, offset=3)
    axes[0].set_ylim(lo, hi + .32 * span)
    return axes


def main(in_dir, out):
    items = pd.read_csv(Path(in_dir) / 'item_gain.tsv', sep='\t', dtype={'subject': str}).query("decoder == 'outline'")
    bars = pd.read_csv(Path(in_dir) / 'bar_distance.tsv', sep='\t', dtype={'subject': str}).query("decoder == 'outline'")
    maps = group_maps(in_dir)
    per_roi = items.groupby('roi').subject.nunique()
    rois = [r for r in ROIS if per_roi.get(r, 0) >= min(2, items.subject.nunique())]
    n = items.subject.nunique()

    fig = plt.figure(figsize=(7.25, 7.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.3, 1.05, .95], hspace=.52, left=.065, right=.985, top=.895, bottom=.07)
    fig.text(.01, .985, f'Prototype, n = {n} subjects, decoding still running', fontsize=9, weight='bold', va='top')

    # a, b: example display and aligned maps
    top = gs[0].subgridspec(1, 6, width_ratios=[1, 1, 1, 1, .05, .3], wspace=.07)
    ex = fig.add_subplot(top[0])
    draw_example(ex)
    ex.text(-.1, 1.17, 'a', transform=ex.transAxes, fontsize=10, weight='bold')
    low, high = maps['low'], maps['high']
    vmax = np.percentile(np.abs(np.r_[low.mean(0).ravel(), high.mean(0).ravel()]), 99.5)
    a0 = fig.add_subplot(top[1])
    draw_map(a0, low.mean(0), vmax, 'Low value − no distractor', LOW, signflip_p(ring_mean(low)))
    a1 = fig.add_subplot(top[2])
    draw_map(a1, high.mean(0), vmax, 'High value − no distractor', HIGH, signflip_p(ring_mean(high)))
    im = draw_map(fig.add_subplot(top[3]), (high - low).mean(0), vmax, 'High − low value', TARGET,
                  signflip_p(ring_mean(high - low)))
    cax = fig.add_subplot(top[4])
    cb = fig.colorbar(im, cax=cax, format='%.4f')
    cb.outline.set_visible(False), cb.ax.tick_params(labelsize=6)
    cax.set_title('Δ intensity', fontsize=6.5, pad=3, loc='left')
    a0.text(-.04, 1.17, 'b', transform=a0.transAxes, fontsize=10, weight='bold')
    a1.text(.5, 1.17, f'Decoded input in {"/".join(MAP_ROIS)}: distractor trials minus no-distractor trials',
            transform=a1.transAxes, ha='center', fontsize=8)

    # c: items
    row = gs[1].subgridspec(1, 2, width_ratios=[1, 3.3], wspace=.3)
    a = illustrate_items(fig, row[0])
    a.text(-.2, 1.3, 'c', transform=a.transAxes, fontsize=10, weight='bold')
    ax = fig.add_subplot(row[1])
    panel_items(ax, items, rois)
    ax.set_title('Item intensity compared with the same item when neutral, at the same bar position', loc='left', pad=9, fontsize=8)
    for i, r in enumerate(ROLE):
        ax.text(.005 + i * .2, 1.0, ROLE[r], color=COL[r], transform=ax.transAxes, fontsize=6.5, va='bottom')

    # d: bar
    row = gs[2].subgridspec(1, 2, width_ratios=[1, 3.3], wspace=.3)
    a = illustrate_bar(fig, row[0])
    a.text(-.2, 1.3, 'd', transform=a.transAxes, fontsize=10, weight='bold')
    axes = panel_bar(fig, row[1], bars, rois)
    axes[0].text(0, 1.2, 'Bar intensity compared with no-distractor trials, by distance from the distractor',
                 transform=axes[0].transAxes, fontsize=8, va='bottom')
    box = axes[2].get_position()
    fig.text(box.x0 + box.width / 2, .012, 'Distance of bar from distractor (deg)', ha='center', fontsize=8)
    fig.text(.02, .012, 'High value', color=HIGH, ha='left', fontsize=8)
    fig.text(.115, .012, 'Low value', color=LOW, ha='left', fontsize=8)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f'{out}.pdf')
    fig.savefig(f'{out}.png', dpi=200)
    print(f'{out}.pdf', f'n = {n} subjects', sorted(items.subject.unique()))
    plot_pathways(items, bars, n, out)


def plot_pathways(items, bars, n, out):
    """Items and bar responses averaged over the areas of each visual pathway (within subject)."""
    group = {r: g for g, rs in PATHWAYS.items() for r in rs}
    have = {g: [r for r in rs if r in set(items.roi)] for g, rs in PATHWAYS.items()}
    names = {g: f'{g} ({", ".join(rs)})' for g, rs in have.items() if rs}
    label = lambda d: d.assign(roi=d.roi.map(group).map(names)).dropna(subset=['roi'])  # noqa: E731
    it = label(items).groupby(['subject', 'roi', 'role'], as_index=False).pct.mean()
    ba = label(bars).groupby(['subject', 'roi', 'aligned_to', 'value', 'distance'], as_index=False).pct.mean()
    order = list(names.values())
    fig = plt.figure(figsize=(7.25, 4.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 1], hspace=.55, left=.08, right=.985, top=.87, bottom=.1)
    fig.text(.01, .985, f'Prototype, n = {n} subjects, decoding still running', fontsize=9, weight='bold', va='top')
    ax = fig.add_subplot(gs[0])
    panel_items(ax, it, order)
    ax.set_title('Item intensity compared with the same item when neutral, at the same bar position', loc='left', pad=9, fontsize=8)
    for i, r in enumerate(ROLE):
        ax.text(.005 + i * .2, 1.0, ROLE[r], color=COL[r], transform=ax.transAxes, fontsize=6.5, va='bottom')
    ax.text(-.07, 1.14, 'a', transform=ax.transAxes, fontsize=10, weight='bold')
    axes = panel_bar(fig, gs[1], ba, order)
    axes[0].text(0, 1.2, 'Bar intensity compared with no-distractor trials, by distance from the distractor',
                 transform=axes[0].transAxes, fontsize=8, va='bottom')
    axes[0].text(-.2, 1.2, 'b', transform=axes[0].transAxes, fontsize=10, weight='bold')
    fig.text(.53, .015, 'Distance of bar from distractor (deg)', ha='center', fontsize=8)
    fig.text(.02, .015, 'High value', color=HIGH, fontsize=8), fig.text(.115, .015, 'Low value', color=LOW, fontsize=8)
    fig.savefig(f'{out}_pathways.pdf')
    fig.savefig(f'{out}_pathways.png', dpi=200)
    print(f'{out}_pathways.pdf', names)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in-dir', default='notes/data/decoded_gain')
    ap.add_argument('--out', default='notes/figures/decoded_gain_prototype')
    a = ap.parse_args()
    main(a.in_dir, a.out)
