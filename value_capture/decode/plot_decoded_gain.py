"""Figure of the model-free gain readout: aligned decoded maps, item gain and bar gain around the distractor.

Reads the summaries of aggregate_decoded.py. Conditions are averaged within subject first (over held-out sessions).

    python value_capture/decode/plot_decoded_gain.py --in-dir notes/data/decoded_gain --out notes/figures/decoded_gain_prototype
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Circle

mpl.rcParams.update({'font.family': 'Helvetica', 'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': .8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'xtick.major.size': 3, 'ytick.major.size': 3,
                     'xtick.major.width': .8, 'ytick.major.width': .8, 'pdf.fonttype': 42, 'savefig.dpi': 300})
ROIS = ['V1', 'V2', 'V3', 'hV4', 'LO', 'V3AB']
MAP_ROIS = ['V1', 'V2', 'V3']
HIGH, LOW, TARGET = '#C8102E', '#1F6FB2', '#2B2B2B'
COL = {'distractor_high': HIGH, 'distractor_low': LOW, 'target': TARGET, 'high': HIGH, 'low': LOW}
ROLE = {'target': 'Target', 'distractor_low': 'Low-value distractor', 'distractor_high': 'High-value distractor'}
ITEM_R, ECC = 1.0, 4.0


def group_maps(in_dir):
    z = np.load(Path(in_dir) / 'aligned_maps.npz')
    rows = [dict(zip(('subject', 'session', 'roi', 'decoder', 'value'), k.split('|')), key=k) for k in z.files]
    t = pd.DataFrame(rows).query("decoder == 'free' and roi in @MAP_ROIS")
    out = {}
    for value, g in t.groupby('value'):
        out[value] = np.mean([np.mean([z[k] for k in s.key], 0) for _, s in g.groupby('subject')], 0)
    return out, t.subject.nunique()


def draw_map(ax, img, vmax, title, col):
    im = ax.imshow(img, origin='lower', extent=(-5, 5, -5, 5), cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='bilinear')
    d = ECC / np.sqrt(2)
    ax.add_patch(Circle((d, d), ITEM_R, fill=False, ec=col, lw=1.3))
    for x, y in ((-d, d), (-d, -d)):
        ax.add_patch(Circle((x, y), ITEM_R, fill=False, ec=TARGET, lw=.9, ls=(0, (2, 1.5))))
    ax.plot(0, 0, '+', color='k', ms=4, mew=.8)
    ax.text(d, d + 1.2, 'Distractor', color=col, ha='center', va='bottom', fontsize=6.5)
    ax.text(-d, d + 1.2, 'Target', color=TARGET, ha='center', va='bottom', fontsize=6.5)
    ax.text(-d, -d - 1.2, 'Target', color=TARGET, ha='center', va='top', fontsize=6.5)
    ax.set_title(title, color=col, pad=3)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def main(in_dir, out):
    items = pd.read_csv(Path(in_dir) / 'item_gain.tsv', sep='\t', dtype={'subject': str}).query("decoder == 'outline'")
    bars = pd.read_csv(Path(in_dir) / 'bar_distance.tsv', sep='\t', dtype={'subject': str}).query("decoder == 'outline'")
    maps, n_map = group_maps(in_dir)
    per_roi = items.groupby('roi').subject.nunique()
    rois = [r for r in ROIS if per_roi.get(r, 0) >= min(2, items.subject.nunique())]   # no single-subject ROIs
    n = items.subject.nunique()

    fig = plt.figure(figsize=(7.25, 7.3))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.75, 1, 1], hspace=.42, left=.085, right=.985, top=.925, bottom=.07)
    fig.text(.01, .985, f'Prototype, n = {n} subject{"s" if n != 1 else ""}, decoding still running', fontsize=9, weight='bold', va='top')

    # a: aligned maps
    top = gs[0].subgridspec(1, 5, width_ratios=[1, 1, 1, .045, .27], wspace=.06)
    vmax = np.percentile(np.abs(np.r_[maps['low'].ravel(), maps['high'].ravel()]), 99.5)
    a0 = fig.add_subplot(top[0])
    draw_map(a0, maps['low'], vmax, 'Low-value distractor', LOW)
    draw_map(fig.add_subplot(top[1]), maps['high'], vmax, 'High-value distractor', HIGH)
    im = draw_map(fig.add_subplot(top[2]), maps['high'] - maps['low'], vmax, 'High − low value', TARGET)
    cb = fig.colorbar(im, cax=fig.add_subplot(top[3]), format='%.3f')
    cb.set_label('Decoded intensity vs no distractor', fontsize=7, labelpad=2), cb.outline.set_visible(False)
    a0.text(-.13, 1.07, 'a', transform=a0.transAxes, fontsize=10, weight='bold')
    a0.text(-.03, .5, f'{"/".join(MAP_ROIS)}, aligned to distractor', transform=a0.transAxes, rotation=90, va='center', ha='right', fontsize=7.5)

    # b: item gain
    ax = fig.add_subplot(gs[1])
    s = items.groupby(['subject', 'roi', 'role'], as_index=False).pct.mean()
    order = list(ROLE)
    sns.stripplot(s, x='roi', y='pct', hue='role', order=rois, hue_order=order, palette=COL, dodge=True, size=2.6,
                  alpha=.55, jitter=.08, legend=False, ax=ax, zorder=1)
    sns.pointplot(s, x='roi', y='pct', hue='role', order=rois, hue_order=order, palette=COL, dodge=.53, linestyle='none',
                  errorbar=('se', 1), markersize=4, err_kws={'linewidth': 1.2}, legend=False, ax=ax, zorder=3)
    ax.axhline(0, color='.35', lw=.8, ls=(0, (3, 2)))
    ax.set(xlabel='', ylabel='Item intensity vs neutral (%)')
    for i, r in enumerate(order):
        ax.text(.01 + i * .17, 1.0, ROLE[r], color=COL[r], transform=ax.transAxes, fontsize=7, va='bottom')
    ax.text(-.07, 1.06, 'b', transform=ax.transAxes, fontsize=10, weight='bold')
    sns.despine(ax=ax, offset=4, trim=False)

    # c: bar gain by distance from the distractor
    sub = gs[2].subgridspec(1, len(rois), wspace=.12)
    d = bars.query("aligned_to == 'distractor'").groupby(['subject', 'roi', 'value', 'distance'], as_index=False).pct.mean()
    axes = []
    for i in range(len(rois)):
        axes.append(fig.add_subplot(sub[i], sharey=axes[0] if axes else None))
    for i, (r, ax) in enumerate(zip(rois, axes)):
        sns.lineplot(d.query('roi == @r'), x='distance', y='pct', hue='value', hue_order=['low', 'high'],
                     palette=COL, errorbar=('se', 1), marker='o', markersize=3, markeredgewidth=0, lw=1.2, legend=False, ax=ax)
        ax.axhline(0, color='.35', lw=.8, ls=(0, (3, 2)))
        ax.set_title(r, pad=2)
        ax.set(xlabel='', ylabel='Bar vs no distractor (%)' if i == 0 else '')
        if i:
            ax.tick_params(labelleft=False)
        sns.despine(ax=ax, offset=3)
    fig.text(.53, .012, 'Distance of bar from distractor (deg)', ha='center', fontsize=8)
    axes[0].text(-.27, 1.1, 'c', transform=axes[0].transAxes, fontsize=10, weight='bold')
    axes[-1].text(0, 1.03, 'High value', color=HIGH, transform=axes[-1].transAxes, ha='left', fontsize=7)
    axes[-1].text(1, 1.03, 'Low value', color=LOW, transform=axes[-1].transAxes, ha='right', fontsize=7)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f'{out}.pdf')
    fig.savefig(f'{out}.png', dpi=200)
    print(f'{out}.pdf', f'n = {n} subjects', sorted(items.subject.unique()), 'maps n =', n_map)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in-dir', default='notes/data/decoded_gain')
    ap.add_argument('--out', default='notes/figures/decoded_gain_prototype')
    a = ap.parse_args()
    main(a.in_dir, a.out)
