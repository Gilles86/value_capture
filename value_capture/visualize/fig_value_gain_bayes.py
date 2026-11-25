"""Figure: the distractor value effect on the gain field, Bayesian group models with a noise level per participant.

a  Every visual area. Dots: participants' high - low gain (all-runs fit), shaded by precision (1 / SE of the
   per-run gains), so the noisy participants are pale. Purple: area mean and 95% HDI of the SE-weighted meta-analysis
   over all areas (``gainfield/meta_value_gain.py``, retsupp model). Open circle: the per-run model with a residual
   scale per participant (``gainfield/hierarchical_value_gain.py``). Right: pathway means and the overall mean.
b  hV4 per participant, sorted: estimate +- 95% interval from the per-run SE (grey) and the participant's slope in the
   hierarchical model (purple), which shrinks noisy participants towards the group.
c  hV4 across model variants.

Primary fit: gain fields on DoG pRFs fitted jointly, points trials excluded, one width per effect.

    python value_capture/visualize/fig_value_gain_bayes.py
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import numpy as np
import pandas as pd

from value_capture.visualize.fig_value_gain import DOUBLE, FG, HIGH, MUTED, offset_spines

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'notes' / 'data'
OUT = ROOT / 'notes' / 'figures'
FIT = 'gain_fields_dog_fbevents_free_exclude_widths'
GROUPS = [('Early', ['V1', 'V2', 'V3']), ('Dorsal', ['V3AB', 'IPS']), ('Occipito-temporal', ['hV4', 'LO', 'TO', 'VO'])]
LOW = '#009E73'


def read(name):
    f = DATA / name
    return pd.read_csv(f, sep='\t', dtype={'subject': str}) if f.exists() else pd.DataFrame()


def clip_scatter(ax, x, v, lo, hi, **kw):
    inside = (v >= lo) & (v <= hi)
    c = np.asarray(kw.pop('c'))
    ax.scatter(x[inside], v[inside], c=c[inside], **kw)
    for xx, vv, cc in zip(x[~inside], v[~inside], c[~inside]):
        ax.scatter(xx, hi if vv > hi else lo, marker='^' if vv > hi else 'v', c=[cc], clip_on=False,
                   **{k: w for k, w in kw.items() if k != 'marker'})


def panel_areas(ax, meta, data, hier):
    pos, x = {}, 0.
    for _, rois in GROUPS:
        for r in rois:
            if r in set(data.roi):
                pos[r] = x
                x += 1
        x += .7
    gx = x + .3
    m = meta[meta.variant == 'normal']
    lo, hi = -260, 360
    rng = np.random.default_rng(1)
    for r, xr in pos.items():
        d = data[data.roi == r]
        prec = 1 / d.se.to_numpy()
        a = .15 + .85 * (prec / prec.max()) ** .7
        cols = np.array([mpl.colors.to_rgba(FG, aa) for aa in a])
        clip_scatter(ax, xr - .14 + rng.uniform(-.1, .1, len(d)), d.y.to_numpy(), lo, hi, c=cols, s=9, lw=0, zorder=2)
        row = m[(m.level == 'area') & (m.unit == r)]
        if len(row):
            row = row.iloc[0]
            col = HIGH if row.hdi_low > 0 else (LOW if row.hdi_high < 0 else FG)
            ax.plot([xr + .12] * 2, [row.hdi_low, row.hdi_high], color=col, lw=1.6, solid_capstyle='butt', zorder=3)
            ax.plot(xr + .12, row['mean'], 'o', ms=4.2, color=col, zorder=4)
            ax.text(xr, hi * 1.02, f'P = {row.p_gt0:.3f}'.replace('0.', '.') if .001 <= row.p_gt0 <= .999 else
                    ('P > .999' if row.p_gt0 > .999 else 'P < .001'), ha='center', va='bottom', fontsize=5.5,
                    color=col if col != FG else '.4', fontweight='bold' if col != FG else 'normal')
        h = hier[hier.roi == r]
        if len(h):
            ax.plot(xr + .3, h['mean'].iloc[0], 'o', ms=3.6, mfc='white', mec=FG, mew=.8, zorder=4)
    # pathway and overall means
    gpos = {}
    for i, (g, _) in enumerate(GROUPS + [('Overall', None)]):
        unit, level = (g, 'group') if g != 'Overall' else ('all', 'overall')
        row = m[(m.level == level) & (m.unit == unit)]
        if not len(row):
            continue
        row, xg = row.iloc[0], gx + i * .9
        gpos[g] = xg
        ax.plot([xg] * 2, [max(row.hdi_low, lo), min(row.hdi_high, hi)], color=FG, lw=1.6, solid_capstyle='butt')
        ax.plot(xg, row['mean'], 'D' if g == 'Overall' else 's', ms=3.8, color=FG)
    ax.axhline(0, color=MUTED, lw=.7, ls='--', zorder=0)
    ax.set_ylim(lo, hi)
    offset_spines(ax)
    ticks = list(pos.values()) + list(gpos.values())
    labels = list(pos) + [g.replace('Occipito-temporal', 'Occ.-temp.') for g in gpos]
    ax.set_xticks(ticks, labels, fontsize=6.5)
    plt.setp(ax.get_xticklabels()[len(pos):], rotation=30, ha='right', rotation_mode='anchor')
    ax.set_xlim(-.6, ticks[-1] + .5)
    ax.set_ylabel('High − low value\ndistractor gain (%)')
    ax.text(-.6, 1.02, 'P(high > low):', transform=mtrans.blended_transform_factory(ax.transData, ax.transAxes),
            fontsize=5.5, color='.4', ha='right', va='bottom')
    tr = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
    for g, rois in GROUPS:
        xs = [pos[r] for r in rois if r in pos]
        if xs:
            ax.plot([xs[0] - .35, xs[-1] + .35], [1.13, 1.13], transform=tr, color=MUTED, lw=.7, clip_on=False)
            ax.text(np.mean(xs), 1.15, g, transform=tr, ha='center', va='bottom', fontsize=6.5, color='.3')
    ax.plot([gx - .35, ticks[-1] + .35], [1.13, 1.13], transform=tr, color=MUTED, lw=.7, clip_on=False)
    ax.text((gx + ticks[-1]) / 2, 1.15, 'Pathway means', transform=tr, ha='center', va='bottom', fontsize=6.5,
            color='.3')
    # key
    k = dict(transform=ax.transAxes, clip_on=False)
    y0, x0 = .97, .735
    for i, (a, lab) in enumerate([(1., 'Participant, precise'), (.2, 'Participant, noisy')]):
        ax.scatter([x0 + i * .0], [y0 - i * .075], s=9, color=mpl.colors.to_rgba(FG, a), lw=0, **k)
        ax.text(x0 + .012, y0 - i * .075, lab, fontsize=6, va='center', transform=ax.transAxes)
    ax.plot([x0 + .2] * 2, [y0 - .04, y0 + .04], color=HIGH, lw=1.6, **k)
    ax.plot(x0 + .2, y0, 'o', ms=4.2, color=HIGH, **k)
    ax.text(x0 + .212, y0, 'Meta-analysis, SE-weighted', fontsize=6, va='center', transform=ax.transAxes)
    ax.plot(x0 + .2, y0 - .075, 'o', ms=3.6, mfc='white', mec=FG, mew=.8, **k)
    ax.text(x0 + .212, y0 - .075, 'Per-run model, noise per participant', fontsize=6, va='center',
            transform=ax.transAxes)


def panel_subjects(ax, data, subj):
    d = data[data.roi == 'hV4'].merge(subj[subj.roi == 'hV4'], on='subject', suffixes=('', '_h'))
    d = d.sort_values('y').reset_index(drop=True)
    lo, hi = -300, 700
    for i, r in d.iterrows():
        a, b = r.y - 1.96 * r.se, r.y + 1.96 * r.se
        ax.plot([max(a, lo), min(b, hi)], [i - .15] * 2, color=MUTED, lw=1.1, solid_capstyle='butt')
        if b > hi:
            ax.plot(hi, i - .15, '>', ms=3, color=MUTED, clip_on=False)
        ax.plot(min(r.y, hi), i - .15, 'o', ms=3, color='.45')
        ax.plot([r.hdi_low, r.hdi_high], [i + .15] * 2, color=HIGH, lw=1.1, solid_capstyle='butt')
        ax.plot(r['mean'], i + .15, 'o', ms=3, color=HIGH)
    ax.axvline(0, color=MUTED, lw=.7, ls='--', zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_yticks(range(len(d)), [f'{s}' for s in d.subject], fontsize=5.5)
    ax.set_ylabel('Participant (sorted)')
    ax.set_xlabel('hV4 high − low gain (%)')
    offset_spines(ax)
    ax.text(.97, .2, 'Own data ± 1.96 SE', color='.45', fontsize=6, ha='right', transform=ax.transAxes)
    ax.text(.97, .1, 'Hierarchical model', color=HIGH, fontsize=6, ha='right', transform=ax.transAxes)


def panel_variants(ax, meta, hier, shared):
    rows = []
    h = hier[hier.roi == 'hV4'].iloc[0]
    rows.append(('Per-run, noise per participant', h['mean'], h.hdi_low, h.hdi_high, h.p_high_gt_low, True))
    if len(shared):
        s = shared[shared.roi == 'hV4'].iloc[0]
        rows.append(('Per-run, one noise level', s['mean'], s.hdi_low, s.hdi_high, s.p_high_gt_low, False))
    for v, lab in [('normal', 'Meta-analysis, SE-weighted'), ('t', 'Meta-analysis, Student-t'),
                   ('nose', 'Meta-analysis, unweighted')]:
        r = meta[(meta.variant == v) & (meta.unit == 'hV4')]
        if len(r):
            r = r.iloc[0]
            rows.append((lab, r['mean'], r.hdi_low, r.hdi_high, r.p_gt0, v == 'normal'))
    for i, (lab, m, a, b, p, main) in enumerate(rows[::-1]):
        col = HIGH if main else FG
        ax.plot([a, b], [i, i], color=col, lw=1.6 if main else 1.1, solid_capstyle='butt')
        ax.plot(m, i, 'o', ms=4 if main else 3.3, color=col)
        ax.text(1.02, i, f'P = {p:.3f}'.replace('0.', '.'), transform=mtrans.blended_transform_factory(
            ax.transAxes, ax.transData), va='center', fontsize=6, color=col)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]], fontsize=6.5)
    ax.axvline(0, color=MUTED, lw=.7, ls='--', zorder=0)
    ax.set_xlabel('hV4 high − low gain (%)')
    ax.set_ylim(-.6, len(rows) - .4)
    offset_spines(ax)


def main():
    meta = read(f'meta_value_gain_{FIT}.tsv')
    data = read(f'meta_value_gain_{FIT}_data.tsv')
    hier = read(f'hierarchical_value_gain_{FIT}.tsv')
    subj = read(f'hierarchical_value_gain_{FIT}_subjects.tsv')
    shared = read(f'hierarchical_value_gain_{FIT}_sharednoise.tsv')
    fig = plt.figure(figsize=(DOUBLE, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], width_ratios=[1, 1.15], hspace=.75, wspace=.75)
    axa = fig.add_subplot(gs[0, :])
    axb = fig.add_subplot(gs[1, 0])
    axc = fig.add_subplot(gs[1, 1])
    panel_areas(axa, meta, data, hier)
    panel_subjects(axb, data, subj)
    panel_variants(axc, meta, hier, shared)
    axa.set_title('The high-value distractor gets a larger gain in hV4 and a smaller one in LO', loc='left',
                  fontsize=7.5, pad=26)
    axb.set_title('Noisy participants count for less', loc='left', fontsize=7.5)
    axc.set_title('hV4: every model agrees', loc='left', fontsize=7.5)
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    for ax, l in [(axa, 'a'), (axb, 'b'), (axc, 'c')]:
        bb = inv.transform(ax.get_tightbbox(fig.canvas.get_renderer()))     # incl. tick labels and title
        fig.text(bb[0, 0] - .01, bb[1, 1], l, fontsize=10, fontweight='bold', va='top', ha='right')
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'fig_value_gain_bayes.{ext}', dpi=300, bbox_inches='tight')
    print(OUT / 'fig_value_gain_bayes.png')


if __name__ == '__main__':
    main()
