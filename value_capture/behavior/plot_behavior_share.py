"""Shareable behaviour figure (main study), in the retsupp paper-figure style.

a, response time per condition: subject dots (no connecting lines), mean ± within-subject SEM
   (Cousineau-Morey), brackets with the mean within-subject difference and one-sided p.
b, within-subject differences as horizontal strips: subject dots, mean (diamond) with its
   bootstrapped 95% CI, count of subjects in the predicted direction.
c, accuracy, as a.

    python value_capture/behavior/plot_behavior_share.py
Reads notes/data/behavior_trials_main.tsv (from fit_logrt_bambi.py export).
One-sided tests: directional predictions (distractors cost, high value more than low).
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
mpl.rcParams.update({
    'font.family': 'Helvetica', 'font.size': 7, 'axes.labelsize': 7.5, 'xtick.labelsize': 7,
    'ytick.labelsize': 7, 'axes.linewidth': .8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': .8, 'ytick.major.width': .8, 'pdf.fonttype': 42, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': .03})
FG, MUTED = '#1F2933', '#9AA5B1'
COL = {'none': MUTED, 'low': '#4A7FB5', 'high': '#B2182B'}
LAB = {'none': 'Absent', 'low': 'Low', 'high': 'High'}
ORDER = ['none', 'low', 'high']


def stars(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else f'p = {p:.2f}'.replace('0.', '.')


def ws_sem(w):
    k = w.shape[1]
    return (w.sub(w.mean(1), axis=0) + w.values.mean()).sem() * np.sqrt(k / (k - 1))


def cond_panel(ax, w, ylim, yticks, pairs, unit, seed=3):
    lo, hi = ylim
    m, se = w.mean(), ws_sem(w)
    rng = np.random.default_rng(seed)
    for i, c in enumerate(ORDER):
        v = w[c].to_numpy()
        xx = i + rng.uniform(-.14, .14, len(v))
        ok, over, under = (v >= lo) & (v <= hi), v > hi, v < lo
        ax.scatter(xx[ok], v[ok], s=9, color=COL[c], alpha=.35, lw=0, zorder=1)
        ax.scatter(xx[over], np.full(over.sum(), hi), s=10, marker='^', color=COL[c], alpha=.6, lw=0, clip_on=False)
        ax.scatter(xx[under], np.full(under.sum(), lo), s=10, marker='v', color=COL[c], alpha=.6, lw=0, clip_on=False)
        ax.errorbar(i, m[c], se[c], color=COL[c], marker='o', ms=6, mec='w', mew=.7, lw=1.4, capsize=0, zorder=4)
    ax.plot(range(3), m.values, color=FG, lw=.8, zorder=3)
    span = hi - lo
    for n, (a, b, alt) in enumerate(pairs):
        d = w[a] - w[b]
        p = stats.ttest_1samp(d, 0, alternative=alt).pvalue
        i, j = ORDER.index(a), ORDER.index(b)
        y = hi + (.12 + n * .14) * span
        ax.plot([min(i, j), max(i, j)], [y, y], color=FG, lw=.7, clip_on=False)
        ax.text((i + j) / 2, y + .01 * span, f'{d.mean():+.{1 if unit == "%" else 0}f} {unit} {stars(p)}'.replace('-', '−'),
                ha='center', va='bottom', fontsize=6.5, color=FG, clip_on=False)
    ax.set_xticks(range(3), [LAB[c] for c in ORDER])
    ax.set_xlabel('Distractor value', labelpad=2)
    ax.set_xlim(-.45, 2.45)
    ax.set_ylim(lo, hi)
    ax.set_yticks(yticks)
    for s in ('left', 'bottom'):
        ax.spines[s].set_position(('outward', 4))


def strip(ax, d, color, label, lim, word, alt, seed=0):
    v = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, (10000, len(v))).mean(1)
    m, (clo, chi) = v.mean(), np.percentile(boots, [2.5, 97.5])
    p = stats.ttest_1samp(v, 0, alternative=alt).pvalue
    yy = rng.uniform(-.32, .32, len(v))
    ax.axvline(0, color='.75', lw=.6, zorder=0)
    ax.scatter(np.clip(v, *lim), yy, s=7, color=color, alpha=.45, lw=0, zorder=2, clip_on=False)
    ax.plot([clo, chi], [0, 0], color=FG, lw=1.4, solid_capstyle='butt', zorder=3)
    ax.plot(m, 0, 'D', ms=4.2, color=color, mec=FG, mew=.8, zorder=4)
    good = (v > 0) if alt == 'greater' else (v < 0)
    ax.text(0, 1.15, f'{label}  {stars(p)}', transform=ax.transAxes, fontsize=6.3, color=FG, va='bottom')
    ax.text(1, 1.15, f'{good.sum()}/{len(v)} {word}', transform=ax.transAxes, fontsize=6.3, color=FG,
            ha='right', va='bottom')
    ax.set_xlim(*lim)
    ax.set_ylim(-.5, .5)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_position(('outward', 2))


def load_behaviour():
    df = pd.read_csv(ROOT / 'notes/data/behavior_trials_main.tsv', sep='\t', dtype={'subject': str})
    df['correct'] = df['correct'].map({True: 1., False: 0., 'True': 1., 'False': 0.})
    return df


if __name__ == '__main__':
    df = pd.read_csv(ROOT / 'notes/data/behavior_trials_main.tsv', sep='\t', dtype={'subject': str})
    df['correct'] = df['correct'].map({True: 1., False: 0., 'True': 1., 'False': 0.})
    n = df.subject.nunique()
    rt = (df[df.correct == 1].groupby(['subject', 'cond']).rt.mean() * 1000).unstack()[ORDER]
    acc = (df.groupby(['subject', 'cond']).correct.mean() * 100).unstack()[ORDER]

    W, H = 7.09, 2.35
    fig = plt.figure(figsize=(W, H))
    at = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])  # noqa: E731 (inches)
    Y0, PH = .45, 1.25

    ax_a = at(.45, Y0, 1.45, PH)
    cond_panel(ax_a, rt, (700, 1050), [700, 800, 900, 1000],
               [('low', 'none', 'greater'), ('high', 'none', 'greater'), ('high', 'low', 'greater')], 'ms')
    ax_a.set_ylabel('Response time (ms)')

    x0, SW, SH = 2.55, 1.55, .2
    for k, (lab, d, c, word, alt) in enumerate([
            ('Low − absent', rt.low - rt.none, COL['low'], 'slower', 'greater'),
            ('High − absent', rt.high - rt.none, COL['high'], 'slower', 'greater'),
            ('High − low', rt.high - rt.low, COL['high'], 'slower', 'greater')]):
        s = at(x0, Y0 + PH - .3 - k * .5, SW, SH)
        strip(s, d, c, lab, (-60, 110), word, alt, seed=k)
        s.set_xticks([-50, 0, 50, 100])
        if k < 2:
            s.set_xticklabels([])
        else:
            s.set_xlabel('RT difference (ms)', labelpad=1)
    fig.text((x0 + SW / 2) / W, (Y0 + PH + .12) / H, 'Within subject', fontsize=6.5, color='.4', ha='center')

    ax_c = at(4.75, Y0, 1.45, PH)
    cond_panel(ax_c, acc, (85, 100), [85, 90, 95, 100],
               [('low', 'none', 'less'), ('high', 'none', 'less'), ('high', 'low', 'less')], '%')
    ax_c.set_ylabel('Accuracy (% correct)')

    for a, l, x in ((ax_a, 'a', -.42), (ax_c, 'c', -.42)):
        a.text(x, 1.45, l, transform=a.transAxes, fontsize=8, weight='bold', va='bottom')
    fig.text((x0 - .25) / W, (Y0 + PH + .12) / H + .0, 'b', fontsize=8, weight='bold')
    out = ROOT / 'notes/figures/behavior_share'
    fig.savefig(f'{out}.pdf')
    fig.savefig(f'{out}.png', dpi=300)
    print(f"""Caption: Value-driven capture, main study (n = {n}, sub-04–15). a, Response time on correct trials when the
    colour-singleton distractor signalled low or high reward, or no distractor was present. Dots: subjects (triangles:
    beyond the axis); large points and bars: mean ± within-subject SEM (Cousineau–Morey). Brackets: mean within-subject
    difference, one-sided paired t-test (distractor slower; high slower than low; * p < .05, ** p < .01, *** p < .001).
    b, Within-subject RT differences: dots subjects, diamond mean, bar bootstrapped 95% CI; counts: subjects in the
    predicted direction. c, Accuracy, as a (one-sided: distractor and high value lower).""")
