"""Design-matrix figure for the reward GLM (one subject): structure, time courses and collinearity.

Page per design ('logreward', 'rpe'):
  a, the design matrix of one run (task regressors, then nuisance), columns scaled for display;
  b, the task regressors over the first 90 s of that run (convolved, with event onsets);
  c, correlations between the task regressors, pooled over all runs (collinearity check);
  d, regressor counts: events per condition per run.

    python value_capture/glm/plot_design_matrix.py 04 --out notes/figures/glm_reward_design_sub-04.pdf
"""
import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from nilearn import image
from nilearn.glm.first_level import make_first_level_design_matrix

from value_capture.glm.fit_nilearn_glm_reward import TR, build_events, build_events_rpe, confounds
from value_capture.utils.data import Subject

mpl.rcParams.update({'font.family': 'Helvetica', 'font.size': 7, 'pdf.fonttype': 42, 'axes.spines.top': False,
                     'axes.spines.right': False})


def designs(sub, design):
    out = []
    for ses in sub.get_sessions():
        runs = sub.get_runs(ses)
        bolds = sub.get_preprocessed_bold(ses, runs, space='MNI152NLin2009cAsym', res='2')
        for run, bold in zip(runs, bolds):
            raw = sub.get_onsets(ses, run)
            n = image.load_img(str(bold)).shape[3]
            ev = (build_events_rpe if design == 'rpe' else build_events)(raw, len(raw[raw.event_type == 'pulse']) - n)
            st = json.loads(Path(str(bold).replace('.nii.gz', '.json')).read_text()).get('StartTime', 0.)
            ft = (np.arange(n) + st / TR) * TR
            conf = confounds(sub, ses, run, 'fmriprep')
            dm = make_first_level_design_matrix(ft, ev, hrf_model='spm', drift_model=None, add_regs=conf.to_numpy(),
                                                add_reg_names=list(conf.columns))
            out.append((ses, run, ev, dm))
    return out


def page(pdf, out, subject, design, runs):
    ses, run, ev, dm = runs[0]
    task = [c for c in dm.columns if not (c.startswith(('trans', 'rot', 'a_comp', 'cosine', 'framewise', 'dvars'))
                                          or c == 'constant')]
    fig = plt.figure(figsize=(7.2, 8.6))
    fig.suptitle(f'Reward GLM, design "{design}" — sub-{subject}, ses-{ses} run-{run}', x=.02, ha='left', fontsize=9,
                 weight='bold')
    ax = fig.add_axes([.07, .47, .5, .46])
    x = dm.to_numpy()
    x = (x - x.min(0)) / np.where(np.ptp(x, 0) > 0, np.ptp(x, 0), 1)
    ax.imshow(x, aspect='auto', cmap='Greys', interpolation='nearest')
    ax.set_xticks(range(dm.shape[1]), dm.columns, rotation=90, fontsize=4.5)
    for t in ax.get_xticklabels():
        t.set_color('#1F2933' if t.get_text() in task else '#9AA5B1')
    ax.set_ylabel('Volume (TR = 1.6 s)')
    ax.axvline(len(task) - .5, color='#B2182B', lw=.8)
    ax.text(.5 * len(task), -4, 'task', ha='center', fontsize=6.5, color='#B2182B')
    ax.text(len(task) + .5 * (dm.shape[1] - len(task)), -4, 'nuisance (motion, aCompCor, DCT cosines)', ha='center',
            fontsize=6.5, color='.4')
    ax.set_title('a  Design matrix (columns scaled 0-1)', loc='left', fontsize=7.5, pad=12)

    # correlations of task regressors over all runs
    allx = pd.concat([d[[c for c in task if c in d]] for *_, d in runs]).fillna(0)
    r = allx.corr()
    ax = fig.add_axes([.66, .6, .3, .3])
    im = ax.imshow(r, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(r)), r.columns, rotation=90, fontsize=5)
    ax.set_yticks(range(len(r)), r.columns, fontsize=5)
    for i in range(len(r)):
        for j in range(len(r)):
            if i != j and abs(r.iloc[i, j]) >= .3:
                ax.text(j, i, f'{r.iloc[i, j]:.1f}', ha='center', va='center', fontsize=4.5)
    ax.set_title('c  Correlations of task regressors\n(all runs; |r| ≥ .3 printed)', loc='left', fontsize=7.5)
    cb = fig.colorbar(im, ax=ax, fraction=.04, pad=.02)
    cb.ax.tick_params(labelsize=5)

    # time courses
    ax = fig.add_axes([.07, .07, .62, .3])
    t = dm.index.to_numpy()
    show = t < 90 + t[0]
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    for k, c in enumerate(task):
        ax.plot(t[show], dm[c].to_numpy()[show] - 1.4 * k, color=cols[k % 10], lw=1)
        ax.text(t[show][-1] + 1, -1.4 * k, c, fontsize=5.5, va='center', color=cols[k % 10])
        on = ev[ev.trial_type == c]
        on = on[on.onset < 90]
        ax.plot(on.onset, np.full(len(on), -1.4 * k + .9), '|', color=cols[k % 10], ms=4)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Time in run (s)')
    ax.set_title('b  Task regressors, first 90 s (ticks: event onsets; modulators scaled by their values)', loc='left',
                 fontsize=7.5)

    # counts
    ax = fig.add_axes([.8, .07, .17, .42])
    cnt = pd.DataFrame([e.trial_type.value_counts() for *_, e, _ in runs]).fillna(0)
    m = cnt.mean().sort_index()
    ax.barh(range(len(m)), m.values, color='#7B8794')
    ax.set_yticks(range(len(m)), m.index, fontsize=5.5)
    ax.invert_yaxis()
    ax.set_xlabel('Events per run (mean)')
    ax.set_title('d  Events per run', loc='left', fontsize=7.5)
    pdf.savefig(fig)
    fig.savefig(str(Path(out).with_suffix('')) + f'_{design}.png', dpi=150)
    plt.close(fig)


def confounds_cols(dm):
    return [c for c in dm.columns if c.startswith(('trans', 'rot', 'a_comp', 'cosine', 'framewise', 'dvars'))]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('subject')
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-valuecapture')
    ap.add_argument('--out', default='notes/figures/glm_reward_design_sub-04.pdf')
    a = ap.parse_args()
    sub = Subject(a.subject, bids_folder=a.bids_folder)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(a.out) as pdf:
        for design in ('logreward', 'rpe'):
            page(pdf, a.out, a.subject, design, designs(sub, design))
    print(a.out)
