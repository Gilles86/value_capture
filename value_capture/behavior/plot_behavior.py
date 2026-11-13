"""Quick cohort-level behaviour figure: RT and accuracy by distractor condition.

Usage:
    python -m value_capture.behavior.plot_behavior --bids-folder /data/ds-valuecapture \
        --out notes/figures/behavior_cohort.pdf
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from value_capture.utils.data import get_all_behavioral_data

ORDER = ['No Distractor', 'Low Value', 'High Value']
PALETTE = {'No Distractor': '#8f8f8f', 'Low Value': '#7fb3d5',
           'Medium Value': '#b58800', 'High Value': '#c0392b'}


def main(bids_folder, out):
    df = get_all_behavioral_data(bids_folder=bids_folder).reset_index()
    # Main study only: pilots sub-01..03 had a 3-rank design and are not
    # independent of the main sample (see participants.tsv)
    df = df[~df['subject'].isin(['01', '02', '03'])]
    df = df[df['condition'].isin(ORDER)]
    df['condition'] = pd.Categorical(df['condition'], ORDER, ordered=True)
    df['rt_ms'] = df['rt'] * 1000
    correct = df[df['correct'] == 1]

    # Per-subject means
    rt = correct.groupby(['subject', 'condition'], observed=True)['rt_ms'].mean().reset_index()
    rt['rt_capture'] = rt['rt_ms'] - rt.groupby('subject')['rt_ms'].transform(
        lambda x: x[rt.loc[x.index, 'condition'] == 'No Distractor'].mean())
    acc = df.groupby(['subject', 'condition'], observed=True)['correct'].mean().reset_index()
    acc['correct'] *= 100

    n = df['subject'].nunique()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), constrained_layout=True)
    for ax, data, y, label in [
        (axes[0], rt, 'rt_ms', 'RT correct trials (ms)'),
        (axes[1], rt[rt['condition'] != 'No Distractor'], 'rt_capture',
         'Capture: RT − no-distractor RT (ms)'),
        (axes[2], acc, 'correct', 'Accuracy (%)'),
    ]:
        sns.lineplot(data=data, x='condition', y=y, units='subject', estimator=None,
                     color='0.8', lw=0.6, ax=ax, legend=False)
        sns.pointplot(data=data, x='condition', y=y, hue='condition', palette=PALETTE,
                      errorbar=('ci', 95), ax=ax, legend=False, markersize=6)
        ax.set_xlabel('')
        ax.set_ylabel(label)
        ax.tick_params(axis='x', rotation=30)
        if y == 'rt_capture':
            ax.axhline(0, color='k', lw=0.6, ls='--')
    sns.despine(fig)
    fig.suptitle(f'Value-driven capture, main study N = {n} (grey: subjects; dots: mean ± 95% CI)',
                 fontsize=10)
    fig.savefig(out)
    fig.savefig(out.replace('.pdf', '.png'), dpi=200)

    # Printed summary
    print(rt.groupby('condition', observed=True)[['rt_ms', 'rt_capture']].agg(['mean', 'sem']).round(1))
    print(acc.groupby('condition', observed=True)['correct'].agg(['mean', 'sem']).round(1))
    print(f'Trials: {len(df)}, subjects: {n}')
    print(df.groupby('subject')['correct'].mean().round(3).to_string())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bids-folder', default='/data/ds-valuecapture')
    p.add_argument('--out', default='notes/figures/behavior_cohort.pdf')
    a = p.parse_args()
    main(a.bids_folder, a.out)
