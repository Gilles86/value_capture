"""Plot the quick (uncorrected) group z maps of the reward GLM: one row per contrast, a glass brain and two slices
through vmPFC / ventral striatum (x = 0 sagittal, y = 10 coronal), thresholded at |z| > 3.1 (p < .001, two-sided
≈ .002), uncorrected.

    python value_capture/glm/plot_quick_group_maps.py --in-dir notes/data/glm_reward_quick --out notes/figures/glm_reward_quick.pdf
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from nilearn import plotting

mpl.rcParams.update({'font.family': 'Helvetica', 'font.size': 7, 'pdf.fonttype': 42})
TITLES = {'outcome_logreward': 'Reward received, log(1 + points)', 'outcome_logreward_cued': 'Log reward, cued (distractor present)',
          'outcome_logreward_uncued': 'Log reward, uncued (distractor absent)',
          'outcome_rpe': 'Prediction error: log-reward slope uncued − cued',
          'cue_value': 'Search: high- vs low-value distractor', 'capture': 'Search: distractor vs absent',
          'feedback_vs_omit': 'Points shown vs omitted', 'omission_value': 'Omission of ×100 vs ×1'}

ap = argparse.ArgumentParser()
ap.add_argument('--in-dir', default='notes/data/glm_reward_quick')
ap.add_argument('--out', default='notes/figures/glm_reward_quick.pdf')
ap.add_argument('--threshold', type=float, default=3.1)
a = ap.parse_args()
files = {c: Path(a.in_dir) / f'contrast-{c}_stat-z.nii.gz' for c in TITLES}
files = {c: f for c, f in files.items() if f.exists()}
fig = plt.figure(figsize=(7.2, 1.25 * len(files)))
for i, (c, f) in enumerate(files.items()):
    y0 = 1 - (i + 1) / len(files)
    h = 1 / len(files)
    ax = fig.add_axes([0, y0, .5, h * .85])
    plotting.plot_glass_brain(str(f), threshold=a.threshold, plot_abs=False, colorbar=False, display_mode='lyrz',
                              axes=ax, vmax=6, cmap='cold_hot')
    ax2 = fig.add_axes([.5, y0, .25, h * .85])
    plotting.plot_stat_map(str(f), threshold=a.threshold, display_mode='x', cut_coords=[0], axes=ax2,
                           colorbar=False, vmax=6, annotate=False, draw_cross=False, cmap='cold_hot')
    ax3 = fig.add_axes([.75, y0, .25, h * .85])
    plotting.plot_stat_map(str(f), threshold=a.threshold, display_mode='y', cut_coords=[10], axes=ax3,
                           colorbar=i == 0, vmax=6, annotate=False, draw_cross=False, cmap='cold_hot')
    fig.text(.005, y0 + h * .9, TITLES[c], fontsize=7.5, weight='bold', va='bottom')
fig.text(.995, .002, f'n = 12, one-sample t, |z| > {a.threshold} uncorrected; x = 0, y = 10', fontsize=6, ha='right',
         color='.4')
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(a.out, dpi=200)
fig.savefig(str(a.out).replace('.pdf', '.png'), dpi=150)
print(a.out)
