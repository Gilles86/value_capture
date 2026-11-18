"""Annotate the significant group clusters of the reward GLM: table + one slice view per cluster.

Significant = FWE p < .05 by permutation cluster mass OR TFCE (fit_second_level_perm.py, whole brain or reward
SVC). For every cluster: size, peak t and MNI coordinates, the best permutation p, the Harvard-Oxford label at
the peak (highest probability, cortical or subcortical), and the SPM/RFT cluster p_FWE of the SPM cluster
containing that peak (second_level_spm.py), as the robustness check.

Figure: one page per contrast, slices through each cluster's peak (significant voxels only), annotated.

Run locally after syncing <bids>/derivatives/nilearn_glm_reward/{group,group_spm} to notes/data/glm_reward/:
    python value_capture/glm/annotate_clusters.py --in-dir notes/data/glm_reward --out notes/figures/glm_reward_clusters
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn import datasets, image, plotting
from scipy import ndimage

mpl.rcParams.update({'font.family': 'Helvetica', 'font.size': 7, 'pdf.fonttype': 42})
SPACE = 'space-MNI152NLin2009cAsym_res-2'
TITLES = {'outcome_logreward': 'Reward received, log(1 + points)', 'outcome_logreward_cued': 'Log reward, cued (distractor present)',
          'outcome_logreward_uncued': 'Log reward, uncued (distractor absent)',
          'outcome_rpe': 'Prediction error: log-reward slope uncued − cued',
          'cue_value': 'Search: high- vs low-value distractor', 'capture': 'Search: distractor vs absent',
          'feedback_vs_omit': 'Points shown vs omitted', 'omission_value': 'Omission of ×100 vs ×1'}
ALPHA = -np.log10(.05)


def ho_label(xyz, cache={}):
    if not cache:
        for kind in ('cort', 'sub'):
            a = datasets.fetch_atlas_harvard_oxford(f'{kind}-prob-2mm')
            cache[kind] = (image.load_img(a.maps), a.labels)
    best = ('unlabelled', 0.)
    for kind, (img, labels) in cache.items():
        ijk = np.round(np.linalg.inv(img.affine) @ np.r_[xyz, 1])[:3].astype(int)
        try:
            probs = np.asarray(img.dataobj[ijk[0], ijk[1], ijk[2], :], float)
        except IndexError:
            continue
        for k in np.argsort(probs)[::-1]:
            lab = labels[k + 1] if len(labels) == probs.size + 1 else labels[k]
            if kind == 'sub' and ('Cortex' in lab or 'White Matter' in lab):
                continue          # the subcortical atlas also labels cortex / WM wholesale
            if probs[k] > best[1]:
                best = (lab, probs[k])
            break
    return f'{best[0]} ({best[1]:.0f}%)' if best[1] else 'unlabelled'


def spm_cluster_p(spm_dir, xyz):
    """Cluster p_FWE (RFT) of the SPM cluster whose listed peak lies closest (< 8 mm) to xyz, pos or neg."""
    best = None
    for sign in ('pos', 'neg'):
        f = spm_dir / f'table_{sign}.tsv'
        if not f.exists():
            continue
        t = pd.read_csv(f, sep='\t')
        t['clus_pFWE'] = pd.to_numeric(t.clus_pFWE, errors='coerce')   # empty tables come back as strings
        t = t[np.isfinite(t.clus_pFWE)]
        if t.empty:
            continue
        d = np.sqrt(((t[['x', 'y', 'z']].to_numpy() - xyz) ** 2).sum(1))
        i = d.argmin()
        if d[i] < 8 and (best is None or d[i] < best[1]):
            best = (float(t.clus_pFWE.iloc[i]), d[i])
    return np.nan if best is None else best[0]


def main(in_dir, out):
    gdir, sdir = Path(in_dir) / 'group', Path(in_dir) / 'group_spm'
    rows = []
    for con in TITLES:
        for scope in ('wholebrain', 'reward'):
            f = lambda k: gdir / f'group_{SPACE}_contrast-{con}_scope-{scope}_stat-{k}.nii.gz'  # noqa: E731
            if not f('t').exists():
                continue
            t_img = image.load_img(str(f('t')))
            t = t_img.get_fdata()
            logp = np.maximum(image.load_img(str(f('logp_max_mass'))).get_fdata(),
                              image.load_img(str(f('logp_max_tfce'))).get_fdata())
            sig = logp > ALPHA
            for sign in (1, -1):
                lab, n = ndimage.label(sig & (np.sign(t) == sign))      # 6-connected, as nilearn
                for k in range(1, n + 1):
                    m = lab == k
                    if m.sum() < 5:
                        continue
                    i = np.argmax(np.abs(t) * m)
                    ijk = np.unravel_index(i, t.shape)
                    xyz = (t_img.affine @ np.r_[ijk, 1])[:3]
                    rows.append(dict(contrast=con, scope=scope, sign='+' if sign > 0 else '−', k=int(m.sum()),
                                     peak_t=float(t[ijk]), x=xyz[0], y=xyz[1], z=xyz[2],
                                     p_perm=10 ** -float(logp[m].max()), label=ho_label(xyz),
                                     p_spm_rft=spm_cluster_p(sdir / (con + ('_svc' if scope == 'reward' else '')), xyz)))
    tab = pd.DataFrame(rows).sort_values(['contrast', 'scope', 'p_perm', 'k'], ascending=[True, True, True, False])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    tab.round(4).to_csv(f'{out}.tsv', sep='\t', index=False)
    print(tab.round(3).to_string(index=False) if len(tab) else 'no significant clusters')
    if tab.empty:
        return
    # One page per contrast: a grid of slice views (sagittal, coronal, axial through each cluster's peak), only the
    # significant voxels coloured, each with its annotation. No glass brains (Gilles prefers targeted slices).
    from matplotlib.backends.backend_pdf import PdfPages
    ncol, pw, ph = 2, 3.55, 1.45                      # panel width/height (in)
    pngs = []
    with PdfPages(f'{out}.pdf') as pdf:
        for con, g in tab.groupby('contrast', sort=False):
            g = g.sort_values(['scope', 'p_perm', 'k'], ascending=[False, True, False]).head(14)
            nrow = int(np.ceil(len(g) / ncol))
            W, H = ncol * pw + .1, nrow * (ph + .32) + .35
            fig = plt.figure(figsize=(W, H))
            fig.text(.01, 1 - .12 / H, TITLES[con], fontsize=8.5, weight='bold', va='top')
            maps = {}
            for k_, (_, r) in enumerate(g.iterrows()):
                if r.scope not in maps:
                    stem = gdir / f'group_{SPACE}_contrast-{con}_scope-{r.scope}_stat-'
                    t_img = image.load_img(str(stem) + 't.nii.gz')
                    lp = np.maximum(image.load_img(str(stem) + 'logp_max_mass.nii.gz').get_fdata(),
                                    image.load_img(str(stem) + 'logp_max_tfce.nii.gz').get_fdata())
                    maps[r.scope] = image.new_img_like(t_img, np.where(lp > ALPHA, t_img.get_fdata(), 0))
                row, col = divmod(k_, ncol)
                x0 = .05 + col * pw
                y0 = H - .35 - (row + 1) * (ph + .32)
                ax = fig.add_axes([x0 / W, y0 / H, (pw - .1) / W, ph / H])
                plotting.plot_stat_map(maps[r.scope], cut_coords=(r.x, r.y, r.z), display_mode='ortho', axes=ax,
                                       colorbar=False, vmax=6, cmap='cold_hot', annotate=False, draw_cross=False,
                                       black_bg=False)
                spm = 'n/a' if np.isnan(r.p_spm_rft) else f'{r.p_spm_rft:.3f}'
                svc = ' (reward SVC)' if r.scope == 'reward' else ''
                fig.text((x0 + .02) / W, (y0 + ph + .02) / H,
                         f'{r.sign} {r.label}{svc}\n({r.x:.0f}, {r.y:.0f}, {r.z:.0f}), k = {r.k}, peak t = {r.peak_t:.1f}; '
                         f'p perm = {r.p_perm:.3f}, p RFT = {spm}', fontsize=6, va='bottom', linespacing=1.15)
            pdf.savefig(fig, dpi=200)
            png = f'{out}_{con}.png'
            fig.savefig(png, dpi=150)
            pngs.append(png)
            plt.close(fig)
    print(f'{out}.pdf', *pngs, sep='\n')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', default='notes/data/glm_reward')
    ap.add_argument('--out', default='notes/figures/glm_reward_clusters')
    a = ap.parse_args()
    main(a.in_dir, a.out)
