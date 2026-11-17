#!/usr/bin/env python3
"""SPM12 second level on the nilearn reward-GLM effect maps: random field theory (GRF) inference.

The robustness check next to the permutation inference (fit_second_level_perm.py): same first-level effect
maps, SPM one-sample t-test, contrasts +1 and −1, and SPM's own tables with
  - voxel-level FWE (RFT) and FDR,
  - cluster-level FWE (RFT) at a cluster-forming threshold of p < .001 uncorrected (Eklund et al. 2016: RFT
    cluster inference is valid at .001, liberal at .01),
whole brain and within the a-priori reward mask (small-volume correction: the SPM model restricted to the mask,
so smoothness and resels are those of the mask).

Writes per contrast <bids>/derivatives/nilearn_glm_reward/group_spm/<contrast>[_svc]/ with SPM.mat, spmT_*,
and table_{pos,neg}.tsv (spm_list 'Table'). Needs MATLAB (module) and SPM12 at /shares/zne.uzh/containers/spm12,
with APPTAINER_BIND=/shares/zne.uzh (see the fmri-glm skill, references/spm_nipype.md).

    python value_capture/glm/second_level_spm.py --matlab $(command -v matlab)
"""
import argparse
import gzip
import shutil
import subprocess
from pathlib import Path

from value_capture.glm.fit_second_level_perm import CONTRASTS, SPACE, reward_rois

SPM = '/shares/zne.uzh/containers/spm12'

MATLAB = r"""
addpath('{spm}'); spm('defaults', 'fmri'); spm_jobman('initcfg'); spm_get_defaults('cmdline', true);
outdir = '{outdir}';
scans = {scans};
mb = {{}};
mb{{1}}.spm.stats.factorial_design.dir = {{outdir}};
mb{{1}}.spm.stats.factorial_design.des.t1.scans = scans;
mb{{1}}.spm.stats.factorial_design.masking.tm.tm_none = 1;
mb{{1}}.spm.stats.factorial_design.masking.im = 1;
mb{{1}}.spm.stats.factorial_design.masking.em = {{'{mask}'}};
mb{{1}}.spm.stats.factorial_design.globalc.g_omit = 1;
mb{{1}}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
mb{{1}}.spm.stats.factorial_design.globalm.glonorm = 1;
mb{{2}}.spm.stats.fmri_est.spmmat = {{fullfile(outdir, 'SPM.mat')}};
mb{{2}}.spm.stats.fmri_est.method.Classical = 1;
mb{{3}}.spm.stats.con.spmmat = {{fullfile(outdir, 'SPM.mat')}};
mb{{3}}.spm.stats.con.consess{{1}}.tcon.name = 'pos';
mb{{3}}.spm.stats.con.consess{{1}}.tcon.weights = 1;
mb{{3}}.spm.stats.con.consess{{2}}.tcon.name = 'neg';
mb{{3}}.spm.stats.con.consess{{2}}.tcon.weights = -1;
spm_jobman('run', mb);
names = {{'pos', 'neg'}};
for ic = 1:2
    xSPM = struct('swd', outdir, 'title', names{{ic}}, 'Ic', ic, 'n', 1, 'Im', [], 'pm', [], 'Ex', [], ...
                  'u', 0.001, 'k', 0, 'thresDesc', 'none');
    [SPM, xSPM] = spm_getSPM(xSPM);
    T = spm_list('Table', xSPM);
    fid = fopen(fullfile(outdir, ['table_' names{{ic}} '.tsv']), 'w');
    fprintf(fid, 'set_p\tset_c\tclus_pFWE\tclus_qFDR\tclus_k\tclus_punc\tpeak_pFWE\tpeak_qFDR\tpeak_T\tpeak_Z\tpeak_punc\tx\ty\tz\n');
    for r = 1:size(T.dat, 1)
        row = T.dat(r, :);
        v = cell(1, 12);
        for c = 1:11
            if isempty(row{{c}}), v{{c}} = 'NaN'; else, v{{c}} = num2str(row{{c}}, 6); end
        end
        xyz = row{{12}};
        fprintf(fid, '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%g\t%g\t%g\n', v{{1:11}}, xyz(1), xyz(2), xyz(3));
    end
    fclose(fid);
    fprintf('%s %s: FWEc (k for cluster pFWE<.05) = %g, FWE voxel T = %.2f\n', outdir, names{{ic}}, ...
            xSPM.uc(3), xSPM.uc(1));
end
"""


def main(bids, subjects, matlab, contrasts):
    root = Path(bids) / 'derivatives' / 'nilearn_glm_reward'
    mask_gz = root / 'group_spm' / 'reward_mask.nii.gz'
    if not mask_gz.exists():          # same a-priori mask as the permutation SVC (Harvard-Oxford reward ROIs)
        import numpy as np
        from nilearn import image
        ref = image.load_img(str(next((root / f'sub-{subjects[0]}' / 'func').glob(f'*{SPACE}*_stat-effect_statmap.nii.gz'))))
        rois = reward_rois(ref)
        m = (sum(np.asarray(r.dataobj) for r in rois.values()) > 0).astype(np.uint8)
        mask_gz.parent.mkdir(parents=True, exist_ok=True)
        image.new_img_like(ref, m).to_filename(str(mask_gz))
    for con in contrasts:
        for svc in (False, True):
            outdir = root / 'group_spm' / (con + ('_svc' if svc else ''))
            if outdir.exists():
                shutil.rmtree(outdir)          # SPM refuses to overwrite an existing SPM.mat
            (outdir / 'scans').mkdir(parents=True)
            scans = []
            for s in subjects:   # SPM12 cannot read .nii.gz
                f = root / f'sub-{s}' / 'func' / f'sub-{s}_task-valuecapture_{SPACE}_contrast-{con}_stat-effect_statmap.nii.gz'
                if not f.exists():
                    continue
                dst = outdir / 'scans' / f.name.replace('.nii.gz', '.nii')
                with gzip.open(f) as fi, open(dst, 'wb') as fo:
                    shutil.copyfileobj(fi, fo)
                scans.append(f"'{dst},1'")
            mask = ''
            if svc:
                mask = str(outdir / 'reward_mask.nii')
                with gzip.open(mask_gz) as fi, open(mask, 'wb') as fo:
                    shutil.copyfileobj(fi, fo)
            code = MATLAB.format(spm=SPM, outdir=outdir, scans='{' + '; '.join(scans) + '}', mask=mask)
            (outdir / 'batch.m').write_text(code)
            print(f'{con}{" (SVC)" if svc else ""}: {len(scans)} subjects', flush=True)
            subprocess.run([matlab, '-batch', f"run('{outdir / 'batch.m'}')"], check=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-valuecapture')
    ap.add_argument('--subjects', nargs='+', default=[f'{i:02d}' for i in range(4, 16)])
    ap.add_argument('--contrasts', nargs='+', default=CONTRASTS)
    ap.add_argument('--matlab', required=True)
    a = ap.parse_args()
    main(a.bids_folder, a.subjects, a.matlab, a.contrasts)
