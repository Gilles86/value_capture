# PRFs on averaged vs single-trial betas (experiment, 2026-11-17)

**Question.** Does fitting the PRF to the mean beta per stimulus image (20 bar images, ~32 trials each) give
different PRFs than fitting all 640 single-trial betas? Gilles's expectation: it should not matter much.

**Prediction.** If every trial of an image has the same model prediction (true for the bar-only paradigm), the
least-squares loss over trials splits into a within-image term that does not depend on the parameters and
n × the loss on the image means. Same minimiser, so the same x, y, σ, amplitude, baseline (up to optimiser
noise and unequal counts per image). R² does differ: on the means the within-image noise is gone from the
denominator, so R² is much higher, and an R² ≥ .1 threshold picks a different, much larger voxel set. Higher
areas (IPS, VO, TO) that look empty at single-trial R² ≥ .1 may appear, with the same PRFs.

**Method.**
- `fit_prf.py --paradigm bar --out-suffix _grid5` (all trials) and the same with `--average-trials`
  (`derivatives/prf_glmsingle_grid5`, `..._grid5_avg`), ±5° grid at 0.1°/px, all 12 subjects.
  Slurm jobs 8479594 (trials), 8479595 (averaged).
- `prf/compare_avg_vs_trials.py` (job 8479908): per subject × neuropythy ROI, R² of each fit on its own data,
  single-trial R² of the averaged fit's parameters, parameter agreement (Pearson r, median |Δ|) over voxels with
  single-trial R² ≥ .1, and voxel counts passing R² ≥ .1 under each R². Output `notes/data/prf_avg_vs_trials.tsv`
  (+ `_voxels.tsv`).

**Results.** _Pending (jobs queued)._
