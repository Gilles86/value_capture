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

**Results (2026-11-18, 12 subjects; medians over subjects).** Gilles's expectation holds: the parameters do
not change. What changes is R², and an R² threshold on averaged data selects noise.

| ROI | voxels R² ≥ .1, single trials (sum) | voxels R² ≥ .1, averaged (sum) | median R² single trials | median R² averaged | r(x) | r(y) | r(σ) | median abs Δ x, y, σ (°) |
|---|---|---|---|---|---|---|---|---|
| V1 | 3,907 | 20,435 | .013 | .41 | .998 | .998 | .996 | .000 |
| V2 | 3,270 | 18,005 | .014 | .41 | .999 | .999 | .999 | .000 |
| V3 | 2,892 | 14,295 | .018 | .46 | 1.000 | 1.000 | 1.000 | .000 |
| hV4 | 661 | 7,879 | .016 | .44 | 1.000 | 1.000 | 1.000 | .000 |
| LO | 944 | 8,607 | .016 | .42 | 1.000 | 1.000 | 1.000 | .000 |
| V3AB | 193 | 10,845 | .013 | .38 | 1.000 | 1.000 | 1.000 | .005 |
| rest of grey matter | 7,174 | 1,513,041 | .010 | .34 | .999 | .998 | .999 | .000 |

(parameter agreement over voxels with single-trial R² ≥ .1; voxels with R² ≥ .02 in either fit enter the table)

- **Same PRFs.** x, y and σ agree to r ≥ .996 with a median difference of 0.000°; the single-trial R² of the
  averaged fit's parameters equals that of the single-trial fit (median Δ = 0). Amplitude r is lower in V1/V2
  (.65/.79) but its median difference is 0.001: a few outlying voxels.
- **R² on averages is not comparable.** With 20 image means and 5 free parameters, even pure noise reaches
  R² ≈ .3: in the rest of grey matter 1.5 million voxels (summed over subjects) pass R² ≥ .1 on the averages
  against 7,174 on single trials. "IPS appears after averaging" would therefore be an artefact of the threshold,
  not more signal. Thresholds must be set on single-trial (or cross-validated) R², or against a null.
- Cost: the averaged fit takes ~11 min per subject on an A100 (vs ~45 min for single trials) — a cheap way to get
  the same parameters, as long as voxels are selected by single-trial R².

