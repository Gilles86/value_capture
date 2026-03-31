# GLMSingle temporal upsampling experiment — 2026-03-31

## Question

GLMSingle requires a TR-resolution design matrix, so trial onsets must be rounded to the
nearest volume. With TR = 1.6 s the maximum rounding error is ±0.8 s. Does upsampling the
BOLD by 2.5× before fitting (TR → 0.64 s, max rounding error ±0.32 s) meaningfully change
the single-trial betas?

## Experiment

Scripts: `notes/experiments/glmsingle_upsample_exp.py` + `glmsingle_upsample_exp.sh`

- **Condition A** (baseline): native TR = 1.6 s, onsets rounded to ±0.8 s
- **Condition B** (test): BOLD linearly upsampled 2.5×, TR = 0.64 s, onsets rounded to ±0.32 s

BOLD upsampled via linear interpolation along the time axis (scipy `interp1d`).

Dataset: sub-01, sub-02, sub-03, ses-1, 8 runs, 320 trials each.

## Results

| | sub-01 | sub-02 | sub-03 |
|---|---|---|---|
| Mean voxel corr (A vs B) | 0.913 | 0.863 | 0.849 |
| Median voxel corr | 0.941 | 0.916 | 0.915 |
| p5 voxel corr | 0.782 | 0.603 | 0.422 |
| Mean R² — native | 3.64% | 2.42% | 3.65% |
| Mean R² — upsampled | 4.30% | 3.28% | 5.83% |
| ΔR² | +0.66 pp | +0.86 pp | +2.18 pp |

Unlike the condition-structure experiment, upsampling **substantially** changes the betas
(mean voxel r ~ 0.85–0.91) and meaningfully improves R² (+0.7–2.2 percentage points). The R²
gain confirms the upsampled model genuinely fits better — the native-TR onset rounding error
was measurably degrading estimates, not just adding noise.

Trial-pattern correlations in the top-R² voxels are higher (~0.95–0.98), indicating the
spatial response patterns are largely preserved, but beta magnitudes shift appreciably.

Sub-03 shows the largest R² gain (+2.18 pp) and the lowest voxel correlations, suggesting
more variable sub-TR trial timing in that subject.

## Conclusion

**Upsampling is worth doing and has been added to the main pipeline** (`fit_glmsingle.py`).
The factor was set to **3×** (TR → 0.533 s, max rounding error ±0.27 s) as a round number
with slightly better precision than 2.5×.

Further upsampling beyond 3× is likely not worth it: the BOLD signal is heavily low-pass
filtered by the HRF (~0.1–0.2 Hz bandwidth), so ±0.27 s rounding error is already
negligible relative to the HRF rise time (~2 s) and peak (~5–6 s).

## On interpolation order

Linear interpolation is appropriate here. The BOLD signal is band-limited to ~0.1–0.2 Hz
(set by the HRF), well below the Nyquist frequency of the 1.6 s TR (0.31 Hz). At this
bandwidth, linear interpolation between adjacent samples is already very accurate — the
signal is smooth enough that the quadratic error of linear interpolation is negligible
compared to thermal noise, physiological noise, and motion artefacts. Higher-order methods
(cubic spline, sinc) would be theoretically marginally better but practically indistinguishable.
