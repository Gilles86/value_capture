# Blank trials & GLMSingle filtering — 2026-03-27

## Run structure (default settings, n_trials=40)

- **Per-trial mean duration**: 6.75 s (trial_start 0.5 + pre_target 0.5 + target 1.75 + iti1 1.5 + feedback 1.0 + iti2 1.5)
- **Total run duration**: ~330 s / ~206 TRs at TR=1.6 s
  - Leading blank: 20 s
  - 40 trials: 270 s
  - 2 × 10 s mid-run rests (after trial 13 and 26): 20 s
  - Trailing blank: 20 s
- **Mean SOA**: 6.75 s (4.2 TRs)
- **Dedicated blank time**: 60 s total = 18% of run

## GLMSingle high-pass filtering

GLMSingle detrends with polynomials; default degree = `round(run_s / 2 / 60)` = **3** for a 330 s run.

Effective HP cutoff ≈ 2 × 330 / (3+1) ≈ **165 s** — comparable to SPM/nilearn's 128 s default, slightly more lenient.

## Does ridge regression reduce the need for blank trials?

Yes, partially — and this is the key argument against adding more blanks.

- The polynomial terms are **not** ridge-regularised; only single-trial betas are.
- However, fracridge makes it **costly** to explain slow BOLD drift via many small task betas (high Σβ²), while the polynomial explains drift for free.
- This implicitly pushes slow components onto the polynomial and fast trial-by-trial variance onto the betas — exactly the separation blank trials would otherwise enforce.
- In unregularised GLM: blank trials are important to break polynomial–task collinearity.
- With fracridge: that disambiguation is largely handled by the penalty itself.

## What blank periods still help with

**GLMdenoise** (part of GLMSingle Model B/D) identifies noise-pool voxels — those with low task-related variance — to extract structured noise regressors. Without any baseline, all voxels look task-related and the noise pool degrades. The 20 s leading/trailing blanks are the most important piece here.

## Conclusion / advice

- Current design is **adequate**. The 20 s leading/trailing blanks anchor GLMdenoise and the polynomial; the 2 × 10 s mid-run rests add within-run anchor points; jittered ITIs (max 4 s, 5× per run) provide deconvolution leverage.
- **Do not trade trials for blank time.** More trials = better PRF coverage and more single-trial betas, which matters more here than marginal baseline improvement.
- If co-author wants a visible concession: extend the two mid-run rests from 10 s → ~16 s (+~12 s per run, negligible cost). This adds one clean rest-period "anchor" in the GLMSingle diagnostic figures without cutting trials.
