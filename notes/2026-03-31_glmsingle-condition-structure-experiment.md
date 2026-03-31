# GLMSingle condition structure experiment — 2026-03-31

## Question

Does the choice of condition labels in the GLMSingle design matrix affect single-trial beta
estimates? Specifically: should we group trials by **bar position × orientation** (20 conditions,
current pipeline) or by **bar × reward condition** (80 conditions, task-aware grouping)?

GLMSingle uses condition labels only for two things:
1. Grouping trials in the fracridge cross-validation (`calcbadness`) to select the per-voxel
   regularisation parameter λ (frac).
2. The HRF library cross-validation (same scheme).

It does NOT affect the single-trial design matrix itself — every trial always gets its own column
in `designSINGLE`.

## Experiment

Scripts: `notes/experiments/glmsingle_condition_exp.py` + `submit_glmsingle_condition_exp.sh`

- **Condition A** (baseline): 20 conditions — bar position × orientation only, same as
  `fit_glmsingle.py`. Each condition repeats ~2×/run, giving a well-powered CV.
- **Condition B** (test): 80 conditions — bar × value_rank × distractor_present.
  Each condition repeats ~0.5×/run (many conditions appear only once per run).

Dataset: sub-01, sub-02, sub-03, ses-1, all 8 valuecapture runs each (320 trials/subject).

Comparison metric: Pearson r per voxel across the 320 trial betas (A vs B).

## Results

Across ~890k brain voxels in three subjects (brain-masked):

| Metric | Value |
|--------|-------|
| Mean voxel correlation (A vs B) | r = 0.9997 |
| Median voxel correlation | r = 1.000 |
| 1% quantile | r = 0.9994 |
| Minimum (within brain mask) | r = 0.959 |

The trial-wise spatial pattern correlations are equally high.

### FRACvalue (regularisation parameter)

Despite near-identical betas, ~10–15% of brain voxels select a different frac between A and B:

| Subject | Voxels with different frac | Mean \|ΔFrac\| |
|---------|---------------------------|----------------|
| sub-01  | 13.0%                     | 0.060          |
| sub-02  | 10.1%                     | 0.060          |
| sub-03  | 15.1%                     | 0.060          |

This is expected: with 80 conditions and 0.5 reps/run, the leave-one-run-out CV that selects
frac is noisier, so the optimal λ estimate shifts for some voxels. However, the fracridge
regularisation path is smooth — a small shift in frac (mean Δ = 0.06) moves the betas
negligibly. The voxels where correlation dips below 0.999 are those where the frac change
happens to land on a more sensitive part of the regularisation path.

### R² note

The R² stored in GLMSingle's output (`results['typed']['R2']`) is **in-sample**, not
cross-validated. The cross-validation (`calcbadness` / `rrbadness`) is used only to *select*
the frac, not to evaluate the final model fit. R² is the in-sample fit of the full model with
the chosen λ. It can therefore overfit, though the fracridge regularisation limits this.

## Conclusion

**The 20-condition structure is fine. No need to include reward condition in the design matrix.**

The condition labels affect only the fracridge CV, and even where they change the selected λ,
the betas are virtually identical (r > 0.999 at p1, min r = 0.96 across the whole brain).
More conditions means fewer reps/condition/run, which makes the CV noisier without any benefit
to the betas. Keep the current 20-condition pipeline.

## Technical notes / bugs fixed along the way

### Events TSV path bug

The experiment script originally read events from the BIDS `func/` events TSVs:
```
sub-XX/ses-Y/func/sub-XX_ses-Y_task-valuecapture_run-Z_events.tsv
```
These files do **not** contain `event_type == 'pulse'` rows, so `total_pulses = 0` and
`n_removed = 0 - n_vols = -207`. Every trial onset then mapped to the last clipped volume,
causing all conditions to collide at the same timepoint → GLMSingle assertion error.

Fix: use `Subject.get_onsets(session, run)` (same as `fit_glmsingle.py`), which reads from
`sourcedata/behavior/` — the raw behavioral TSVs that do include pulse events — and normalises
t = 0 to the first scanner pulse.

### n_removed for upsampled TR (upsample experiment)

In `glmsingle_upsample_exp.py`, condition B upsamples the BOLD by 2.5× (TR = 0.64 s).
The naive `n_removed = total_pulses - n_vols_up` mixes native pulse count (~219) with
upsampled volume count (518), giving n_removed = −299. This shifts all onsets ~300 volumes
too high, again causing clipping collisions.

Fix: convert via time —
```python
n_vols_native = int(round(n_vols * tr / TR))
n_removed = int(round((total_pulses - n_vols_native) * TR / tr))
```
