# GLM Contrasts — Value Capture fMRI Task

## Design overview

Each trial has two modelled phases, both convolved with the SPM canonical HRF:

| Phase | Duration | Regressors |
|-------|----------|------------|
| **Target** (search array on screen) | 1.75 s | `dist_absent`, `dist_rank0`, `dist_rank1`, `dist_rank2` |
| **Feedback** | 1.0 s | `feedback_omitted`, `feedback_shown`, `feedback_points` |

Nuisance regressors per run: 7 aCompCor components, 6 rigid-body motion
parameters (trans/rot x/y/z), framewise displacement, DVARS, and fmriprep DCT
cosines (low-frequency drift; nilearn's own drift model is therefore disabled).

---

## Regressors

### Target phase — proximate reward signal

The distractor colour tells the subject what reward *could* be earned on this
trial. This is the proximate reward signal: it is visible at stimulus onset,
before any response or outcome.

| Regressor | Condition | Value rank |
|-----------|-----------|-----------|
| `dist_absent` | No distractor present | — |
| `dist_rank0` | Low-value distractor | 0 (lowest) |
| `dist_rank1` | Medium-value distractor | 1 |
| `dist_rank2` | High-value distractor | 2 (highest) |

The colour-to-rank mapping is counterbalanced across subjects × sessions
(see `value_capture/utils/data.py` → `get_value_condition()`).

### Feedback phase — reward prediction error proxy

Feedback is shown on ~50% of scanning trials (randomised). Three regressors
decompose the feedback response:

| Regressor | Type | What it captures |
|-----------|------|-----------------|
| `feedback_omitted` | Unmodulated | Baseline BOLD on no-feedback trials (blank fixation) |
| `feedback_shown` | Unmodulated | Mean BOLD response to *receiving* any explicit feedback |
| `feedback_points` | Parametric modulator | How BOLD **scales** with reward magnitude |

**Parametric modulation details:** amplitude = `log(1 + earned_points) − mean_run`,
mean-centred within each run. `feedback_shown` therefore captures the *average*
neural response to feedback delivery, while `feedback_points` captures the
*deviation* from that average driven by reward magnitude — the closest available
proxy to a scalar reward prediction error without an explicit value model.

---

## Contrasts

### Proximate reward (target phase)

#### `value_linear`
```
dist_rank2 − dist_rank0
```
**Question:** Does BOLD scale linearly with distractor value rank?
**Positive:** Higher activity for high-value than low-value distractors.
Expected in frontoparietal attention areas, possibly striatum/caudate.
**Use:** Primary proximate-reward contrast.

---

#### `value_capture`
```
dist_rank2 − dist_absent
```
**Question:** Does the *high-value* distractor drive more BOLD than having no
distractor at all?
**Positive:** The high-value distractor captures neural resources above the
no-distractor baseline — the signature value-capture effect.
**Note:** Stronger than `value_linear` because it tests against a no-distractor
baseline rather than just relative to the low-value condition.

---

#### `value_nonlinear`
```
dist_rank1 − (dist_rank0 + dist_rank2) / 2
```
**Question:** Does the medium-value distractor response deviate from a linear
value scale?
**Positive (inverted-U):** More response to medium-value than expected from
linear interpolation between low and high — could reflect diminishing returns,
a compressed value scale, or categorical mid-value representation.
**Negative (U-shaped):** Less response to medium-value — suggests categorical
high/low coding rather than graded value, or response compression at the middle.
**Note:** Together with `value_linear`, these two contrasts form a complete
orthogonal decomposition of the three distractor value conditions.

---

#### `value_step_low`
```
dist_rank1 − dist_rank0
```
**Question:** Is the low→medium value step reflected in BOLD?

---

#### `value_step_high`
```
dist_rank2 − dist_rank1
```
**Question:** Is the medium→high value step reflected in BOLD?

`value_step_low` and `value_step_high` together decompose `value_linear`.
If both steps are similar in magnitude, value coding is approximately linear.
If `value_step_high` >> `value_step_low`, the BOLD response is driven
primarily by the transition to the highest value (convex scaling).
If `value_step_low` >> `value_step_high`, the transition away from the lowest
value dominates (concave scaling).

---

#### `distractor_any`
```
(dist_rank0 + dist_rank1 + dist_rank2) / 3 − dist_absent
```
**Question:** Does any distractor, regardless of value, drive extra BOLD?
**Use:** Isolates generic attentional capture from value-specific modulation.
Compare with `value_linear` to separate generic capture from value scaling.

---

### Reward prediction error proxy (feedback phase)

#### `feedback`
```
feedback_shown − feedback_omitted
```
**Question:** Does *receiving* any explicit feedback drive BOLD above the
no-feedback baseline?
**Positive:** Regions responding to outcome delivery broadly — RPE-related
regions (ventral striatum, ACC, vmPFC) are expected here.
**Use:** Identifies the feedback-responsive network before asking about
value scaling.

---

#### `feedback_value`
```
feedback_points  [parametric modulator]
```
**Question:** Does BOLD scale with the *magnitude* of received reward?
**Positive:** Regions where activity increases with log(earned points) —
expected in ventral striatum, OFC, vmPFC.
**Use:** Scalar RPE magnitude proxy. The closest available measure to a
reward prediction error without an explicit computational model of expectations.

---

## Output files

Per-subject NIfTIs in `derivatives/nilearn_glm/sub-<id>/func/`:

```
sub-<id>_task-valuecapture_space-<space>_contrast-<name>_stat-z_statmap.nii.gz
sub-<id>_task-valuecapture_space-<space>_contrast-<name>_stat-effect_statmap.nii.gz
```

Per-subject figures in `derivatives/nilearn_glm/sub-<id>/figures/`:
- `*_design_matrix.png` — design matrix for each run
- `*_fdr-report.pdf` — FDR-thresholded maps for all contrasts (this report)

## Notes on multiple comparisons

- **Single-subject exploration:** FDR q < 0.05 + 10-voxel cluster minimum
  (see `report_nilearn_glm.py`). Controls false discovery rate, not FWE.
- **Group analysis:** Use effect-size maps (not z-maps) as input to second-level
  models. Z-maps are subject-specific and do not pool correctly across subjects.
  For FWE-corrected group inference, use FSL `randomise` with TFCE
  (`randomise -T`) on the effect-size maps.
- **GRF-based FWE at subject level** is not implemented in nilearn. Use SPM
  (load the z-maps as a single-image one-sample t-test) if voxel-level FWE
  is required at the subject level.
