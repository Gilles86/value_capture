# Gain fields in value_capture: plan (2026-11-17)

Port of the retsupp gain-field ("attention field") model and its model-free decoding check to single-trial
GLMsingle betas. retsupp references: `retsupp/modeling/local_models.py` (factorial items model),
`fit_cv_ladder_nested.py` (ladder, shared L-BFGS fits), `fit_trialwise_af_gains.py` (linear per-trial gains),
`retsupp/decode/decode_final.py` + `visualize/paper/fig6_decoding.py` (decoding).

## What is different from retsupp

- **One beta per trial, no HRF, no time axis.** The bar is static during the 1.75 s search display, and the
  search display is on at the same time, so the bar and the 8 items are in every beta together.
- **No sustained/transient split.** In retsupp the sustained field (whole run, HP location) and the transient
  T/D fields are separated in time. Here everything multiplies the same trial's input. There is no HP location;
  the manipulated variable is the distractor's **value** (high/low colour), reversed between sessions.
- **The PRFs are mapped in the same trials** (bar), with the items in the paradigm (`fit_prf.py --paradigm
  bar_items`). A field term that is the same on every trial is a reshaping of every PRF and is not identified
  when the PRFs are free (retsupp `notes/joint_optimisation_2026-11-13.md`). Here only location-specific terms
  vary over trials (target and distractor move over the 4 diagonals), so they are identified; anything global
  must be zero-sum or fixed.

## Model (per subject × ROI; PRFs frozen from the bar+items fit, optional polish later)

    S_k(g)   = bar_k(g) + items_k(g)                                   input of trial k (the PRF paradigm)
    A_ℓ(g)   = exp(-||g - r_ℓ||² / 2σ²)                                peak-1 field at diagonal ℓ (4°)
    F_k(g)   = 1 + g_T · A_{T_k}(g) + g_D[v_k] · A_{D_k}(g)              D term absent on distractor-absent trials
    β_kv     = baseline_v + amplitude_v · Σ_g S_k(g) F_k(g) rf_v(g)

- Shared per subject × ROI: σ (one width, retsupp ties σ_T to σ_D), g_T, g_D[low], g_D[high].
- The field multiplies the input before the PRF pools it (as retsupp), so it acts on the item at T/D and on
  the bar where the bar passes near them.
- **Ladder** (nested, parent start, L-BFGS on the shared terms as in retsupp `shared_fit`):
  m0 PRF only → m1 + g_T → m2 + g_D (high = low) → m3 + g_D high ≠ low (value) → m4 + learning (below).
- **Value contrast** dV = g_D[high] − g_D[low]. Prediction: dV < 0 if high value is suppressed more (proactive)
  or > 0 if it captures more; behaviour says high-value distractors slow RT a little more (+6 ms, n.s. overall,
  bigger later in session 1).

## Learning: does the value gain change over time?

With σ and the PRFs fixed, the prediction is **linear in the gains** (retsupp `fit_trialwise_af_gains.py`, but
simpler: no HRF kernel, no overlap between trials):

    β_kv − pred0_kv = amplitude_v · [ g_T X_T,kv + g_D,k X_D,kv ]
    X_ℓ,kv = Σ_g S_k(g) A_ℓ(g) rf_v(g)

1. **Per run** (16 per subject): OLS/weighted LS of g_T, g_D[high], g_D[low] per run → trajectories over runs
   within each session; test dV(run) slope and session 1 vs session 2 (mapping reversed: does dV flip sign at the
   start of session 2 and re-learn?).
2. **Parametric** (retsupp (c)): g_D,k = g_D + β_v·value_k + β_t·run_k + β_vt·value_k·run_k (+ session),
   jackknife SEs over runs; group via the hierarchical meta-analysis (retsupp `trialwise_meta_bayes.py`).
3. Relate to behaviour: per-subject dV against the RT value effect (and its time course in
   `behavior_practice_learning`).

## Model-free check: decoding (`value_capture/decode/decode_trials.py`)

- Encoding model = PRFs only (Gaussian, bar + items paradigm), **no gain terms**, PRFs and noise model
  (ResidualFitter on betas) from the **other session** (`fit_prf.py --train-sessions`), voxels with training
  R² ≥ .1 per neuropythy ROI.
- Decode each held-out trial's image (StimulusFitter). `outline` decoder: only the true input's pixels
  (bar + 8 items) are free. The decoded intensity of the distractor item vs the neutral diagonal items vs the
  target item, and of the bar in the quadrant facing each, is a gain estimate without a gain model.
- Readout: per trial and ROI, then contrasts high − low value distractor, distractor − neutral, target − neutral,
  over runs (learning). Normalise per run (mean decoded intensity on the bar) before condition contrasts
  (retsupp trap: run-level drift).
- retsupp lesson: its decoder used a bar-only paradigm because modelled items pulled decoded mass to the item
  corners. Here the items are on in every trial with the bar, so they must be in the model; the outline
  decoder pins the input's shape and decodes only intensities, which avoids that.
- Later: `simulate_decode --mode fitted` equivalent (predict betas from the fitted gain model with and without
  gains, add resampled residuals, decode both the same way).

## Status

- PRF fits with items (`prf_glmsingle_items`) and session-CV fits (`_train-ses1`, `_train-ses2`) submitted
  2026-11-17; test decode of sub-07, session 2 chained (job 8472597).
- Gain-field model fitting code: not written yet (next).
