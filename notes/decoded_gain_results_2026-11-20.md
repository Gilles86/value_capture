# Decoded gain, gain fields and behaviour: where we stand (2026-11-20)

Main study, n = 12 (sub-04..15; ventral areas n = 11). p-values are uncorrected sign-flip tests across subjects
unless noted. Question: is the stimulus at a distractor represented more (or less) strongly in visual cortex, and
more so for a high-value distractor? And does that link to behaviour or to striatum?

## Pipelines

- **Betas.** `glmsingle` (plain), `glmsingle_fbreg` (points as condition-level regressors, canonical HRF) and
  `glmsingle_fbevents` (points as their own single-trial events). **Primary: `glmsingle_fbevents`.**
- **PRFs.** Must be fitted on the same betas that are decoded ("matched"). Mismatched PRFs (fitted on plain
  betas, decoding fbreg betas) produced the strongest DoG value effect, which shrank once matched.
- **Points display.** Shown on ~50% of scan trials, 2.5-4 s after search, in the distractor's colour, with 100x
  more points for high value. It leaks into the search betas even when GLMsingle models it. Handled by
  (a) excluding points trials, or (b) subtracting the points response estimated on the no-distractor trials,
  where the multiplier is random (`decode/aggregate_decoded.remove_points`; per voxel for gain fields).
  A points term fitted on all trials would absorb a real value effect (multiplier follows value).

## Results

1. **Attention boost: robust.** Targets and distractors decode brighter than neutral items (early and ventral)
   in every pipeline. Gain fields give a positive target gain, mostly V3 and ventral (e.g. DoG, points
   excluded: early +0.15 p = .009, ventral +0.61 p = .001).
2. **Foveal high - low effect = the points display.** Present on points trials only (p = .006; without points
   p = .69), not removed by modelling the points in GLMsingle (fbevents +1.55 p = .006), removed by excluding
   points trials or subtracting them.
3. **Value at the distractor (decoding): not robust.** DoG with mismatched PRFs: early p = .002, V2 p = .007.
   Matched, points subtracted: DoG early +0.8% p = .21, V2 +1.7% p = .25; Gaussian V2 +3.6% p = .014, early
   p = .75. V2 is positive in most variants: a hint only. DN decoding is unreliable (outliers).
4. **Bar near vs far a distractor (decoding):** ventral p ~ .005-.03; gain-field predictions do not match it.
5. **Gain fields (fbevents betas, PRFs fitted jointly with the gains on the same betas, both sessions together;
   `gain_fields{,_dog}_fbevents_free_{subtract,exclude}`).** No value gain in V1-V3 in any fit. Ventral
   high - low is positive with points subtracted (Gaussian +0.26 p = .029, DoG +0.29 p = .038) but not with
   points trials excluded (+0.03 p = .84, +0.09 p = .45): most likely residual points (hV4/LO cover the fovea).
   R² gain over PRF only <= .005.
6. **Colour (pilot, V1-V3, Gaussian, points subtracted):** green - orange -0.003 p = .87; colour x value p >= .10.
7. **Learning (per-run gains, sigma and PRFs fixed):** no change of the value gain over runs, early vs late
   within a session, or between sessions (all p > .06, mostly > .2).
8. **Behaviour.** Log-RT mixed models: no effect of trial-wise decoded gains (95% HDIs include 0, bounds about
   +-1.3%). Race model (Wald race, PyMC/numpyro): high value slows drift ~4%, distractor ~8%; trial-wise gains
   no effect (<= +-4.5% drift per SD). Subject-wise: neural high - low does not predict the drift value effect
   (subject-level predictor on the primary pipeline: all 95% HDIs include 0, BF01 1.8-5.9; the earlier
   gamma < 0 came from the mismatched pipeline).
9. **Striatum.** Decoded distractor gain as a trial-wise modulator in a first-level GLM (`rpe_rt_gain`): no
   striatal effect (all p > .2; dist_gain_value accumbens p = .051). Cue value replicates (caudate p = .003,
   accumbens p = .026, vmPFC negative p = .043).

## Bottom line

Attention to the target raises the gain of its representation. Distractor value does not detectably change gain
in early visual cortex once the points display is dealt with, and the neural measures do not explain the
behavioural value effect. Value effects that looked significant traced back to the points display or to PRFs
fitted on other betas.

## Open

- A main figure on the primary pipeline (fbevents, matched PRFs, points subtracted / excluded).
- Ventral value effect with points subtracted vs excluded: a better points model for foveal areas, or a ventral
  analysis restricted to peripheral voxels (eccentricity > 2 deg).
- Ladder model comparison for the gain fields (plan in `gain_field_plan_2026-11-17.md`), cross-validated.
