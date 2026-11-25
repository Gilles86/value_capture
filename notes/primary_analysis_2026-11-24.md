# Primary analysis for the distractor value effect (declared 2026-11-24)

**Status of this declaration.** It was written after the main-study data (n = 12) had been analysed in many ways, so
for those data it is a choice of headline analysis, not a pre-registration. It becomes confirmatory only for data
analysed after this date: the pilot sessions once processed, and any new scans. Those must be analysed with exactly
this pipeline, and the result reported whatever it is.

## Question

Does a high-value distractor get a larger attentional gain in visual cortex than a low-value one?

## Primary test

- **Betas:** `glmsingle_fbevents` (points display as its own single-trial events).
- **Trials:** distractor-present and -absent trials **without the points display** (`--points exclude`). Excluding
  the points trials removes the points leak by construction (it is retinotopic, grows with the HRF overlap and the
  number of characters, `notes/decoded_gain_results_2026-11-20.md`).
- **Model:** gain fields on DoG pRFs, pRFs fitted jointly with the gains on the same betas (`--free-prf`), one field
  per effect with its own amplitude and width (`--separate-widths`): target, low-value distractor, high-value
  distractor. Both sessions together, so colour is balanced.
- **Area:** hV4 (neuropythy), voxels with pRF R² >= .1; a subject enters with >= 10 such voxels.
- **Statistic:** hierarchical Bayesian model of the per-run gains
  (`gainfield/hierarchical_value_gain.py`: `gain ~ high_minus_low + (high_minus_low | subject) + (1 | run)`,
  Student-t, **residual scale per subject**: `log sigma ~ 1 + (1 | subject)`), **P(high > low)**, posterior mean and
  95% HDI. Directional: high > low. (Amended 2026-11-25, see below.)
- **Secondary statistic, all areas jointly:** the retsupp meta-analysis model (`gainfield/meta_value_gain.py`): one
  estimate per subject x area with its standard error from the per-run gains, group (Early / Dorsal /
  Occipito-temporal), area and subject effects.
- **Secondary statistic (same fit):** one-sided Wilcoxon on the per-subject all-trials amplitudes, high > low.

Command:

    EXTRA_ARGS="--glmsingle-deriv glmsingle_fbevents --free-prf --points exclude --prf-model dog --separate-widths \
        --tag fbevents_free_exclude_widths" sbatch --time=08:00:00 --array=<subjects> fit_gain_fields_cpu.sh
    python value_capture/gainfield/hierarchical_value_gain.py \
        --folder notes/data/gain_fields_dog_fbevents_free_exclude_widths --rois hV4

## Amendment 2026-11-25: noise per subject

The run-to-run spread of the hV4 gains differs about 15-fold between subjects (SD of per-run high - low 50% to 875%).
With one shared residual scale the noisiest subjects' runs count as much as the cleanest subjects' runs, so the
primary model now gives every subject its own residual scale (as in retsupp, where the group models weight each
subject by its own standard error). Made before any pilot gain field had been fitted. The shared-scale result is kept
as a robustness row (`--noise shared`).

## Everything else is robustness

Other areas and pooled pathways; points subtracted / kept / pRF-overlap cut-off; measured response at the item;
decoding of the item and of the bar near it; colour x value split. Reported, uncorrected, labelled as secondary.

## Result on the main study (n = 11 in hV4; sub-12 has < 10 hV4 voxels with R² >= .1 in every fit)

- Hierarchical, residual scale per subject (primary): high - low = +78% gain (95% HDI 26 to 135),
  P(high > low) = .996. Converged (R-hat <= 1.01, no divergences). Shared scale: +123% (6 to 241), P = .98.
- Meta-analysis over areas, SE-weighted: hV4 +54% (15 to 96), P = .995; Student-t +34% (1 to 71), P = .986; without
  SE weighting +83% (11 to 155), P = .986.
- Wilcoxon: high +77% (FWHM 1.4 deg) vs low +30% (4.0 deg), p = .012 one-sided.
- Same model in the other areas (secondary, per-subject scale): V1 +1 (P = .65), V2 +6 (.81), V3 -14 (.11),
  LO -78 (HDI -155 to 3, P = .02, the opposite direction), V3AB n = 5. In the SE-weighted meta-analysis LO is
  credibly negative (-80, HDI -123 to -37); the LO reversal must be reported next to the hV4 effect. The pathway
  means are null (overall -8, HDI -26 to 11): hV4 and LO cancel within the occipito-temporal group.
