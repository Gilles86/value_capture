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
  Student-t), **P(high > low)**, posterior mean and 95% HDI. Directional: high > low.
- **Secondary statistic (same fit):** one-sided Wilcoxon on the per-subject all-trials amplitudes, high > low.

Command:

    EXTRA_ARGS="--glmsingle-deriv glmsingle_fbevents --free-prf --points exclude --prf-model dog --separate-widths \
        --tag fbevents_free_exclude_widths" sbatch --time=08:00:00 --array=<subjects> fit_gain_fields_cpu.sh
    python value_capture/gainfield/hierarchical_value_gain.py \
        --folder notes/data/gain_fields_dog_fbevents_free_exclude_widths --rois hV4

## Everything else is robustness

Other areas and pooled pathways; points subtracted / kept / pRF-overlap cut-off; measured response at the item;
decoding of the item and of the bar near it; colour x value split. Reported, uncorrected, labelled as secondary.

## Result on the main study (n = 11 in hV4; sub-12 has < 10 hV4 voxels with R² >= .1 in every fit)

- Hierarchical: high - low = +123% gain (95% HDI 6 to 241), P(high > low) = .98. Converged (R-hat <= 1.01, no
  divergences).
- Wilcoxon: high +77% (FWHM 1.4 deg) vs low +30% (4.0 deg), p = .012 one-sided.
- Same model in the other areas (secondary): V1 +1 (P = .72), V2 +10 (.85), V3 -22 (.08), LO -77 (HDI -149 to 3,
  P = .02, i.e. the opposite direction), V3AB n = 5. The LO reversal is not significant either way but must be
  reported next to the hV4 effect.
