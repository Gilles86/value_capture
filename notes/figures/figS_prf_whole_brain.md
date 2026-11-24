# Supplementary Figure: whole-brain PRF maps

Files: `notes/figures/figS_prf_whole_brain.pdf` / `.png` (180 × 309 mm).
Script: `value_capture/visualize/figS_prf_whole_brain.py` (its docstring has the render and compose commands).
Renders: `notes/figures/prf_inflated_wholebrain/` (colour settings in `render_settings.json`).

## Legend

**Well-fitted population receptive fields lie in the retinotopic visual areas.** Variance explained (R²)
by the population receptive field (PRF) model, shown on each participant's inflated cortical surface
(n = 12, sub-04 to sub-15; one row per participant). Columns show the left hemisphere from the side
(lateral) and from the midline (medial), the right hemisphere from the midline and from the side, both
hemispheres from below (ventral; occipital pole at the top) and from behind (posterior). Together the
views cover the whole cortex. PRFs were fitted to GLMsingle single-trial response amplitudes of both
scanning sessions, with a 2D Gaussian PRF and a stimulus model containing the mapping bar and the eight
search-array items (bar + items fit). R² is in-sample and was sampled to each participant's FreeSurfer
surface. Colour (inferno, 0 to .15) fades in over the curvature (light grey, gyri; dark grey, sulci)
from R² = .01 (transparent) to R² = .06 (opaque); values above .15 are drawn at the top colour.
White lines are the borders of the visual areas: V1–V3, hV4, VO, LO, TO and V3A/B from each
participant's Bayesian retinotopic atlas fit (neuropythy), with IPS0–5 and FEF from the Wang et al.
(2015) probabilistic atlas where neuropythy assigns no area. Top row: the visual areas of one
participant (sub-04), coloured as in the key below it.

Across participants, of the 68,393 vertices with R² > .06 (opaque in the maps), 83 (0.12%) lie outside
these areas, all within a few millimetres of an area border (mostly at the edges of hV4, VO and TO; most
in sub-07, left hemisphere, 33 vertices). At R² > .1, 2 of ~51,000 vertices lie outside.

Note on the colour ramp: the median R² over all cortical vertices is ≈ .01, so the start of the ramp
sits at the noise floor of the fit. The faint, speckled purple tint seen across non-visual cortex
(e.g. on the medial and lateral frontal surface) comes from vertices with R² between .01 and ~.03 and
does not form retinotopic structure.
