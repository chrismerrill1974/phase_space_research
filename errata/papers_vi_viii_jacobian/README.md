# Erratum to Papers VI, VII and VIII: the missing Jacobian in the per-lens perturber count

Paper VI counts detectable other-branch galaxies per strong-lens
sightline by integrating a stellar mass function, written per dex of
stellar mass, over the logarithm of halo mass. The conversion carries the
Jacobian d log M_star / d log M_halo, which for the paper's single
power-law abundance matching is the constant 1/1.4, and the script
omitted it. Every count downstream is high by exactly 1.4 and every
Poisson significance by sqrt(1.4): Paper VI's 1.08 per lens is 0.77,
Paper VII's C_sub = 0.514 is 0.367, Paper VIII's wide-survey expectation
4,463-5,955 is 3,188-4,253 and its 312-system in-principle test
12.6-16.8 at 5.0-5.8 sigma is 9.0-12.0 at 4.2-4.9 sigma. The joint
weak-lensing fit of Paper VII, Paper VI's profiles, Paper IX and Paper
X's own counts are untouched. The audit also found that Paper VI's text
quotes Schechter parameters its script does not use; the text is
corrected to the script's parameters. The note (`erratum_vi_viii.pdf`,
`erratum_vi_viii.tex`) has every changed number, published beside
corrected, and the restated falsification statements.

## Scripts (run in this order; each gated on reproducing the published product)

- `step0_audit_jacobian.py` — reproduces Paper VI's published per-lens
  counts with its own functions (gate), recomputes them with the
  Jacobian, tabulates the three Schechter parameter sets in play, and
  lists the files in the Paper VI-IX workspaces that carry a fed
  quantity. Printout: `audit_jacobian.txt`.
- `step1_paper7_forecast_v2.py` — Paper VII's Phase 5a Euclid forecast
  as a function of C_sub: reproduces `euclid_forecast.npz` at the
  published C_sub (gate), writes `euclid_forecast_v2.npz` at C_sub/1.4
  beside it, prints every changed number old beside new. Printout:
  `paper7_forecast_v2.txt`.
- `step2_paper8_forecast_v2.py` — Paper VIII's recalibrated forecast
  from the v2 input: reproduces `recalibrated_forecast.npz` from the v1
  input (gate), writes `recalibrated_forecast_v2.npz` beside it, prints
  every changed number old beside new. Printout: `paper8_forecast_v2.txt`.

All three run in seconds, single-threaded. The two `_v2` products are in
this repo as `data/paper7/euclid_forecast_v2.npz` and
`data/paper8/recalibrated_forecast_v2.npz`; the originals are unchanged.

## Expected directory layout

The scripts use relative paths from a series root holding the three
paper workspaces, with this folder two levels below it:

```
<series root>/
  dark_matter_predictions_6_strong_lensing/003_phase2/paper6_phase2_results.json
                                            (in this repo: data/paper10/paper6_phase2_results.json)
  dark_matter_predictions_7/data/            joint_constraint.npz, lnL_sub_grid.npz,
                                            euclid_forecast.npz  (in this repo: data/paper7/)
  dark_matter_predictions_8/data/            q1_demographics.npz, recalibrated_forecast.npz
                                            (in this repo: data/paper8/)
  dark_matter_predictions_8/data/from_paper_vii/  copies of the three Paper VII products
  <erratum folder>/01_audit/step0_audit_jacobian.py
  <erratum folder>/02_propagate/step1_paper7_forecast_v2.py, step2_paper8_forecast_v2.py
  <erratum folder>/data/                     printouts and audit_jacobian.npz
```

The audit's file listing (its part d) walks the workspace trees and is
informational; its gate and corrected numbers need only the four data
files named above.
