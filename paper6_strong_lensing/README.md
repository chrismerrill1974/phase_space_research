# Paper VI: Galaxy-Mass Dark Substructure from Strong Lensing

Scripts for the ISF strong-lensing predictions paper. **Version 2
(2026-10-11)**: version 1 counted detectable other-branch galaxies per
lens by evaluating the stellar mass function, per dex of stellar mass,
at the abundance-matched stellar mass and summing over d log M_halo
without the Jacobian d log M_star / d log M_halo = 1/1.4 of the matching
relation, so every count downstream was high by exactly 1.4 (1.08 per
lens -> 0.77; ~4,000 Euclid detections at 5 percent efficiency ->
~2,900). Version 1 also quoted Schechter parameters (phi* 0.012, alpha
-1.05) the script did not use (3.96e-3, -1.16). The scripts here are the
version-2 scripts; the version-1 scripts are in this repository's
history before 2026-10-11, and the audit that found and propagated the
error is in `errata/papers_vi_viii_jacobian/`.

## Scripts (run in this order from this folder; seconds each, single-threaded)

- `paper6_phase2_isf_predictions.py` — epsilon(k) at halo scales, the ISF
  excess mass function (with the Jacobian, `JACOBIAN = 1/1.4` applied in
  `isf_excess_dndm`), per-lens abundances, LOS fraction, density
  profiles and convergence, the Meneghetti bracket. Before writing, it
  GATES: with the Jacobian set to 1 it must reproduce the version-1
  results file to 1e-6 in the per-lens counts (see layout below).
  Output: `paper6_phase2_results.json`.
- `paper6_phase2c_figures.py` — Figures 1 and 2 (epsilon vs mass; the
  mass function, ISF excess vs LCDM subhalos).
- `paper6_phase2d_profiles.py` — Figure 3 (density profiles and
  convergence; unchanged by the correction).
- `paper6_phase3b_figure4.py` — Figure 4 (mass function vs published
  constraints) and `paper6_phase3b_metrics.json`: sub-galactic
  fractions, crossover mass, integrated ISF/LCDM ratio.
- `paper6_phase3c_meneghetti.py` — Figure 5 (Meneghetti GGSL bracket;
  uses the excess factor, unchanged) and `paper6_phase3c_metrics.json`.
- `paper6_phase3d_los_profile.py` — Figure 6 (LOS fraction vs mass,
  convergence profile) and `paper6_phase3d_metrics.json`.
- `paper6_phase3e_synthesis.py` — Figure 7 (scorecard) and
  `paper6_phase3e_summary.json`; reads every number from the results
  and metrics files.
- `check_paper6_v2_text.py` — checks the paper's tex carries the
  version-2 numbers and none of the version-1 ones.

## Expected directory layout

The version-2 scripts read and write in their own folder (`figures/`
for the PNGs). Two inputs are expected at these relative paths:

```
../004_phase3/paper6_handoff/phase2_outputs/paper6_phase2_results.json
        the version-1 results, for the phase-2 gate
        (in this repo: data/paper10/paper6_phase2_results.json is an
        earlier version-1 run with older key names; the gate wants the
        copy the paper used, which the version-2 results file in
        this folder reproduces to 1e-6 when JACOBIAN is set to 1)
../005_writeup/paper6.tex
        for check_paper6_v2_text.py
```

Figures and the version-2 results file are not stored here; run the
scripts to regenerate them.
