# Paper X: The Galaxy-Scale Dark Mass Profile Under ISF

Scripts for the theory paper deriving the dark mass profile around a
single galaxy from the Paper IX capture ontology, with a dated prediction
for the Milky Way's outer rotation-curve slope and a corrected detection
threshold for the Euclid Q1 perturber search (Paper XI).

## Scripts (run in this order)

- `step1_hydrostatic_halo.py` — Other-branch diffuse gas as an isothermal
  hydrostatic atmosphere in the shared well, with the Milky Way B2 baryons
  (de Salas et al. 2019; razor-thin exponential discs). Level 0 gate on the
  singular isothermal sphere; Levels 1-2 over M_200 and T/T_vir. Shows the
  cold bound and that the atmosphere is too diffuse to be the halo.
  Output: `hydrostatic_halo.npz`.
- `step2_dissipated_component.py` — Captured other-branch collapsed matter
  as the halo: M_cap = (N-1) eps_inf(N) M_bar with eps_inf(N) from the
  Paper VII product; two geometries; the required-mass table; the chain
  N -> extent -> outer slope beta(15-30 kpc).
  Input: `joint_constraint_paper7.npz` (= `data/paper7/joint_constraint.npz`).
  Output: `dissipated_component.npz`.
- `step2b_combine_hot_remainder.py` — Adds the hot remainder
  (M_200 - M_bar - M_cap at >= T_vir) and re-solves the captured extent to
  the solar-radius datum; curves to R_200, the 100 kpc dip, enclosed
  masses at 50 and 100 kpc. Output: `combined_halo.npz`.
  Runtime about 2-4 minutes single-threaded (hundreds of shooting solves).
- `step3_perturber_profile.py` — Reproduces Paper VI's convergence table
  and 1.08 per lens, and Paper VIII's 312-system row (three gates), then
  the ISF perturber as a Hernquist spheroid under two extent scalings, the
  aperture-mass threshold, corrected C_sub, and the Paper XI numbers,
  with the promoted mass-to-number mapping.
  Inputs: `paper6_phase2_results.json`, `euclid_forecast_paper7.npz`,
  `q1_demographics_paper8.npz`. Output: `perturber_profile.npz`.
- `step4_figures.py` — Produces the four paper figures.

## Expected directory layout

Scripts use relative paths from a project root containing:

```
01_theory/     # steps 1-3 (this folder's contents)
02_figures/    # step4_figures.py
data/          # inputs listed above and the .npz outputs
               # (in this repo: data/paper10/)
figures/       # figure outputs
```

The only external input is the Milky Way baryonic model, whose
parameters are written into the scripts. Steps 1, 2, 3 and 4 run in
seconds; step 2b takes minutes. All single-threaded.
