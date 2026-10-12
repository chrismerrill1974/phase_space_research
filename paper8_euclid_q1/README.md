# Paper VIII: Euclid Q1 Demographics Recalibration

Scripts for anchoring the Paper VII Euclid falsification forecast to
measured Q1 Strong Lensing Discovery Engine demographics. **Version 2
(2026-10-11)**: the forecast inherited Paper VI's missing Jacobian 1/1.4
through Paper VII's per-lens counts; the scripts here read Paper VII's
version-2 product (`euclid_forecast_v2.npz`) and write
`recalibrated_forecast_v2.npz`, with every count 1/1.4 of version 1 and
every significance 1/sqrt(1.4) of it (wide-survey fiducial 3,188-4,253,
was 4,463-5,955; 312-system test 9.0-12.0 at 4.2-4.9 sigma, was
12.6-16.8 at 5.0-5.8). The version-1 scripts are in this repository's
history before 2026-10-11; the audit is in `errata/papers_vi_viii_jacobian/`.

## Scripts

- `step1_q1_demographics.py` — Parses the Q1 SLDE lens catalogue and
  automated mass models: grade A/B counts, rate per deg^2, Einstein-radius
  distribution, DR1/wide-survey extrapolations.
  Output: `q1_demographics.npz`.
- `step2_recalibrated_forecast.py` — Reruns the Paper VII Phase 5a Poisson
  forecast with measured inputs, over three detection-efficiency models
  (flat / linear / area in theta_E) and three survey stages (Q1 / DR1 /
  Wide). Also computes the Q1 in-principle bound from the 312 modelled
  grade-A/B systems. Reads `euclid_forecast_v2.npz`; gates on the input
  counts being version 1's times 1/1.4 and on an existing v2 product
  matching the run. Output: `recalibrated_forecast_v2.npz`.
- `step3_figures.py` — Produces the three paper figures from the v2
  product into a `figures/` folder beside the script.
- `check_paper8_v2_text.py` — Checks the paper's tex carries the
  version-2 numbers and none of the version-1 ones.

## Expected directory layout

The scripts use relative paths from a project root containing:

```
data/raw/                 # Q1 SLDE CSVs (see below)
data/from_paper_vii/      # joint_constraint.npz, lnL_sub_grid.npz,
                          # euclid_forecast.npz and euclid_forecast_v2.npz
                          # (in this repo: data/paper7/)
data/                     # outputs (in this repo: data/paper8/, which holds
                          # recalibrated_forecast.npz and _v2.npz)
<script folder>/figures/  # figure outputs (version 2)
```

## Input data

Q1 SLDE catalogue (not redistributed here; ~1 MB of CSVs from the Euclid
Collaboration's public record, Zenodo 10.5281/zenodo.15003116 /
record 15025832):

- `q1_discovery_engine_lens_catalog.csv`
- `modeling_lens_mass.csv`

Paper VII likelihood grids are already in this repo under `data/paper7/`.

All scripts run in seconds, single-threaded.
