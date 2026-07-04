# Paper VIII: Euclid Q1 Demographics Recalibration

Scripts for anchoring the Paper VII Euclid falsification forecast to
measured Q1 Strong Lensing Discovery Engine demographics.

## Scripts

- `step1_q1_demographics.py` — Parses the Q1 SLDE lens catalogue and
  automated mass models: grade A/B counts, rate per deg^2, Einstein-radius
  distribution, DR1/wide-survey extrapolations.
  Output: `q1_demographics.npz`.
- `step2_recalibrated_forecast.py` — Reruns the Paper VII Phase 5a Poisson
  forecast with measured inputs, over three detection-efficiency models
  (flat / linear / area in theta_E) and three survey stages (Q1 / DR1 /
  Wide). Also computes the Q1 in-principle bound from the 312 modelled
  grade-A/B systems. Output: `recalibrated_forecast.npz`.
- `step3_figures.py` — Produces the three paper figures.

## Expected directory layout

The scripts use relative paths from a project root containing:

```
data/raw/                 # Q1 SLDE CSVs (see below)
data/from_paper_vii/      # joint_constraint.npz, lnL_sub_grid.npz,
                          # euclid_forecast.npz  (in this repo: data/paper7/)
data/                     # outputs (in this repo: data/paper8/)
figures/                  # figure outputs
```

## Input data

Q1 SLDE catalogue (not redistributed here; ~1 MB of CSVs from the Euclid
Collaboration's public record, Zenodo 10.5281/zenodo.15003116 /
record 15025832):

- `q1_discovery_engine_lens_catalog.csv`
- `modeling_lens_mass.csv`

Paper VII likelihood grids are already in this repo under `data/paper7/`.

All scripts run in seconds, single-threaded.
