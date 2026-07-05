# Paper IX: Black Holes as Branch-Blind Objects

Scripts for the theory paper proposing a gravitational-capture origin
for the ISF cross-branch correlation epsilon(k), with black holes as
branch-blind anchor objects and a dated falsification calendar.

## Scripts

- `step1_eps_evolution.py` — Computes the capture-mechanism predictions:
  eps_inf(z) proportional to the Press-Schechter collapsed mass fraction
  (M_min = 1e10-1e12 Msun/h), and k_t(z) tracking the nonlinear scale
  k_NL(z), compared against the Paper VII fitted values.
  Input: `pknl_camb.npz` (linear P(k,z) from Paper VII Phase 1a; included
  in `data/paper9/`). Output: `eps_evolution.npz`.
- `step2_bh_accounting.py` — Soltan-argument bound on the cross-branch
  capture efficiency gamma(N), and the early-universe overmassive
  headroom vs the JWST z > 6 anomaly. Output: `bh_accounting.npz`.
- `step3_figures.py` — Produces the three paper figures (mechanism
  schematic, eps(k,z) evolution/discriminant, gamma(N) + JWST panels).

## Expected directory layout

Scripts use relative paths from a project root containing:

```
data/       # pknl_camb.npz input and .npz outputs
            # (in this repo: data/paper9/)
figures/    # figure outputs
```

All scripts run in seconds, single-threaded. No external data downloads
required — `pknl_camb.npz` is the only input and is included.
