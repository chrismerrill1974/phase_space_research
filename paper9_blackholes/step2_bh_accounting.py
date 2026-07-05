#!/usr/bin/env python3
"""
Paper IX — Step 2: Black-hole growth accounting vs N

If N branches feed shared black holes but each branch observes only its
own accretion luminosity, the local BH mass density exceeds the
Soltan-argument accounting by

    rho_BH = [1 + (N-1) * gamma] * rho_acc_visible,

where gamma is the mean cross-branch capture efficiency (suppressed by
partial co-location: branch j usually hosts no galaxy at branch i's BH).

This script:
  (a) derives the allowed gamma(N) region from the observed
      rho_BH / rho_acc consistency band (literature values with
      generous uncertainties, radiative efficiency eff = 0.1);
  (b) reports the gamma upper bound at the lensing-preferred N values
      (N = 6 fiducial, N = 10 joint best-fit from Paper VII);
  (c) computes the early-universe headroom: the overmassive factor
      1 + (N-1)*gamma_early for gamma_early -> 1 (maximally shared
      early wells), compared with the JWST z > 6 overmassive-BH range.

Literature anchors (generous bands, see paper Sect. 5 for refs):
  rho_BH(z=0)  = (4.5 +/- 1.5) x 10^5 Msun/Mpc^3
  rho_acc(eff=0.1) = (4.0 +/- 1.2) x 10^5 Msun/Mpc^3

Output: ../data/bh_accounting.npz
"""

import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
OUTPATH = os.path.join(ROOT, 'data', 'bh_accounting.npz')

RHO_BH, SIG_BH = 4.5e5, 1.5e5
RHO_ACC, SIG_ACC = 4.0e5, 1.2e5

# Ratio r = rho_BH / rho_acc with propagated (uncorrelated) errors
r = RHO_BH / RHO_ACC
sig_r = r * np.sqrt((SIG_BH/RHO_BH)**2 + (SIG_ACC/RHO_ACC)**2)
print(f"Soltan ratio r = rho_BH/rho_acc = {r:.2f} +/- {sig_r:.2f}")

# r = 1 + (N-1) gamma  ->  gamma = (r - 1)/(N - 1)
# 95% upper bound on gamma at each N (one-sided Gaussian)
r_95 = r + 1.645 * sig_r
N_grid = np.array([2, 3, 4, 6, 8, 10])
gamma_best = np.clip((r - 1) / (N_grid - 1), 0, None)
gamma_95 = (r_95 - 1) / (N_grid - 1)

print(f"\n95% upper bound r < {r_95:.2f}")
print(f"\n  {'N':>3s}  {'gamma_best':>10s}  {'gamma_95':>9s}")
for N, gb, g95 in zip(N_grid, gamma_best, gamma_95):
    print(f"  {N:3d}  {gb:10.3f}  {g95:9.3f}")

# Early-universe headroom: overmassive factor if gamma_early -> 1
overmassive_max = 1 + (N_grid - 1) * 1.0
print(f"\nMaximal early overmassive factor 1+(N-1) (gamma_early=1):")
for N, f in zip(N_grid, overmassive_max):
    print(f"  N={N:2d}: {f:4.0f}x")
print("\nJWST z>6 overmassive range (BH/M* vs local relation): ~10-100x")
print("-> N >= 10 reaches the low edge of the JWST band with gamma_early=1;")
print("   smaller N requires part of the excess to be astrophysical (seed")
print("   models etc.). Framed as consistency-of-sign, not explanation.")

# Consistency check at the Paper VII/VIII preferred N values
for N_pref in (6, 10):
    i = int(np.where(N_grid == N_pref)[0][0])
    print(f"\nAt N={N_pref}: late-time gamma_95 = {gamma_95[i]:.3f} "
          f"(cross-branch capture must be <~{100*gamma_95[i]:.0f}% "
          f"of visible accretion today)")

np.savez(
    OUTPATH,
    rho_bh=RHO_BH, sig_bh=SIG_BH, rho_acc=RHO_ACC, sig_acc=SIG_ACC,
    r=r, sig_r=sig_r, r_95=r_95,
    N_grid=N_grid, gamma_best=gamma_best, gamma_95=gamma_95,
    overmassive_max=overmassive_max,
    jwst_overmassive_range=np.array([10.0, 100.0]),
)
print(f"\nSaved {OUTPATH}")
