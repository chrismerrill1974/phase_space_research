#!/usr/bin/env python3
"""
Paper VII — Phase 5a: Euclid Forecast

Constructs Fisher forecast for Euclid substructure counts
(~170,000 strong lenses, ~75,000 grade A/B) and shows how the
joint constraint collapses with future data.

Two scenarios:
  (i)  ISF-true: fiducial N=6, N_det=1.08/lens, eta=0.05
  (ii) Null: zero ISF perturbers detected

Also computes the falsification threshold: minimum count per lens
such that N>=4 is excluded at 95% confidence.

Depends on:
  ../data/joint_constraint.npz  (Phase 4)
  ../data/lnL_sub_grid.npz     (Phase 3)

Output: ../data/euclid_forecast.npz
"""

import numpy as np
import os

# ── Paths ─────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA_DIR = os.path.join(REPO_ROOT, 'data')

JOINT_PATH = os.path.join(DATA_DIR, 'joint_constraint.npz')
SUB_PATH = os.path.join(DATA_DIR, 'lnL_sub_grid.npz')
OUTPATH = os.path.join(DATA_DIR, 'euclid_forecast.npz')

for label, path in [('Joint constraint', JOINT_PATH), ('Sub grid', SUB_PATH)]:
    if not os.path.isfile(path):
        print(f"ERROR: {label} not found at:\n  {path}")
        raise SystemExit(1)


# ══════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("Phase 5a: Euclid Forecast")
print("=" * 70)

jc = np.load(JOINT_PATH, allow_pickle=True)
sub = np.load(SUB_PATH, allow_pickle=True)

N_grid = jc['N_grid']              # [2, 3, 4, 6, 8, 10]
eps_inf_grid = jc['eps_inf_grid']  # [0.00, 0.25, 0.40, 0.58, 0.68, 0.74]
k_t_grid = jc['k_t_grid']         # 30 values
chi2_wl_grid = jc['chi2_wl_grid']  # (6, 30)
chi2_lcdm = float(jc['chi2_lcdm'])

n_N = len(N_grid)
n_kt = len(k_t_grid)

print(f"\n  Grid: {n_N} N x {n_kt} k_t")


# ══════════════════════════════════════════════════════════════════
# Euclid parameters
# ══════════════════════════════════════════════════════════════════
print("\n[1] Euclid survey parameters")

N_lenses = 75000        # grade A/B strong lenses
eta = 0.05              # conservative detection efficiency
M_threshold_log = 8.8   # O'Riordan+2022: 10^(8.8) M_sun

# Per-lens ISF detectable count at each N
# N_det(N, eps_inf) = (N-1)(1-eps_inf) * C_sub
# where C_sub is the integral over the Schechter mass function * detection volume
# From Paper VI: N_det = 1.08 at N=6, eps_inf=0.58
# So C_sub = 1.08 / [(6-1)(1-0.58)] = 1.08 / 2.10 = 0.5143
C_sub = 1.08 / ((6 - 1) * (1 - 0.58))

N_det_per_lens = np.zeros(n_N)
for iN, (N, eps) in enumerate(zip(N_grid, eps_inf_grid)):
    N_det_per_lens[iN] = (N - 1) * (1 - eps) * C_sub

print(f"\n  N_lenses = {N_lenses} (Euclid grade A/B)")
print(f"  eta = {eta} (conservative detection efficiency)")
print(f"  M_threshold = 10^{M_threshold_log} M_sun")
print(f"  C_sub = {C_sub:.4f} (from Paper VI N=6 calibration)")

print(f"\n  Per-lens ISF detectable count:")
print(f"  {'N':>3s}  {'eps_inf':>7s}  {'(N-1)(1-e)':>10s}  {'N_det':>7s}  {'N_Euclid':>10s}")
for iN in range(n_N):
    N_euclid = N_det_per_lens[iN] * N_lenses * eta
    print(f"  {N_grid[iN]:3d}  {eps_inf_grid[iN]:7.2f}  "
          f"{(N_grid[iN]-1)*(1-eps_inf_grid[iN]):10.2f}  "
          f"{N_det_per_lens[iN]:7.3f}  {N_euclid:10.0f}")


# ══════════════════════════════════════════════════════════════════
# Scenario 1: ISF-true (fiducial N=6)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("[2] Scenario 1: ISF-true (fiducial N=6)")
print("=" * 70)

# Mock observation: the "true" value is at N=6
iN_fid = np.where(N_grid == 6)[0][0]
N_det_true = N_det_per_lens[iN_fid]
n_expected = N_det_true * N_lenses * eta  # expected total detections

print(f"\n  Fiducial: N=6, N_det = {N_det_true:.3f}/lens")
print(f"  Expected Euclid detections: {n_expected:.0f}")
print(f"  Poisson sigma: sqrt(n) = {np.sqrt(n_expected):.1f}")

# For large counts, Poisson -> Gaussian
# The measurement is: n_obs = n_expected +/- sqrt(n_expected)
# Which constrains N_det/lens = n_obs / (N_lenses * eta)
#   sigma(N_det) = sqrt(n_expected) / (N_lenses * eta)
sigma_Ndet = np.sqrt(n_expected) / (N_lenses * eta)

print(f"  sigma(N_det/lens) = {sigma_Ndet:.5f}")
print(f"  Fractional precision: {sigma_Ndet/N_det_true*100:.1f}%")

# Euclid substructure chi^2 for ISF-true scenario
chi2_euclid_isf = np.zeros((n_N, n_kt))
for iN in range(n_N):
    # chi^2 = (N_det_model - N_det_true)^2 / sigma^2
    chi2_val = (N_det_per_lens[iN] - N_det_true)**2 / sigma_Ndet**2
    chi2_euclid_isf[iN, :] = chi2_val  # constant in k_t

print(f"\n  Euclid chi^2 (ISF-true scenario):")
print(f"  {'N':>3s}  {'N_det':>7s}  {'chi2_Euclid':>11s}  {'sigma_away':>11s}")
for iN in range(n_N):
    sig = abs(N_det_per_lens[iN] - N_det_true) / sigma_Ndet
    print(f"  {N_grid[iN]:3d}  {N_det_per_lens[iN]:7.3f}  "
          f"{chi2_euclid_isf[iN, 0]:11.2f}  {sig:11.1f}")

# Joint + Euclid chi^2 (ISF-true)
chi2_joint_euclid_isf = chi2_wl_grid + sub['chi2_sub_grid'] + chi2_euclid_isf

# Best-fit from joint+Euclid
idx_best = np.unravel_index(np.argmin(chi2_joint_euclid_isf),
                             chi2_joint_euclid_isf.shape)
print(f"\n  Joint+Euclid best-fit: N={N_grid[idx_best[0]]}, "
      f"k_t={k_t_grid[idx_best[1]]:.4f}")
print(f"  Dchi^2 vs LCDM: {chi2_joint_euclid_isf[idx_best] - chi2_lcdm:.2f}")

# Marginalised P(N) with Euclid (ISF-true)
dchi2_from_min = chi2_joint_euclid_isf - chi2_joint_euclid_isf.min()
L_euclid_isf = np.exp(-0.5 * dchi2_from_min)

dlog_kt = np.diff(np.log(k_t_grid))
wt = np.zeros(n_kt)
wt[0] = dlog_kt[0] / 2
wt[-1] = dlog_kt[-1] / 2
wt[1:-1] = (dlog_kt[:-1] + dlog_kt[1:]) / 2

P_N_euclid_isf = np.array([np.sum(L_euclid_isf[iN] * wt) for iN in range(n_N)])
P_N_euclid_isf /= P_N_euclid_isf.sum()

print(f"\n  Marginalised P(N) with Euclid (ISF-true):")
print(f"  {'N':>3s}  {'P(N) current':>12s}  {'P(N) +Euclid':>12s}")
P_N_current = jc['P_N_joint']
for iN in range(n_N):
    print(f"  {N_grid[iN]:3d}  {P_N_current[iN]:12.6f}  "
          f"{P_N_euclid_isf[iN]:12.6f}")


# ══════════════════════════════════════════════════════════════════
# Scenario 2: Null (zero ISF perturbers)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("[3] Scenario 2: Null (zero ISF perturbers)")
print("=" * 70)

# If no ISF perturbers are detected, n_obs = 0 (or consistent with 0)
# Conservative: assume n_obs = 0, sigma from Poisson upper limit
# For n=0 Poisson, 95% upper limit is 3.0 events
# But with N_lenses * eta = 3750 effective sightlines, even n_obs = 0
# is extremely constraining.

# Actually, the model predicts n_model = N_det * N_lenses * eta.
# If n_obs = 0, the Poisson likelihood is:
#   L = exp(-n_model) * n_model^0 / 0! = exp(-n_model)
# So: -2 ln L = 2 * n_model = chi^2

# For the null scenario, we use n_obs = 0:
chi2_euclid_null = np.zeros((n_N, n_kt))
for iN in range(n_N):
    n_model = N_det_per_lens[iN] * N_lenses * eta
    # -2 ln L(Poisson) = 2 * (n_model - n_obs * ln(n_model) + ln(n_obs!))
    # For n_obs = 0: -2 ln L = 2 * n_model
    chi2_euclid_null[iN, :] = 2.0 * n_model

# LCDM has N_det = 0 (no ISF contribution), so chi2_LCDM_null = 0
# Delta_chi^2 = chi2_ISF_null - 0 = 2 * n_model

print(f"\n  Euclid exclusion (null scenario, n_obs=0):")
print(f"  {'N':>3s}  {'n_model':>9s}  {'chi2_null':>9s}  {'sigma':>7s}")
for iN in range(n_N):
    n_model = N_det_per_lens[iN] * N_lenses * eta
    sig = np.sqrt(chi2_euclid_null[iN, 0])
    print(f"  {N_grid[iN]:3d}  {n_model:9.0f}  {chi2_euclid_null[iN, 0]:9.0f}  "
          f"{sig:7.0f}")

print(f"\n  ALL N values are excluded at enormous significance if n_obs=0.")
print(f"  Even N=2 (minimum ISF): {np.sqrt(chi2_euclid_null[0, 0]):.0f} sigma.")
print(f"  ISF is decisively falsifiable with Euclid.")


# ══════════════════════════════════════════════════════════════════
# Falsification threshold
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("[4] Falsification threshold")
print("=" * 70)

# What is the minimum N_det/lens such that each N is excluded at 95%?
# 95% exclusion requires Delta_chi^2 > 3.84 (1 dof)
# (N_det_model - N_det_obs)^2 / sigma^2 > 3.84
# where sigma = sqrt(N_det_obs * N_lenses * eta) / (N_lenses * eta)
#             = sqrt(N_det_obs / (N_lenses * eta))

# For each N, find the N_det_obs threshold:
# If N_det_obs < N_det_model:
#   (N_det_model - N_det_obs)^2 > 3.84 * N_det_obs / (N_lenses * eta)
# This is a quadratic in N_det_obs.

print(f"\n  Falsification: minimum observed N_det/lens to exclude each N at 95%")
print(f"  (i.e., if observed count is BELOW this, that N is excluded)")
print(f"\n  {'N':>3s}  {'N_det_model':>11s}  {'N_det_min_95':>12s}  {'n_det_total':>11s}")

N_det_threshold_95 = np.zeros(n_N)
for iN in range(n_N):
    mu = N_det_per_lens[iN]
    # Solve: (mu - x)^2 / [x / (N_lenses * eta)] = 3.84
    # (mu - x)^2 * N_lenses * eta = 3.84 * x
    # Let a = N_lenses * eta = 3750
    a = N_lenses * eta
    # a*(mu-x)^2 = 3.84*x
    # a*mu^2 - 2*a*mu*x + a*x^2 = 3.84*x
    # a*x^2 - (2*a*mu + 3.84)*x + a*mu^2 = 0
    A_coef = a
    B_coef = -(2 * a * mu + 3.84)
    C_coef = a * mu**2
    disc = B_coef**2 - 4 * A_coef * C_coef
    if disc > 0:
        x1 = (-B_coef - np.sqrt(disc)) / (2 * A_coef)
        x2 = (-B_coef + np.sqrt(disc)) / (2 * A_coef)
        # We want the lower root (x < mu)
        N_det_threshold_95[iN] = x1
    else:
        N_det_threshold_95[iN] = 0.0

    n_total = N_det_threshold_95[iN] * N_lenses * eta
    print(f"  {N_grid[iN]:3d}  {mu:11.3f}  {N_det_threshold_95[iN]:12.4f}  "
          f"{n_total:11.0f}")

# The key falsification statement: what count per lens excludes N>=4?
# N>=4 has the lowest N_det at N=4
iN4 = np.where(N_grid == 4)[0][0]
print(f"\n  Key falsification threshold:")
print(f"    If observed N_det/lens < {N_det_threshold_95[iN4]:.4f}")
print(f"    (total < {N_det_threshold_95[iN4] * N_lenses * eta:.0f} detections)")
print(f"    then ISF with N>=4 is excluded at 95% confidence.")

# For the paper's closing sentence:
# N=2 threshold
iN2 = 0
print(f"\n    If observed N_det/lens < {N_det_threshold_95[iN2]:.4f}")
print(f"    (total < {N_det_threshold_95[iN2] * N_lenses * eta:.0f} detections)")
print(f"    then ALL ISF models (N>=2) are excluded at 95% confidence.")


# ══════════════════════════════════════════════════════════════════
# ISF-true scenario: N discrimination
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("[5] ISF-true: N discrimination power")
print("=" * 70)

# If ISF is true at N=6, how well can Euclid distinguish N values?
# sigma_N from Fisher: d(N_det)/dN at discrete grid points
# For each pair of N values, compute the discrimination:

print(f"\n  Pairwise discrimination (sigma) at fiducial N=6:")
print(f"  {'N_test':>6s}  {'dN_det':>8s}  {'sigma':>7s}")
for iN in range(n_N):
    if iN == iN_fid:
        continue
    delta_Ndet = abs(N_det_per_lens[iN] - N_det_per_lens[iN_fid])
    sigma = delta_Ndet / sigma_Ndet
    print(f"  {N_grid[iN]:6d}  {delta_Ndet:8.4f}  {sigma:7.1f}")

# Note: the N_det values are close for N>=4, so N discrimination
# within N>=4 is limited even with Euclid
print(f"\n  Euclid strongly distinguishes N=2 and N=3 from N>=4.")
print(f"  N=4 vs N=6: moderate discrimination.")
print(f"  N=6 vs N=8 vs N=10: weak discrimination (N_det saturates).")


# ══════════════════════════════════════════════════════════════════
# Construct Euclid forecast likelihood grids for figure
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("[6] Constructing forecast grids for figure")
print("=" * 70)

# Current joint chi^2 (WL + sub)
chi2_current = jc['chi2_joint_grid']
dchi2_current = chi2_current - chi2_current.min()

# ISF-true: joint + Euclid
dchi2_isf_true = chi2_joint_euclid_isf - chi2_joint_euclid_isf.min()

# Null: for the exclusion region, we show Delta_chi^2 of each ISF point
# relative to LCDM (which has chi^2 = 0 in the null Euclid scenario)
# Combined: current WL chi^2 + sub chi^2 + Euclid null chi^2
# vs LCDM: chi2_lcdm + 0 (LCDM has no Euclid contribution)
chi2_null_total = chi2_wl_grid + sub['chi2_sub_grid'] + chi2_euclid_null
dchi2_null_vs_lcdm = chi2_null_total - chi2_lcdm

print(f"  Current joint: min Dchi^2 = {dchi2_current.min():.2f}, "
      f"at N={N_grid[np.unravel_index(np.argmin(dchi2_current), dchi2_current.shape)[0]]}")
print(f"  ISF-true forecast: min Dchi^2 = {dchi2_isf_true.min():.2f}")
print(f"  Null exclusion: min Dchi^2 vs LCDM = {dchi2_null_vs_lcdm.min():.1f}")
print(f"                  max Dchi^2 vs LCDM = {dchi2_null_vs_lcdm.max():.1f}")
print(f"  (All ISF models excluded at >>{np.sqrt(dchi2_null_vs_lcdm.min()):.0f} sigma in null)")


# ══════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY — Phase 5a")
print("=" * 70)

print(f"""
  Euclid forecast for ISF substructure:
    Survey: {N_lenses} grade A/B strong lenses
    Detection efficiency: eta = {eta}
    Mass threshold: 10^{M_threshold_log} M_sun

  ISF-true scenario (fiducial N=6):
    Expected detections: {n_expected:.0f} (Poisson sigma: {np.sqrt(n_expected):.1f})
    Fractional precision on N_det: {sigma_Ndet/N_det_true*100:.1f}%
    N=2 excluded at {abs(N_det_per_lens[0] - N_det_true)/sigma_Ndet:.1f} sigma
    N=3 excluded at {abs(N_det_per_lens[1] - N_det_true)/sigma_Ndet:.1f} sigma
    N=4 vs N=6: {abs(N_det_per_lens[iN4] - N_det_true)/sigma_Ndet:.1f} sigma

  Null scenario (zero ISF perturbers):
    N=2 excluded at {np.sqrt(chi2_euclid_null[0, 0]):.0f} sigma
    N=6 excluded at {np.sqrt(chi2_euclid_null[iN_fid, 0]):.0f} sigma
    N=10 excluded at {np.sqrt(chi2_euclid_null[-1, 0]):.0f} sigma
    -> ISF is decisively falsifiable

  Falsification thresholds (95% exclusion):
    ALL ISF (N>=2): N_det/lens < {N_det_threshold_95[0]:.4f} ({N_det_threshold_95[0]*N_lenses*eta:.0f} total)
    N>=4:           N_det/lens < {N_det_threshold_95[iN4]:.4f} ({N_det_threshold_95[iN4]*N_lenses*eta:.0f} total)

  Paper closing sentence:
    "If Euclid substructure counts at 10^{{{M_threshold_log}}} M_sun
     yield fewer than {N_det_threshold_95[0]*N_lenses*eta:.0f} detections across
     {N_lenses} grade-A/B strong lenses, the ISF model is excluded
     at >95% confidence for all N >= 2."
""")


# ══════════════════════════════════════════════════════════════════
# Save checkpoint
# ══════════════════════════════════════════════════════════════════
print("[Saving checkpoint...]")

np.savez(
    OUTPATH,
    # Grid axes
    N_grid=N_grid,
    eps_inf_grid=eps_inf_grid,
    k_t_grid=k_t_grid,
    # Euclid parameters
    N_lenses=N_lenses,
    eta=eta,
    M_threshold_log=M_threshold_log,
    C_sub=C_sub,
    N_det_per_lens=N_det_per_lens,
    sigma_Ndet=sigma_Ndet,
    # ISF-true scenario
    chi2_euclid_isf=chi2_euclid_isf,
    chi2_joint_euclid_isf=chi2_joint_euclid_isf,
    dchi2_isf_true=dchi2_isf_true,
    P_N_euclid_isf=P_N_euclid_isf,
    n_expected_isf=n_expected,
    # Null scenario
    chi2_euclid_null=chi2_euclid_null,
    chi2_null_total=chi2_null_total,
    dchi2_null_vs_lcdm=dchi2_null_vs_lcdm,
    # Falsification thresholds
    N_det_threshold_95=N_det_threshold_95,
    # Current joint (for comparison)
    chi2_current_joint=jc['chi2_joint_grid'],
    dchi2_current=dchi2_current,
    chi2_lcdm=chi2_lcdm,
)

print(f"  Saved to {OUTPATH}")
print("\nPhase 5a complete.")
