#!/usr/bin/env python3
"""
Phase B, Step B1: Check for ell-dependent calibration driving k_t

The question: does the data/LCDM ratio have a systematic slope with ell
that could be mimicking the ISF signal and pulling k_t to 0.81?

Step A1 tested this per-pair and found r = -0.09 (no degeneracy).
This script goes further:

  1. Covariance-weighted pair-averaged ratio vs ell
     (what the chi^2 fitting actually "sees")
  2. Residual pattern: data - A_opt*LCDM, as a function of ell
     (the residual that ISF is trying to fit)
  3. Compare that residual to the ISF improvement pattern
  4. Per-ell chi^2 contribution breakdown

Run: python3 step_B1_ell_calibration.py

Depends on:
  ../../data/kids1000_bandpowers.npz
  ../../data/response_operators_tomo.npz
"""

import numpy as np
import os

# ── Paths ─────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
DATA_DIR = os.path.join(REPO_ROOT, 'data')

KIDS_PATH = os.path.join(DATA_DIR, 'kids1000_bandpowers.npz')
TOMO_PATH = os.path.join(DATA_DIR, 'response_operators_tomo.npz')

for label, path in [('KiDS data', KIDS_PATH), ('Tomo operators', TOMO_PATH)]:
    if not os.path.isfile(path):
        print(f"ERROR: {label} not found at:\n  {path}")
        raise SystemExit(1)

# ── Load ──────────────────────────────────────────────────────────
print("=" * 70)
print("Step B1: ell-dependent calibration check")
print("=" * 70)

dk = np.load(KIDS_PATH, allow_pickle=True)
dr = np.load(TOMO_PATH, allow_pickle=True)

bp_data = dk['bp_data']             # (15, 8)
ell_eff = dk['ell_eff']             # (8,)
cov_ee = dk['cov_ee']               # (120, 120)
pair_labels = dk['pair_labels']     # (15, 2)

A_tomo = dr['A_tomo']               # (15, 8, 500)
k_hMpc = dr['k_hMpc']              # (500,)
cl_lcdm_tomo = dr['cl_lcdm_tomo']  # (15, 8)

n_pairs = len(pair_labels)
n_ell = len(ell_eff)
n_data = n_pairs * n_ell
n_k = len(k_hMpc)

# ── Inverse covariance and LCDM profiled amplitude ───────────────
data_flat = bp_data.flatten()
cov_inv = np.linalg.inv(cov_ee)

m_lcdm = cl_lcdm_tomo.flatten()
Cinv_m = cov_inv @ m_lcdm
Cinv_d = cov_inv @ data_flat
dCd = data_flat @ Cinv_d
dCm = data_flat @ Cinv_m
mCm = m_lcdm @ Cinv_m
A_opt_lcdm = dCm / mCm
chi2_lcdm = dCd - dCm**2 / mCm

print(f"\n  LCDM profiled amplitude: A_opt = {A_opt_lcdm:.1f}")
print(f"  LCDM profiled chi2 = {chi2_lcdm:.2f}")
print(f"  chi2/Ndof = {chi2_lcdm/(n_data-1):.3f}")

# Best-fit LCDM model vector (amplitude-scaled)
model_lcdm_scaled = A_opt_lcdm * m_lcdm


# ══════════════════════════════════════════════════════════════════
# TEST 1: Residual data - A*LCDM, grouped by ell
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 1: Residuals (data - A_opt*LCDM) grouped by ell-bin")
print("=" * 70)

residual = data_flat - model_lcdm_scaled
diag_sigma = np.sqrt(np.diag(cov_ee))
norm_residual = residual / diag_sigma

# Group by ell-bin: each ell-bin has 15 data points (one per pair)
# The flat ordering is pair-major: [pair0_ell0, pair0_ell1, ..., pair0_ell7,
#                                    pair1_ell0, ..., pair14_ell7]
print(f"\n  Normalized residuals (data - A*LCDM)/sigma, averaged per ell-bin:")
print(f"  (A positive mean = data systematically above scaled LCDM at that ell)")
print(f"\n  {'ell':>7s}  {'mean_r':>8s}  {'std_r':>8s}  {'median_r':>8s}  {'N>0':>4s}/{'':<4s}")

ell_mean_residual = np.zeros(n_ell)
ell_std_residual = np.zeros(n_ell)

for il in range(n_ell):
    # Indices for this ell-bin across all pairs
    indices = [ip * n_ell + il for ip in range(n_pairs)]
    r_ell = norm_residual[indices]
    ell_mean_residual[il] = np.mean(r_ell)
    ell_std_residual[il] = np.std(r_ell)
    n_pos = np.sum(r_ell > 0)
    print(f"  {ell_eff[il]:7.1f}  {np.mean(r_ell):8.3f}  {np.std(r_ell):8.3f}  "
          f"{np.median(r_ell):8.3f}  {n_pos:4d}/{n_pairs:<4d}")

# Fit a linear trend to the ell-averaged residual
log_ell = np.log(ell_eff / 500.0)
coeffs = np.polyfit(log_ell, ell_mean_residual, 1)
slope = coeffs[0]
intercept = coeffs[1]

print(f"\n  Linear fit to mean residual vs log(ell/500):")
print(f"    slope = {slope:.4f}")
print(f"    intercept = {intercept:.4f}")
print(f"    (slope=0 means no systematic ell trend)")

if abs(slope) < 0.5:
    print(f"    -> SMALL slope: no significant ell-dependent systematic")
elif abs(slope) < 1.0:
    print(f"    -> MODERATE slope: possible mild ell trend")
else:
    print(f"    -> LARGE slope: systematic ell trend present")


# ══════════════════════════════════════════════════════════════════
# TEST 2: Per-ell chi^2 contribution
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 2: Per-ell chi^2 contribution (LCDM vs ISF)")
print("=" * 70)

# Compute ISF model at k_t = 0.81 h/Mpc, N=6
N_isf = 6
eps_inf = 0.58
kt_isf = 0.81
alpha = 2

eps_k = eps_inf / (1.0 + (kt_isf / k_hMpc)**alpha)
M = (1.0 + (N_isf - 1) * eps_k) / N_isf
M_sq = M**2

cl_isf = np.array([A_tomo[ip] @ M_sq for ip in range(n_pairs)])
m_isf = cl_isf.flatten()

# Profile amplitude for ISF
Cinv_m_isf = cov_inv @ m_isf
dCm_isf = data_flat @ Cinv_m_isf
mCm_isf = m_isf @ Cinv_m_isf
A_opt_isf = dCm_isf / mCm_isf
chi2_isf = dCd - dCm_isf**2 / mCm_isf

model_isf_scaled = A_opt_isf * m_isf

print(f"\n  ISF (N=6, k_t=0.81): A_opt = {A_opt_isf:.1f}, chi2 = {chi2_isf:.2f}")
print(f"  Delta_chi2 (ISF - LCDM) = {chi2_isf - chi2_lcdm:.2f}")

# Compute per-data-point contribution to chi^2 for both models
# chi^2 = sum_ij (d-m)_i C^{-1}_{ij} (d-m)_j
# Per-point contribution (approximate, using diagonal only for interpretability):
resid_lcdm = data_flat - model_lcdm_scaled
resid_isf = data_flat - model_isf_scaled

# Full chi^2 contribution per data point: (d-m)_i * sum_j C^{-1}_{ij} (d-m)_j
chi2_per_point_lcdm = resid_lcdm * (cov_inv @ resid_lcdm)
chi2_per_point_isf = resid_isf * (cov_inv @ resid_isf)
delta_chi2_per_point = chi2_per_point_isf - chi2_per_point_lcdm

# Sum per ell-bin
print(f"\n  Per-ell chi^2 contribution:")
print(f"  {'ell':>7s}  {'chi2_LCDM':>10s}  {'chi2_ISF':>10s}  {'Delta':>8s}  {'ISF_better?':>12s}")

chi2_per_ell_lcdm = np.zeros(n_ell)
chi2_per_ell_isf = np.zeros(n_ell)

for il in range(n_ell):
    indices = [ip * n_ell + il for ip in range(n_pairs)]
    c_lcdm = np.sum(chi2_per_point_lcdm[indices])
    c_isf = np.sum(chi2_per_point_isf[indices])
    chi2_per_ell_lcdm[il] = c_lcdm
    chi2_per_ell_isf[il] = c_isf
    delta = c_isf - c_lcdm
    better = "YES" if delta < -0.5 else ("marginal" if delta < 0 else "no")
    print(f"  {ell_eff[il]:7.1f}  {c_lcdm:10.2f}  {c_isf:10.2f}  {delta:8.2f}  {better:>12s}")

print(f"\n  Total: LCDM={np.sum(chi2_per_ell_lcdm):.2f}, "
      f"ISF={np.sum(chi2_per_ell_isf):.2f}, "
      f"Delta={np.sum(delta_chi2_per_point):.2f}")
print(f"  (Note: per-point chi2 sums to full chi2 by construction)")

# Which ell-bins drive the ISF preference?
delta_per_ell = chi2_per_ell_isf - chi2_per_ell_lcdm
il_best = np.argmin(delta_per_ell)
il_worst = np.argmax(delta_per_ell)
print(f"\n  ISF gains most at ell={ell_eff[il_best]:.0f} "
      f"(Delta_chi2 = {delta_per_ell[il_best]:.2f})")
print(f"  ISF loses most at ell={ell_eff[il_worst]:.0f} "
      f"(Delta_chi2 = {delta_per_ell[il_worst]:.2f})")


# ══════════════════════════════════════════════════════════════════
# TEST 3: Per-pair chi^2 contribution
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 3: Per-pair chi^2 contribution (LCDM vs ISF)")
print("=" * 70)

print(f"\n  {'Pair':>6s}  {'chi2_LCDM':>10s}  {'chi2_ISF':>10s}  {'Delta':>8s}  {'ISF_better?':>12s}")

chi2_per_pair_lcdm = np.zeros(n_pairs)
chi2_per_pair_isf = np.zeros(n_pairs)

for ip in range(n_pairs):
    indices = [ip * n_ell + il for il in range(n_ell)]
    c_lcdm = np.sum(chi2_per_point_lcdm[indices])
    c_isf = np.sum(chi2_per_point_isf[indices])
    chi2_per_pair_lcdm[ip] = c_lcdm
    chi2_per_pair_isf[ip] = c_isf
    delta = c_isf - c_lcdm
    b1, b2 = pair_labels[ip]
    better = "YES" if delta < -0.5 else ("marginal" if delta < 0 else "no")
    print(f"  ({b1},{b2})  {c_lcdm:10.2f}  {c_isf:10.2f}  {delta:8.2f}  {better:>12s}")

delta_per_pair = chi2_per_pair_isf - chi2_per_pair_lcdm
ip_best = np.argmin(delta_per_pair)
ip_worst = np.argmax(delta_per_pair)
b1b, b2b = pair_labels[ip_best]
b1w, b2w = pair_labels[ip_worst]
print(f"\n  ISF gains most at pair ({b1b},{b2b}) "
      f"(Delta = {delta_per_pair[ip_best]:.2f})")
print(f"  ISF loses most at pair ({b1w},{b2w}) "
      f"(Delta = {delta_per_pair[ip_worst]:.2f})")


# ══════════════════════════════════════════════════════════════════
# TEST 4: Covariance-weighted pair-averaged ratio
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 4: Covariance-weighted effective data/LCDM ratio vs ell")
print("=" * 70)

# For each ell-bin, compute the inverse-variance-weighted average
# of data_i / model_LCDM_i across all 15 pairs.
# This shows what the chi^2 "sees" as a function of ell.

print(f"\n  Weighted ratio = sum_pairs [w_ip * (data/A_LCDM_model)_ip] / sum w_ip")
print(f"  where w_ip = 1/sigma^2 for that data point")
print(f"\n  {'ell':>7s}  {'wt_ratio':>10s}  {'unwt_ratio':>10s}  {'N_pairs':>8s}")

for il in range(n_ell):
    indices = [ip * n_ell + il for ip in range(n_pairs)]
    d_ell = data_flat[indices]
    m_ell = model_lcdm_scaled[indices]
    sig_ell = diag_sigma[indices]

    # Only use pairs where model > 0
    valid = m_ell > 0
    if np.sum(valid) == 0:
        print(f"  {ell_eff[il]:7.1f}  {'---':>10s}  {'---':>10s}  {0:8d}")
        continue

    ratio = d_ell[valid] / m_ell[valid]
    weights = 1.0 / sig_ell[valid]**2

    wt_avg = np.average(ratio, weights=weights)
    unwt_avg = np.mean(ratio)

    print(f"  {ell_eff[il]:7.1f}  {wt_avg:10.4f}  {unwt_avg:10.4f}  {np.sum(valid):8d}")


# ══════════════════════════════════════════════════════════════════
# TEST 5: ISF residual shape — what pattern does ISF fit?
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 5: ISF improvement pattern — where does ISF help?")
print("=" * 70)

# The ISF model changes the prediction shape. Show the difference
# between scaled ISF and scaled LCDM, normalized by sigma.
isf_lcdm_diff = (model_isf_scaled - model_lcdm_scaled) / diag_sigma

print(f"\n  (ISF_model - LCDM_model) / sigma, averaged per ell-bin:")
print(f"  (Shows HOW ISF differs from LCDM in data space)")
print(f"\n  {'ell':>7s}  {'mean_diff':>10s}  {'std_diff':>10s}")

for il in range(n_ell):
    indices = [ip * n_ell + il for ip in range(n_pairs)]
    diff_ell = isf_lcdm_diff[indices]
    print(f"  {ell_eff[il]:7.1f}  {np.mean(diff_ell):10.4f}  {np.std(diff_ell):10.4f}")

# Correlation between ISF-LCDM difference and the LCDM residual
corr_global = np.corrcoef(isf_lcdm_diff, norm_residual)[0, 1]
print(f"\n  Correlation: (ISF-LCDM diff) vs (LCDM residual) = {corr_global:.4f}")
print(f"  (r > 0 means ISF shifts the model TOWARD the data)")
print(f"  (r ~ 0 means ISF changes are orthogonal to where LCDM fails)")

if corr_global > 0.3:
    print(f"  -> ISF is genuinely fitting a pattern in the LCDM residuals")
elif corr_global > 0.1:
    print(f"  -> Mild alignment between ISF and LCDM residual structure")
else:
    print(f"  -> ISF changes are mostly orthogonal to LCDM residuals")


# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
  B1 Question: Does ell-dependent calibration drive k_t?

  Test 1 (residual slope): slope = {slope:.4f}
    {'SMALL' if abs(slope) < 0.5 else 'MODERATE' if abs(slope) < 1.0 else 'LARGE'} — {'no significant ell trend' if abs(slope) < 0.5 else 'possible trend'}

  Test 2 (per-ell chi2): ISF gains most at ell={ell_eff[il_best]:.0f}, loses at ell={ell_eff[il_worst]:.0f}

  Test 3 (per-pair chi2): ISF gains most at ({b1b},{b2b}), loses at ({b1w},{b2w})

  Test 5 (ISF-LCDM correlation with residuals): r = {corr_global:.4f}

  Combined with A1 result (per-pair tilt vs ISF: r = -0.09):
""")

if abs(slope) < 0.5 and abs(corr_global) < 0.3:
    print(f"  CONCLUSION: No evidence that ell-dependent calibration drives k_t.")
    print(f"  The k_t = 0.81 shift appears to be a genuine shape preference.")
elif abs(slope) < 1.0 and abs(corr_global) < 0.5:
    print(f"  CONCLUSION: Weak evidence of ell-dependent structure, but")
    print(f"  insufficient to explain the k_t shift. Proceed with caution.")
else:
    print(f"  CONCLUSION: Significant ell-dependent structure detected.")
    print(f"  k_t may be partially driven by calibration. Investigate further.")

print(f"\nScript complete.")
