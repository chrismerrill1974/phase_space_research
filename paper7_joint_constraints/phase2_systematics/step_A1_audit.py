#!/usr/bin/env python3
"""
Phase A, Step A1: Unit/Convention Audit — Diagnostic Script

Five diagnostics to characterize the ~5x calibration offset between
our LCDM theory predictions and KiDS-1000 band powers.

Diagnostic 1: Data/theory ratio vs ell for every pair
Diagnostic 2: Fit a power-law tilt to the ratio — does it mimic ISF?
Diagnostic 3: Tomographic sum cross-check vs Phase 1 single-bin
Diagnostic 4: n(z) interpolation quality check per bin
Diagnostic 5: What ISF k_t would the tilt alone produce?

Run: python3 step_A1_audit.py

Depends on:
  ../data/kids1000_bandpowers.npz          (Step 2a)
  ../data/response_operators_tomo.npz      (Step 2b)
  handoffs/.../phase1/data/response_operator.npz   (Phase 1)
  handoffs/.../phase1/data/cl_kappa_lcdm.npz       (Phase 1)
"""

import numpy as np
import os

# ── Paths ─────────────────────────────────────────────────────────
# Try two possible base dirs (run from step_A1_audit/ or from repo root)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# Data from Steps 2a/2b
DATA_DIR = os.path.join(REPO_ROOT, 'data')
KIDS_PATH = os.path.join(DATA_DIR, 'kids1000_bandpowers.npz')
TOMO_PATH = os.path.join(DATA_DIR, 'response_operators_tomo.npz')

# Phase 1 data
P1_DIR = os.path.join(REPO_ROOT, 'handoffs', 'paper7_handoff_phase2',
                       'phase1', 'data')
P1_RESP_PATH = os.path.join(P1_DIR, 'response_operator.npz')
P1_CL_PATH = os.path.join(P1_DIR, 'cl_kappa_lcdm.npz')

# Output
OUT_DIR = SCRIPT_DIR
OUT_FILE = os.path.join(OUT_DIR, 'audit_results.txt')

# ── Verify files exist ────────────────────────────────────────────
for label, path in [('KiDS data', KIDS_PATH),
                    ('Tomo operators', TOMO_PATH),
                    ('Phase 1 response', P1_RESP_PATH),
                    ('Phase 1 C_kappa', P1_CL_PATH)]:
    if not os.path.isfile(path):
        print(f"ERROR: {label} not found at:\n  {path}")
        print(f"  (REPO_ROOT = {REPO_ROOT})")
        raise SystemExit(1)

# ── Load everything ───────────────────────────────────────────────
print("=" * 70)
print("Step A1: Unit/Convention Audit — Diagnostic Script")
print("=" * 70)

print("\nLoading data...")
dk = np.load(KIDS_PATH, allow_pickle=True)
dr = np.load(TOMO_PATH, allow_pickle=True)
dp1 = np.load(P1_RESP_PATH, allow_pickle=True)
dp1c = np.load(P1_CL_PATH, allow_pickle=True)

bp_data = dk['bp_data']             # (15, 8)
ell_eff = dk['ell_eff']             # (8,)
pair_labels = dk['pair_labels']     # (15, 2)
nz_bins = dk['nz_bins']             # (5, 119)
z_mid = dk['z_mid']                 # (119,)
z_low = dk['z_low']
z_high = dk['z_high']

A_tomo = dr['A_tomo']               # (15, 8, 500)
k_hMpc = dr['k_hMpc']              # (500,)
cl_lcdm_tomo = dr['cl_lcdm_tomo']  # (15, 8)
W_all = dr['W_all']                 # (5, 800)
chi_W = dr['chi_W']                 # (800,)

A_p1 = dp1['A']                     # (100, 500)
ell_p1 = dp1['ell']                 # (100,)
cl_p1 = dp1['cl_lcdm_from_A']      # (100,)
k_hMpc_p1 = dp1['k_hMpc']          # (500,)

cl_p1c = dp1c['cl_kappa_lcdm']     # (100,)
ell_p1c = dp1c['ell']              # (100,)
nz_p1_z = dp1c['z_nz']            # (5000,)
nz_p1_vals = dp1c['nz_vals']      # (5000,)

n_pairs = len(pair_labels)
n_ell = len(ell_eff)

print(f"  KiDS: {n_pairs} pairs x {n_ell} ell-bins = {n_pairs*n_ell} points")
print(f"  ell range: [{ell_eff[0]:.1f}, {ell_eff[-1]:.1f}]")
print(f"  Phase 1: {len(ell_p1)} ell-bins, k-grid: {len(k_hMpc_p1)} points")


# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC 1: Data/theory ratio vs ell for every pair
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DIAGNOSTIC 1: Data/theory ratio vs ell (convention B: ell(ell+1)/(2pi))")
print("=" * 70)

ell_factor = ell_eff * (ell_eff + 1) / (2 * np.pi)
cl_lcdm_scaled = cl_lcdm_tomo * ell_factor[np.newaxis, :]

# Ratio: data / scaled_theory
ratio_all = np.full_like(bp_data, np.nan)
for ip in range(n_pairs):
    for il in range(n_ell):
        if cl_lcdm_scaled[ip, il] > 0:
            ratio_all[ip, il] = bp_data[ip, il] / cl_lcdm_scaled[ip, il]

# Print full table
print(f"\n  {'Pair':>6s}", end="")
for il in range(n_ell):
    print(f"  ell={ell_eff[il]:.0f}", end="")
print("    median")

for ip in range(n_pairs):
    b1, b2 = pair_labels[ip]
    print(f"  ({b1},{b2})", end="")
    for il in range(n_ell):
        r = ratio_all[ip, il]
        if np.isfinite(r):
            print(f"  {r:9.2f}", end="")
        else:
            print(f"  {'---':>9s}", end="")
    valid = ratio_all[ip][np.isfinite(ratio_all[ip])]
    if len(valid) > 0:
        print(f"  {np.median(valid):9.2f}")
    else:
        print(f"  {'---':>9s}")

# Summary stats
valid_mask = np.isfinite(ratio_all) & (cl_lcdm_scaled > 0) & (bp_data > 0)
ratios_pos = ratio_all[valid_mask]
print(f"\n  Over all positive-data points:")
print(f"    N = {len(ratios_pos)}")
print(f"    Median = {np.median(ratios_pos):.3f}")
print(f"    Mean   = {np.mean(ratios_pos):.3f}")
print(f"    Std    = {np.std(ratios_pos):.3f}")
print(f"    Min    = {np.min(ratios_pos):.3f}")
print(f"    Max    = {np.max(ratios_pos):.3f}")


# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC 2: Power-law tilt fit to the ratio
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DIAGNOSTIC 2: Power-law tilt fit  ratio = A * (ell/ell_pivot)^gamma")
print("=" * 70)

ell_pivot = 500.0

# Fit per auto-correlation pair
print(f"\n  Fitting log(ratio) = log(A) + gamma * log(ell/{ell_pivot:.0f})")
print(f"  (using only positive data points)\n")
print(f"  {'Pair':>6s}  {'A':>8s}  {'gamma':>8s}  {'N_pts':>6s}  {'resid_rms':>10s}")

gamma_per_pair = {}
for ip in range(n_pairs):
    b1, b2 = pair_labels[ip]
    mask = (ratio_all[ip] > 0) & np.isfinite(ratio_all[ip])
    if np.sum(mask) < 3:
        print(f"  ({b1},{b2})  {'skip':>8s}  {'skip':>8s}  {np.sum(mask):6d}")
        continue

    x = np.log(ell_eff[mask] / ell_pivot)
    y = np.log(ratio_all[ip, mask])

    # Simple linear regression: y = a + b*x
    coeffs = np.polyfit(x, y, 1)
    gamma = coeffs[0]
    logA = coeffs[1]
    A_fit = np.exp(logA)

    residuals = y - (logA + gamma * x)
    rms = np.sqrt(np.mean(residuals**2))

    gamma_per_pair[(b1, b2)] = gamma
    print(f"  ({b1},{b2})  {A_fit:8.3f}  {gamma:8.4f}  {np.sum(mask):6d}  {rms:10.4f}")

# Average gamma across high-SNR auto pairs
auto_gammas = []
for ip in range(n_pairs):
    b1, b2 = pair_labels[ip]
    if b1 == b2 and (b1, b2) in gamma_per_pair:
        auto_gammas.append(gamma_per_pair[(b1, b2)])

if auto_gammas:
    print(f"\n  Auto-pair mean gamma = {np.mean(auto_gammas):.4f} "
          f"+/- {np.std(auto_gammas):.4f}")
    print(f"  (gamma > 0 means ratio increases with ell)")
    print(f"  (gamma ~ 0 means ratio is flat = pure amplitude offset)")


# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC 3: Tomographic sum vs Phase 1 single-bin
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DIAGNOSTIC 3: Tomographic sum cross-check vs Phase 1")
print("=" * 70)

# The total convergence power from all 5 bins is:
#   C_kappa_total(ell) = (1/N_bins^2) * [sum_i C_ii + 2*sum_{i<j} C_ij]
# where N_bins = 5 (equal-weight bins).
# This should roughly match Phase 1 C_kappa if Phase 1 used n(z) = sum of bins.

# Method 1: weighted sum with 1/25 for auto, 2/25 for cross
cl_tomo_sum = np.zeros(n_ell)
for ip in range(n_pairs):
    b1, b2 = pair_labels[ip]
    if b1 == b2:
        w = 1.0 / 25.0
    else:
        w = 2.0 / 25.0
    cl_tomo_sum += w * cl_lcdm_tomo[ip]

# Method 2: simple sum (no weighting)
cl_tomo_simple_sum = np.sum(cl_lcdm_tomo, axis=0)

# Interpolate Phase 1 C_kappa to KiDS ell values
from scipy.interpolate import interp1d
cl_p1_interp = interp1d(ell_p1, cl_p1, kind='cubic',
                          bounds_error=False, fill_value=np.nan)
cl_p1_at_kids = cl_p1_interp(ell_eff)

print(f"\n  Phase 1 used analytic n(z) ~ z^2 exp(-(z/0.70)^1.5), <z>=0.70")
print(f"  KiDS bins have <z> = [0.26, 0.40, 0.56, 0.79, 0.98]")
print(f"\n  Comparison at KiDS ell values:")
print(f"  {'ell':>7s}  {'P1_C_kappa':>12s}  {'tomo_wt_sum':>12s}  "
      f"{'tomo_raw_sum':>12s}  {'ratio_wt':>10s}  {'ratio_raw':>10s}")

for il in range(n_ell):
    p1_val = cl_p1_at_kids[il]
    wt_val = cl_tomo_sum[il]
    raw_val = cl_tomo_simple_sum[il]

    r_wt = wt_val / p1_val if p1_val > 0 else np.nan
    r_raw = raw_val / p1_val if p1_val > 0 else np.nan

    print(f"  {ell_eff[il]:7.1f}  {p1_val:12.4e}  {wt_val:12.4e}  "
          f"{raw_val:12.4e}  {r_wt:10.4f}  {r_raw:10.4f}")

if np.all(np.isfinite(cl_p1_at_kids)):
    ratio_wt = cl_tomo_sum / cl_p1_at_kids
    ratio_raw = cl_tomo_simple_sum / cl_p1_at_kids
    print(f"\n  Weighted sum / Phase 1: mean = {np.mean(ratio_wt):.4f}, "
          f"range = [{ratio_wt.min():.4f}, {ratio_wt.max():.4f}]")
    print(f"  Raw sum / Phase 1:     mean = {np.mean(ratio_raw):.4f}, "
          f"range = [{ratio_raw.min():.4f}, {ratio_raw.max():.4f}]")
    print(f"\n  Note: ratio != 1 is expected because the Phase 1 n(z) is an")
    print(f"  analytic approximation, not the sum of KiDS tomographic bins.")
    print(f"  What matters: is the SHAPE (ell-dependence) consistent?")

    shape_ratio_wt = ratio_wt / ratio_wt.mean()
    print(f"\n  Shape consistency (wt_sum/P1, normalized to mean=1):")
    print(f"    Range: [{shape_ratio_wt.min():.4f}, {shape_ratio_wt.max():.4f}]")
    print(f"    Max deviation from unity: {np.max(np.abs(shape_ratio_wt - 1))*100:.2f}%")
    if np.max(np.abs(shape_ratio_wt - 1)) < 0.05:
        print(f"    -> PASS: shapes agree to <5%")
    else:
        print(f"    -> WARNING: shapes differ by >{np.max(np.abs(shape_ratio_wt-1))*100:.0f}%")


# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC 4: n(z) interpolation quality
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DIAGNOSTIC 4: n(z) interpolation quality per bin")
print("=" * 70)

dz_kids = z_high - z_low  # bin widths (should all be 0.05)

print(f"\n  KiDS n(z) grid: {len(z_mid)} points, dz = {dz_kids[0]:.4f}")
print(f"  z range: [{z_mid[0]:.3f}, {z_mid[-1]:.3f}]")

for b in range(5):
    nz = nz_bins[b]
    norm_raw = np.sum(nz * dz_kids)

    # After normalization to integral=1
    nz_normed = nz / norm_raw

    # Basic stats
    i_peak = np.argmax(nz_normed)
    z_peak = z_mid[i_peak]
    nz_peak = nz_normed[i_peak]
    z_mean = np.sum(z_mid * nz_normed * dz_kids)

    # Effective width (FWHM-like)
    half_max = nz_peak / 2.0
    above_half = z_mid[nz_normed >= half_max]
    if len(above_half) > 1:
        fwhm = above_half[-1] - above_half[0]
    else:
        fwhm = 0.0

    # Number of non-zero points
    n_nonzero = np.sum(nz > 0)

    # Smoothness check: max second derivative relative to peak
    d2nz = np.diff(nz_normed, n=2)
    if nz_peak > 0:
        roughness = np.max(np.abs(d2nz)) / nz_peak
    else:
        roughness = 0.0

    print(f"\n  Bin {b+1}:")
    print(f"    raw integral  = {norm_raw:.4f}")
    print(f"    <z>           = {z_mean:.3f}")
    print(f"    z_peak        = {z_peak:.3f}")
    print(f"    n(z)_peak     = {nz_peak:.3f}  (normed)")
    print(f"    FWHM          = {fwhm:.3f}")
    print(f"    N_nonzero     = {n_nonzero} / {len(z_mid)}")
    print(f"    Roughness     = {roughness:.4f}")
    print(f"    Points in FWHM = {np.sum(nz_normed >= half_max)}")

    if np.sum(nz_normed >= half_max) < 4:
        print(f"    ⚠ WARNING: Only {np.sum(nz_normed >= half_max)} points "
              f"above half-max — interpolation may be unreliable")


# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC 5: Does the ell-tilt mimic an ISF signal?
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DIAGNOSTIC 5: Can the ell-dependent tilt mimic ISF?")
print("=" * 70)

# For each auto pair, compute what ISF k_t would best reproduce
# the observed ratio tilt (relative to the LCDM shape).
# ratio(ell) ~ LCDM_scaled(ell) * ratio_fit(ell) ~ ISF_model(ell)
#
# If the ratio tilt is degenerate with ISF, then the ratio
# R(ell) = data / (A * LCDM) should look like M^2_ISF(k(ell)).
#
# Under Limber: k ~ ell/chi_peak, so the ell-dependence maps to k-dependence.

# Use the N=6, eps_inf=0.58 ISF model
N_test = 6
eps_inf_test = 0.58
alpha = 2

# For the (5,5) pair, estimate the effective k at each ell
# using the peak of the lensing kernel for bin 5
# W_all is (5, 800) on chi_W (800,)
W5 = W_all[4]  # bin 5
i_peak_W5 = np.argmax(W5)
chi_peak_5 = chi_W[i_peak_W5]
h_val = float(dr['h'])

print(f"\n  Bin 5 lensing kernel peaks at chi = {chi_peak_5:.0f} Mpc")
print(f"  h = {h_val:.3f}")

k_eff_hMpc = (ell_eff + 0.5) / chi_peak_5 / h_val  # convert 1/Mpc to h/Mpc
print(f"\n  Effective k at each ell (using Limber at kernel peak):")
print(f"  {'ell':>7s}  {'k_eff (h/Mpc)':>14s}")
for il in range(n_ell):
    print(f"  {ell_eff[il]:7.1f}  {k_eff_hMpc[il]:14.4f}")

# Now scan k_t values and see which best fits the RATIO pattern
# for the (5,5) pair
ip_55 = -1
for ip in range(n_pairs):
    if pair_labels[ip][0] == 5 and pair_labels[ip][1] == 5:
        ip_55 = ip
        break

if ip_55 >= 0:
    mask_55 = (ratio_all[ip_55] > 0) & np.isfinite(ratio_all[ip_55])
    ratio_55 = ratio_all[ip_55, mask_55]
    ell_55 = ell_eff[mask_55]

    # Normalize ratio to its mean (we care about the tilt, not amplitude)
    ratio_55_normed = ratio_55 / np.mean(ratio_55)

    k_t_scan = np.logspace(np.log10(0.01), np.log10(5.0), 200)
    best_chi2 = np.inf
    best_kt = 0.0

    for kt in k_t_scan:
        # ISF M^2 at effective k values
        k_eff_at_ell = (ell_55 + 0.5) / chi_peak_5 / h_val
        eps_k = eps_inf_test / (1.0 + (kt / k_eff_at_ell)**alpha)
        M = (1.0 + (N_test - 1) * eps_k) / N_test
        M_sq = M**2

        # Normalize M^2 to its mean for shape comparison
        M_sq_normed = M_sq / np.mean(M_sq)

        chi2 = np.sum((ratio_55_normed - M_sq_normed)**2)
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_kt = kt

    print(f"\n  (5,5) pair: ratio tilt best matched by ISF k_t = {best_kt:.3f} h/Mpc")
    print(f"    (residual chi2 = {best_chi2:.6f})")

    # Show the comparison
    # M_sq_approx: evaluated at the 8 approximate k_eff values (for shape comparison)
    eps_k_approx = eps_inf_test / (1.0 + (best_kt / k_eff_hMpc)**alpha)
    M_approx = (1.0 + (N_test - 1) * eps_k_approx) / N_test
    M_sq_approx = M_approx**2

    # M_sq_full: evaluated on the full 500-point k-grid (for response operator)
    eps_k_full = eps_inf_test / (1.0 + (best_kt / k_hMpc)**alpha)
    M_full = (1.0 + (N_test - 1) * eps_k_full) / N_test
    M_sq_full = M_full**2

    # Compute ISF C_ell from the ACTUAL response operator (not Limber approx)
    cl_isf_best = A_tomo[ip_55] @ M_sq_full   # (8,500) @ (500,) -> (8,)
    cl_lcdm_55 = cl_lcdm_tomo[ip_55]          # (8,)

    # Normalize to mean over valid ell-bins for shape comparison
    cl_isf_normed = cl_isf_best / np.mean(cl_isf_best[mask_55])
    cl_lcdm_normed = cl_lcdm_55 / np.mean(cl_lcdm_55[mask_55])
    isf_lcdm_shape_ratio = cl_isf_normed / cl_lcdm_normed

    print(f"\n  Shape comparison at each ell (all normalized to mean=1):")
    print(f"  {'ell':>7s}  {'data/theory':>12s}  {'ISF_approx':>12s}  "
          f"{'ISF_exact':>12s}")
    for il in range(n_ell):
        r_data = ratio_all[ip_55, il] / np.mean(ratio_55) if mask_55[il] else np.nan
        m_sq_n = M_sq_approx[il] / np.mean(M_sq_approx[mask_55])
        isf_ex = isf_lcdm_shape_ratio[il] if cl_lcdm_normed[il] > 0 else np.nan
        print(f"  {ell_eff[il]:7.1f}  {r_data:12.4f}  {m_sq_n:12.4f}  {isf_ex:12.4f}")

    # Key question: is the tilt consistent with ISF or with something else?
    # Compute correlation between data ratio tilt and ISF shape
    # Use the exact ISF/LCDM ratio from the response operator, not the approx
    if np.sum(mask_55) >= 3:
        d_vals = ratio_all[ip_55, mask_55] / np.mean(ratio_55)
        m_vals_approx = M_sq_approx[mask_55] / np.mean(M_sq_approx[mask_55])
        m_vals_exact = isf_lcdm_shape_ratio[mask_55] / np.mean(isf_lcdm_shape_ratio[mask_55])
        corr_approx = np.corrcoef(d_vals, m_vals_approx)[0, 1]
        corr_exact = np.corrcoef(d_vals, m_vals_exact)[0, 1]
        print(f"\n  Correlation (data tilt vs ISF shape):")
        print(f"    Approx (Limber k_eff):  r = {corr_approx:.4f}")
        print(f"    Exact  (response op.):  r = {corr_exact:.4f}")
        corr = corr_exact
        if abs(corr) > 0.8:
            print(f"  ⚠ HIGH CORRELATION — the calibration tilt IS degenerate with ISF")
            print(f"    k_t from Step 2c may be driven by the calibration tilt,")
            print(f"    not the underlying physics.")
        elif abs(corr) > 0.5:
            print(f"  ~ MODERATE CORRELATION — partial degeneracy")
        else:
            print(f"  ✓ LOW CORRELATION — calibration tilt is orthogonal to ISF shape")

    # Also test at Step 2c's best-fit k_t = 0.81 h/Mpc for comparison
    kt_2c = 0.81
    eps_k_2c = eps_inf_test / (1.0 + (kt_2c / k_hMpc)**alpha)
    M_2c = (1.0 + (N_test - 1) * eps_k_2c) / N_test
    M_sq_2c = M_2c**2
    cl_isf_2c = A_tomo[ip_55] @ M_sq_2c
    isf_2c_shape = (cl_isf_2c / np.mean(cl_isf_2c[mask_55])) / cl_lcdm_normed

    print(f"\n  For reference — ISF shape at Step 2c best-fit k_t={kt_2c}:")
    print(f"  {'ell':>7s}  {'data_tilt':>12s}  {'ISF_0.81':>12s}")
    for il in range(n_ell):
        r_data = ratio_all[ip_55, il] / np.mean(ratio_55) if mask_55[il] else np.nan
        isf_val = isf_2c_shape[il] if cl_lcdm_normed[il] > 0 else np.nan
        print(f"  {ell_eff[il]:7.1f}  {r_data:12.4f}  {isf_val:12.4f}")

    if np.sum(mask_55) >= 3:
        m_2c_vals = isf_2c_shape[mask_55] / np.mean(isf_2c_shape[mask_55])
        corr_2c = np.corrcoef(d_vals, m_2c_vals)[0, 1]
        print(f"\n  Correlation (data tilt vs ISF k_t=0.81): r = {corr_2c:.4f}")


# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC 6: Raw numbers table for the README
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DIAGNOSTIC 6: Raw C_ell values at key points")
print("=" * 70)

print(f"\n  For cross-referencing with published values:")
print(f"\n  {'Pair':>6s}  {'ell':>7s}  {'cl_raw':>12s}  {'cl_scaled':>12s}  "
      f"{'bp_data':>12s}  {'ratio':>8s}")
for ip in [0, 5, 9, 12, 14]:  # (1,1), (2,2), (3,3), (4,4), (5,5)
    b1, b2 = pair_labels[ip]
    for il in [0, 3, 7]:  # ell_min, ell_mid, ell_max
        r = ratio_all[ip, il] if np.isfinite(ratio_all[ip, il]) else 0.0
        print(f"  ({b1},{b2})  {ell_eff[il]:7.1f}  {cl_lcdm_tomo[ip,il]:12.4e}  "
              f"{cl_lcdm_scaled[ip,il]:12.4e}  {bp_data[ip,il]:12.4e}  {r:8.2f}")
    print()


# ══════════════════════════════════════════════════════════════════
# Save results
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("SUMMARY")
print("=" * 70)

# Compute key summary numbers
auto_pairs = [(ip, pair_labels[ip]) for ip in range(n_pairs)
              if pair_labels[ip][0] == pair_labels[ip][1]]

auto_medians = []
for ip, (b1, b2) in auto_pairs:
    valid = ratio_all[ip][np.isfinite(ratio_all[ip]) & (ratio_all[ip] > 0)]
    if len(valid) > 0:
        auto_medians.append((b1, b2, np.median(valid)))

print(f"\n  Auto-pair median ratios (data / ell(ell+1)C_theory/(2pi)):")
for b1, b2, med in auto_medians:
    print(f"    ({b1},{b2}): {med:.2f}")

if auto_gammas:
    mean_gamma = np.mean(auto_gammas)
    print(f"\n  Mean power-law tilt (auto pairs): gamma = {mean_gamma:.4f}")
    if abs(mean_gamma) < 0.1:
        print(f"    -> Tilt is small: offset is mostly a constant")
    elif abs(mean_gamma) < 0.3:
        print(f"    -> Moderate tilt: could partially mimic ISF")
    else:
        print(f"    -> Strong tilt: likely driving k_t shift")

print(f"\n  Key question answered: is the calibration offset degenerate")
print(f"  with the ISF signal? See Diagnostic 5 correlation value above.")

print(f"\nScript complete.")
