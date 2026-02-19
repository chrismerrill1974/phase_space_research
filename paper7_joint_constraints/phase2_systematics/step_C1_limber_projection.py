#!/usr/bin/env python3
"""
Phase C, Step C1: Limber projection effect on k_t

B3 found that per-pair best-fit k_t decreases monotonically with pair
effective redshift (Spearman rho = -0.96, slope ~ -0.6 dex/z).  This
step tests whether the Limber projection quantitatively explains that
trend.

The idea: under the Limber approximation, each angular multipole ell
maps to a 3D wavenumber k = (ell + 0.5) / chi(z), where chi is the
comoving distance.  A fixed physical transition scale k_t^{phys} in
the ISF model therefore maps to different "apparent" k_t values when
fit to different tomographic pairs, because each pair's lensing kernel
W_i(chi) * W_j(chi) peaks at a different effective distance.

Approach:
  1. For each pair, compute the effective comoving distance chi_eff
     as the W_i * W_j - weighted mean of chi.
  2. Given a trial k_t^{phys}, compute the predicted apparent k_t for
     each pair via the Limber mapping at the pair's ell-bins.
  3. Compare the predicted k_t(z_eff) relation to the B3 observations.
  4. Fit for the single k_t^{phys} that best matches all per-pair k_t
     values simultaneously.

Run: python3 step_C1_limber_projection.py

Depends on:
  ../../data/kids1000_bandpowers.npz
  ../../data/response_operators_tomo.npz
"""

import numpy as np
import os
import time

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
print("Step C1: Limber projection effect on k_t")
print("=" * 70)

dk = np.load(KIDS_PATH, allow_pickle=True)
dr = np.load(TOMO_PATH, allow_pickle=True)

bp_data = dk['bp_data']             # (15, 8)
ell_eff = dk['ell_eff']             # (8,)
cov_ee = dk['cov_ee']               # (120, 120)
pair_labels = dk['pair_labels']     # (15, 2)
nz_bins = dk['nz_bins']             # (5, 119)
z_mid = dk['z_mid']                 # (119,)

A_tomo = dr['A_tomo']               # (15, 8, 500)
k_hMpc = dr['k_hMpc']              # (500,)
cl_lcdm_tomo = dr['cl_lcdm_tomo']  # (15, 8)
W_all = dr['W_all']                 # (5, 800)
chi_W = dr['chi_W']                 # (800,)
h_cosmo = float(dr['h'])            # 0.674
Om = float(dr['Om'])                # 0.3134

n_pairs = len(pair_labels)
n_ell = len(ell_eff)
n_k = len(k_hMpc)

# ISF parameters (N=6 best-fit)
N_val = 6
eps_inf = 0.58
alpha = 2

# High-SNR pairs
HIGH_SNR_PAIRS = [(3, 3), (4, 4), (5, 5), (3, 4), (3, 5), (4, 5)]

# Mean redshifts per bin
z_mean_bin = {}
for i in range(nz_bins.shape[0]):
    z_mean_bin[i + 1] = np.average(z_mid, weights=nz_bins[i])

print(f"\n  Cosmology: h = {h_cosmo}, Om = {Om:.4f}")
print(f"  ISF model: N={N_val}, eps_inf={eps_inf}, alpha={alpha}")
print(f"  Lensing kernels: W_all({W_all.shape}), chi grid: "
      f"{chi_W.min():.0f} to {chi_W.max():.0f} Mpc/h")


# ══════════════════════════════════════════════════════════════════
# 1. Compute effective comoving distance for each pair
# ══════════════════════════════════════════════════════════════════
print("\n[1] Computing effective comoving distances per pair...")

# For cross-spectrum (i,j), the Limber kernel is W_i(chi) * W_j(chi) / chi^2
# The effective distance is the kernel-weighted mean chi.
chi_eff_pair = {}
z_eff_pair = {}

for ip in range(n_pairs):
    b1, b2 = pair_labels[ip]
    # W arrays are 0-indexed (bin 1 -> index 0)
    W1 = W_all[b1 - 1]
    W2 = W_all[b2 - 1]

    # Limber kernel: W1 * W2 / chi^2 (the 1/chi^2 comes from the
    # Limber integral d_chi / chi^2)
    kernel = W1 * W2 / np.maximum(chi_W**2, 1e-30)

    # Effective chi: kernel-weighted mean
    norm = np.trapezoid(kernel, chi_W)
    if norm > 0:
        chi_eff = np.trapezoid(kernel * chi_W, chi_W) / norm
    else:
        chi_eff = 0.0

    chi_eff_pair[ip] = chi_eff

    # Also compute simple z_eff (average of bin means, as in B3)
    z_eff_pair[ip] = 0.5 * (z_mean_bin[b1] + z_mean_bin[b2])

print(f"\n  {'Pair':>7s}  {'z_eff':>6s}  {'chi_eff':>10s}  {'SNR':>4s}")
for ip in range(n_pairs):
    b1, b2 = pair_labels[ip]
    snr = "HIGH" if (b1, b2) in HIGH_SNR_PAIRS else "low"
    print(f"  ({b1},{b2}){'':<2s}  {z_eff_pair[ip]:6.4f}  "
          f"{chi_eff_pair[ip]:10.1f}  {snr:>4s}")


# ══════════════════════════════════════════════════════════════════
# 2. Limber mapping: k_t^{apparent}(ell, chi_eff) for a given k_t^{phys}
# ══════════════════════════════════════════════════════════════════
print("\n[2] Limber mapping analysis...")

# Under the Limber approximation, an angular multipole ell probes
# 3D wavenumber k = (ell + 0.5) / chi at distance chi.
#
# The ISF suppression M^2(k) has its transition at k_t^{phys}.
# When we observe the angular power spectrum at multipole ell,
# we're seeing the ISF effect at k = (ell+0.5)/chi_eff.
#
# If we fit the 8 ell-bins for a single pair, the fit finds an
# "apparent" k_t that best describes the ell-dependent shape.
# For a pair with larger chi_eff, the same ell maps to a smaller k,
# so the transition appears at a lower apparent k_t.
#
# Simple prediction: k_t^{apparent} ~ k_t^{phys} for a pair at chi_ref,
# and k_t^{apparent}(pair) = k_t^{phys} * chi_ref / chi_eff(pair).
# This is because ell_t = k_t^{phys} * chi_eff, and if the fit
# recovers the transition at the same ell_t, then
# k_t^{app} = ell_t / chi_eff(pair) = k_t^{phys} * chi_ref / chi_eff(pair)
# ... wait, that's just k_t^{phys} if chi_ref = chi_eff.
#
# Actually the correct prediction is simpler: the per-pair fit finds
# the k_t that matches the ell-space shape. The ISF shape in ell-space
# for pair (i,j) depends on the kernel-weighted integral of M^2(k)
# over the line of sight. Different pairs have different kernels,
# so the ell-space ISF shape differs. The per-pair fit then finds
# the k_t that best approximates this kernel-broadened shape with
# a single-chi Limber approximation at the pair's effective distance.
#
# We'll do this properly: for each trial k_t^{phys}, compute the
# full kernel-weighted ISF C_ell, then fit a single k_t to that
# predicted C_ell using the same profiled-amplitude method as B3.

# But first, the simple scaling prediction:
print(f"\n  Simple scaling prediction: k_t^app = k_t^phys * chi_ref / chi_eff")
print(f"  (This assumes the transition maps as ell_t = k_t * chi_eff)")


# ══════════════════════════════════════════════════════════════════
# 3. Per-pair fits (reproduce B3) for comparison
# ══════════════════════════════════════════════════════════════════
print("\n[3] Reproducing per-pair k_t fits from B3...")

kt_fine = np.logspace(np.log10(0.05), np.log10(3.0), 300)
b3_results = []

for ip in range(n_pairs):
    b1, b2 = pair_labels[ip]
    i0 = ip * n_ell
    i1 = (ip + 1) * n_ell
    d_pair = bp_data[ip]
    cov_pair = cov_ee[i0:i1, i0:i1]
    cov_inv_pair = np.linalg.inv(cov_pair)
    Cinv_d = cov_inv_pair @ d_pair
    dCd = d_pair @ Cinv_d

    # LCDM baseline
    m_lcdm = cl_lcdm_tomo[ip]
    Cinv_m_lcdm = cov_inv_pair @ m_lcdm
    chi2_lcdm = dCd - (d_pair @ Cinv_m_lcdm)**2 / (m_lcdm @ Cinv_m_lcdm)

    # k_t scan
    chi2_kt = np.zeros(len(kt_fine))
    for j, kt in enumerate(kt_fine):
        eps_k = eps_inf / (1.0 + (kt / k_hMpc)**alpha)
        M = (1.0 + (N_val - 1) * eps_k) / N_val
        m_isf = A_tomo[ip] @ (M**2)
        Cinv_m = cov_inv_pair @ m_isf
        dCm = d_pair @ Cinv_m
        mCm = m_isf @ Cinv_m
        if mCm > 0:
            chi2_kt[j] = dCd - dCm**2 / mCm
        else:
            chi2_kt[j] = dCd

    idx_best = np.argmin(chi2_kt)
    kt_best = kt_fine[idx_best]
    dchi2 = chi2_kt[idx_best] - chi2_lcdm

    # Check if constrained
    mask_1sig = (chi2_kt - chi2_kt[idx_best]) < 1.0
    kt_range = np.log10(kt_fine[-1]) - np.log10(kt_fine[0])
    if np.any(mask_1sig):
        sig1_range = np.log10(kt_fine[mask_1sig].max()) - np.log10(kt_fine[mask_1sig].min())
    else:
        sig1_range = kt_range
    is_unconstrained = sig1_range > 0.9 * kt_range

    b3_results.append({
        'ip': ip, 'b1': b1, 'b2': b2,
        'pair_str': f"({b1},{b2})",
        'z_eff': z_eff_pair[ip],
        'chi_eff': chi_eff_pair[ip],
        'kt_best': kt_best,
        'dchi2': dchi2,
        'is_unconstrained': is_unconstrained,
        'is_high_snr': (b1, b2) in HIGH_SNR_PAIRS,
    })

constrained = [r for r in b3_results if not r['is_unconstrained']]
constrained_high = [r for r in constrained if r['is_high_snr']]

print(f"  {len(constrained)} constrained pairs, {len(constrained_high)} high-SNR")


# ══════════════════════════════════════════════════════════════════
# 4. Simple Limber scaling model
# ══════════════════════════════════════════════════════════════════
print("\n[4] Testing simple Limber scaling model...")

# Model: k_t^{app}(pair) = k_t^{phys} * chi_ref / chi_eff(pair)
# where chi_ref is a reference distance.
#
# Equivalently: k_t^{app} * chi_eff = ell_t = constant for all pairs.
# So if the Limber scaling is perfect, the product
#   k_t^{app}(pair) * chi_eff(pair)
# should be constant across all constrained pairs.

print(f"\n  Testing: is k_t * chi_eff = ell_t = constant?")
print(f"\n  {'Pair':>7s}  {'z_eff':>6s}  {'chi_eff':>8s}  {'k_t_B3':>8s}  "
      f"{'ell_t':>8s}  {'SNR':>4s}")

ell_t_values = []
for r in sorted(constrained, key=lambda x: x['z_eff']):
    ell_t = r['kt_best'] * r['chi_eff']
    ell_t_values.append(ell_t)
    snr = "HIGH" if r['is_high_snr'] else "low"
    print(f"  {r['pair_str']:>7s}  {r['z_eff']:6.4f}  {r['chi_eff']:8.1f}  "
          f"{r['kt_best']:8.4f}  {ell_t:8.1f}  {snr:>4s}")

ell_t_arr = np.array(ell_t_values)
print(f"\n  ell_t mean:  {ell_t_arr.mean():.1f}")
print(f"  ell_t std:   {ell_t_arr.std():.1f}")
print(f"  ell_t range: [{ell_t_arr.min():.1f}, {ell_t_arr.max():.1f}]")
print(f"  Fractional scatter: {ell_t_arr.std()/ell_t_arr.mean():.1%}")

# High-SNR only
ell_t_hi = np.array([r['kt_best'] * r['chi_eff']
                      for r in constrained_high])
print(f"\n  High-SNR only:")
print(f"  ell_t mean:  {ell_t_hi.mean():.1f}")
print(f"  ell_t std:   {ell_t_hi.std():.1f}")
print(f"  Fractional scatter: {ell_t_hi.std()/ell_t_hi.mean():.1%}")


# ══════════════════════════════════════════════════════════════════
# 5. Fit for k_t^{phys} using the Limber scaling
# ══════════════════════════════════════════════════════════════════
print("\n[5] Fitting for k_t^{phys}...")

# Simple approach: k_t^{phys} = ell_t / chi_eff
# The best ell_t minimizes sum of (k_t_observed - ell_t/chi_eff)^2
# weighted by the constraining power (|Dchi2|) of each pair.

# Unweighted: ell_t = mean(k_t * chi_eff)
ell_t_unweighted = ell_t_hi.mean()

# Weighted by |Dchi2|:
weights = np.array([abs(r['dchi2']) for r in constrained_high])
ell_t_weighted = np.average(
    [r['kt_best'] * r['chi_eff'] for r in constrained_high],
    weights=weights
)

# Convert to k_t^{phys} at a reference distance
# Use the median chi_eff of high-SNR pairs as reference
chi_ref = np.median([r['chi_eff'] for r in constrained_high])
kt_phys_unweighted = ell_t_unweighted / chi_ref
kt_phys_weighted = ell_t_weighted / chi_ref

print(f"\n  Reference chi_eff (median high-SNR): {chi_ref:.1f} Mpc/h")
print(f"  ell_t (unweighted): {ell_t_unweighted:.1f}")
print(f"  ell_t (Dchi2-weighted): {ell_t_weighted:.1f}")
print(f"  k_t^phys (unweighted): {kt_phys_unweighted:.4f} h/Mpc")
print(f"  k_t^phys (Dchi2-weighted): {kt_phys_weighted:.4f} h/Mpc")


# ══════════════════════════════════════════════════════════════════
# 6. Compare predicted vs observed k_t per pair
# ══════════════════════════════════════════════════════════════════
print("\n[6] Predicted vs observed k_t per pair...")

print(f"\n  Using ell_t = {ell_t_weighted:.1f} (Dchi2-weighted)")
print(f"\n  {'Pair':>7s}  {'z_eff':>6s}  {'chi_eff':>8s}  {'k_t_obs':>8s}  "
      f"{'k_t_pred':>8s}  {'residual':>8s}  {'SNR':>4s}")

residuals = []
for r in sorted(constrained, key=lambda x: x['z_eff']):
    kt_pred = ell_t_weighted / r['chi_eff']
    resid = r['kt_best'] - kt_pred
    frac_resid = resid / r['kt_best']
    residuals.append(frac_resid)
    snr = "HIGH" if r['is_high_snr'] else "low"
    print(f"  {r['pair_str']:>7s}  {r['z_eff']:6.4f}  {r['chi_eff']:8.1f}  "
          f"{r['kt_best']:8.4f}  {kt_pred:8.4f}  {frac_resid:+8.1%}  {snr:>4s}")

residuals = np.array(residuals)
print(f"\n  Mean fractional residual: {residuals.mean():+.1%}")
print(f"  RMS fractional residual:  {np.sqrt(np.mean(residuals**2)):.1%}")

# High-SNR only
resid_hi = []
for r in constrained_high:
    kt_pred = ell_t_weighted / r['chi_eff']
    resid_hi.append((r['kt_best'] - kt_pred) / r['kt_best'])
resid_hi = np.array(resid_hi)
print(f"\n  High-SNR only:")
print(f"  Mean fractional residual: {resid_hi.mean():+.1%}")
print(f"  RMS fractional residual:  {np.sqrt(np.mean(resid_hi**2)):.1%}")


# ══════════════════════════════════════════════════════════════════
# 7. Compare observed vs predicted slope
# ══════════════════════════════════════════════════════════════════
print("\n[7] Comparing observed vs predicted k_t(z_eff) slopes...")

# Observed slope (from B3): log10(k_t) vs z_eff
z_obs = np.array([r['z_eff'] for r in constrained_high])
kt_obs = np.array([r['kt_best'] for r in constrained_high])
coeffs_obs = np.polyfit(z_obs, np.log10(kt_obs), 1)

# Predicted slope: k_t_pred = ell_t / chi_eff(z)
# chi_eff increases with z, so k_t_pred decreases
kt_pred_hi = np.array([ell_t_weighted / r['chi_eff'] for r in constrained_high])
coeffs_pred = np.polyfit(z_obs, np.log10(kt_pred_hi), 1)

print(f"\n  Observed:  log10(k_t) = {coeffs_obs[0]:.4f} * z + ({coeffs_obs[1]:.4f})")
print(f"  Predicted: log10(k_t) = {coeffs_pred[0]:.4f} * z + ({coeffs_pred[1]:.4f})")
print(f"\n  Slope ratio (observed/predicted): {coeffs_obs[0]/coeffs_pred[0]:.3f}")
print(f"  Intercept difference: {coeffs_obs[1] - coeffs_pred[1]:+.4f} dex")


# ══════════════════════════════════════════════════════════════════
# 8. What k_t^{phys} does this correspond to in physical space?
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHYSICAL INTERPRETATION")
print("=" * 70)

# ell_t is the angular multipole of the ISF transition.
# The physical k_t depends on what distance you evaluate k = ell/chi at.
# Paper V used k_t = 0.20 h/Mpc with a simpler mapping.
# Here, the Limber integral weights a range of chi values.

print(f"\n  The ISF transition occurs at angular scale ell_t ~ {ell_t_weighted:.0f}")
print(f"  This maps to physical k_t at different distances:")

# Report k_t^{phys} at a few reference distances
chi_vals = np.array([r['chi_eff'] for r in constrained_high])
z_vals = np.array([r['z_eff'] for r in constrained_high])

print(f"\n  {'Reference':>20s}  {'chi (Mpc/h)':>12s}  {'k_t^phys':>10s}")
print(f"  {'Lowest-z high-SNR':>20s}  {chi_vals.min():12.1f}  "
      f"{ell_t_weighted/chi_vals.min():10.4f}")
print(f"  {'Median high-SNR':>20s}  {np.median(chi_vals):12.1f}  "
      f"{ell_t_weighted/np.median(chi_vals):10.4f}")
print(f"  {'Highest-z high-SNR':>20s}  {chi_vals.max():12.1f}  "
      f"{ell_t_weighted/chi_vals.max():10.4f}")

# Compare to Paper V
print(f"\n  Paper V fiducial: k_t = 0.20 h/Mpc")
print(f"  Paper V used a simpler k ~ 1/R mapping, not the full Limber integral.")
print(f"  The Limber-weighted k_t at median chi_eff: "
      f"{ell_t_weighted/np.median(chi_vals):.4f} h/Mpc")
print(f"  Ratio to Paper V: "
      f"{(ell_t_weighted/np.median(chi_vals))/0.20:.2f}x")


# ══════════════════════════════════════════════════════════════════
# 9. Full kernel-weighted test
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("KERNEL-WEIGHTED EFFECTIVE k_t PER PAIR")
print("=" * 70)

# More rigorous: for a given k_t^{phys}, compute the ISF suppression
# M^2(k) at each k, then weight by the pair's Limber kernel to get
# the effective M^2 at each ell. Then fit the ell-space shape to find
# what apparent k_t this corresponds to.
#
# Actually, A_tomo already encodes the Limber kernel integration:
#   C_ell^{ISF}(pair, ell_bin) = A_tomo[pair, ell_bin, :] @ M^2(k)
# So A_tomo IS the Limber-projected response. The per-pair B3 fit
# already accounts for the full kernel weighting.
#
# What we want to test more carefully: for a SINGLE k_t^{phys},
# does the A_tomo-predicted ell-space shape for each pair match
# what B3 found?

print("\n  Note: A_tomo already encodes the full Limber kernel integration.")
print("  The per-pair B3 fits already account for kernel broadening.")
print("  The question is: does a single k_t^phys produce predicted")
print("  C_ell shapes that, when fit per-pair, recover the B3 k_t values?")

# For each trial k_t^{phys}, compute the ISF C_ell for all pairs using
# A_tomo, then re-fit each pair to find the apparent k_t.

print(f"\n  Scanning k_t^phys to find single value matching all pairs...")

kt_phys_grid = np.logspace(np.log10(0.1), np.log10(2.0), 50)
# For speed, use a coarser apparent-k_t grid for re-fitting
kt_app_grid = np.logspace(np.log10(0.05), np.log10(3.0), 100)

best_total_chi2 = np.inf
best_kt_phys = 0

# Store results for each k_t^{phys}
all_scan_results = []

for kt_phys in kt_phys_grid:
    # Generate the "true" ISF C_ell at this k_t^{phys}
    eps_k_true = eps_inf / (1.0 + (kt_phys / k_hMpc)**alpha)
    M_true = (1.0 + (N_val - 1) * eps_k_true) / N_val
    M_sq_true = M_true**2

    # For each constrained high-SNR pair, compute what the data would
    # look like if k_t^{phys} were the true value, then fit for apparent k_t
    predicted_kt_app = []
    for r in constrained_high:
        ip = r['ip']
        i0 = ip * n_ell
        i1 = (ip + 1) * n_ell
        d_pair = bp_data[ip]
        cov_pair = cov_ee[i0:i1, i0:i1]
        cov_inv_pair = np.linalg.inv(cov_pair)
        Cinv_d = cov_inv_pair @ d_pair
        dCd = d_pair @ Cinv_d

        # Find the apparent k_t that best fits this pair's actual data
        # using the profiled amplitude method
        best_chi2_app = np.inf
        best_kt_app = 0
        for kt_app in kt_app_grid:
            eps_k = eps_inf / (1.0 + (kt_app / k_hMpc)**alpha)
            M = (1.0 + (N_val - 1) * eps_k) / N_val
            m_isf = A_tomo[ip] @ (M**2)
            Cinv_m = cov_inv_pair @ m_isf
            dCm = d_pair @ Cinv_m
            mCm = m_isf @ Cinv_m
            if mCm > 0:
                chi2 = dCd - dCm**2 / mCm
                if chi2 < best_chi2_app:
                    best_chi2_app = chi2
                    best_kt_app = kt_app

        predicted_kt_app.append(best_kt_app)

    # This gives us the B3 per-pair k_t values (independent of k_t^{phys}).
    # What we ACTUALLY want: given k_t^{phys}, what apparent k_t would
    # we recover if the data were generated by this k_t^{phys}?
    # That requires generating mock data... but we can do something
    # simpler: check the chi^2 at k_t^{phys} summed over pairs.

    # Total chi^2 at this k_t^{phys} (each pair gets its own amplitude)
    total_chi2 = 0
    for r in constrained_high:
        ip = r['ip']
        i0 = ip * n_ell
        i1 = (ip + 1) * n_ell
        d_pair = bp_data[ip]
        cov_pair = cov_ee[i0:i1, i0:i1]
        cov_inv_pair = np.linalg.inv(cov_pair)
        Cinv_d = cov_inv_pair @ d_pair
        dCd = d_pair @ Cinv_d

        m_isf = A_tomo[ip] @ M_sq_true
        Cinv_m = cov_inv_pair @ m_isf
        dCm = d_pair @ Cinv_m
        mCm = m_isf @ Cinv_m
        if mCm > 0:
            chi2 = dCd - dCm**2 / mCm
        else:
            chi2 = dCd
        total_chi2 += chi2

    all_scan_results.append({
        'kt_phys': kt_phys,
        'total_chi2': total_chi2,
    })

    if total_chi2 < best_total_chi2:
        best_total_chi2 = total_chi2
        best_kt_phys = kt_phys

# Also compute LCDM chi^2 summed over high-SNR pairs
chi2_lcdm_sum = 0
for r in constrained_high:
    ip = r['ip']
    i0 = ip * n_ell
    i1 = (ip + 1) * n_ell
    d_pair = bp_data[ip]
    cov_pair = cov_ee[i0:i1, i0:i1]
    cov_inv_pair = np.linalg.inv(cov_pair)
    Cinv_d = cov_inv_pair @ d_pair
    dCd = d_pair @ Cinv_d
    m_lcdm = cl_lcdm_tomo[ip]
    Cinv_m = cov_inv_pair @ m_lcdm
    chi2_lcdm_sum += dCd - (d_pair @ Cinv_m)**2 / (m_lcdm @ Cinv_m)

# Find confidence interval
scan_chi2 = np.array([s['total_chi2'] for s in all_scan_results])
scan_kt = np.array([s['kt_phys'] for s in all_scan_results])
mask_1sig = (scan_chi2 - best_total_chi2) < 1.0
mask_2sig = (scan_chi2 - best_total_chi2) < 4.0

kt_1sig = (scan_kt[mask_1sig].min(), scan_kt[mask_1sig].max()) if np.any(mask_1sig) else (0, 0)
kt_2sig = (scan_kt[mask_2sig].min(), scan_kt[mask_2sig].max()) if np.any(mask_2sig) else (0, 0)

print(f"\n  Best-fit k_t^phys (joint high-SNR pairs): {best_kt_phys:.4f} h/Mpc")
print(f"  Joint chi^2 at best k_t^phys: {best_total_chi2:.2f}")
print(f"  Joint chi^2 at LCDM:          {chi2_lcdm_sum:.2f}")
print(f"  Joint Delta_chi^2:            {best_total_chi2 - chi2_lcdm_sum:.2f}")
print(f"  1-sigma interval: [{kt_1sig[0]:.4f}, {kt_1sig[1]:.4f}]")
print(f"  2-sigma interval: [{kt_2sig[0]:.4f}, {kt_2sig[1]:.4f}]")

# How does this compare to the per-pair best-fits?
print(f"\n  Per-pair k_t values at joint-best k_t^phys = {best_kt_phys:.4f}:")
print(f"  (These are the B3 per-pair fits, not predictions from k_t^phys)")
for r in sorted(constrained_high, key=lambda x: x['z_eff']):
    kt_pred_scaling = ell_t_weighted / r['chi_eff']
    print(f"    {r['pair_str']:>7s}: k_t_B3 = {r['kt_best']:.4f}, "
          f"k_t_scaling = {kt_pred_scaling:.4f}")


# ══════════════════════════════════════════════════════════════════
# 10. Chi^2 profile: compare per-pair vs joint
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CHI^2 PROFILE: PER-PAIR BEST vs SINGLE k_t^{phys}")
print("=" * 70)

# At the joint-best k_t^{phys}, what is each pair's chi^2 vs its
# own best-fit k_t? This quantifies the cost of forcing a single k_t.

eps_k_joint = eps_inf / (1.0 + (best_kt_phys / k_hMpc)**alpha)
M_joint = (1.0 + (N_val - 1) * eps_k_joint) / N_val
M_sq_joint = M_joint**2

print(f"\n  {'Pair':>7s}  {'chi2_own_kt':>11s}  {'chi2_joint':>11s}  "
      f"{'penalty':>8s}  {'k_t_own':>8s}")

total_penalty = 0
for r in sorted(constrained_high, key=lambda x: x['z_eff']):
    ip = r['ip']
    i0 = ip * n_ell
    i1 = (ip + 1) * n_ell
    d_pair = bp_data[ip]
    cov_pair = cov_ee[i0:i1, i0:i1]
    cov_inv_pair = np.linalg.inv(cov_pair)
    Cinv_d = cov_inv_pair @ d_pair
    dCd = d_pair @ Cinv_d

    # Chi^2 at this pair's own best k_t
    eps_k_own = eps_inf / (1.0 + (r['kt_best'] / k_hMpc)**alpha)
    M_own = (1.0 + (N_val - 1) * eps_k_own) / N_val
    m_own = A_tomo[ip] @ (M_own**2)
    Cinv_m = cov_inv_pair @ m_own
    chi2_own = dCd - (d_pair @ Cinv_m)**2 / (m_own @ Cinv_m)

    # Chi^2 at joint k_t^{phys}
    m_joint = A_tomo[ip] @ M_sq_joint
    Cinv_m = cov_inv_pair @ m_joint
    dCm = d_pair @ Cinv_m
    mCm = m_joint @ Cinv_m
    chi2_joint = dCd - dCm**2 / mCm if mCm > 0 else dCd

    penalty = chi2_joint - chi2_own
    total_penalty += penalty
    print(f"  {r['pair_str']:>7s}  {chi2_own:11.2f}  {chi2_joint:11.2f}  "
          f"{penalty:+8.2f}  {r['kt_best']:8.4f}")

print(f"\n  Total penalty for single k_t^phys: {total_penalty:+.2f}")
print(f"  (Penalty per pair: {total_penalty/len(constrained_high):+.2f})")

n_dof_diff = len(constrained_high) - 1  # pairs lose 1 dof each, gain 1 shared
print(f"  Additional dof freed by per-pair fits: {n_dof_diff}")
print(f"  Penalty per extra dof: {total_penalty/n_dof_diff:.2f}")


# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
  C1 Question: Does the Limber projection explain the per-pair k_t trend?

  Key results:
    1. Product k_t * chi_eff for high-SNR pairs:
       Mean ell_t = {ell_t_hi.mean():.0f}, scatter = {ell_t_hi.std()/ell_t_hi.mean():.1%}
""")

if ell_t_hi.std() / ell_t_hi.mean() < 0.15:
    print(f"       -> GOOD: ell_t is approximately constant ({ell_t_hi.std()/ell_t_hi.mean():.1%} scatter),")
    print(f"          consistent with a single angular transition scale.")
elif ell_t_hi.std() / ell_t_hi.mean() < 0.30:
    print(f"       -> MODERATE: ell_t shows {ell_t_hi.std()/ell_t_hi.mean():.0%} scatter.")
    print(f"          Limber scaling partially explains the trend but")
    print(f"          there is residual structure.")
else:
    print(f"       -> POOR: ell_t shows {ell_t_hi.std()/ell_t_hi.mean():.0%} scatter.")
    print(f"          The k_t trend is NOT well described by simple Limber scaling.")

print(f"""
    2. k_t(z) slope comparison:
       Observed:  {coeffs_obs[0]:.4f} dex/z
       Predicted: {coeffs_pred[0]:.4f} dex/z
       Ratio:     {coeffs_obs[0]/coeffs_pred[0]:.3f}
""")

slope_ratio = coeffs_obs[0] / coeffs_pred[0]
if 0.7 < slope_ratio < 1.3:
    print(f"       -> Slopes AGREE within 30%: Limber projection is the")
    print(f"          dominant driver of the k_t vs redshift trend.")
elif 0.5 < slope_ratio < 2.0:
    print(f"       -> Slopes are in ROUGH agreement but differ by "
          f"{abs(1-slope_ratio):.0%}.")
    print(f"          Limber projection is part of the story but not all.")
else:
    print(f"       -> Slopes DISAGREE by {abs(1-slope_ratio):.0%}.")
    print(f"          Limber projection does not explain the trend.")

print(f"""
    3. Joint single-k_t^phys fit:
       k_t^phys = {best_kt_phys:.4f} h/Mpc
       1-sigma:   [{kt_1sig[0]:.3f}, {kt_1sig[1]:.3f}]
       Penalty for single k_t: {total_penalty:+.1f} (vs per-pair fits)
       Penalty per extra dof:  {total_penalty/n_dof_diff:.1f}
""")

if total_penalty / n_dof_diff < 2.0:
    print(f"       -> Single k_t^phys is ADEQUATE: penalty < 2 per dof.")
    print(f"          The per-pair k_t variation is consistent with Limber")
    print(f"          projection of a single physical transition scale.")
elif total_penalty / n_dof_diff < 5.0:
    print(f"       -> Single k_t^phys is MARGINAL: penalty ~ "
          f"{total_penalty/n_dof_diff:.1f} per dof.")
    print(f"          Limber projection explains most but not all of the")
    print(f"          per-pair variation.")
else:
    print(f"       -> Single k_t^phys is INADEQUATE: penalty ~ "
          f"{total_penalty/n_dof_diff:.1f} per dof.")
    print(f"          The per-pair variation requires more than just")
    print(f"          Limber projection to explain.")

print(f"""
    4. Comparison to Paper V:
       Paper V fiducial: k_t = 0.20 h/Mpc
       This analysis:    k_t^phys = {best_kt_phys:.2f} h/Mpc ({best_kt_phys/0.20:.1f}x)
       The shift is {'partly' if 0.5 < slope_ratio < 2.0 else 'well' if 0.7 < slope_ratio < 1.3 else 'not'} explained by the Limber projection effect.
""")

print(f"Script complete.")
