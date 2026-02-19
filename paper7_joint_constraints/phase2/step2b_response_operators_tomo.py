#!/usr/bin/env python3
"""
Paper VII - Phase 2, Step 2b
Build tomographic response operators for KiDS-1000.

For each tomographic bin pair (i,j), the Limber integral is:
  C_ij(ℓ) = ∫ dχ W_i(χ)W_j(χ)/χ² × P_NL(k=(ℓ+½)/χ, z(χ))

The ISF version is:
  C_ij^ISF(ℓ) = ∫ dχ W_i(χ)W_j(χ)/χ² × M²(k) × P_NL(k, z)

Discretizing with change of variables χ→k:
  C_ij^ISF(ℓ) = Σ_k A_ij(ℓ,k) × M²(k)

where A_ij(ℓ,k) = Δk/(ℓ+½) × W_i(χ_ℓk)W_j(χ_ℓk) × P_NL(k, z_ℓk)
and χ_ℓk = (ℓ+½)/k

We build A_ij at the 8 KiDS effective ℓ values, giving A_ij(8, 500).

Validation:
  (a) A_ij @ ones gives ΛCDM C_ij(ℓ) — compare shape to data
  (b) Compare single-bin A to Phase 1 response operator

Output: data/response_operators_tomo.npz
Depends on: data/kids1000_bandpowers.npz (Step 2a), CAMB
"""

import numpy as np
from scipy.interpolate import interp1d
import camb
import os
import time

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

OUTPATH = os.path.join(DATA_DIR, 'response_operators_tomo.npz')
KIDS_PATH = os.path.join(DATA_DIR, 'kids1000_bandpowers.npz')

# ── Load KiDS data from Step 2a ───────────────────────────────────
print("Loading KiDS data from Step 2a...")
dk = np.load(KIDS_PATH, allow_pickle=True)

ell_eff = dk['ell_eff']           # (8,) effective ℓ values
pair_labels = dk['pair_labels']   # (15, 2) bin pair indices
bp_data = dk['bp_data']           # (15, 8) band power values
z_mid = dk['z_mid']               # (119,) redshift bin centers
z_low = dk['z_low']               # (119,)
z_high = dk['z_high']             # (119,)
nz_bins = dk['nz_bins']           # (5, 119) n(z) per source bin

n_pairs = int(dk['n_pairs'])
n_ell = len(ell_eff)
n_zbins = 5

print(f"  ℓ values: {ell_eff}")
print(f"  Pairs: {n_pairs}, ℓ-bins: {n_ell}")
print(f"  n(z) bins: {n_zbins}, z-points: {len(z_mid)}")

# ══════════════════════════════════════════════════════════════════
# PART 1: CAMB setup — distances and P(k,z)
# ══════════════════════════════════════════════════════════════════
print("\n[1] Setting up CAMB cosmology...")

H0 = 67.4
h = H0 / 100.0
ombh2 = 0.02237
omch2 = 0.1200
Om = (ombh2 + omch2) / h**2
c_km_s = 2.998e5

pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, tau=0.054,
                   mnu=0.06, num_massive_neutrinos=1)
pars.InitPower.set_params(As=2.1e-9, ns=0.9649)
pars.set_matter_power(
    redshifts=sorted([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0],
                     reverse=True),
    kmax=100.0, nonlinear=True
)
pars.NonLinear = camb.model.NonLinear_both
results = camb.get_results(pars)

print(f"  σ8 = {results.get_sigma8_0():.4f}")

# Distance relations
z_fine = np.linspace(0, 5.0, 10000)
chi_fine = results.comoving_radial_distance(z_fine)  # Mpc
chi_of_z = interp1d(z_fine, chi_fine, kind='cubic')
z_of_chi = interp1d(chi_fine, z_fine, kind='cubic')

Hz_fine = np.array([results.hubble_parameter(z) for z in z_fine])
H_of_z = interp1d(z_fine, Hz_fine, kind='cubic')

chi_max_source = chi_of_z(min(z_high.max(), 4.0))
print(f"  χ(z=3) = {chi_of_z(3.0):.0f} Mpc")
print(f"  χ_max_source = {chi_max_source:.0f} Mpc")

# P(k,z) in Mpc units
PK_nl = results.get_matter_power_interpolator(
    nonlinear=True, var1='delta_tot', var2='delta_tot',
    hubble_units=False, k_hunit=False, log_interp=True
)

# ══════════════════════════════════════════════════════════════════
# PART 2: Lensing kernels W_i(χ) for each tomographic bin
# ══════════════════════════════════════════════════════════════════
print("\n[2] Computing lensing kernels W_i(χ) for 5 bins...")

# Prefactor: (3/2)(H0/c)² Ωm
H0_over_c = H0 / c_km_s  # 1/Mpc
prefactor = 1.5 * H0_over_c**2 * Om
print(f"  Lensing prefactor = {prefactor:.3e} /Mpc²")

# χ grid for kernel computation
n_chi_W = 800
chi_W_min = 10.0
chi_W_max = chi_max_source - 50
chi_W = np.linspace(chi_W_min, chi_W_max, n_chi_W)

# Source distribution on χ grid for each bin
# n_χ(χ) = n(z(χ)) × H(z(χ))/c  [so ∫ n_χ dχ = 1]
# Use KiDS z_mid grid converted to χ

# For the source integral, use a fine χ grid
chi_src_fine = np.linspace(1.0, chi_W_max + 100, 5000)
z_src_fine = z_of_chi(chi_src_fine)
Hz_src_fine = H_of_z(z_src_fine)

# Interpolate each n(z) bin onto the fine z grid
dz_kids = z_high - z_low  # KiDS bin widths
nz_interps = []
for b in range(n_zbins):
    # Normalize: ∫ n(z) dz = 1
    norm = np.sum(nz_bins[b] * dz_kids)
    nz_normed = nz_bins[b] / norm
    nz_func = interp1d(z_mid, nz_normed, kind='linear',
                        bounds_error=False, fill_value=0.0)
    nz_interps.append(nz_func)

# Compute W_i(χ) for each bin
W_all = np.zeros((n_zbins, n_chi_W))

for b in range(n_zbins):
    nz_func = nz_interps[b]

    # n_χ(χ) on the fine source grid
    n_chi_fine = nz_func(z_src_fine) * Hz_src_fine / c_km_s

    for ic, chi_val in enumerate(chi_W):
        z_val = z_of_chi(chi_val)

        # g(χ) = ∫_χ^∞ dχ' n_χ(χ') (χ'-χ)/χ'
        mask = chi_src_fine > chi_val
        if np.sum(mask) < 5:
            W_all[b, ic] = 0.0
            continue

        chi_s = chi_src_fine[mask]
        n_s = n_chi_fine[mask]
        integrand = n_s * (chi_s - chi_val) / chi_s
        g_val = np.trapezoid(integrand, chi_s)

        W_all[b, ic] = prefactor * chi_val * (1 + z_val) * g_val

    # Report peak
    ipk = np.argmax(W_all[b])
    z_pk = z_of_chi(chi_W[ipk])
    print(f"  Bin {b+1}: W peak at z={z_pk:.2f}, "
          f"χ={chi_W[ipk]:.0f} Mpc, "
          f"W_max={W_all[b,ipk]:.3e}")

# Build interpolators
W_interps = []
for b in range(n_zbins):
    W_func = interp1d(chi_W, W_all[b], kind='cubic',
                       bounds_error=False, fill_value=0.0)
    W_interps.append(W_func)

# ══════════════════════════════════════════════════════════════════
# PART 3: Build response operators A_ij for all 15 pairs
# ══════════════════════════════════════════════════════════════════
print(f"\n[3] Building {n_pairs} response operators, "
      f"each ({n_ell}, 500)...")

# k-grid: same as Phase 1
k_hMpc = np.logspace(-4, np.log10(50.0), 500)
k_Mpc = k_hMpc * h
n_k = len(k_Mpc)

# Trapezoid weights
dk = np.zeros(n_k)
dk[0] = k_Mpc[1] - k_Mpc[0]
dk[-1] = k_Mpc[-1] - k_Mpc[-2]
for j in range(1, n_k - 1):
    dk[j] = 0.5 * (k_Mpc[j+1] - k_Mpc[j-1])

# Valid χ range
chi_min_valid = chi_W_min
chi_max_valid = chi_W_max
z_max_pk = 3.0

# Store all response operators: shape (15, 8, 500)
A_tomo = np.zeros((n_pairs, n_ell, n_k))

# Also compute ΛCDM predictions: A_ij @ ones
cl_lcdm_tomo = np.zeros((n_pairs, n_ell))

t0 = time.time()

for ip in range(n_pairs):
    b1, b2 = pair_labels[ip]
    b1_idx = b1 - 1  # 0-indexed
    b2_idx = b2 - 1

    W1 = W_interps[b1_idx]
    W2 = W_interps[b2_idx]

    for il, ell in enumerate(ell_eff):
        ell_half = ell + 0.5

        for jk in range(n_k):
            k_j = k_Mpc[jk]
            chi_ij = ell_half / k_j

            if chi_ij < chi_min_valid or chi_ij > chi_max_valid:
                continue

            z_ij = z_of_chi(chi_ij)
            if z_ij < 0 or z_ij > z_max_pk:
                continue

            W1_val = W1(chi_ij)
            W2_val = W2(chi_ij)
            if W1_val <= 0 or W2_val <= 0:
                continue

            try:
                pk_ij = PK_nl.P(z_ij, k_j)
            except Exception:
                continue

            # Key formula: W_i × W_j instead of W²
            A_tomo[ip, il, jk] = (dk[jk] / ell_half) * \
                                  W1_val * W2_val * pk_ij

    # ΛCDM: M² = 1
    cl_lcdm_tomo[ip] = A_tomo[ip] @ np.ones(n_k)

    elapsed = time.time() - t0
    print(f"  Pair ({b1},{b2}): done "
          f"[{elapsed:.1f}s elapsed, "
          f"ΛCDM C_ℓ range: "
          f"{cl_lcdm_tomo[ip].min():.3e} to "
          f"{cl_lcdm_tomo[ip].max():.3e}]")

total_time = time.time() - t0
print(f"\n  All operators built in {total_time:.1f}s")
print(f"  A_tomo shape: {A_tomo.shape}")

# ══════════════════════════════════════════════════════════════════
# PART 4: Convention check — compare ΛCDM to KiDS data
# ══════════════════════════════════════════════════════════════════
print("\n[4] Convention check: ΛCDM vs KiDS data...")

# Our C_ij(ℓ) is the raw convergence power spectrum (dimensionless).
# KiDS band powers might be ℓ(ℓ+1)C_E/(2π) or raw C_E.
# Test both conventions.

# Convention A: data = raw C_E(ℓ)
ratio_raw = bp_data / cl_lcdm_tomo

# Convention B: data = ℓ(ℓ+1)C_E(ℓ)/(2π)
ell_factor = ell_eff * (ell_eff + 1) / (2 * np.pi)
cl_lcdm_scaled = cl_lcdm_tomo * ell_factor[np.newaxis, :]
ratio_scaled = bp_data / cl_lcdm_scaled

print(f"\n  Convention A (data = raw C_E):")
print(f"    Ratio range: [{ratio_raw.min():.3f}, {ratio_raw.max():.3f}]")
med_A = np.median(ratio_raw[bp_data > 0])
print(f"    Median ratio (positive data): {med_A:.3f}")

print(f"\n  Convention B (data = ℓ(ℓ+1)C_E/(2π)):")
print(f"    Ratio range: [{ratio_scaled.min():.3f}, {ratio_scaled.max():.3f}]")
med_B = np.median(ratio_scaled[bp_data > 0])
print(f"    Median ratio (positive data): {med_B:.3f}")

# The correct convention should give ratios near 1
# (our ΛCDM uses Planck cosmology; KiDS prefers slightly lower S8)
if abs(med_A - 1.0) < abs(med_B - 1.0):
    convention = 'raw'
    print(f"\n  → Convention A (raw C_E) is closer: median ratio = {med_A:.3f}")
    cl_lcdm_compare = cl_lcdm_tomo.copy()
else:
    convention = 'scaled'
    print(f"\n  → Convention B (ℓ(ℓ+1)C_E/(2π)) is closer: "
          f"median ratio = {med_B:.3f}")
    cl_lcdm_compare = cl_lcdm_scaled.copy()

# Detailed comparison for a few pairs
print(f"\n  Pair-by-pair comparison ({convention}):")
print(f"  {'Pair':>8s}  {'med ratio':>10s}  {'mean |ratio-1|':>15s}")
for ip in range(n_pairs):
    b1, b2 = pair_labels[ip]
    if convention == 'raw':
        r = bp_data[ip] / cl_lcdm_tomo[ip]
    else:
        r = bp_data[ip] / cl_lcdm_scaled[ip]
    valid = (cl_lcdm_tomo[ip] > 0) & np.isfinite(r)
    if np.any(valid):
        med = np.median(r[valid])
        dev = np.mean(np.abs(r[valid] - 1.0))
        print(f"  ({b1},{b2}){'':<3s}  {med:10.3f}  {dev:15.3f}")

# ══════════════════════════════════════════════════════════════════
# PART 5: Cross-check with Phase 1 response operator
# ══════════════════════════════════════════════════════════════════
print("\n[5] Cross-check: comparison with Phase 1 operator...")

# Load Phase 1 response operator (single effective n(z))
p1_path = os.path.join(DATA_DIR, 'response_operator.npz')
if os.path.isfile(p1_path):
    dp1 = np.load(p1_path)
    A_p1 = dp1['A']           # (100, 500)
    ell_p1 = dp1['ell']       # (100,)
    cl_p1_lcdm = dp1['cl_lcdm_from_A']  # (100,)

    # Interpolate Phase 1 ΛCDM to KiDS ℓ values
    cl_p1_interp = interp1d(ell_p1, cl_p1_lcdm, kind='cubic')
    cl_p1_at_kids_ell = cl_p1_interp(ell_eff)

    # The Phase 1 operator used a KiDS-like analytic n(z).
    # Sum over all tomographic pairs (weighted) should roughly match.
    # Simple sum of auto-correlations:
    cl_sum_auto = np.zeros(n_ell)
    for ip in range(n_pairs):
        b1, b2 = pair_labels[ip]
        if b1 == b2:
            cl_sum_auto += cl_lcdm_tomo[ip]

    print(f"  Phase 1 ΛCDM C_κ at ℓ~500: {cl_p1_at_kids_ell[3]:.3e}")
    print(f"  Sum of auto-pair ΛCDM: {cl_sum_auto[3]:.3e}")
    print(f"  Ratio (sum_auto / P1): {cl_sum_auto[3]/cl_p1_at_kids_ell[3]:.3f}")
    print(f"  (This ratio is expected to differ since P1 used a")
    print(f"   different broader n(z). Shape consistency matters.)")
else:
    print(f"  Phase 1 data not found at {p1_path} — skipping cross-check")

# ══════════════════════════════════════════════════════════════════
# PART 6: Save checkpoint
# ══════════════════════════════════════════════════════════════════
print("\n[6] Saving checkpoint...")

np.savez(
    OUTPATH,
    # Response operators
    A_tomo=A_tomo,                  # (15, 8, 500) response operators
    ell_eff=ell_eff,                # (8,) effective ℓ values
    k_hMpc=k_hMpc,                 # (500,) k in h/Mpc
    k_Mpc=k_Mpc,                   # (500,) k in 1/Mpc
    pair_labels=pair_labels,        # (15, 2)
    n_pairs=n_pairs,
    # ΛCDM predictions
    cl_lcdm_tomo=cl_lcdm_tomo,     # (15, 8)
    convention=convention,          # 'raw' or 'scaled'
    # Lensing kernels
    chi_W=chi_W,                    # (n_chi_W,)
    W_all=W_all,                    # (5, n_chi_W)
    # Cosmology
    h=h, Om=Om, H0=H0,
)

print(f"  ✓ Saved to {OUTPATH}")
print(f"  A_tomo: {A_tomo.shape}")
print(f"  Convention: {convention}")

# ══════════════════════════════════════════════════════════════════
# PART 7: Summary
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY — Ready for Step 2c (Likelihood)")
print("=" * 70)
print(f"""
Built 15 tomographic response operators:
  A_tomo shape: {A_tomo.shape} = (pairs, ℓ-bins, k-bins)
  ℓ range: [{ell_eff[0]:.1f}, {ell_eff[-1]:.1f}]
  k range: [{k_hMpc[0]:.4f}, {k_hMpc[-1]:.1f}] h/Mpc
  Band power convention: {convention}

Usage for any ISF model (N, k_t, ε∞):
  eps_k = eps_inf / (1 + (k_t / k_hMpc)**2)
  M = (1 + (N-1)*eps_k) / N
  M_sq = M**2
  cl_isf = np.array([A_tomo[ip] @ M_sq for ip in range(15)])

  Then: χ² = (data - cl_isf_flat) @ C⁻¹ @ (data - cl_isf_flat)
  where cl_isf_flat = cl_isf.flatten() (120 entries)

Time per ISF model: ~microseconds (15 matrix-vector products)
Full 6×20 grid: ~milliseconds
""")

print("Script complete.")
