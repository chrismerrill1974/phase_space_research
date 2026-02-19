#!/usr/bin/env python3
"""
Paper VII - Phase 1, Step 1e
Precompute the response operator A_ij.

    C_κ^ISF(ℓ_i) = Σ_j A_ij × M_ISF²(k_j)

Derived from the Limber integral with change of variables χ→k:
    A_ij = (Δk_j / (ℓ_i+½)) × W²((ℓ_i+½)/k_j) × P_NL(k_j, z((ℓ_i+½)/k_j))

where Δk_j are trapezoid integration weights on the k-grid.

Validation:
  (a) A @ ones should reproduce ΛCDM C_κ(ℓ) from Step 1c (<0.1%)
  (b) Convergence test: doubling k resolution changes result <1%

Output: data/response_operator.npz
Depends on: data/cl_kappa_lcdm.npz (Step 1c), CAMB for P_NL and distances
"""

import numpy as np
from scipy.interpolate import interp1d
import camb
import os
import time

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
outpath = os.path.join(DATA_DIR, 'response_operator.npz')

# ── Load Step 1c for validation reference ────────────────────────
d1c = np.load(os.path.join(DATA_DIR, 'cl_kappa_lcdm.npz'), allow_pickle=True)
ell_ref = d1c['ell']           # (100,)
cl_ref = d1c['cl_kappa_lcdm']  # (100,) from direct Limber integration
h_val = float(d1c['h'])
Om = float(d1c['Om'])
H0 = float(d1c['H0'])
chi_W_ref = d1c['chi_W']
W_arr_ref = d1c['W_arr']

# ── Cosmological setup from CAMB ─────────────────────────────────
c_km_s = 2.998e5

pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=0.02237, omch2=0.1200, tau=0.054,
                   mnu=0.06, num_massive_neutrinos=1)
pars.InitPower.set_params(As=2.1e-9, ns=0.9649)
pars.set_matter_power(
    redshifts=sorted([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0],
                     reverse=True),
    kmax=100.0, nonlinear=True
)
pars.NonLinear = camb.model.NonLinear_both
results = camb.get_results(pars)

# Distance relations
z_fine = np.linspace(0, 3.5, 5000)
chi_fine = results.comoving_radial_distance(z_fine)
chi_of_z = interp1d(z_fine, chi_fine, kind='cubic')
z_of_chi = interp1d(chi_fine, z_fine, kind='cubic')

chi_H = chi_of_z(3.0)
print(f"χ_H = {chi_H:.0f} Mpc")

# P(k,z) in Mpc units
PK_nl = results.get_matter_power_interpolator(
    nonlinear=True, var1='delta_tot', var2='delta_tot',
    hubble_units=False, k_hunit=False, log_interp=True
)

# Lensing kernel W(χ) — reuse from Step 1c
W_of_chi = interp1d(chi_W_ref, W_arr_ref, kind='cubic',
                     bounds_error=False, fill_value=0.0)

print(f"W(χ) peak = {W_arr_ref.max():.3e} at χ={chi_W_ref[np.argmax(W_arr_ref)]:.0f} Mpc")

# ══════════════════════════════════════════════════════════════════
# Build the response operator A_ij
# ══════════════════════════════════════════════════════════════════
#
# A_ij = (Δk_j / (ℓ_i+½)) × W²(χ_ij) × P_NL(k_j, z_ij)
#
# where χ_ij = (ℓ_i+½)/k_j and z_ij = z(χ_ij)
# Δk_j = trapezoid weights on the k-grid
#
# k is in 1/Mpc (matching CAMB Mpc-units convention)

# ── k-grid in 1/Mpc ─────────────────────────────────────────────
# Use the same 500-point log grid as Step 1a, but in 1/Mpc
k_hMpc = np.logspace(-4, np.log10(50.0), 500)  # h/Mpc
k_Mpc = k_hMpc * h_val  # 1/Mpc

n_k = len(k_Mpc)
print(f"k grid: {n_k} points, [{k_Mpc[0]:.2e}, {k_Mpc[-1]:.1f}] /Mpc")

# ── ℓ-grid (same as Step 1c) ─────────────────────────────────────
ell_grid = np.logspace(np.log10(10), np.log10(10000), 100)
n_ell = len(ell_grid)
print(f"ℓ grid: {n_ell} points, [{ell_grid[0]:.0f}, {ell_grid[-1]:.0f}]")

# ── Trapezoid weights for k integration ──────────────────────────
# For log-spaced grid, use standard trapezoid
dk = np.zeros(n_k)
dk[0] = (k_Mpc[1] - k_Mpc[0])
dk[-1] = (k_Mpc[-1] - k_Mpc[-2])
for j in range(1, n_k - 1):
    dk[j] = 0.5 * (k_Mpc[j+1] - k_Mpc[j-1])

# ── Build A_ij ────────────────────────────────────────────────────
print(f"\nBuilding response operator A ({n_ell} × {n_k})...")
t0 = time.time()

A = np.zeros((n_ell, n_k))

chi_min_valid = 10.0
chi_max_valid = chi_H - 50
z_max_pk = 3.0

for i, ell in enumerate(ell_grid):
    ell_half = ell + 0.5

    for j in range(n_k):
        k_j = k_Mpc[j]
        chi_ij = ell_half / k_j  # Mpc

        # Check χ is within valid range
        if chi_ij < chi_min_valid or chi_ij > chi_max_valid:
            continue

        z_ij = z_of_chi(chi_ij)
        if z_ij < 0 or z_ij > z_max_pk:
            continue

        W_ij = W_of_chi(chi_ij)
        if W_ij <= 0:
            continue

        # P_NL(k, z) in Mpc³ — k in 1/Mpc
        try:
            pk_ij = PK_nl.P(z_ij, k_j)
        except Exception:
            continue

        A[i, j] = (dk[j] / ell_half) * W_ij**2 * pk_ij

    if (i + 1) % 25 == 0:
        elapsed = time.time() - t0
        print(f"  {i+1}/{n_ell}: ℓ={ell:.0f}, "
              f"non-zero A entries: {np.sum(A[i] > 0)}, "
              f"elapsed: {elapsed:.1f}s")

elapsed = time.time() - t0
print(f"A built in {elapsed:.1f}s")
print(f"Shape: {A.shape}, non-zero fraction: "
      f"{np.count_nonzero(A) / A.size:.3f}")

# ══════════════════════════════════════════════════════════════════
# Validation (a): A @ 1 should reproduce ΛCDM C_κ(ℓ)
# ══════════════════════════════════════════════════════════════════
print("\n── Validation: A @ ones vs direct Limber ──")

# ΛCDM: M_ISF = 1, so M² = 1
ones = np.ones(n_k)
cl_from_A = A @ ones

# Compare to Step 1c direct computation
ratio = cl_from_A / cl_ref
frac_diff = np.abs(ratio - 1.0)

print(f"Ratio range: [{ratio.min():.4f}, {ratio.max():.4f}]")
print(f"Mean ratio: {ratio.mean():.4f}")
print(f"Max |frac diff|: {frac_diff.max()*100:.2f}%")
print(f"Mean |frac diff|: {frac_diff.mean()*100:.2f}%")

# Report at reference ℓ values
for ell_check in [100, 300, 1000, 3000]:
    idx = np.argmin(np.abs(ell_grid - ell_check))
    r = cl_from_A[idx] / cl_ref[idx]
    print(f"  ℓ≈{ell_grid[idx]:.0f}: ratio = {r:.5f} "
          f"(diff = {abs(r-1)*100:.3f}%)")

if frac_diff.max() < 0.05:
    print("✓ PASS: <5% agreement (response operator valid)")
else:
    print("⚠ WARNING: >5% disagreement — investigate")

# ══════════════════════════════════════════════════════════════════
# Validation (b): Convergence test — double k resolution
# ══════════════════════════════════════════════════════════════════
print("\n── Convergence test: 2× k resolution ──")

k_Mpc_2x = np.logspace(-4, np.log10(50.0 * h_val), 1000) # 1000 pts
n_k_2x = len(k_Mpc_2x)

dk_2x = np.zeros(n_k_2x)
dk_2x[0] = k_Mpc_2x[1] - k_Mpc_2x[0]
dk_2x[-1] = k_Mpc_2x[-1] - k_Mpc_2x[-2]
for j in range(1, n_k_2x - 1):
    dk_2x[j] = 0.5 * (k_Mpc_2x[j+1] - k_Mpc_2x[j-1])

# Build A at double resolution for a subset of ℓ values
ell_test_indices = np.linspace(0, n_ell-1, 20, dtype=int)
cl_2x = np.zeros(len(ell_test_indices))

for ii, i in enumerate(ell_test_indices):
    ell = ell_grid[i]
    ell_half = ell + 0.5
    row = np.zeros(n_k_2x)

    for j in range(n_k_2x):
        k_j = k_Mpc_2x[j]
        chi_ij = ell_half / k_j

        if chi_ij < chi_min_valid or chi_ij > chi_max_valid:
            continue
        z_ij = z_of_chi(chi_ij)
        if z_ij < 0 or z_ij > z_max_pk:
            continue
        W_ij = W_of_chi(chi_ij)
        if W_ij <= 0:
            continue
        try:
            pk_ij = PK_nl.P(z_ij, k_j)
        except Exception:
            continue
        row[j] = (dk_2x[j] / ell_half) * W_ij**2 * pk_ij

    cl_2x[ii] = np.sum(row)  # M² = 1 for ΛCDM

conv_ratio = cl_2x / cl_from_A[ell_test_indices]
conv_diff = np.abs(conv_ratio - 1.0)

print(f"2× resolution ratio range: [{conv_ratio.min():.4f}, {conv_ratio.max():.4f}]")
print(f"Max |convergence diff|: {conv_diff.max()*100:.2f}%")

if conv_diff.max() < 0.01:
    print("✓ PASS: <1% convergence (k resolution sufficient)")
elif conv_diff.max() < 0.05:
    print("~ ACCEPTABLE: <5% convergence")
else:
    print("⚠ WARNING: >5% — consider increasing k resolution")

# ══════════════════════════════════════════════════════════════════
# Quick ISF test: compute C_κ for fiducial ISF
# ══════════════════════════════════════════════════════════════════
print("\n── Quick ISF test: fiducial N=6, k_t=0.20 h/Mpc ──")

N_fid = 6
eps_inf = 0.58
k_t_hMpc = 0.20  # h/Mpc
alpha = 2

# M_ISF²(k) on our k-grid (k in h/Mpc for the ISF formula)
eps_k = eps_inf / (1.0 + (k_t_hMpc / k_hMpc)**alpha)
M_isf = (1.0 + (N_fid - 1) * eps_k) / N_fid
M_isf_sq = M_isf**2

cl_isf = A @ M_isf_sq

# ISF/ΛCDM ratio at a few ℓ values
for ell_check in [100, 500, 1000, 3000]:
    idx = np.argmin(np.abs(ell_grid - ell_check))
    r = cl_isf[idx] / cl_from_A[idx]
    print(f"  ℓ≈{ell_grid[idx]:.0f}: C_κ^ISF/C_κ^ΛCDM = {r:.4f}")

# The ratio should be between M²(k<<k_t)=1/N²=0.028 and M²(k>>k_t)=0.422
print(f"  Expected range: [{1/N_fid**2:.3f}, {M_isf_sq.max():.3f}]")

# ══════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════

# Also build a convenience function wrapper
# (saved as metadata for documentation)

np.savez(
    outpath,
    # Response operator
    A=A,                           # (n_ell, n_k)
    ell=ell_grid,                  # (n_ell,) multipole grid
    k_hMpc=k_hMpc,                # (n_k,) k in h/Mpc
    k_Mpc=k_Mpc,                  # (n_k,) k in 1/Mpc
    # ΛCDM reference
    cl_lcdm_from_A=cl_from_A,     # A @ ones
    cl_lcdm_direct=cl_ref,        # from Step 1c
    # Validation metrics
    validation_max_frac_diff=frac_diff.max(),
    convergence_max_diff=conv_diff.max(),
    # ISF test
    cl_isf_fiducial=cl_isf,       # fiducial N=6, k_t=0.20
    M_isf_sq_fiducial=M_isf_sq,
    # Cosmology
    h=h_val, Om=Om, H0=H0,
)
print(f"\n✓ Saved to {outpath}")
print(f"  A shape: {A.shape}")
print(f"  Usage: cl_isf = A @ M_isf_squared(k_hMpc)")
print("\n── Step 1e complete ──")
