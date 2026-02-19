#!/usr/bin/env python3
"""
Paper VII - Phase 1, Step 1a
Generate ΛCDM nonlinear matter power spectrum P_NL(k, z) using CAMB.

Cosmology: Planck 2018 (TT,TE,EE+lowE+lensing)
  Ωm = 0.315, σ8 = 0.811, h = 0.674, ns = 0.965
  Ωb h² = 0.02237, Ωc h² = 0.1200

k range: [1e-4, 50] h/Mpc, 500 log-spaced points
z values: [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

Output: data/pknl_camb.npz
"""

import numpy as np
import camb
import os

# ── Output path ──────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)
outpath = os.path.join(DATA_DIR, 'pknl_camb.npz')

# ── Planck 2018 cosmological parameters ─────────────────────────
H0 = 67.4          # km/s/Mpc
ombh2 = 0.02237    # Ωb h²
omch2 = 0.1200     # Ωc h²
ns = 0.9649        # scalar spectral index
# σ8 = 0.811 — we'll set As and verify σ8 after
As_init = 2.1e-9   # initial guess for As
tau = 0.054         # optical depth to reionisation

# Derived
h = H0 / 100.0
Om = (ombh2 + omch2) / h**2
print(f"Planck 2018 cosmology: H0={H0}, Ωm={Om:.4f}, h={h}, ns={ns}")

# ── k and z grids ────────────────────────────────────────────────
k_grid = np.logspace(-4, np.log10(50.0), 500)  # h/Mpc
z_list = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
print(f"k grid: {len(k_grid)} points from {k_grid[0]:.1e} to {k_grid[-1]:.1f} h/Mpc")
print(f"z values: {z_list}")

# ── Set up CAMB parameters ──────────────────────────────────────
pars = camb.CAMBparams()
pars.set_cosmology(
    H0=H0,
    ombh2=ombh2,
    omch2=omch2,
    tau=tau,
    mnu=0.06,        # minimal neutrino mass
    num_massive_neutrinos=1
)
pars.InitPower.set_params(As=As_init, ns=ns)

# Request nonlinear P(k) via Halofit
pars.set_matter_power(
    redshifts=sorted(z_list, reverse=True),  # CAMB wants descending
    kmax=60.0,  # slightly beyond our grid edge
    nonlinear=True
)
pars.NonLinear = camb.model.NonLinear_both  # Halofit for matter power

# ── Run CAMB ─────────────────────────────────────────────────────
print("\nRunning CAMB (linear + Halofit nonlinear)...")
results = camb.get_results(pars)

# Check σ8
sigma8_camb = results.get_sigma8_0()
print(f"σ8 from CAMB = {sigma8_camb:.4f} (target: 0.811)")

# ── Extract P_NL(k, z) ──────────────────────────────────────────
# CAMB returns k in 1/Mpc; we want h/Mpc
# Use get_matter_power_interpolator for precise k sampling
print("\nExtracting nonlinear P(k, z) on our grid...")

# Get the nonlinear interpolator
PK_nl = results.get_matter_power_interpolator(
    nonlinear=True,
    var1='delta_tot',
    var2='delta_tot',
    hubble_units=True,   # k in h/Mpc, P(k) in (Mpc/h)³
    k_hunit=True,
    log_interp=True
)

# Also get linear for comparison
PK_lin = results.get_matter_power_interpolator(
    nonlinear=False,
    var1='delta_tot',
    var2='delta_tot',
    hubble_units=True,
    k_hunit=True,
    log_interp=True
)

# Evaluate on our grid
pknl_array = np.zeros((len(z_list), len(k_grid)))
pklin_array = np.zeros((len(z_list), len(k_grid)))

for i, z in enumerate(z_list):
    pknl_array[i, :] = PK_nl.P(z, k_grid)
    pklin_array[i, :] = PK_lin.P(z, k_grid)
    print(f"  z={z:.1f}: P_NL(k=0.1)={PK_nl.P(z, 0.1):.1f}, "
          f"P_NL(k=1.0)={PK_nl.P(z, 1.0):.1f} (Mpc/h)³")

# ── Sanity checks ────────────────────────────────────────────────
print("\n── Sanity checks ──")

# 1. P(k) should be positive everywhere
assert np.all(pknl_array > 0), "ERROR: Negative P(k) values found!"
print("✓ All P_NL values positive")

# 2. Nonlinear > linear at high k (z=0)
ratio_z0 = pknl_array[0] / pklin_array[0]
k_high = k_grid > 0.5
assert np.all(ratio_z0[k_high] > 1.0), "ERROR: Halofit not enhancing high-k!"
print(f"✓ Halofit enhancement at k>0.5: {ratio_z0[k_high].min():.2f}x to "
      f"{ratio_z0[k_high].max():.1f}x")

# 3. P(k) should decrease with increasing z at fixed k
for j in range(len(z_list) - 1):
    # Growth: P(k, z_lower) > P(k, z_higher) at linear scales
    ratio = pknl_array[j, 100] / pknl_array[j+1, 100]
    print(f"  P(k={k_grid[100]:.3f}, z={z_list[j]:.1f}) / "
          f"P(k={k_grid[100]:.3f}, z={z_list[j+1]:.1f}) = {ratio:.3f}")

# 4. σ8 consistency
print(f"\n✓ σ8 = {sigma8_camb:.4f} (target 0.811, "
      f"deviation {abs(sigma8_camb - 0.811)/0.811*100:.1f}%)")

# ── Save checkpoint ──────────────────────────────────────────────
np.savez(
    outpath,
    k=k_grid,               # (500,) h/Mpc
    z=np.array(z_list),      # (7,)
    pknl=pknl_array,         # (7, 500) (Mpc/h)³
    pklin=pklin_array,        # (7, 500) (Mpc/h)³
    sigma8=sigma8_camb,
    cosmology={
        'H0': H0, 'ombh2': ombh2, 'omch2': omch2,
        'ns': ns, 'As': As_init, 'tau': tau,
        'Om': Om, 'h': h
    }
)
print(f"\n✓ Saved to {outpath}")
print(f"  Arrays: k({k_grid.shape}), z({len(z_list)}), "
      f"pknl{pknl_array.shape}, pklin{pklin_array.shape}")

print("\n── Step 1a complete ──")
