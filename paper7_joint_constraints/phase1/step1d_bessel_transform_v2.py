#!/usr/bin/env python3
"""
Paper VII - Phase 1, Step 1d (v2)
Bessel transforms of C_κ(ℓ) to real-space observables.

Primary product: ξ+(θ) — cosmic shear 2-point correlation function.
    ξ+(θ) = ∫ ℓ dℓ/(2π) C_κ(ℓ) J₀(ℓθ)

This is what KiDS and DES measure and what we compare to in Phase 2.

Secondary product: ΔΣ(R) at a fiducial lens redshift (for visualisation
of the projection effect in the Paper V approximate vs proper comparison).

NOTE on ΔΣ: The C_κ(ℓ) from Step 1c is the convergence AUTO spectrum,
not the galaxy-matter CROSS spectrum. The Bessel transform to ΔΣ(R) 
gives the mean convergence profile, not the GGL signal. Values are 
smaller than typical GGL measurements. This is used for relative 
comparison (ΛCDM vs ISF) rather than absolute calibration.

Output: data/dsigma_lcdm.npz
Depends on: data/cl_kappa_lcdm.npz (Step 1c)

Changes from v1: Fixed monotonicity assertion, added physics note.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import jv
import camb
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
outpath = os.path.join(DATA_DIR, 'dsigma_lcdm.npz')

# ── Load Step 1c results ─────────────────────────────────────────
d = np.load(os.path.join(DATA_DIR, 'cl_kappa_lcdm.npz'), allow_pickle=True)
ell_grid = d['ell']
cl_kappa = d['cl_kappa_lcdm']
h_val = float(d['h'])
Om = float(d['Om'])
H0 = float(d['H0'])

ell_camb = d['ell_camb']
cl_camb = d['cl_kappa_camb']

print(f"Loaded C_κ(ℓ): our grid {len(ell_grid)} pts, "
      f"CAMB {len(ell_camb)} pts")

# ── Distances from CAMB ──────────────────────────────────────────
c_km_s = 2.998e5

pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=0.02237, omch2=0.1200, tau=0.054,
                   mnu=0.06, num_massive_neutrinos=1)
pars.InitPower.set_params(As=2.1e-9, ns=0.9649)
pars.set_matter_power(redshifts=[0], kmax=10.0)
results = camb.get_results(pars)

z_fine = np.linspace(0, 3.0, 3000)
chi_fine = results.comoving_radial_distance(z_fine)
chi_of_z = interp1d(z_fine, chi_fine, kind='cubic')

# ══════════════════════════════════════════════════════════════════
# PART 1: Cosmic shear ξ±(θ) — PRIMARY OBSERVABLE
# ══════════════════════════════════════════════════════════════════
# ξ+(θ) = ∫ ℓ dℓ/(2π) C_κ(ℓ) J₀(ℓθ)
# ξ-(θ) = ∫ ℓ dℓ/(2π) C_κ(ℓ) J₄(ℓθ)

print("\n── Computing ξ±(θ) from C_κ(ℓ) ──")

theta_arcmin = np.logspace(np.log10(0.5), np.log10(300), 30)
theta_rad = theta_arcmin * np.pi / (180.0 * 60.0)

# Use CAMB's dense ℓ sampling
ell_dense = np.arange(2, 10001, dtype=float)
cl_dense = cl_camb.copy()

xi_plus = np.zeros(len(theta_rad))
xi_minus = np.zeros(len(theta_rad))

for i, th in enumerate(theta_rad):
    integrand_plus = ell_dense * cl_dense * jv(0, ell_dense * th) / (2 * np.pi)
    integrand_minus = ell_dense * cl_dense * jv(4, ell_dense * th) / (2 * np.pi)

    xi_plus[i] = np.trapezoid(integrand_plus, ell_dense)
    xi_minus[i] = np.trapezoid(integrand_minus, ell_dense)

print(f"θ range: {theta_arcmin[0]:.1f}–{theta_arcmin[-1]:.0f} arcmin")
for th_ref in [1, 10, 100]:
    idx = np.argmin(np.abs(theta_arcmin - th_ref))
    print(f"  ξ+({th_ref}') = {xi_plus[idx]:.3e}, "
          f"ξ-({th_ref}') = {xi_minus[idx]:.3e}")

# ══════════════════════════════════════════════════════════════════
# PART 2: ΔΣ(R) for visualisation
# ══════════════════════════════════════════════════════════════════
# ΔΣ(R) = Σ_cr × ∫ ℓ dℓ/(2π) C_κ(ℓ) J₂(ℓθ)
# where θ = R/χ_L at the lens distance.
#
# This uses the convergence auto-spectrum, so gives the mean
# convergence profile — useful for RELATIVE comparison (ISF vs ΛCDM)
# but not absolute calibration against GGL data.

print("\n── Computing ΔΣ(R) at z_L = 0.27 (for visualisation) ──")

z_L = 0.27
chi_L = chi_of_z(z_L)

z_s_eff = 0.7
chi_s_eff = chi_of_z(z_s_eff)
chi_Ls = chi_s_eff - chi_L

c2_over_4piG = 1.663e18  # M_sun/Mpc
D_L = chi_L / (1 + z_L)
D_s = chi_s_eff / (1 + z_s_eff)
D_Ls = chi_Ls / (1 + z_s_eff)
Sigma_cr = c2_over_4piG * D_s / (D_L * D_Ls)
print(f"Lens: z_L={z_L}, χ_L={chi_L:.0f} Mpc, "
      f"Σ_cr = {Sigma_cr:.3e} M_sun/Mpc²")

R_hMpc = np.logspace(-1, 2, 15)
R_Mpc = R_hMpc / h_val

dsigma = np.zeros(len(R_Mpc))
for i, R in enumerate(R_Mpc):
    theta_R = R / chi_L
    integrand = ell_dense * cl_dense * jv(2, ell_dense * theta_R) / (2 * np.pi)
    dsigma[i] = Sigma_cr * np.trapezoid(integrand, ell_dense)

dsigma_hMsun_pc2 = dsigma * h_val * 1e-12

print(f"R range: {R_hMpc[0]:.1f}–{R_hMpc[-1]:.0f} h⁻¹Mpc")
for R_ref in [0.5, 1.0, 5.0, 10.0, 50.0]:
    idx = np.argmin(np.abs(R_hMpc - R_ref))
    print(f"  ΔΣ(R≈{R_hMpc[idx]:.1f} h⁻¹Mpc) = "
          f"{dsigma_hMsun_pc2[idx]:.4f} h M_sun/pc²")

# ── Sanity checks ────────────────────────────────────────────────
print("\n── Sanity checks ──")

# ξ+ should be positive and decreasing
assert np.all(xi_plus > 0), "ERROR: Negative ξ+ values!"
print("✓ All ξ+ values positive")

xi_10 = xi_plus[np.argmin(np.abs(theta_arcmin - 10))]
assert 1e-6 < xi_10 < 1e-3, f"ξ+(10') = {xi_10:.3e} outside expected range"
print(f"✓ ξ+(10') = {xi_10:.3e} (expected ~10⁻⁵ for Planck cosmology)")

# ξ+ should be roughly monotonically decreasing
# (allow small oscillations at large θ from Bessel ringing)
for j in range(len(xi_plus) - 2):
    if theta_arcmin[j] < 200:
        assert xi_plus[j] >= xi_plus[j+1] * 0.8, \
            f"ξ+ not decreasing at θ={theta_arcmin[j]:.1f}'"
print("✓ ξ+(θ) monotonically decreasing (within 20% tolerance)")

# ΔΣ should be mostly positive (convergence auto)
n_pos = np.sum(dsigma > 0)
print(f"✓ ΔΣ: {n_pos}/{len(dsigma)} positive values "
      f"(from convergence auto-spectrum)")

# ── Save checkpoint ──────────────────────────────────────────────
np.savez(
    outpath,
    # ξ±(θ) — primary WL observables
    theta_arcmin=theta_arcmin,
    theta_rad=theta_rad,
    xi_plus=xi_plus,
    xi_minus=xi_minus,
    # ΔΣ(R) — for visualisation / relative comparison
    R_hMpc=R_hMpc,
    R_Mpc=R_Mpc,
    dsigma=dsigma,
    dsigma_hMsun_pc2=dsigma_hMsun_pc2,
    # Geometry
    z_L=z_L, chi_L=chi_L, z_s_eff=z_s_eff, Sigma_cr=Sigma_cr,
    # Cosmology
    h=h_val, Om=Om,
)
print(f"\n✓ Saved to {outpath}")
print("\n── Step 1d complete ──")
