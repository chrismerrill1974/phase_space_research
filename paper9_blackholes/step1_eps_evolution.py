#!/usr/bin/env python3
"""
Paper IX — Step 1: eps(k,z) from gravitational capture

Computes the two quantitative predictions of the capture mechanism:

  (a) eps_inf(z) proportional to the Press-Schechter collapsed mass
      fraction F(>M_min, z), normalised to the Paper V/VII fitted
      eps_inf(0) = 0.58, for M_min in {1e10, 1e11, 1e12} Msun/h.
  (b) k_t(z) identified with the nonlinear scale k_NL(z) defined by
      Delta^2(k_NL, z) = 1; compared against the Paper VII fitted
      k_t = 0.52 [0.46, 0.55] h/Mpc.

sigma(M,z) is computed from the Paper VII CAMB linear P(k, z=0),
rescaled to arbitrary z with the Carroll-Press-Turner growth factor
(flat LCDM, Om=0.315). This avoids rerunning CAMB and is accurate to
<<1% for this cosmology over z=0-6.

Input:  ../data/pknl_camb.npz  (k, z, pklin from Paper VII Phase 1a)
Output: ../data/eps_evolution.npz
"""

import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA = os.path.join(ROOT, 'data')
OUTPATH = os.path.join(DATA, 'eps_evolution.npz')

OM, OL = 0.315, 0.685            # Planck 2018, as Papers V-VIII
RHO_CRIT = 2.775e11              # h^2 Msun / Mpc^3
DELTA_C = 1.686                  # spherical collapse threshold
EPS_INF_0 = 0.58                 # Paper V/VII fitted value at z=0
KT_FIT = (0.52, 0.46, 0.55)      # Paper VII k_t^phys and 1-sigma range

d = np.load(os.path.join(DATA, 'pknl_camb.npz'), allow_pickle=True)
k = d['k']                        # h/Mpc
pklin0 = d['pklin'][0]            # z=0 linear P(k), (Mpc/h)^3

# ── Growth factor (Carroll, Press & Turner 1992 approximation) ────
def growth(z):
    a = 1.0 / (1.0 + z)
    Ez2 = OM / a**3 + OL
    om_a = OM / a**3 / Ez2
    ol_a = OL / Ez2
    g = 2.5 * om_a / (om_a**(4/7.) - ol_a + (1 + om_a/2.) * (1 + ol_a/70.))
    return a * g

def D_norm(z):
    return growth(z) / growth(0.0)

# ── sigma(M, z) from tophat-filtered linear P(k) ──────────────────
rho_m = OM * RHO_CRIT             # comoving matter density, h^2 Msun/Mpc^3

def sigma_R(R, z):
    x = k * R
    W = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3
    integrand = pklin0 * W**2 * k**2
    s2 = np.trapezoid(integrand, k) / (2 * np.pi**2)
    return np.sqrt(s2) * D_norm(z)

def sigma_M(M, z):
    R = (3 * M / (4 * np.pi * rho_m))**(1/3.)   # Mpc/h, M in Msun/h
    return sigma_R(R, z)

# sanity: sigma_8 at z=0
s8 = sigma_R(8.0, 0.0)
print(f"sigma_8(z=0) from pklin: {s8:.3f}  (CAMB file value: {float(d['sigma8']):.3f})")

# ── (a) Collapsed fraction and eps_inf(z) ─────────────────────────
z_grid = np.linspace(0, 6, 61)
M_min_list = [1e10, 1e11, 1e12]   # Msun/h

F_coll = np.zeros((len(M_min_list), len(z_grid)))
from math import erfc
for iM, Mmin in enumerate(M_min_list):
    for iz, z in enumerate(z_grid):
        s = sigma_M(Mmin, z)
        F_coll[iM, iz] = erfc(DELTA_C / (np.sqrt(2) * s))

eps_inf_z = EPS_INF_0 * F_coll / F_coll[:, :1]   # normalised at z=0

print("\neps_inf(z) prediction (normalised to 0.58 at z=0):")
print(f"  {'z':>4s}" + "".join(f"  Mmin=1e{int(np.log10(M)):d}" for M in M_min_list))
for z_show in [0.0, 0.5, 1.0, 2.0, 4.0, 6.0]:
    iz = np.argmin(np.abs(z_grid - z_show))
    row = "  ".join(f"{eps_inf_z[iM, iz]:9.3f}" for iM in range(len(M_min_list)))
    print(f"  {z_grid[iz]:4.1f}  {row}")

# ── (b) Nonlinear scale k_NL(z): Delta^2(k_NL, z) = 1 ─────────────
def k_NL(z):
    Delta2 = k**3 * pklin0 * D_norm(z)**2 / (2 * np.pi**2)
    # first upward crossing of 1
    idx = np.where(Delta2 >= 1.0)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    # log-linear interpolation
    lk = np.interp(0.0, np.log(Delta2[i-1:i+1]), np.log(k[i-1:i+1]))
    return np.exp(lk)

kNL_z = np.array([k_NL(z) for z in z_grid])
print(f"\nk_NL(z=0) = {kNL_z[0]:.3f} h/Mpc")
print(f"Paper VII fitted k_t^phys = {KT_FIT[0]:.2f} [{KT_FIT[1]:.2f}, {KT_FIT[2]:.2f}] h/Mpc")
print(f"Ratio k_t^fit / k_NL(0) = {KT_FIT[0]/kNL_z[0]:.2f}")
for z_show in [0.5, 1.0, 2.0]:
    iz = np.argmin(np.abs(z_grid - z_show))
    print(f"k_NL(z={z_grid[iz]:.1f}) = {kNL_z[iz]:.3f} h/Mpc "
          f"(ratio to z=0: {kNL_z[iz]/kNL_z[0]:.2f})")

# ── Contrast curve: baryonic-feedback-like evolution ──────────────
# Feedback suppression tracks the star-formation/AGN history, commonly
# peaked near z~2 (Madau-Lickhtman shape); we use the Madau-Dickinson
# SFRD shape as the representative mimic to show the two z-evolutions
# differ in shape, not just amplitude.
sfrd = (1 + z_grid)**2.7 / (1 + ((1 + z_grid)/2.9)**5.6)   # Madau-Dickinson 2014
sfrd_norm = sfrd / sfrd[0]

np.savez(
    OUTPATH,
    z_grid=z_grid, M_min_list=np.array(M_min_list),
    F_coll=F_coll, eps_inf_z=eps_inf_z, eps_inf_0=EPS_INF_0,
    kNL_z=kNL_z, kt_fit=np.array(KT_FIT),
    sigma8_check=s8, sfrd_shape=sfrd_norm,
    k=k, pklin0=pklin0,
)
print(f"\nSaved {OUTPATH}")
