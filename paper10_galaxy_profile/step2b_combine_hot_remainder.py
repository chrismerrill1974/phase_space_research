#!/usr/bin/env python3
"""
Paper X — Step 2b: captured component plus the hot remainder, out to R_200

Step 2 built the dark halo inside the disc region from captured
other-branch collapsed matter (a Hernquist spheroid of mass
f M_bar, f = (N-1) eps_inf(N), with half-mass radius r_half fixed by
the solar-radius circular speed). The rest of the well's mass,
M_200 - M_bar - M_cap, is still other-branch diffuse gas at or above
the virial temperature (step 1). This script combines the two:

  total potential = baryons (B2) + captured Hernquist + hot isothermal
  gas in hydrostatic equilibrium in that total potential, with the gas
  normalised so that the total mass inside R_200 equals M_200.

Two passes:
  Pass 1  Captured extent held at the step 2 value. Reports whether the
          gas, now in a deeper central potential, piles inward and lifts
          v_c(8.2) above the 230 km/s datum, and by how much.
  Pass 2  Captured extent re-solved with the gas present so that the
          total v_c(8.2) = 230 again (nested root find). This is the
          self-consistent model; its outer curve to R_200 is the output.

Reported per (N, M_200, T/T_vir): v_c at 8.2, 30, 50, 100, 200 kpc
and at R_200; beta(15-30); gas fraction inside 30 kpc; enclosed mass
at 50 and 100 kpc. Data to be tested, printed beside the models and
not used: Orphan-Chenab stream M(<50 kpc) = 3.77e11 (Koposov+2023),
Sagittarius stream M(<100 kpc) = 5.6e11 (Vasiliev+2021), and the
weak-lensing finding of flat circular velocities to several hundred
kpc for isolated galaxies (Mistele+2024).

Branch-mass option (added 2026-10-10 after peer review): `--m 2` or
`--m 3` runs the same combination for the chain at m = 2 or 3 (other
branches holding m times our collapsed mass per well), reading the
captured mass f = (N-1) eps_inf(N) m and the step 2 extents for that m,
and writing to combined_halo_m{m}.npz so the m = 1 file is untouched.

Inputs:  ../data/hydrostatic_halo.npz (step 1), ../data/dissipated_component.npz (step 2)
Output:  ../data/combined_halo.npz (m = 1) or ../data/combined_halo_m{m}.npz
Runtime: a few minutes, single-threaded (hundreds of shooting solves at
0.2 dex scan resolution); hand off if it grows beyond that.
"""

import os
import sys
import time
import argparse
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

T0 = time.time()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA = os.path.join(ROOT, 'data')
_ap = argparse.ArgumentParser(description='captured component + hot remainder')
_ap.add_argument('--m', type=int, default=1, choices=[1, 2, 3],
                 help='collapsed mass per other branch, in units of ours (default 1)')
M_BRANCH = _ap.parse_args().m
OUTPATH = os.path.join(DATA, 'combined_halo.npz' if M_BRANCH == 1 else f'combined_halo_m{M_BRANCH}.npz')

G = 4.30091e-6
H0 = 67.4
RHO_CRIT = 3 * (H0 / 1000.0)**2 / (8 * np.pi * G)

# ── Baryons (B2), identical to steps 1-2 ────────────────────────────
BULGE_M, BULGE_A = 1.55e10, 0.7
DISCS = [(3.65e10, 2.35), (8.2e9, 18.24), (1.3e9, 2.57), (7.0e7, 5.00), (2.2e5, 3.30)]
M_BAR = BULGE_M + sum(m for m, _ in DISCS)

def m_bar(r):
    """Spherical-equivalent baryonic mass inside r [kpc]: Hernquist bulge
    mass plus, for each exponential disc, r v_c^2 / G with the razor-thin
    Freeman (1970) circular speed v_c^2 = 4 pi G Sigma_0 R_d y^2
    [I0(y)K0(y) - I1(y)K1(y)], y = r / 2R_d (concession 4 fixed
    2026-10-10; previously the discs were spherically averaged)."""
    from scipy.special import i0, i1, k0, k1
    r = np.asarray(r, dtype=float)
    m = BULGE_M * r**2 / (r + BULGE_A)**2
    for md, rd in DISCS:
        sigma0 = md / (2.0 * np.pi * rd**2)
        y = np.maximum(r, 1e-6) / (2.0 * rd)
        vc2 = 4.0 * np.pi * G * sigma0 * rd * y**2 * (i0(y) * k0(y) - i1(y) * k1(y))
        m = m + r * vc2 / G
    return m

def m_hern(r, mass, r_half):
    a = r_half / (1.0 + np.sqrt(2.0))
    return mass * r**2 / (r + a)**2

def r200_of(m200):
    return (3.0 * m200 / (800.0 * np.pi * RHO_CRIT))**(1.0 / 3.0)

def outer_slope(r, vc, r_lo=15.0, r_hi=30.0):
    sel = (r >= r_lo) & (r <= r_hi)
    return np.polyfit(np.log(r[sel]), np.log(vc[sel]), 1)[0]

# ── Hydrostatic gas in an external potential (as step 1, plus captured) ──
def integrate_gas(sigma2, ln_rho0, r_min, r_max, m_ext, n_out=500):
    m_gas0 = 4.0 / 3.0 * np.pi * r_min**3 * np.exp(ln_rho0)
    def rhs(s, y):
        r = np.exp(s)
        return [-G * (y[1] + m_ext(r)) / (sigma2 * r), 4.0 * np.pi * r**3 * np.exp(y[0])]
    s_grid = np.linspace(np.log(r_min), np.log(r_max), n_out)
    sol = solve_ivp(rhs, (s_grid[0], s_grid[-1]), [ln_rho0, m_gas0],
                    t_eval=s_grid, method='LSODA', rtol=1e-8, atol=1e-10)
    if not sol.success:
        raise RuntimeError(sol.message)
    return np.exp(sol.t), np.exp(sol.y[0]), sol.y[1]

def solve_gas(m200, t_ratio, m_ext, r_min=0.01, r_max_factor=5.0):
    """Isothermal gas at sigma^2 = t_ratio * sigma_vir^2 in the external
    mass m_ext(r) plus its own gravity, normalised so that
    M_gas(<R_200) = M_200 - m_ext(R_200). Returns (r, rho, m_gas) or None."""
    r200 = r200_of(m200)
    sigma2 = t_ratio * G * m200 / (2.0 * r200)
    target = m200 - m_ext(r200)
    if target <= 0:
        return None
    r_max = r_max_factor * r200
    def f(log10_rho0):
        r, rho, mg = integrate_gas(sigma2, np.log(10.0**log10_rho0), r_min, r_max, m_ext)
        return np.interp(r200, r, mg) - target
    grid = np.linspace(-10.0, 14.0, 121)    # 0.2 dex: the feasible window in rho0 is narrow
    f_prev, x_prev = f(grid[0]), grid[0]
    if f_prev >= 0:
        return None
    bracket = None
    for x in grid[1:]:
        fx = f(x)
        if fx > 0:
            bracket = (x_prev, x); break
        f_prev, x_prev = fx, x
    if bracket is None:
        return None
    lr = brentq(f, *bracket, xtol=1e-8, rtol=1e-10)
    r, rho, mg = integrate_gas(sigma2, np.log(10.0**lr), r_min, r_max, m_ext)
    return r, rho, mg

# ── Inputs and gates ────────────────────────────────────────────────
h1 = np.load(os.path.join(DATA, 'hydrostatic_halo.npz'))
h2 = np.load(os.path.join(DATA, 'dissipated_component.npz'))
assert bool(h1['level0_pass']) and bool(h2['gate_eps']) and bool(h2['gate_step1']), "upstream gates"
N_ALL = h2['N_GRID'].astype(int)
if M_BRANCH == 1:
    F_AVAIL = h2['f_avail_m1']
    RH_PRED = h2['rh_pred']
else:
    F_AVAIL = h2['f_avail_m1'] * M_BRANCH
    RH_PRED = h2['rh_pred_m23'][M_BRANCH - 2]     # row 0 is m = 2, row 1 is m = 3
VC_DATUM = float(h2['vc_solar_datum'][0])
sel = np.isfinite(RH_PRED)
N_GRID = N_ALL[sel]; F_GRID = F_AVAIL[sel]; RH_GRID = RH_PRED[sel]
M200_GRID = np.array([5e11, 1e12, 1.5e12, 2e12, 3e12])
T_GRID = np.array([1.0, 1.2, 1.5, 2.0, 2.5, 3.0])
R_EVAL = np.array([8.178, 30.0, 50.0, 100.0, 200.0])
R0 = R_EVAL[0]
DATA_TO_TEST = {"M(<50 kpc) Orphan-Chenab": 3.77e11, "M(<100 kpc) Sagittarius": 5.6e11}

print(f"Paper X step 2b — captured component + hot remainder, m = {M_BRANCH}")
print(f"  N, f, r_half from step 2: " + ", ".join(f"N={n}: f={f:.2f}, r_half={rh:.1f} kpc" for n, f, rh in zip(N_GRID, F_GRID, RH_GRID)))

# ── Pass 1: captured extent fixed; does the gas pile in? ────────────
shape = (len(N_GRID), len(M200_GRID), len(T_GRID))
vc8_p1 = np.full(shape, np.nan); fgas30_p1 = np.full(shape, np.nan)
print("\nPass 1 — captured extent fixed at the step 2 value; total v_c(8.2) with the gas added, and gas share inside 30 kpc")
print(f"  {'N':>3s} {'M_200':>8s} | " + " ".join(f"T={t:<4.1f}" for t in T_GRID) + "   (v_c(8.2) km/s | M_gas(<30)/M_tot(<30))")
for i, (N, f, rh) in enumerate(zip(N_GRID, F_GRID, RH_GRID)):
    for j, m200 in enumerate(M200_GRID):
        m_ext = lambda r, f=f, rh=rh: m_bar(r) + m_hern(r, f * M_BAR, rh)
        cells = []
        for k, t in enumerate(T_GRID):
            g = solve_gas(m200, t, m_ext)
            if g is None:
                cells.append("   ---   "); continue
            r, rho, mg = g
            mtot = mg + m_ext(r)
            vc8 = np.sqrt(G * np.interp(R0, r, mtot) / R0)
            vc8_p1[i, j, k] = vc8
            fgas30_p1[i, j, k] = np.interp(30.0, r, mg) / np.interp(30.0, r, mtot)
            cells.append(f"{vc8:4.0f}|{fgas30_p1[i,j,k]:.2f}")
        print(f"  {N:3d} {m200:8.1e} | " + " ".join(cells))
print(f"  datum: v_c(8.2) = {VC_DATUM:.0f} km/s.  --- : no hydrostatic solution at that T.")

# ── Pass 2: re-solve the captured extent with the gas present ───────
rh_p2 = np.full(shape, np.nan); beta_p2 = np.full(shape, np.nan)
vc_p2 = np.full(shape + (len(R_EVAL),), np.nan); vc200_p2 = np.full(shape, np.nan)
m50_p2 = np.full(shape, np.nan); m100_p2 = np.full(shape, np.nan)
fgas30_p2 = np.full(shape, np.nan)
R_PROF = np.logspace(-1, np.log10(400.0), 400)
prof_p2 = np.full(shape + (len(R_PROF),), np.nan)

def total_profile(f, rh, m200, t):
    m_ext = lambda r: m_bar(r) + m_hern(r, f * M_BAR, rh)
    g = solve_gas(m200, t, m_ext)
    if g is None:
        return None
    r, rho, mg = g
    return r, mg, mg + m_ext(r)

print("\nPass 2 — captured extent re-solved so that total v_c(8.2) = datum with the gas present")
print(f"  {'N':>3s} {'M_200':>8s} {'T/Tv':>4s} | {'r_half':>6s} {'(was)':>6s} | {'beta':>6s} | v_c at 8.2    30    50   100   200  R_200 | gas(<30) | M(<50) M(<100) [1e11]")
for i, (N, f, rh0) in enumerate(zip(N_GRID, F_GRID, RH_GRID)):
    for j, m200 in enumerate(M200_GRID):
        r200 = r200_of(m200)
        for k, t in enumerate(T_GRID):
            if not np.isfinite(vc8_p1[i, j, k]):
                continue
            def resid(rh):
                out = total_profile(f, rh, m200, t)
                if out is None:
                    return np.nan
                r, mg, mt = out
                return np.sqrt(G * np.interp(R0, r, mt) / R0) - VC_DATUM
            # the captured component is more compact for smaller r_half; the
            # datum is reached by shrinking or growing r_half from the step 2 value
            # Pass 1 showed the gas lifts v_c(8.2) only slightly at the step 2
            # extent, so the root lies just above it; bracket from there.
            lo, hi = rh0, 400.0
            f_lo, f_hi = resid(lo), resid(hi)
            if np.isfinite(f_lo) and f_lo < 0:       # gas lowered it: look inward
                lo, hi = max(2.0, 0.5 * rh0), rh0
                f_lo, f_hi = resid(lo), resid(hi)
            if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo * f_hi > 0:
                print(f"  {N:3d} {m200:8.1e} {t:4.1f} | no root for r_half in [{lo}, {hi}] kpc (resid {f_lo:+.0f}, {f_hi:+.0f})")
                continue
            rh = brentq(resid, lo, hi, xtol=1e-3, rtol=1e-6)
            r, mg, mt = total_profile(f, rh, m200, t)
            vc = np.sqrt(G * mt / r)
            rh_p2[i, j, k] = rh
            beta_p2[i, j, k] = outer_slope(r, vc)
            vc_p2[i, j, k] = np.interp(R_EVAL, r, vc)
            vc200_p2[i, j, k] = np.interp(r200, r, vc)
            m50_p2[i, j, k] = np.interp(50.0, r, mt)
            m100_p2[i, j, k] = np.interp(100.0, r, mt)
            fgas30_p2[i, j, k] = np.interp(30.0, r, mg) / np.interp(30.0, r, mt)
            prof_p2[i, j, k] = np.interp(R_PROF, r, vc)
            print(f"  {N:3d} {m200:8.1e} {t:4.1f} | {rh:6.1f} {rh0:6.1f} | {beta_p2[i,j,k]:+.3f} | "
                  + " ".join(f"{v:5.0f}" for v in vc_p2[i, j, k]) + f"  {vc200_p2[i,j,k]:5.0f} |   {fgas30_p2[i,j,k]:.2f}   |"
                  f" {m50_p2[i,j,k]/1e11:5.2f}  {m100_p2[i,j,k]/1e11:5.2f}")
print("  Data to be tested (not used): " + "; ".join(f"{k} = {v/1e11:.2f}e11" for k, v in DATA_TO_TEST.items()))
print("  Weak lensing (Mistele+2024): circular velocities of isolated galaxies stay flat to several hundred kpc.")

print("\nSummary per N (pass 2, over feasible M_200 and T):")
print(f"  {'N':>3s} | beta(15-30) range | v_c(100)/v_c(8.2) range | v_c(200)/v_c(8.2) range | M(<50) range [1e11] | M(<100) range [1e11]")
for i, N in enumerate(N_GRID):
    b = beta_p2[i]; ok = np.isfinite(b)
    if not ok.any():
        print(f"  {N:3d} | no feasible solution"); continue
    r100 = (vc_p2[i, :, :, 3] / vc_p2[i, :, :, 0])[ok]
    r200 = (vc_p2[i, :, :, 4] / vc_p2[i, :, :, 0])[ok]
    print(f"  {N:3d} | [{b[ok].min():+.2f}, {b[ok].max():+.2f}]    | [{r100.min():.2f}, {r100.max():.2f}]            "
          f"| [{r200.min():.2f}, {r200.max():.2f}]            | [{m50_p2[i][ok].min()/1e11:.2f}, {m50_p2[i][ok].max()/1e11:.2f}]      "
          f"| [{m100_p2[i][ok].min()/1e11:.2f}, {m100_p2[i][ok].max()/1e11:.2f}]")

np.savez(OUTPATH, m_branch=M_BRANCH, N_GRID=N_GRID, F_GRID=F_GRID, RH_GRID=RH_GRID, M200_GRID=M200_GRID,
         T_GRID=T_GRID, R_EVAL=R_EVAL, R_PROF=R_PROF, vc8_p1=vc8_p1, fgas30_p1=fgas30_p1,
         rh_p2=rh_p2, beta_p2=beta_p2, vc_p2=vc_p2, vc200_p2=vc200_p2, m50_p2=m50_p2,
         m100_p2=m100_p2, fgas30_p2=fgas30_p2, prof_p2=prof_p2, vc_datum=VC_DATUM,
         data_to_test_names=np.array(list(DATA_TO_TEST.keys())),
         data_to_test_values=np.array(list(DATA_TO_TEST.values())))
print(f"\nSaved {OUTPATH}   ({time.time() - T0:.0f} s)")
