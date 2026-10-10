#!/usr/bin/env python3
"""
Paper X — Step 1: the extended component as a hydrostatic gas halo

Under ISF (Paper IX ontology), the dark mass around a galaxy is, by mass,
mostly the diffuse gas of the other branches held in the shared
gravitational well. That gas is pressure-supported at its own temperature
and cannot cool through our electromagnetism. Its radial profile is
therefore a hydrostatic-equilibrium problem, not a collisionless one.
This script builds that profile in levels of escalating complexity and
reports the outer logarithmic slope of the rotation curve,
beta = d ln v_c / d ln R over 15-30 kpc, as the paper's main prediction.

  Level 0  Isothermal gas in its own gravity, no baryons: the singular
           isothermal sphere (SIS), rho = sigma^2/(2 pi G r^2),
           v_c = sqrt(2) sigma, a flat rotation curve. GATE: the ODE
           solver started on the analytic SIS must stay on it to
           ~1e-8; the nonsingular (cored) isothermal sphere must
           approach it at large radius.
  Level 1  Self-consistent: isothermal gas at the virial temperature of
           the well, in the potential of a fixed Milky Way baryonic model
           (Jiao et al. 2026 / de Salas et al. 2019 "B2" model,
           spherically averaged) plus the gas's own gravity. The gas mass
           inside R_200 is M_200 - M_bar: every branch's share of the
           well, all in the hot phase at this level (the dissipated
           part of the other branches' share is moved out by step 2).
  Level 2  Gas temperature not pinned: T_gas/T_vir over [0.3, 1.5].
           Output: the beta(15-30 kpc) band versus M_200 and T ratio.

ISF consistency bound reported alongside: each branch's share of the
well is M_200/N, which must exceed our own galaxy's baryonic mass
M_bar on average, so M_200 >= N * M_bar. Reported as N_max(M_200).

Reference curves for the figure: NFW halo with c = 12 and the same
baryons, at each M_200 (the standard comparison in the review).

No fitting to any Milky Way rotation-curve measurement is performed.
The Jiao et al. 2026 Keplerian slope and the Cepheid mild decline are
recorded in the plan as data to be tested, not used here.

Output: ../data/hydrostatic_halo.npz
Runtime: seconds, single-threaded.
"""

import os
import sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA = os.path.join(ROOT, 'data')
os.makedirs(DATA, exist_ok=True)
OUTPATH = os.path.join(DATA, 'hydrostatic_halo.npz')

# ── Constants and conventions ───────────────────────────────────────
G = 4.30091e-6            # kpc (km/s)^2 / Msun
H0 = 67.4                 # km/s/Mpc, Planck 2018 (as Papers V-IX)
RHO_CRIT = 3 * (H0 / 1000.0)**2 / (8 * np.pi * G)   # Msun / kpc^3
# (= 2.775e11 h^2 Msun/Mpc^3 = 126 Msun/kpc^3 at h = 0.674)

# Milky Way baryonic model B2 (de Salas et al. 2019, as adopted by
# Jiao et al. 2026 Table 2): Hernquist bulge + exponential discs.
# Used spherically averaged: M_bar(<r) = bulge(<r) + sum of disc(<R=r).
BULGE_M, BULGE_A = 1.55e10, 0.7            # Msun, kpc
DISCS = [  # (mass Msun, radial scale length kpc)
    (3.65e10, 2.35),    # stellar disc
    (8.2e9, 18.24),     # HI gas
    (1.3e9, 2.57),      # H2 gas
    (7.0e7, 5.00),      # cold dust
    (2.2e5, 3.30),      # warm dust
]
M_BAR = BULGE_M + sum(m for m, _ in DISCS)   # 6.16e10 Msun

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

def r200_of(m200):
    """Virial radius [kpc] for mean enclosed density 200 rho_crit."""
    return (3.0 * m200 / (800.0 * np.pi * RHO_CRIT))**(1.0 / 3.0)

# ── Hydrostatic isothermal gas in a well ────────────────────────────
# State y = [ln rho_gas, M_gas(<r)] integrated in s = ln r.
#   d ln rho / d ln r = - G [M_bar(r) + M_gas(r)] / (sigma^2 r)
#   d M_gas  / d ln r = 4 pi r^3 rho
# sigma^2 = k T / (mu m_p) in (km/s)^2.

def integrate_gas(sigma2, ln_rho0, r_min, r_max, with_baryons=True,
                  n_out=600, m_gas0=None):
    if m_gas0 is None:   # uniform-density core inside r_min
        m_gas0 = 4.0 / 3.0 * np.pi * r_min**3 * np.exp(ln_rho0)
    def rhs(s, y):
        r = np.exp(s)
        rho = np.exp(y[0])
        m_in = y[1] + (m_bar(r) if with_baryons else 0.0)
        return [-G * m_in / (sigma2 * r), 4.0 * np.pi * r**3 * rho]
    s_grid = np.linspace(np.log(r_min), np.log(r_max), n_out)
    sol = solve_ivp(rhs, (s_grid[0], s_grid[-1]), [ln_rho0, m_gas0],
                    t_eval=s_grid, method='LSODA', rtol=1e-9, atol=1e-12)
    if not sol.success:
        raise RuntimeError(sol.message)
    r = np.exp(sol.t)
    return r, np.exp(sol.y[0]), sol.y[1]

def solve_halo(m200, t_ratio, r_min=0.01, r_max_factor=5.0):
    """Level 1/2: isothermal gas of mass M_200 - M_bar inside R_200,
    at sigma^2 = t_ratio * sigma_vir^2, with the baryons. Shoots on the
    central density. Returns dict of profiles, or None when no
    hydrostatic solution exists.

    Why None can happen (a physics bound, not a numerical one): a
    self-gravitating isothermal sphere encloses at most ~2 sigma^2 R / G
    inside any radius R (the singular-sphere value, which the cored
    solutions oscillate about). Hence M_gas(<R_200) <= ~t_ratio * M_200.
    Gas colder than about (M_200 - M_bar)/M_200 times T_vir cannot hold
    the required mass inside R_200 in equilibrium; it would collapse."""
    r200 = r200_of(m200)
    sigma2_vir = G * m200 / (2.0 * r200)     # kT_vir/(mu m_p)
    sigma2 = t_ratio * sigma2_vir
    m_gas_target = m200 - M_BAR
    if m_gas_target <= 0:
        raise ValueError("M_200 below the baryonic mass")
    r_max = r_max_factor * r200

    def mass_in_r200(log10_rho0):
        r, rho, mg = integrate_gas(sigma2, np.log(10.0**log10_rho0),
                                   r_min, r_max)
        return np.interp(r200, r, mg) - m_gas_target

    lo, hi = -10.0, 14.0
    f_lo = mass_in_r200(lo)
    assert f_lo < 0, "low-density end of the shooting bracket"
    # The enclosed mass is bounded above; scan upward for a sign change
    # so that a cold, infeasible case is reported rather than asserted.
    grid = np.linspace(lo, hi, 49)
    f_prev, x_prev = f_lo, lo
    bracket = None
    for x in grid[1:]:
        f = mass_in_r200(x)
        if f > 0:
            bracket = (x_prev, x)
            break
        f_prev, x_prev = f, x
    if bracket is None:
        return None
    log10_rho0 = brentq(mass_in_r200, *bracket, xtol=1e-10, rtol=1e-12)
    r, rho, mg = integrate_gas(sigma2, np.log(10.0**log10_rho0),
                               r_min, r_max)
    m_tot = mg + m_bar(r)
    vc = np.sqrt(G * m_tot / r)
    return dict(r=r, rho_gas=rho, m_gas=mg, m_tot=m_tot, vc=vc,
                r200=r200, sigma2=sigma2, sigma2_vir=sigma2_vir,
                rho0=10.0**log10_rho0)

def outer_slope(r, vc, r_lo=15.0, r_hi=30.0):
    """beta = d ln v_c / d ln R by least squares over [r_lo, r_hi]."""
    sel = (r >= r_lo) & (r <= r_hi)
    x, y = np.log(r[sel]), np.log(vc[sel])
    return np.polyfit(x, y, 1)[0]

# ── NFW reference (c = 12) with the same baryons ────────────────────
def nfw_vc(r, m200, c=12.0):
    r200 = r200_of(m200)
    rs = r200 / c
    fc = np.log(1 + c) - c / (1 + c)
    x = r / rs
    m_dm = (m200 - M_BAR) * (np.log(1 + x) - x / (1 + x)) / fc
    return np.sqrt(G * (m_dm + m_bar(r)) / r)

# ═══════════════════════════════════════════════════════════════════
print("Paper X step 1 — hydrostatic gas halo")
print(f"  rho_crit = {RHO_CRIT:.2f} Msun/kpc^3 ; M_bar = {M_BAR:.3e} Msun")

# ── Level 0 gates ───────────────────────────────────────────────────
sigma2_test = 100.0**2          # sigma = 100 km/s
r0, r1 = 1.0, 1000.0
# (i) start ON the singular solution; must stay on it
rho_sis0 = sigma2_test / (2 * np.pi * G * r0**2)
m_sis0 = 2 * sigma2_test * r0 / G
r, rho, mg = integrate_gas(sigma2_test, np.log(rho_sis0), r0, r1,
                           with_baryons=False, m_gas0=m_sis0)
rho_sis = sigma2_test / (2 * np.pi * G * r**2)
vc_sis = np.sqrt(G * mg / r)
gate_sis_rho = np.max(np.abs(rho / rho_sis - 1))
gate_sis_vc = np.max(np.abs(vc_sis / np.sqrt(2 * sigma2_test) - 1))
# (ii) nonsingular (cored) isothermal sphere approaches the SIS
r, rho, mg = integrate_gas(sigma2_test, np.log(1e8), 1e-3, 1e4,
                           with_baryons=False, n_out=2000)
vc_core = np.sqrt(G * mg / r)
outer = r > 1e3
gate_core = np.median(np.abs(vc_core[outer] / np.sqrt(2 * sigma2_test) - 1))
print("\nLevel 0 gates (SIS):")
print(f"  on-SIS start, max |rho/rho_SIS - 1|  = {gate_sis_rho:.2e}  (need < 1e-7)")
print(f"  on-SIS start, max |v_c/(sqrt2 s) - 1| = {gate_sis_vc:.2e}  (need < 1e-7)")
print(f"  cored sphere, outer-decade median |v_c/(sqrt2 s) - 1| = {gate_core:.3f}  (need < 0.05)")
LEVEL0_PASS = (gate_sis_rho < 1e-7) and (gate_sis_vc < 1e-7) and (gate_core < 0.05)
print(f"  LEVEL 0 GATE: {'PASS' if LEVEL0_PASS else 'FAIL'}")
if not LEVEL0_PASS:
    sys.exit("Level 0 gate failed; no physics output.")

# ── Level 1 and 2 grid ──────────────────────────────────────────────
M200_GRID = np.array([2e11, 3e11, 5e11, 7e11, 1e12, 1.5e12, 2e12])
T_GRID = np.round(np.arange(0.3, 2.0001, 0.1), 2)   # extended past the plan's 1.5 once the cold bound appeared
N_GRID = np.array([2, 3, 4, 6, 8, 10])
R_EVAL = np.array([8.178, 15.0, 20.0, 25.0, 30.0, 50.0])

beta = np.full((len(M200_GRID), len(T_GRID)), np.nan)
vc_at = np.full((len(M200_GRID), len(T_GRID), len(R_EVAL)), np.nan)
vc_gas_at = np.full_like(vc_at, np.nan)      # gas-only circular speed
r_half = np.full((len(M200_GRID), len(T_GRID)), np.nan)   # gas half-mass radius inside R_200
f_gas30 = np.full((len(M200_GRID), len(T_GRID)), np.nan)  # M_gas(<30 kpc) / M_200
rho0 = np.full((len(M200_GRID), len(T_GRID)), np.nan)
r200_arr = np.array([r200_of(m) for m in M200_GRID])
beta_nfw = np.zeros(len(M200_GRID))
vc_nfw_at = np.zeros((len(M200_GRID), len(R_EVAL)))
R_PROFILE = np.logspace(-1, np.log10(300.0), 300)
vc_profiles = np.full((len(M200_GRID), len(T_GRID), len(R_PROFILE)), np.nan)
rho_profiles = np.full_like(vc_profiles, np.nan)

for i, m200 in enumerate(M200_GRID):
    vnfw = nfw_vc(R_PROFILE, m200)
    beta_nfw[i] = outer_slope(R_PROFILE, vnfw)
    vc_nfw_at[i] = np.interp(R_EVAL, R_PROFILE, vnfw)
    for j, t in enumerate(T_GRID):
        h = solve_halo(m200, t)
        if h is None:          # no hydrostatic solution at this T (cold bound)
            continue
        beta[i, j] = outer_slope(h['r'], h['vc'])
        vc_at[i, j] = np.interp(R_EVAL, h['r'], h['vc'])
        vc_gas_at[i, j] = np.interp(R_EVAL, h['r'], np.sqrt(G * h['m_gas'] / h['r']))
        rho0[i, j] = h['rho0']
        mg200 = np.interp(h['r200'], h['r'], h['m_gas'])
        r_half[i, j] = np.interp(0.5 * mg200, h['m_gas'], h['r'])
        f_gas30[i, j] = np.interp(30.0, h['r'], h['m_gas']) / m200
        vc_profiles[i, j] = np.interp(R_PROFILE, h['r'], h['vc'])
        rho_profiles[i, j] = np.interp(R_PROFILE, h['r'], h['rho_gas'])

# ── Report ──────────────────────────────────────────────────────────
jT1 = int(np.argmin(np.abs(T_GRID - 1.0)))
print("\nLevel 1 (T_gas = T_vir):  total v_c [km/s] at R = 8.2, 15, 20, 25, 30, 50 kpc;  beta(15-30);")
print("  gas-only v_c at 8.2 and 30 kpc; gas half-mass radius inside R_200; M_gas(<30 kpc)/M_200;  NFW c=12 reference")
print(f"  {'M_200':>8s} {'R_200':>6s} | " + " ".join(f"{R:6.1f}" for R in R_EVAL) + " |  beta  | gas(8.2) gas(30) r_half f_gas30 | NFW: v_c(8.2) v_c(30)  beta")
for i, m200 in enumerate(M200_GRID):
    row = " ".join(("{:6.1f}".format(v) if np.isfinite(v) else "   n/a") for v in vc_at[i, jT1])
    print(f"  {m200:8.1e} {r200_arr[i]:6.0f} | {row} | {beta[i, jT1]:+.3f} |"
          f" {vc_gas_at[i,jT1,0]:8.1f} {vc_gas_at[i,jT1,4]:7.1f} {r_half[i,jT1]:6.0f} {f_gas30[i,jT1]:7.3f} |"
          f"      {vc_nfw_at[i,0]:6.1f} {vc_nfw_at[i,4]:6.1f}  {beta_nfw[i]:+.3f}")

print("\nLevel 2:  beta(15-30 kpc) versus T_gas/T_vir")
print(f"  {'M_200':>8s} | " + " ".join(f"{t:5.1f}" for t in T_GRID) + " |  band [min, max]")
t_min_feasible = np.full(len(M200_GRID), np.nan)
for i, m200 in enumerate(M200_GRID):
    row = " ".join(("{:+5.2f}".format(b) if np.isfinite(b) else "  ---") for b in beta[i])
    ok = np.isfinite(beta[i])
    if ok.any():
        t_min_feasible[i] = T_GRID[ok].min()
        band = f"[{np.nanmin(beta[i]):+.2f}, {np.nanmax(beta[i]):+.2f}]  feasible T/T_vir >= {t_min_feasible[i]:.1f}"
    else:
        band = "no hydrostatic solution on the grid"
    print(f"  {m200:8.1e} | {row} | {band}")
print("  --- : no hydrostatic solution (gas too cold to hold M_200 - M_bar inside R_200)")
print("  Analytic expectation for the cold bound: an isothermal atmosphere holds at most")
print("  ~2 sigma^2 R / G of TOTAL mass (baryons included) inside R, i.e. T/T_vir >~ 1;")
print("  cored solutions reach that only on their first overshoot, so the bound sits above 1.")

print("\nLevel 2 at the coldest feasible T (most concentrated hydrostatic halo allowed):")
print(f"  {'M_200':>8s} {'T/Tvir':>6s} | " + " ".join(f"{R:6.1f}" for R in R_EVAL) + " |  beta  | gas(8.2) gas(30) r_half f_gas30")
for i, m200 in enumerate(M200_GRID):
    if not np.isfinite(t_min_feasible[i]):
        print(f"  {m200:8.1e}    --- | no hydrostatic solution")
        continue
    j = int(np.argmin(np.abs(T_GRID - t_min_feasible[i])))
    row = " ".join(f"{v:6.1f}" for v in vc_at[i, j])
    print(f"  {m200:8.1e} {T_GRID[j]:6.1f} | {row} | {beta[i, j]:+.3f} |"
          f" {vc_gas_at[i,j,0]:8.1f} {vc_gas_at[i,j,4]:7.1f} {r_half[i,j]:6.0f} {f_gas30[i,j]:7.3f}")

print("\nISF consistency bound  M_200 >= N * M_bar  ->  N_max(M_200):")
n_max = np.floor(M200_GRID / M_BAR).astype(int)
for m200, nm in zip(M200_GRID, n_max):
    ok = ", ".join(str(n) for n in N_GRID if n <= nm) or "none"
    print(f"  M_200 = {m200:8.1e}:  N_max = {nm:2d}  -> allowed N in grid: {ok}")

# Decision gate A: the plan stops the paper if the beta band spans the
# whole interval [-0.5, 0]. Report the band over the feasible grid and
# the narrower band at the virial temperature; judgement is in HANDOFF.
band_lo, band_hi = np.nanmin(beta), np.nanmax(beta)
print(f"\nDecision gate A: beta range over all feasible (M_200, T) = [{band_lo:+.2f}, {band_hi:+.2f}]")
print(f"                 beta range at T = T_vir only          = "
      f"[{np.nanmin(beta[:, jT1]):+.2f}, {np.nanmax(beta[:, jT1]):+.2f}]")
print(f"                 NFW c=12 reference over M_200          = "
      f"[{beta_nfw.min():+.2f}, {beta_nfw.max():+.2f}]")

np.savez(OUTPATH,
         M200_GRID=M200_GRID, T_GRID=T_GRID, N_GRID=N_GRID, R_EVAL=R_EVAL,
         R_PROFILE=R_PROFILE, beta=beta, vc_at=vc_at, rho0=rho0,
         r200=r200_arr, beta_nfw=beta_nfw, vc_nfw_at=vc_nfw_at,
         vc_profiles=vc_profiles, rho_profiles=rho_profiles,
         vc_gas_at=vc_gas_at, r_half=r_half, f_gas30=f_gas30,
         M_BAR=M_BAR, RHO_CRIT=RHO_CRIT, n_max=n_max,
         gate_sis_rho=gate_sis_rho, gate_sis_vc=gate_sis_vc,
         gate_core=gate_core, level0_pass=LEVEL0_PASS,
         t_min_feasible=t_min_feasible)
print(f"\nSaved {OUTPATH}")
