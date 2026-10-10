#!/usr/bin/env python3
"""
Paper X — Step 2: the dissipated component promoted to the dark halo

Step 1 showed that a single-temperature hydrostatic atmosphere of
other-branch gas cannot be the dark halo: it cannot hold the halo mass
inside R_200 below the virial temperature, and above it the gas is so
diffuse (half-mass radius 138-177 kpc, ~1 percent of M_200 inside 30
kpc) that v_c at the solar radius comes out ~160 km/s against ~230
measured. Chris's decision (2026-10-10): promote the DISSIPATED
component — captured other-branch collapsed matter (stars, remnants,
cooled gas), collisionless on our branch — to the dominant dark mass
inside the disc region, and build it here.

Mass budget (Paper VI/VII convention). Each other branch holds, on
average, a collapsed mass M_coll = m * M_bar in a well like ours (m = 1
by the Paper IX symmetry; m in {0.5, 1, 2} brackets branch-to-branch
scatter). The fraction co-located with our galaxy is eps_inf(N), the
Paper VII value fixed per N by the S8 constraint (loaded from the Paper
VII data product and gated against the published grid). So
    M_cap(N, m) = (N - 1) * eps_inf(N) * m * M_bar.
The non-co-located remainder is Paper VI's dark-perturber population.

Geometry, two limiting cases, neither chosen (plan Section 4b):
  A  INTACT: (N-1) eps m co-centred copies of our own baryonic model.
     v_c^2 = [1 + (N-1) eps m] v_bar^2. The coldest, most compact case.
  B  HEATED SPHEROID: the captured mass violently relaxed into a
     Hernquist sphere of half-mass radius r_h in {4, ..., 128} kpc
     (r_h = (1+sqrt2) a). The disc's own half-mass radius is ~4 kpc.

Hot other-branch gas (the step 1 component) holds the remainder
M_200 - M_bar - M_cap at >= T_vir and is omitted inside 50 kpc; step 1
bounds its gas-only v_c at <= 57 km/s at 30 kpc (<= 3.5 percent in
quadrature at 220 km/s), read from data/hydrostatic_halo.npz.

Outputs per (N, m, geometry): v_c at 8.2, 15, 20, 25, 30, 50 kpc,
beta = d ln v_c / d ln R over 15-30 kpc, dark fraction inside 30 kpc,
and the implied dark-disc mass for the off-plane test. The measured
v_c(8.2 kpc) ~ 230 km/s (review: 229-237) is printed beside the models
as the datum to be tested; nothing is fitted to it.

Inputs:  ../data/joint_constraint_paper7.npz (copied from the public
         mirror's data/paper7/, as Paper IX copied pknl_camb.npz)
         ../data/hydrostatic_halo.npz (step 1)
Output:  ../data/dissipated_component.npz
Runtime: well under a second.
"""

import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA = os.path.join(ROOT, 'data')
OUTPATH = os.path.join(DATA, 'dissipated_component.npz')

G = 4.30091e-6            # kpc (km/s)^2 / Msun

# ── Baryonic model, identical to step 1 (B2: de Salas et al. 2019) ──
BULGE_M, BULGE_A = 1.55e10, 0.7
DISCS = [(3.65e10, 2.35), (8.2e9, 18.24), (1.3e9, 2.57),
         (7.0e7, 5.00), (2.2e5, 3.30)]
M_BAR = BULGE_M + sum(m for m, _ in DISCS)
M_STELLAR_DISC = DISCS[0][0]

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

def m_hernquist(r, mass, r_half):
    a = r_half / (1.0 + np.sqrt(2.0))
    return mass * r**2 / (r + a)**2

def outer_slope(r, vc, r_lo=15.0, r_hi=30.0):
    sel = (r >= r_lo) & (r <= r_hi)
    return np.polyfit(np.log(r[sel]), np.log(vc[sel]), 1)[0]

# ── Gates on inputs ─────────────────────────────────────────────────
jc = np.load(os.path.join(DATA, 'joint_constraint_paper7.npz'), allow_pickle=True)
N_GRID = jc['N_grid'].astype(int)
EPS_INF = jc['eps_inf_grid'].astype(float)
EXPECTED = {2: 0.00, 3: 0.25, 4: 0.40, 6: 0.58, 8: 0.68, 10: 0.74}
gate_eps = all(abs(EPS_INF[i] - EXPECTED[int(N_GRID[i])]) < 1e-9 for i in range(len(N_GRID)))
h1 = np.load(os.path.join(DATA, 'hydrostatic_halo.npz'))
gate_step1 = bool(h1['level0_pass'])
gas_vc30_max = np.nanmax(h1['vc_gas_at'][:, :, 4])     # column 4 is R = 30 kpc
print("Paper X step 2 — dissipated component")
print(f"  eps_inf(N) from Paper VII product: " + ", ".join(f"N={n}: {e:.2f}" for n, e in zip(N_GRID, EPS_INF)))
print(f"  GATE eps_inf grid matches Papers VI-VIII: {'PASS' if gate_eps else 'FAIL'}")
print(f"  GATE step 1 Level 0 passed in its saved output: {'PASS' if gate_step1 else 'FAIL'}")
print(f"  step 1 bound on omitted hot gas: v_c,gas(30 kpc) <= {gas_vc30_max:.1f} km/s "
      f"(<= {100*(np.sqrt(220**2 + gas_vc30_max**2)/220 - 1):.1f} percent on 220 km/s)")
if not (gate_eps and gate_step1):
    sys.exit("input gate failed; no physics output")

# ── Grids ───────────────────────────────────────────────────────────
M_SCALE = np.array([0.5, 1.0, 2.0])          # M_coll,j / M_bar
R_HALF = np.array([4.0, 8.0, 16.0, 32.0, 64.0, 128.0])   # kpc, heated-spheroid half-mass radius;
                                                          # NFW c=12 at 1e12 has a Hernquist-equivalent r_half ~ 70 kpc
R_EVAL = np.array([8.178, 15.0, 20.0, 25.0, 30.0, 50.0])
R_PROFILE = np.logspace(-1, np.log10(100.0), 400)
VC_SOLAR_DATUM = (230.0, 229.0, 237.0)       # measured, review Section 1.5; a datum, not an input

v_bar = np.sqrt(G * m_bar(R_PROFILE) / R_PROFILE)
v_bar_at = np.interp(R_EVAL, R_PROFILE, v_bar)
beta_bar = outer_slope(R_PROFILE, v_bar)

nN, nM, nR = len(N_GRID), len(M_SCALE), len(R_HALF)
M_cap = np.zeros((nN, nM))
vc_A = np.zeros((nN, nM, len(R_EVAL)));   beta_A = np.zeros((nN, nM))
vc_B = np.zeros((nN, nM, nR, len(R_EVAL))); beta_B = np.zeros((nN, nM, nR))
fdm30_A = np.zeros((nN, nM)); fdm30_B = np.zeros((nN, nM, nR))
prof_A = np.zeros((nN, nM, len(R_PROFILE))); prof_B = np.zeros((nN, nM, nR, len(R_PROFILE)))

for i, N in enumerate(N_GRID):
    for j, m in enumerate(M_SCALE):
        f = (N - 1) * EPS_INF[i] * m               # M_cap / M_bar
        M_cap[i, j] = f * M_BAR
        # Geometry A: intact co-centred copies
        vA = np.sqrt(1.0 + f) * v_bar
        prof_A[i, j] = vA
        vc_A[i, j] = np.interp(R_EVAL, R_PROFILE, vA)
        beta_A[i, j] = outer_slope(R_PROFILE, vA)
        fdm30_A[i, j] = f / (1.0 + f)
        # Geometry B: heated Hernquist spheroid
        for k, rh in enumerate(R_HALF):
            m_tot = m_bar(R_PROFILE) + m_hernquist(R_PROFILE, M_cap[i, j], rh)
            vB = np.sqrt(G * m_tot / R_PROFILE)
            prof_B[i, j, k] = vB
            vc_B[i, j, k] = np.interp(R_EVAL, R_PROFILE, vB)
            beta_B[i, j, k] = outer_slope(R_PROFILE, vB)
            md30 = m_hernquist(30.0, M_cap[i, j], rh)
            fdm30_B[i, j, k] = md30 / (md30 + m_bar(30.0))

# ── Report ──────────────────────────────────────────────────────────
print(f"\nBaryons alone (B2, razor-thin discs): v_c(8.2) = {v_bar_at[0]:.1f} km/s, "
      f"v_c(30) = {v_bar_at[4]:.1f}, beta(15-30) = {beta_bar:+.3f}")
print(f"Datum to be tested (not fitted): v_c(8.2 kpc) = {VC_SOLAR_DATUM[0]:.0f} "
      f"[{VC_SOLAR_DATUM[1]:.0f}-{VC_SOLAR_DATUM[2]:.0f}] km/s")

print("\nMass budget  M_cap = (N-1) eps_inf(N) m M_bar   [1e10 Msun]   (M_bar = %.2f)" % (M_BAR/1e10))
print(f"  {'N':>3s} {'eps':>5s} | " + " ".join(f"m={m:<4.1f}" for m in M_SCALE) + " | f_DM(<30) at m=1, geometry A | (N-1)/N")
for i, N in enumerate(N_GRID):
    row = " ".join(f"{M_cap[i,j]/1e10:6.1f}" for j in range(nM))
    print(f"  {N:3d} {EPS_INF[i]:5.2f} | {row} |   {fdm30_A[i,1]:.2f}                      | {(N-1)/N:.2f}")

def table(title, vc, beta, fdm):
    print(f"\n{title}")
    print(f"  {'N':>3s} {'m':>4s} | " + " ".join(f"{R:6.1f}" for R in R_EVAL) + " |  beta  | f_DM(<30) | v_c(8.2)/230")
    for i, N in enumerate(N_GRID):
        for j, m in enumerate(M_SCALE):
            row = " ".join(f"{v:6.1f}" for v in vc[i, j])
            print(f"  {N:3d} {m:4.1f} | {row} | {beta[i,j]:+.3f} |   {fdm[i,j]:.2f}    |   {vc[i,j,0]/VC_SOLAR_DATUM[0]:.2f}")

table("Geometry A — intact co-centred copies:  v_c [km/s] at R = 8.2 ... 50 kpc", vc_A, beta_A, fdm30_A)
for k, rh in enumerate(R_HALF):
    table(f"Geometry B — heated Hernquist spheroid, r_half = {rh:.0f} kpc", vc_B[:, :, k], beta_B[:, :, k], fdm30_B[:, :, k])

print("\nPrediction table: beta(15-30 kpc) band per N over m in {0.5,1,2} and both geometries;")
print("  and whether any model reaches v_c(8.2) within 15 percent of the datum")
print(f"  {'N':>3s} | beta band A      | beta band B (all r_half) | v_c(8.2) range (all) | within 15% of datum?")
within = np.zeros(nN, dtype=bool)
beta_lo = np.zeros(nN); beta_hi = np.zeros(nN)
for i, N in enumerate(N_GRID):
    all_vc8 = np.concatenate([vc_A[i, :, 0], vc_B[i, :, :, 0].ravel()])
    all_beta = np.concatenate([beta_A[i], beta_B[i].ravel()])
    beta_lo[i], beta_hi[i] = all_beta.min(), all_beta.max()
    within[i] = np.any(np.abs(all_vc8 / VC_SOLAR_DATUM[0] - 1) < 0.15)
    print(f"  {N:3d} | [{beta_A[i].min():+.2f}, {beta_A[i].max():+.2f}]   | [{beta_B[i].min():+.2f}, {beta_B[i].max():+.2f}]           "
          f"| [{all_vc8.min():5.0f}, {all_vc8.max():5.0f}]        | {'yes' if within[i] else 'NO'}")


# ── Required captured mass to match the solar-radius datum ─────────
# Closed form: v_datum^2 = v_bar^2 + G f M_bar h(R)/R, with h the fraction
# of the captured mass inside R. Geometry A: h = M_bar(<R)/M_bar.
# Hernquist: h = (R/(R+a))^2. f_req = (N-1) eps m needed; then the outer
# slope that follows is the prediction conditional on the inner curve.
R0 = R_EVAL[0]
need = (VC_SOLAR_DATUM[0]**2 - v_bar_at[0]**2) * R0 / (G * M_BAR)
f_req_A = need / (m_bar(R0) / M_BAR)
beta_req_A = beta_bar
f_req_B = np.zeros(nR); beta_req_B = np.zeros(nR); vc30_req_B = np.zeros(nR)
for k, rh in enumerate(R_HALF):
    a = rh / (1.0 + np.sqrt(2.0))
    f_req_B[k] = need / (R0 / (R0 + a))**2
    m_tot = m_bar(R_PROFILE) + m_hernquist(R_PROFILE, f_req_B[k] * M_BAR, rh)
    v = np.sqrt(G * m_tot / R_PROFILE)
    beta_req_B[k] = outer_slope(R_PROFILE, v)
    vc30_req_B[k] = np.interp(30.0, R_PROFILE, v)
f_avail = (N_GRID - 1) * EPS_INF          # at m = 1
print("\nCaptured mass REQUIRED to reach v_c(8.2) = 230 km/s, as f = (N-1) eps m = M_cap/M_bar,")
print("  and the outer slope that then follows (the prediction conditional on the inner curve):")
print(f"  geometry A (intact copies):           f_req = {f_req_A:5.2f}   beta(15-30) = {beta_req_A:+.3f}")
for k, rh in enumerate(R_HALF):
    print(f"  geometry B, r_half = {rh:5.0f} kpc:        f_req = {f_req_B[k]:5.2f}   beta(15-30) = {beta_req_B[k]:+.3f}   v_c(30) = {vc30_req_B[k]:5.0f}")
print("  available at m = 1 from the series' eps_inf(N): " + ", ".join(f"N={n}: {f:.2f}" for n, f in zip(N_GRID, f_avail)))
print("  (N needed at m = 1 for an NFW-like extent, r_half ~ 64-128 kpc: "
      + ", ".join(f"{f_req_B[k]/EPS_INF[-1] + 1:.0f}" for k in range(nR) if R_HALF[k] >= 64)
      + " using eps_inf = 0.74, the largest on the grid; or m = "
      + ", ".join(f"{f_req_B[k]/f_avail[-1]:.1f}" for k in range(nR) if R_HALF[k] >= 64) + " at N = 10)")


# ── The one-parameter prediction: N -> f (m = 1) -> r_half -> beta ─
# Given N, the series' eps_inf(N) fixes f = M_cap/M_bar. The inner datum
# then fixes the Hernquist extent r_half (solve f_req(r_half) = f), and
# the outer slope follows with nothing left free.
from scipy.optimize import brentq
R_FAR = np.array([50.0, 100.0, 200.0])
R_PROF_FAR = np.logspace(-1, np.log10(300.0), 500)
def f_req_of_rhalf(rh):
    a = rh / (1.0 + np.sqrt(2.0))
    return need / (R0 / (R0 + a))**2
rh_pred = np.full(nN, np.nan); beta_pred = np.full(nN, np.nan)
vc_far_pred = np.full((nN, len(R_FAR)), np.nan); vc30_pred = np.full(nN, np.nan)
print("\nONE-PARAMETER PREDICTION at m = 1:  N -> f -> r_half (from v_c(8.2) = 230) -> beta(15-30)")
print(f"  {'N':>3s} {'f':>5s} | {'r_half':>7s} | {'beta':>6s} | v_c(30) | v_c(50) v_c(100) v_c(200)  [captured + baryons only; hot gas not yet added]")
for i, N in enumerate(N_GRID):
    f = f_avail[i]
    if f < f_req_A * (R0 / (R0 + 1e-6))**2 or f <= 0:   # cannot reach the datum even when fully compact
        print(f"  {N:3d} {f:5.2f} | cannot reach the datum: f below the intact-copy requirement {f_req_A:.2f}")
        continue
    rh_pred[i] = brentq(lambda rh: f_req_of_rhalf(rh) - f, 1e-3, 1e4)
    m_tot = m_bar(R_PROF_FAR) + m_hernquist(R_PROF_FAR, f * M_BAR, rh_pred[i])
    v = np.sqrt(G * m_tot / R_PROF_FAR)
    beta_pred[i] = outer_slope(R_PROF_FAR, v)
    vc30_pred[i] = np.interp(30.0, R_PROF_FAR, v)
    vc_far_pred[i] = np.interp(R_FAR, R_PROF_FAR, v)
    print(f"  {N:3d} {f:5.2f} | {rh_pred[i]:7.1f} | {beta_pred[i]:+.3f} | {vc30_pred[i]:7.0f} |"
          + " ".join(f"{x:7.0f}" for x in vc_far_pred[i]))
print("  Reference: NFW c=12 with the same baryons gives beta = -0.07 (1e12) to -0.17 (5e11), from step 1; a flat curve is 0;")
print("  Keplerian is -0.5. Gaia-era Milky Way linear slopes of -1.7 to -2.2 km/s/kpc at R~20 kpc, v~215 km/s")
print("  correspond to beta ~ -0.16 to -0.20; Jiao et al. 2026 power law beta = -0.50 +/- 0.10 beyond 16 kpc.")
print("  These are data to be tested against the column above, not inputs to it.")


# ── The same chain at m = 2 and m = 3 (added after peer review, 2026-10-10):
# if other branches hold more collapsed mass per well than ours, f grows,
# the extent that reproduces the solar-radius datum grows, and beta
# flattens. Printed so the population comparison (step 5) can be read
# against m, with nothing else changed.
print("\nThe chain at m = 2 and m = 3 (captured + baryons only):")
print(f"  {'N':>3s} | {'m=2: f':>6s} {'r_half':>7s} {'beta':>7s} {'M<50':>5s} {'M<100':>5s} | {'m=3: f':>6s} {'r_half':>7s} {'beta':>7s} {'M<50':>5s} {'M<100':>5s}   [masses 1e11; streams: 3.77, 5.6]")
rh_pred_m = np.full((2, nN), np.nan); beta_pred_m = np.full((2, nN), np.nan)
m50_m = np.full((2, nN), np.nan); m100_m = np.full((2, nN), np.nan)
for i, N in enumerate(N_GRID):
    cells = []
    for k, mm_ in enumerate((2.0, 3.0)):
        f = f_avail[i] * mm_
        if f <= f_req_A or f <= 0:
            cells.append(f"{f:6.2f} {'---':>7s} {'---':>7s} {'---':>5s} {'---':>5s}"); continue
        rh = brentq(lambda r_: f_req_of_rhalf(r_) - f, 1e-3, 1e5)
        m_tot = m_bar(R_PROF_FAR) + m_hernquist(R_PROF_FAR, f * M_BAR, rh)
        v = np.sqrt(G * m_tot / R_PROF_FAR)
        rh_pred_m[k, i] = rh; beta_pred_m[k, i] = outer_slope(R_PROF_FAR, v)
        m50 = np.interp(50.0, R_PROF_FAR, m_tot); m100 = np.interp(100.0, R_PROF_FAR, m_tot)
        m50_m[k, i] = m50; m100_m[k, i] = m100
        cells.append(f"{f:6.2f} {rh:7.1f} {beta_pred_m[k,i]:+7.3f} {m50/1e11:5.2f} {m100/1e11:5.2f}")
    print(f"  {N:3d} | " + " | ".join(cells))

print("\nOff-plane test context (geometry A, m = 1): implied dark-disc mass = (N-1) eps M_stellar_disc")
for i, N in enumerate(N_GRID):
    print(f"  N = {N:2d}: {(N-1)*EPS_INF[i]*M_STELLAR_DISC/1e11:.2f} x 1e11 Msun"
          + ("   (Sylos Labini & Capuzzo-Dolcetta 2026 DMD fit: 1.0e11 in a 7 kpc disc — context, not a target)" if N == 6 else ""))

np.savez(OUTPATH, N_GRID=N_GRID, EPS_INF=EPS_INF, M_SCALE=M_SCALE, R_HALF=R_HALF,
         R_EVAL=R_EVAL, R_PROFILE=R_PROFILE, M_BAR=M_BAR, M_cap=M_cap,
         v_bar=v_bar, beta_bar=beta_bar, vc_A=vc_A, beta_A=beta_A, fdm30_A=fdm30_A,
         vc_B=vc_B, beta_B=beta_B, fdm30_B=fdm30_B, prof_A=prof_A, prof_B=prof_B,
         beta_lo=beta_lo, beta_hi=beta_hi, within15=within,
         vc_solar_datum=np.array(VC_SOLAR_DATUM), gas_vc30_max=gas_vc30_max,
         gate_eps=gate_eps, gate_step1=gate_step1,
         f_req_A=f_req_A, beta_req_A=beta_req_A, f_req_B=f_req_B,
         beta_req_B=beta_req_B, vc30_req_B=vc30_req_B, f_avail_m1=f_avail,
         rh_pred=rh_pred, beta_pred=beta_pred, vc30_pred=vc30_pred,
         R_FAR=R_FAR, vc_far_pred=vc_far_pred,
         rh_pred_m23=rh_pred_m, beta_pred_m23=beta_pred_m, m50_m23=m50_m, m100_m23=m100_m)
print(f"\nSaved {OUTPATH}")
