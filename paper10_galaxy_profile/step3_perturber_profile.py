#!/usr/bin/env python3
"""
Paper X — Step 3: the ISF perturber profile and the Paper XI threshold

Paper VI modelled an ISF dark perturber as an intact NFW halo (c = 12)
and derived ~1.08 detectable perturbers per lens at 10^10-10^12 Msun;
Paper VII folded that into C_sub = 1.08 / [(6-1)(1-0.58)] = 0.514 and
Paper VIII into the Euclid forecasts, including the in-principle test on
the 312 modelled Q1 systems. Under Paper X's promoted picture an ISF
perturber is a branch-j galaxy plus its captured dark component, a
Hernquist spheroid whose extent is anchored to the Milky Way by step 2.
This script asks what that profile does to the detection threshold and
hence to the Paper XI numbers.

GATES (all three must pass before any new number is printed):
  G1  Reproduce Paper VI's convergence table: kappa_ISF/kappa_LCDM =
      1.064 (1 kpc), 1.377 (5 kpc), 2.112 (10 kpc) and the 45 percent
      bound-mass retention, with Paper VI's own grid and conventions.
  G2  Reproduce Paper VI's 1.08 detectable perturbers per lens at N = 6
      with its mass function, abundance matching, and volumes.
  G3  Reproduce Paper VIII's 312-system row: effective sightlines
      15.60 / 12.69 / 11.69 and n_exp(N=6) = 16.85 / 13.71 / 12.63 for
      the flat / linear / area efficiency models.

TRANSLATION (plan Section 5b). Detectability proxy: the projected mass
inside an aperture r_ap around the perturber centre. The Paper VI
threshold is a 10^10 Msun NFW c=12 halo. The ISF threshold M_min is the
total mass whose Hernquist profile puts the same projected mass inside
r_ap. Apertures: 1 kpc, 5 kpc (Paper VI's discriminant radius, the host
Einstein radius; fiducial), 13.5 kpc (3 arcsec at z = 0.3, the LOS
detection radius). C_sub is then recomputed with M_min as the lower
limit of the Paper VI integral (upper limit 10^12 unchanged).

EXTENT SCALING (open question, both computed, neither chosen): the
Milky Way captured extent r_half(N) from step 2 is carried to a
perturber of total mass M either
  (i)  scaled with the well size, r_half ∝ M^(1/3) relative to a
       1.5e12 Msun Milky Way well, or
  (ii) held fixed at the Milky Way value.

BEYOND THE PLAN, reported separately and not folded into the headline:
the promoted picture also changes the mass-to-number mapping (an ISF
perturber of total mass M is a galaxy of baryonic mass M/(1+f), not the
LCDM abundance-matched galaxy). Its effect on C_sub is printed as an
extra line for Chris to rule on.

Inputs:  ../data/paper6_phase2_results.json (Paper VI grid and ratios)
         ../data/euclid_forecast_paper7.npz, ../data/joint_constraint_paper7.npz
         ../data/q1_demographics_paper8.npz, ../data/dissipated_component.npz
Output:  ../data/perturber_profile.npz
Runtime: seconds (the convergence integrals are the slow part).
"""

import os
import sys
import json
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import brentq

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA = os.path.join(ROOT, 'data')
OUTPATH = os.path.join(DATA, 'perturber_profile.npz')

# ═══════════════════════════════════════════════════════════════════
# Paper VI functions, reproduced with their exact constants
# (paper6_phase2_isf_predictions.py, Phase 2 deliverable).
# ═══════════════════════════════════════════════════════════════════
def epsilon_sigmoid(k, k_t=0.20, eps_inf=0.58, alpha=2.0):
    # Paper VI's convention: eps -> 1 at k << k_t, eps_inf at k >> k_t.
    # (Papers VII/IX write the sigmoid the other way; at the halo masses
    # used here k_M >> k_t so both give eps = eps_inf. Immaterial.)
    return eps_inf + (1.0 - eps_inf) / (1.0 + (k / k_t)**alpha)

def epsilon_at_mass(M_halo, k_t=0.20, eps_inf=0.58, alpha=2.0):
    R_vir = 0.26 * (M_halo / 1e12)**(1.0 / 3.0)
    return epsilon_sigmoid(1.0 / R_vir, k_t, eps_inf, alpha)

def schechter_dndm(M, M_star=10**10.7, alpha_sch=-1.16, phi_star=3.96e-3):
    x = M / M_star
    return np.log(10) * phi_star * x**(alpha_sch + 1) * np.exp(-x)

def mstar_of_mhalo(Mh):
    return 10**10.7 * (Mh / 1e12)**(1.0 / 1.4)

def lensing_volume(z_L=0.3, z_S=2.0):
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    R_vir_host = 0.5
    V_host = 4.0 / 3.0 * np.pi * R_vir_host**3
    V_detect_sub = np.pi * 0.050**2 * 2 * R_vir_host
    theta = 3.0 * 4.848e-6
    DcS = cosmo.comoving_distance(z_S).to('Mpc').value
    DcL = cosmo.comoving_distance(z_L).to('Mpc').value
    V_LOS = np.pi * (theta * 0.5 * (DcL + DcS))**2 * (DcS - DcL)
    return V_host, V_detect_sub, V_LOS

def n_detect_per_lens(N, eps_inf, logM_lo=10.0, logM_hi=12.0, k_t=0.20, alpha=2.0,
                      mstar_map=mstar_of_mhalo):
    """Paper VI per_lens_abundance N_detect (sub + LOS channels)."""
    _, V_sub, V_LOS = lensing_volume()
    logM = np.linspace(logM_lo, logM_hi, 201)
    dlogM = logM[1] - logM[0]
    M = 10**logM
    total = 0.0
    for V in (V_sub, V_LOS):
        integrand = np.array([(N - 1) * (1.0 - epsilon_at_mass(Mh, k_t, eps_inf, alpha))
                              * schechter_dndm(mstar_map(Mh)) * V for Mh in M])
        total += np.sum(integrand) * dlogM
    return total

def nfw_profile(r_kpc, M_halo, c, rho_crit=1.4e11, Delta=200):
    R_vir = (3 * M_halo / (4 * np.pi * Delta * rho_crit))**(1. / 3.) * 1e3
    r_s = R_vir / c
    g_c = np.log(1 + c) - c / (1 + c)
    rho_s = M_halo / (4 * np.pi * r_s**3 * g_c)
    x = np.maximum(np.asarray(r_kpc, dtype=float) / r_s, 1e-6)
    return rho_s / (x * (1 + x)**2), r_s, R_vir

def lensing_convergence(r_kpc, rho, Sigma_cr=3.5e9):
    r = np.asarray(r_kpc, dtype=float)
    rho_i = interp1d(r, rho, kind='linear', fill_value=0, bounds_error=False)
    Sigma = np.zeros_like(r)
    for i, R in enumerate(r):
        if R < r[-1] * 0.99:
            f = lambda r3, _R=R: 0.0 if r3 <= _R else rho_i(r3) * r3 / np.sqrt(r3**2 - _R**2)
            val, _ = quad(f, R * 1.001, r[-1], limit=100)
            Sigma[i] = 2.0 * val
    return Sigma / Sigma_cr

# ═══════════════════════════════════════════════════════════════════
# Gate G1 — Paper VI convergence table on Paper VI's own grid
# ═══════════════════════════════════════════════════════════════════
p6 = json.load(open(os.path.join(DATA, 'paper6_phase2_results.json')))['profile_comparison']
r_grid = np.array(p6['r_kpc'])
rho_isf, r_s6, R_vir6 = nfw_profile(r_grid, 1e10, 12.0)
r_t6 = 0.30 * R_vir6
rho_lcdm = rho_isf * r_t6**2 / (r_grid**2 + r_t6**2)
k_isf = lensing_convergence(r_grid, rho_isf)
k_lcdm = lensing_convergence(r_grid, rho_lcdm)
ratios = {R: np.interp(R, r_grid, k_isf) / np.interp(R, r_grid, k_lcdm) for R in (1.0, 5.0, 10.0)}
pub = p6['convergence_ratios']
g_c6 = np.log(13.0) - 12.0 / 13.0
x_t = r_t6 / r_s6
M_bound = 1e10 * (np.log(1 + x_t) - x_t / (1 + x_t)) / g_c6
g1 = (abs(ratios[1.0] / pub['kappa_isf_over_lcdm_at_1kpc'] - 1) < 1e-3 and
      abs(ratios[5.0] / pub['kappa_isf_over_lcdm_at_5kpc'] - 1) < 1e-3 and
      abs(ratios[10.0] / pub['kappa_isf_over_lcdm_at_10kpc'] - 1) < 1e-3 and
      abs(M_bound / p6['mass_comparison']['M_bound_lcdm'] - 1) < 1e-6)
print("Paper X step 3 — perturber profile and the Paper XI threshold")
print(f"\nG1  Paper VI table: R_vir = {R_vir6:.2f} kpc (pub {p6['parameters']['R_vir_kpc']:.2f}), "
      f"r_s = {r_s6:.2f}, r_t = {r_t6:.2f} (pub {p6['parameters']['r_trunc_kpc']:.2f})")
for R in (1.0, 5.0, 10.0):
    key = f"kappa_isf_over_lcdm_at_{int(R)}kpc"
    print(f"    kappa ratio at {R:4.1f} kpc: {ratios[R]:.4f}   (published {pub[key]:.4f})")
print(f"    LCDM bound mass inside r_t: {M_bound:.4e}   (published {p6['mass_comparison']['M_bound_lcdm']:.4e})")
print(f"    GATE G1: {'PASS' if g1 else 'FAIL'}")

# ═══════════════════════════════════════════════════════════════════
# Gate G2 — Paper VI 1.08 per lens; Gate G3 — Paper VIII 312-system row
# ═══════════════════════════════════════════════════════════════════
N_det6 = n_detect_per_lens(6, 0.58)
C_sub_rep = N_det6 / ((6 - 1) * (1 - 0.58))
fc7 = np.load(os.path.join(DATA, 'euclid_forecast_paper7.npz'), allow_pickle=True)
C_sub_pub = float(fc7['C_sub'])
N_GRID = fc7['N_grid'].astype(int); EPS_INF = fc7['eps_inf_grid'].astype(float)
N_det_pub = fc7['N_det_per_lens']
g2 = abs(N_det6 / 1.08 - 1) < 0.01
print(f"\nG2  Paper VI N_detect(N=6) = {N_det6:.4f}  (published ~1.08);  C_sub = {C_sub_rep:.4f} (Paper VII {C_sub_pub:.4f})")
print(f"    GATE G2: {'PASS' if g2 else 'FAIL'}")

demo = np.load(os.path.join(DATA, 'q1_demographics_paper8.npz'), allow_pickle=True)
theta_AB = demo['theta_E_AB']
ETA0, THETA_REF, ETA_CAP = 0.05, 1.2, 0.15
def eta_of_theta(theta, model):
    if model == 'flat':   return np.full_like(theta, ETA0)
    if model == 'linear': return np.minimum(ETA0 * (theta / THETA_REF), ETA_CAP)
    if model == 'area':   return np.minimum(ETA0 * (theta / THETA_REF)**2, ETA_CAP)
ETA_MODELS = ['flat', 'linear', 'area']
REF = {'flat': (15.599999999999998, 16.848), 'linear': (12.690796875457401, 13.70606063), 'area': (11.692887452304415, 12.62831845)}
iN6 = int(np.where(N_GRID == 6)[0][0])
a_q1 = {m: float(np.sum(eta_of_theta(theta_AB, m))) for m in ETA_MODELS}
g3 = True
print("\nG3  Paper VIII 312-system row (effective sightlines; n_exp at N=6; null exclusion sigma):")
for m in ETA_MODELS:
    n_exp = N_det_pub * a_q1[m]
    ok = abs(a_q1[m] / REF[m][0] - 1) < 1e-6 and abs(n_exp[iN6] / REF[m][1] - 1) < 1e-4
    g3 &= ok
    print(f"    {m:7s}: a = {a_q1[m]:7.3f} (pub {REF[m][0]:7.3f});  n_exp(6) = {n_exp[iN6]:6.3f} (pub {REF[m][1]:6.3f});"
          f"  null sigma N=2 {np.sqrt(2*n_exp[0]):.2f}, N=4 {np.sqrt(2*n_exp[2]):.2f}, N=6 {np.sqrt(2*n_exp[iN6]):.2f}")
print(f"    GATE G3: {'PASS' if g3 else 'FAIL'}")
if not (g1 and g2 and g3):
    sys.exit("a gate failed; no new numbers")

# ═══════════════════════════════════════════════════════════════════
# The ISF perturber profile and the aperture-mass threshold
# ═══════════════════════════════════════════════════════════════════
h2 = np.load(os.path.join(DATA, 'dissipated_component.npz'))
RH_MW = h2['rh_pred']                    # per N in h2['N_GRID'] (nan for N=2,3)
assert np.array_equal(h2['N_GRID'].astype(int), N_GRID)
F_MW = h2['f_avail_m1']
M_MW_WELL = 1.5e12                       # anchor well for scaling (i)

def m_proj_in_aperture(m_enclosed, r, R_ap):
    """Projected mass inside a cylinder of radius R_ap from the enclosed
    3-D mass profile m_enclosed(r) on grid r: sum over shells of the
    fraction of each shell inside the cylinder, 1 - sqrt(1 - (R/r)^2)
    for r > R, and 1 for r <= R."""
    dm = np.diff(m_enclosed)
    rm = 0.5 * (r[1:] + r[:-1])
    frac = np.where(rm <= R_ap, 1.0, 1.0 - np.sqrt(np.clip(1.0 - (R_ap / rm)**2, 0, 1)))
    return np.sum(dm * frac) + m_enclosed[0]

r_fine = np.logspace(-3, np.log10(3000.0), 3000)
def m_nfw_enclosed(r, M, c):
    _, r_s, R_vir = nfw_profile(np.array([1.0]), M, c)
    g_c = np.log(1 + c) - c / (1 + c)
    x = np.minimum(r, R_vir) / r_s        # halo mass defined inside R_vir
    return M * (np.log(1 + x) - x / (1 + x)) / g_c
def m_hern_enclosed(r, M, r_half):
    a = r_half / (1.0 + np.sqrt(2.0))
    return M * r**2 / (r + a)**2

APERTURES = np.array([1.0, 5.0, 13.5])
M_REF = 1e10
mp_ref = np.array([m_proj_in_aperture(m_nfw_enclosed(r_fine, M_REF, 12.0), r_fine, R) for R in APERTURES])
print(f"\nReference threshold: NFW c=12, M = 1e10; projected mass fraction inside 1 / 5 / 13.5 kpc = "
      + " / ".join(f"{x/M_REF:.3f}" for x in mp_ref))

def rhalf_of_mass(M, N_index, scaling):
    rh = RH_MW[N_index]
    if scaling == 'i':
        return rh * (M / M_MW_WELL)**(1.0 / 3.0)
    return rh

def m_min_isf(N_index, scaling, ap_index):
    """Total mass whose Hernquist profile puts the reference projected
    mass inside the aperture."""
    R = APERTURES[ap_index]
    def f(logM):
        M = 10**logM
        return m_proj_in_aperture(m_hern_enclosed(r_fine, M, rhalf_of_mass(M, N_index, scaling)), r_fine, R) - mp_ref[ap_index]
    return 10**brentq(f, 6.0, 14.0)

SCALINGS = ['i', 'ii']
nN, nS, nA = len(N_GRID), len(SCALINGS), len(APERTURES)
M_min = np.full((nN, nS, nA), np.nan)
C_sub_isf = np.full((nN, nS, nA), np.nan)
N_det_isf = np.full((nN, nS, nA), np.nan)
print("\nISF perturber: Hernquist with r_half from the step 2 Milky Way anchor;")
print("  M_min = total mass matching the 1e10 NFW projected aperture mass, per N, scaling, aperture")
print(f"  {'N':>3s} {'r_half,MW':>9s} | scaling (i) r_half ∝ M^(1/3):  M_min @1kpc   @5kpc   @13.5kpc  | scaling (ii) fixed r_half:  M_min @1kpc   @5kpc   @13.5kpc")
for i, N in enumerate(N_GRID):
    if not np.isfinite(RH_MW[i]):
        print(f"  {N:3d}   no anchor (step 2: cannot reach the solar-radius datum)")
        continue
    cells = []
    for s_i, sc in enumerate(SCALINGS):
        for a_i in range(nA):
            M_min[i, s_i, a_i] = m_min_isf(i, sc, a_i)
        cells.append("  ".join(f"{M_min[i, s_i, a]:.2e}" for a in range(nA)))
    print(f"  {N:3d} {RH_MW[i]:9.1f} |                                 {cells[0]}  |                             {cells[1]}")

# ═══════════════════════════════════════════════════════════════════
# Corrected C_sub, per-lens counts, and the Paper XI numbers
# ═══════════════════════════════════════════════════════════════════
print("\nCorrected C_sub (Paper VI integral with lower limit M_min, upper 1e12, N=6 normalisation as Paper VII)")
print("  and corrected per-lens N_det = (N-1)(1-eps_inf) C_sub; aperture 5 kpc is the fiducial")
print(f"  {'N':>3s} | {'C_sub (i) @1 @5 @13.5':>24s} | {'C_sub (ii) @1 @5 @13.5':>24s} | N_det(i,@5)  N_det(ii,@5)  (Paper VIII N_det)")
for i, N in enumerate(N_GRID):
    if not np.isfinite(RH_MW[i]):
        continue
    for s_i in range(nS):
        for a_i in range(nA):
            nd6 = n_detect_per_lens(6, 0.58, logM_lo=np.log10(M_min[i, s_i, a_i]))
            C_sub_isf[i, s_i, a_i] = nd6 / ((6 - 1) * (1 - 0.58))
            N_det_isf[i, s_i, a_i] = (N - 1) * (1 - EPS_INF[i]) * C_sub_isf[i, s_i, a_i]
    print(f"  {N:3d} | " + " ".join(f"{C_sub_isf[i,0,a]:7.3f}" for a in range(nA)) + " | "
          + " ".join(f"{C_sub_isf[i,1,a]:7.3f}" for a in range(nA))
          + f" |   {N_det_isf[i,0,1]:6.3f}       {N_det_isf[i,1,1]:6.3f}        ({N_det_pub[i]:.3f})")

print("\nPaper XI in-principle test on the 312 modelled Q1 systems, fiducial aperture 5 kpc:")
print("  n_exp per N and null-exclusion sigma = sqrt(2 n_exp), for each efficiency model; Paper VIII values in brackets")
n_exp_isf = np.full((nN, nS, len(ETA_MODELS)), np.nan)
for s_i, sc in enumerate(SCALINGS):
    print(f"  scaling ({sc}):")
    for m_i, m in enumerate(ETA_MODELS):
        row = []
        for i, N in enumerate(N_GRID):
            if np.isfinite(N_det_isf[i, s_i, 1]):
                n_exp_isf[i, s_i, m_i] = N_det_isf[i, s_i, 1] * a_q1[m]
                row.append(f"N={N}: {n_exp_isf[i,s_i,m_i]:5.1f} ({N_det_pub[i]*a_q1[m]:4.1f}) {np.sqrt(2*n_exp_isf[i,s_i,m_i]):.1f}σ")
        print(f"    {m:7s}: " + "; ".join(row))
brk = (min(REF[m][1] for m in ETA_MODELS), max(REF[m][1] for m in ETA_MODELS))
print(f"\nDecision gate B (plan 5c): Paper VIII's eta bracket for n_exp(N=6) is [{brk[0]:.1f}, {brk[1]:.1f}].")
for s_i, sc in enumerate(SCALINGS):
    vals = n_exp_isf[iN6, s_i]
    inside = (vals.min() >= brk[0]) and (vals.max() <= brk[1])
    print(f"  scaling ({sc}): corrected n_exp(N=6) over eta models = [{vals.min():.1f}, {vals.max():.1f}]  -> "
          + ("INSIDE the bracket: a paragraph in Paper XI" if inside else "OUTSIDE the bracket: Paper XI's gate is restated"))

# ═══════════════════════════════════════════════════════════════════
# Beyond the plan: the promoted mass-to-number mapping (reported only)
# ═══════════════════════════════════════════════════════════════════
print("\nBEYOND THE PLAN (not in the headline): if an ISF perturber of total mass M is a galaxy of")
print("  baryonic mass M/(1+f') with f' = (N-2) eps_inf (captured from the branches other than ours),")
print("  and stellar mass 0.6 of baryonic, the Paper VI abundance matching M_star(M) is replaced.")
print("  C_sub at the ORIGINAL 1e10 threshold under that mapping, per N (Paper VII: 0.514):")
C_sub_map = np.full(nN, np.nan)
for i, N in enumerate(N_GRID):
    fp = (N - 2) * EPS_INF[i] if N >= 2 else 0.0
    mm = lambda Mh, fp=fp: 0.6 * Mh / (1.0 + fp)
    nd6 = n_detect_per_lens(6, 0.58, mstar_map=mm)
    C_sub_map[i] = nd6 / ((6 - 1) * (1 - 0.58))
    print(f"    N = {N:2d}: f' = {fp:4.2f}, C_sub = {C_sub_map[i]:.3f}")


# ═══════════════════════════════════════════════════════════════════
# HEADLINE (Chris, 2026-10-10, revised after peer review the same day):
# BOTH extent scalings are carried as the headline bracket and neither is
# chosen; the promoted mapping enters. Per-lens count computed directly per N with Paper VI's
# integral: lower limit M_min(N; ii; 5 kpc), upper 1e12,
# M_star(M) = 0.6 M/(1+f'), f' = (N-2) eps_inf(N).
# ═══════════════════════════════════════════════════════════════════
print("\nHEADLINE — promoted mapping, aperture 5 kpc; extent scalings (ii) held and (i) scaled, carried as a bracket")
print(f"  {'N':>3s} {'f_prime':>7s} {'M_min(ii)':>9s} | N_det(ii+map)  C_sub,eff | N_det(i+map) | Paper VIII N_det | n_exp(ii+map): flat linear area | null | n_exp(i+map): flat linear area | null")
N_det_head = np.full(nN, np.nan); N_det_fall = np.full(nN, np.nan); C_eff_head = np.full(nN, np.nan)
n_exp_head = np.full((nN, len(ETA_MODELS)), np.nan)
n_exp_fall = np.full((nN, len(ETA_MODELS)), np.nan)
for i, N in enumerate(N_GRID):
    if not np.isfinite(RH_MW[i]):
        continue
    fp = (N - 2) * EPS_INF[i]
    mm = lambda Mh, fp=fp: 0.6 * Mh / (1.0 + fp)
    N_det_head[i] = n_detect_per_lens(N, EPS_INF[i], logM_lo=np.log10(M_min[i, 1, 1]), mstar_map=mm)
    N_det_fall[i] = n_detect_per_lens(N, EPS_INF[i], logM_lo=np.log10(M_min[i, 0, 1]), mstar_map=mm)
    C_eff_head[i] = N_det_head[i] / ((N - 1) * (1 - EPS_INF[i]))
    n_exp_head[i] = [N_det_head[i] * a_q1[m] for m in ETA_MODELS]
    n_exp_fall[i] = [N_det_fall[i] * a_q1[m] for m in ETA_MODELS]
    print(f"  {N:3d} {fp:7.2f} {M_min[i,1,1]:9.2e} |    {N_det_head[i]:6.3f}       {C_eff_head[i]:6.3f}  |   {N_det_fall[i]:6.3f}     |      {N_det_pub[i]:.3f}       | "
          + " ".join(f"{x:5.1f}" for x in n_exp_head[i]) + " | " + " ".join(f"{np.sqrt(2*x):.1f}" for x in n_exp_head[i])
          + " | " + " ".join(f"{x:5.1f}" for x in n_exp_fall[i]) + " | " + " ".join(f"{np.sqrt(2*x):.1f}" for x in n_exp_fall[i]))
print(f"  Paper VIII n_exp(N=6) for reference: " + " ".join(f"{N_det_pub[iN6]*a_q1[m]:5.1f}" for m in ETA_MODELS))
print("  Zero detections in the 312 systems would exclude N = 6 at the sigma shown; N = 2, 3 are excluded by step 2.")

np.savez(OUTPATH, N_GRID=N_GRID, EPS_INF=EPS_INF, APERTURES=APERTURES, SCALINGS=np.array(SCALINGS),
         RH_MW=RH_MW, M_MW_WELL=M_MW_WELL, mp_ref_frac=mp_ref / M_REF, M_min=M_min,
         C_sub_isf=C_sub_isf, N_det_isf=N_det_isf, N_det_pub=N_det_pub, C_sub_pub=C_sub_pub,
         C_sub_rep=C_sub_rep, a_q1=np.array([a_q1[m] for m in ETA_MODELS]), ETA_MODELS=np.array(ETA_MODELS),
         n_exp_isf=n_exp_isf, paper8_bracket=np.array(brk), C_sub_map=C_sub_map,
         g1_ratios=np.array([ratios[1.0], ratios[5.0], ratios[10.0]]), gates=np.array([g1, g2, g3]),
         N_det_head=N_det_head, N_det_fall=N_det_fall, C_eff_head=C_eff_head, n_exp_head=n_exp_head,
         n_exp_fall=n_exp_fall,
         r_grid_p6=r_grid, k_isf_p6=k_isf, k_lcdm_p6=k_lcdm)
print(f"\nSaved {OUTPATH}")
