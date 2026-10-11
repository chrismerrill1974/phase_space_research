#!/usr/bin/env python3
"""
Erratum to ISF Papers VI-VIII — Level 0: audit of the missing Jacobian
(read-only; writes only its own table under ../data/).

Paper VI (003_phase2/paper6_phase2_isf_predictions.py, per_lens_abundance)
counts detectable other-branch galaxies per strong-lens sightline as

    N_detect = sum over log M_halo in [10, 12] of
               (N-1) [1 - eps(k_M)] Phi(M_star(M_halo)) V  d log M_halo

with Phi the stellar mass function PER DEX OF STELLAR MASS and the sum
taken over d log M_halo. The text (paper6.tex, eq. isf_excess) says Phi
was "converted from the stellar mass function via the stellar-to-halo
mass relation"; that conversion carries d log M_star / d log M_halo,
which for the script's single power law M_star = 10^10.7 (M_halo/1e12)^(1/1.4)
is the constant 1/1.4. The script omits it, so every count downstream of
the integral is high by exactly 1.4.

This script:
  (a) GATE  reproduces Paper VI's published per-lens count (1.0806 at
            N = 6, eps_inf = 0.58, from paper6_phase2_results.json) with
            Paper VI's own functions and constants, as Paper X's step 3
            already does;
  (b)       recomputes it with the Jacobian, and propagates the constant
            factor to Paper VII's C_sub and its per-lens counts per N;
  (c)       evaluates the three Schechter parameter sets in play (the
            text's, the script's, and Baldry et al. 2012's double
            Schechter as published) and the per-lens count under each,
            with and without the Jacobian — REPORTED, NOT ADOPTED: the
            erratum corrects the text to the script's parameters;
  (d)       lists every file in the Paper VI-IX workspaces whose text or
            code names a quantity fed by the integral, for the hand
            classification in HANDOFF.md.

Inputs:  ../../dark_matter_predictions_6_strong_lensing/003_phase2/paper6_phase2_results.json
         ../../dark_matter_predictions_7/data/euclid_forecast.npz (C_sub, N_det per lens)
Output:  ../data/audit_jacobian.npz, ../data/audit_jacobian.txt (this printout)
Runtime: seconds, single-threaded.
"""

import glob
import json
import os
import re
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
SERIES = os.path.abspath(os.path.join(ROOT, '..'))
DATA = os.path.join(ROOT, 'data')
os.makedirs(DATA, exist_ok=True)
P6 = os.path.join(SERIES, 'dark_matter_predictions_6_strong_lensing')
P7 = os.path.join(SERIES, 'dark_matter_predictions_7')
P8 = os.path.join(SERIES, 'dark_matter_predictions_8')
P9 = sorted(glob.glob(os.path.join(SERIES, 'dark_matter_predictions_9*')))
P6_JSON = os.path.join(P6, '003_phase2', 'paper6_phase2_results.json')
P7_FC = os.path.join(P7, 'data', 'euclid_forecast.npz')

_lines = []
def say(s=''):
    print(s); _lines.append(s)

# ═══════════════════════════════════════════════════════════════════
# Paper VI functions, reproduced with their exact constants
# (paper6_phase2_isf_predictions.py; same copy as Paper X step 3).
# ═══════════════════════════════════════════════════════════════════
JACOBIAN = 1.0 / 1.4          # d log M_star / d log M_halo for M_star ∝ M_halo^(1/1.4)

def epsilon_sigmoid(k, k_t=0.20, eps_inf=0.58, alpha=2.0):
    return eps_inf + (1.0 - eps_inf) / (1.0 + (k / k_t)**alpha)

def epsilon_at_mass(M_halo, k_t=0.20, eps_inf=0.58, alpha=2.0):
    R_vir = 0.26 * (M_halo / 1e12)**(1.0 / 3.0)
    return epsilon_sigmoid(1.0 / R_vir, k_t, eps_inf, alpha)

def schechter_script(M, M_star=10**10.7, alpha_sch=-1.16, phi_star=3.96e-3):
    """The function that produced the published numbers (script)."""
    x = M / M_star
    return np.log(10) * phi_star * x**(alpha_sch + 1) * np.exp(-x)

def schechter_text(M, M_star=10**10.7, alpha_sch=-1.05, phi_star=0.012):
    """The parameters the text quotes (paper6.tex, eq. schechter)."""
    return schechter_script(M, M_star, alpha_sch, phi_star)

def schechter_baldry2012(M, logMs=10.66, phi1=3.96e-3, a1=-0.35, phi2=0.79e-3, a2=-1.47):
    """Baldry et al. 2012 double Schechter as published (per dex), h = 0.7."""
    x = M / 10**logMs
    return np.log(10) * np.exp(-x) * (phi1 * x**(a1 + 1) + phi2 * x**(a2 + 1))

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

def n_detect_per_lens(N, eps_inf, phi=schechter_script, jacobian=1.0,
                      logM_lo=10.0, logM_hi=12.0, k_t=0.20, alpha=2.0):
    """Paper VI per_lens_abundance N_detect (sub + LOS channels); jacobian
    multiplies the integrand (1.0 reproduces the script, 1/1.4 corrects it)."""
    _, V_sub, V_LOS = lensing_volume()
    logM = np.linspace(logM_lo, logM_hi, 201)
    dlogM = logM[1] - logM[0]
    M = 10**logM
    total = 0.0
    for V in (V_sub, V_LOS):
        integrand = np.array([(N - 1) * (1.0 - epsilon_at_mass(Mh, k_t, eps_inf, alpha))
                              * phi(mstar_of_mhalo(Mh)) * V * jacobian for Mh in M])
        total += np.sum(integrand) * dlogM
    return total

# ═══════════════════════════════════════════════════════════════════
say("Erratum VI-VIII — Level 0 audit of the missing Jacobian")
say("=" * 72)

# (a) GATE ───────────────────────────────────────────────────────────
with open(P6_JSON) as f:
    p6 = json.load(f)
pub = p6['per_lens_abundance']['isf_conservative']['N_total']
pub_opt = p6['per_lens_abundance']['isf_optimistic']['N_total']
pub_max = p6['per_lens_abundance']['isf_maximum']['N_total']
rep = n_detect_per_lens(6, 0.58)
rep_opt = n_detect_per_lens(6, 0.30)
rep_max = n_detect_per_lens(6, 0.00)
g_a = abs(rep - pub) < 1e-3 and abs(rep_opt - pub_opt) < 1e-3 and abs(rep_max - pub_max) < 1e-3
say(f"\n(a) GATE: reproduce Paper VI's published per-lens counts (paper6_phase2_results.json)")
say(f"    eps_inf 0.58: {rep:.4f} (published {pub:.4f});  0.30: {rep_opt:.4f} ({pub_opt:.4f});  0.00: {rep_max:.4f} ({pub_max:.4f})")
say(f"    GATE (a): {'PASS' if g_a else 'FAIL'}")
if not g_a:
    raise SystemExit("gate (a) failed: Paper VI's functions not reproduced")

fc7 = np.load(P7_FC, allow_pickle=True)
N_grid = fc7['N_grid']; eps_grid = fc7['eps_inf_grid'] if 'eps_inf_grid' in fc7 else None
C_sub_pub = 1.08 / ((6 - 1) * (1 - 0.58))
Ndet7 = fc7['N_det_per_lens']
g_a2 = np.allclose(Ndet7, [(N - 1) * (1 - e) * C_sub_pub for N, e in zip(N_grid, eps_grid)], atol=1e-6) if eps_grid is not None else False
say(f"    Paper VII C_sub = 1.08 / [(6-1)(1-0.58)] = {C_sub_pub:.4f}; its N_det per lens over N = {N_grid.tolist()}:")
say("      " + ", ".join(f"{v:.3f}" for v in Ndet7) + f"   reproduced from C_sub: {'PASS' if g_a2 else 'FAIL'}")

# (b) with the Jacobian ─────────────────────────────────────────────
corr = n_detect_per_lens(6, 0.58, jacobian=JACOBIAN)
say(f"\n(b) With the Jacobian d log M_star / d log M_halo = 1/1.4 = {JACOBIAN:.4f}:")
say(f"    per-lens count at N = 6, eps_inf = 0.58: {corr:.4f}  (published {pub:.4f}; ratio {pub / corr:.4f}, exactly 1.4 since the matching is a single power law)")
say(f"    eps_inf 0.30: {n_detect_per_lens(6, 0.30, jacobian=JACOBIAN):.4f};  0.00: {n_detect_per_lens(6, 0.00, jacobian=JACOBIAN):.4f}")
C_sub_corr = C_sub_pub / 1.4
say(f"    Paper VII C_sub: {C_sub_pub:.4f} -> {C_sub_corr:.4f}  (1.08/1.4 = {1.08 / 1.4:.4f} per lens at N = 6)")
say(f"    {'N':>3s} {'eps_inf':>7s} {'N_det published':>16s} {'N_det corrected':>16s}")
Ndet_corr = Ndet7 / 1.4
for N, e, a, b in zip(N_grid, eps_grid, Ndet7, Ndet_corr):
    say(f"    {int(N):3d} {float(e):7.2f} {a:16.3f} {b:16.3f}")
say("    Downstream, every count divides by 1.4 and every Poisson significance by sqrt(1.4) = %.3f;" % np.sqrt(1.4))
say("    the Levels 1-2 reruns give the exact values (eta caps make Paper VIII's not a pure ratio).")

# Paper VI's own tables and sentences that carry the count (paper6.tex), old -> new
say(f"\n    Paper VI tables, published -> corrected:")
rows = [('Conservative', 0.58, pub, corr), ('Optimistic', 0.30, pub_opt, n_detect_per_lens(6, 0.30, jacobian=JACOBIAN)),
        ('Maximum', 0.00, pub_max, n_detect_per_lens(6, 0.00, jacobian=JACOBIAN))]
say(f"    Table 'abundance' (N_det per lens; N_sub is 3.2e-4 x 1/1.4 and stays negligible):")
for name, e, a, b in rows:
    say(f"      {name:13s} eps {e:.2f}: {a:.2f} -> {b:.2f}")
lcdm_det = p6['per_lens_abundance']['lcdm_baseline']['N_sub_detectable']
say(f"    '2.5x the CDM detectable baseline of ~0.43': {pub / lcdm_det:.2f}x -> {corr / lcdm_det:.2f}x")
los = p6['los_subhalo_fraction']
f_los_lcdm, n_lcdm = los['lcdm_baseline']['f_LOS'], los['lcdm_baseline']['N_total_per_lens']
n_los_corr = p6['los_subhalo_fraction']['isf_added']['N_LOS'] * JACOBIAN
n_sub_corr = p6['los_subhalo_fraction']['isf_added']['N_sub'] * JACOBIAN
n_tot_corr = n_lcdm + n_los_corr + n_sub_corr
f_los_corr = (f_los_lcdm * n_lcdm + n_los_corr) / n_tot_corr
say(f"    Table 'los' combined column: N_total/lens {los['combined']['N_total']:.2f} -> {n_tot_corr:.2f}; "
    f"f_LOS {los['combined']['f_LOS'] * 100:.1f}% -> {f_los_corr * 100:.1f}%; f_sub {los['combined']['f_sub'] * 100:.1f}% -> {(1 - f_los_corr) * 100:.1f}%; "
    f"shift from LCDM {los['combined']['shift_from_lcdm'] * 100:.1f} -> {(f_los_corr - f_los_lcdm) * 100:.1f} points")
say(f"    Table 'euclid' (per lens / x 75,000 grade A/B / x 5 percent):")
for name, e, a, b in rows[:2]:
    say(f"      ISF eps {e:.2f}: {a:.3f} / {a * 75000:,.0f} / {a * 75000 * 0.05:,.0f} -> {b:.3f} / {b * 75000:,.0f} / {b * 75000 * 0.05:,.0f}")
say(f"    Abstract '~4,000 at >50 sigma' (2,500 lenses x 1.08 = {pub * 2500:,.0f} -> {corr * 2500:,.0f}; the >50 sigma is the 75,000 x 0.05 row: "
    f"sqrt(2 x {pub * 3750:,.0f}) = {np.sqrt(2 * pub * 3750):.0f} -> sqrt(2 x {corr * 3750:,.0f}) = {np.sqrt(2 * corr * 3750):.0f} sigma)")
say(f"    Criterion (1), absence in >1,000 lenses at eta 0.05: {np.sqrt(2 * pub * 50):.1f} -> {np.sqrt(2 * corr * 50):.1f} sigma (stays > 5)")

# (c) the three Schechter parameter sets ─────────────────────────────
say(f"\n(c) Schechter parameter sets (per dex of stellar mass), REPORTED NOT ADOPTED:")
say("    text   : phi* = 0.012,   log M* = 10.70, alpha = -1.05   (paper6.tex eq. schechter, attributed to Baldry+2012)")
say("    script : phi* = 3.96e-3, log M* = 10.70, alpha = -1.16   (paper6_phase2_isf_predictions.py; produced the published numbers)")
say("    Baldry : double Schechter, log M* = 10.66, phi1 = 3.96e-3 a1 = -0.35, phi2 = 0.79e-3 a2 = -1.47 (Baldry+2012 as published)")
Ms = 10**np.array([9.0, 9.5, 10.0, 10.5, 11.0, 11.5])
say(f"    {'log M_star':>10s} {'text':>10s} {'script':>10s} {'Baldry':>10s}   [Mpc^-3 dex^-1]")
for M in Ms:
    say(f"    {np.log10(M):10.1f} {schechter_text(M):10.2e} {schechter_script(M):10.2e} {schechter_baldry2012(M):10.2e}")
say(f"    per-lens count at N = 6, eps_inf = 0.58, no Jacobian / with Jacobian:")
sets = {'text': schechter_text, 'script': schechter_script, 'Baldry 2012 double': schechter_baldry2012}
cnt = {}
for name, fn in sets.items():
    a = n_detect_per_lens(6, 0.58, phi=fn); b = n_detect_per_lens(6, 0.58, phi=fn, jacobian=JACOBIAN)
    cnt[name] = (a, b)
    say(f"      {name:20s}: {a:.4f} / {b:.4f}")
say("    The published 1.08 is the script set without the Jacobian; the erratum's corrected number is the script set")
say("    with it. The text's parameters and the Baldry double Schechter are shown only so the discrepancy is on record.")

# (d) where the integral's products are used ─────────────────────────
say(f"\n(d) Files in the Paper VI-IX workspaces naming a quantity fed by the per-lens integral")
say("    (hand classification into affected / unaffected is in the erratum HANDOFF.md):")
PATTERNS = [r'per_lens_abundance', r'isf_excess_dndm', r'C_sub', r'N_det_per_lens', r'N_det\b', r'n_exp(?:ected)?\b',
            r'1\.08\b', r'0\.514', r'euclid_forecast', r'recalibrated_forecast', r'4,?054\b', r'4,?463\b', r'5,?955\b',
            r'2,?037\b', r'2,?733\b', r'isf_detectable_per_lens']
rx = re.compile('|'.join(PATTERNS))
SKIP_DIRS = {'archive', '.claude', '__pycache__'}
hits = []
for base in [P6, P7, P8] + P9:
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if not fn.endswith(('.py', '.tex', '.md', '.json')):
                continue
            path = os.path.join(dirpath, fn)
            try:
                with open(path, encoding='utf-8', errors='replace') as f:
                    n = sum(1 for line in f if rx.search(line))
            except OSError:
                continue
            if n:
                hits.append((os.path.relpath(path, SERIES), n))
for path, n in sorted(hits):
    say(f"    {n:4d}  {path}")
say(f"    {len(hits)} files. Paper IX workspace(s) scanned: {[os.path.basename(p) for p in P9] or 'none found'}")

# save ───────────────────────────────────────────────────────────────
np.savez(os.path.join(DATA, 'audit_jacobian.npz'),
         jacobian=JACOBIAN, N6_published=pub, N6_reproduced=rep, N6_corrected=corr,
         C_sub_published=C_sub_pub, C_sub_corrected=C_sub_corr,
         N_grid=N_grid, eps_inf_grid=eps_grid, N_det_published=Ndet7, N_det_corrected=Ndet_corr,
         schechter_sets=np.array(list(cnt.keys())), per_lens_by_set=np.array(list(cnt.values())),
         hit_files=np.array([h[0] for h in hits]), hit_counts=np.array([h[1] for h in hits]))
with open(os.path.join(DATA, 'audit_jacobian.txt'), 'w') as f:
    f.write("\n".join(_lines) + "\n")
say(f"\nSaved {os.path.join(DATA, 'audit_jacobian.npz')} and audit_jacobian.txt")
