#!/usr/bin/env python3
"""
Erratum to ISF Papers VI-VIII — Level 1: propagate the Jacobian through
Paper VII's Euclid forecast (Phase 5a).

Paper VII's step5a_euclid_forecast.py calibrates C_sub = 1.08 / [(6-1)(1-0.58)]
from Paper VI's per-lens count, which omits the Jacobian 1/1.4 (Level 0,
01_audit/step0_audit_jacobian.py). Everything else in that script is a
function of C_sub and of two products the error does not touch
(joint_constraint.npz, lnL_sub_grid.npz). This script reproduces the
Phase 5a computation as a function of C_sub, with the same survey
parameters (75,000 grade-A/B lenses, eta = 0.05, 10^8.8 Msun), then:

  GATE   at the published C_sub it must reproduce every array in the
         published euclid_forecast.npz;
  V2     at C_sub / 1.4 it writes euclid_forecast_v2.npz BESIDE the
         original (never overwriting it), same keys, plus 'jacobian' and
         'C_sub_published';
  TABLE  prints every number Paper VII's text quotes from this step, old
         beside new, and saves the printout;
  CHECK  prints the SHA-256 of the two untouched inputs so the record
         shows they were not rewritten.

Inputs:  ../../dark_matter_predictions_7/data/joint_constraint.npz
         ../../dark_matter_predictions_7/data/lnL_sub_grid.npz
         ../../dark_matter_predictions_7/data/euclid_forecast.npz (gate)
Output:  ../../dark_matter_predictions_7/data/euclid_forecast_v2.npz
         ../data/paper7_forecast_v2.txt (this printout)
Runtime: seconds, single-threaded.
"""

import hashlib
import os

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
SERIES = os.path.abspath(os.path.join(ROOT, '..'))
DATA = os.path.join(ROOT, 'data')
P7DATA = os.path.join(SERIES, 'dark_matter_predictions_7', 'data')
JOINT_PATH = os.path.join(P7DATA, 'joint_constraint.npz')
SUB_PATH = os.path.join(P7DATA, 'lnL_sub_grid.npz')
PUB_PATH = os.path.join(P7DATA, 'euclid_forecast.npz')
OUTPATH = os.path.join(P7DATA, 'euclid_forecast_v2.npz')

JACOBIAN = 1.0 / 1.4
N_LENSES, ETA, M_THRESHOLD_LOG = 75000, 0.05, 8.8        # Paper VII Phase 5a
C_SUB_PUB = 1.08 / ((6 - 1) * (1 - 0.58))

_lines = []
def say(s=''):
    print(s); _lines.append(s)

jc = np.load(JOINT_PATH, allow_pickle=True)
sub = np.load(SUB_PATH, allow_pickle=True)
pub = np.load(PUB_PATH, allow_pickle=True)
N_grid = jc['N_grid']; eps_inf_grid = jc['eps_inf_grid']; k_t_grid = jc['k_t_grid']
chi2_wl_grid = jc['chi2_wl_grid']; chi2_sub_grid = sub['chi2_sub_grid']; chi2_lcdm = float(jc['chi2_lcdm'])
n_N, n_kt = len(N_grid), len(k_t_grid)
iN_fid = int(np.where(N_grid == 6)[0][0]); iN4 = int(np.where(N_grid == 4)[0][0])

def forecast(C_sub):
    """Paper VII Phase 5a, step for step, as a function of C_sub."""
    a = N_LENSES * ETA
    N_det = np.array([(N - 1) * (1 - e) * C_sub for N, e in zip(N_grid, eps_inf_grid)])
    N_det_true = N_det[iN_fid]
    n_expected = N_det_true * a
    sigma_Ndet = np.sqrt(n_expected) / a
    chi2_euclid_isf = np.repeat(((N_det - N_det_true)**2 / sigma_Ndet**2)[:, None], n_kt, axis=1)
    chi2_joint_euclid_isf = chi2_wl_grid + chi2_sub_grid + chi2_euclid_isf
    idx_best = np.unravel_index(np.argmin(chi2_joint_euclid_isf), chi2_joint_euclid_isf.shape)
    dchi2_isf_true = chi2_joint_euclid_isf - chi2_joint_euclid_isf.min()
    L = np.exp(-0.5 * dchi2_isf_true)
    dlog_kt = np.diff(np.log(k_t_grid)); wt = np.zeros(n_kt)
    wt[0] = dlog_kt[0] / 2; wt[-1] = dlog_kt[-1] / 2; wt[1:-1] = (dlog_kt[:-1] + dlog_kt[1:]) / 2
    P_N = np.array([np.sum(L[iN] * wt) for iN in range(n_N)]); P_N /= P_N.sum()
    chi2_euclid_null = np.repeat((2.0 * N_det * a)[:, None], n_kt, axis=1)
    chi2_null_total = chi2_wl_grid + chi2_sub_grid + chi2_euclid_null
    dchi2_null_vs_lcdm = chi2_null_total - chi2_lcdm
    thr = np.zeros(n_N)
    for iN in range(n_N):
        mu = N_det[iN]
        B = -(2 * a * mu + 3.84); disc = B**2 - 4 * a * (a * mu**2)
        thr[iN] = (-B - np.sqrt(disc)) / (2 * a) if disc > 0 else 0.0
    chi2_current = jc['chi2_joint_grid']
    return dict(N_grid=N_grid, eps_inf_grid=eps_inf_grid, k_t_grid=k_t_grid,
                N_lenses=N_LENSES, eta=ETA, M_threshold_log=M_THRESHOLD_LOG,
                C_sub=C_sub, N_det_per_lens=N_det, sigma_Ndet=sigma_Ndet,
                chi2_euclid_isf=chi2_euclid_isf, chi2_joint_euclid_isf=chi2_joint_euclid_isf,
                dchi2_isf_true=dchi2_isf_true, P_N_euclid_isf=P_N, n_expected_isf=n_expected,
                chi2_euclid_null=chi2_euclid_null, chi2_null_total=chi2_null_total,
                dchi2_null_vs_lcdm=dchi2_null_vs_lcdm, N_det_threshold_95=thr,
                chi2_current_joint=chi2_current, dchi2_current=chi2_current - chi2_current.min(),
                chi2_lcdm=chi2_lcdm, best_idx=idx_best)

say("Erratum VI-VIII — Level 1: Paper VII Euclid forecast with the Jacobian")
say("=" * 72)

# GATE ────────────────────────────────────────────────────────────────
old = forecast(C_SUB_PUB)
bad = []
for key in pub.files:
    if key not in old:
        bad.append(f"{key}: not recomputed"); continue
    if not np.allclose(np.asarray(pub[key], dtype=float), np.asarray(old[key], dtype=float), rtol=1e-9, atol=1e-12):
        bad.append(key)
say(f"\nGATE: published euclid_forecast.npz reproduced at C_sub = {C_SUB_PUB:.4f} "
    f"({len(pub.files)} arrays): {'PASS' if not bad else 'FAIL ' + ', '.join(bad)}")
if bad:
    raise SystemExit("gate failed")

# V2 ──────────────────────────────────────────────────────────────────
C_SUB_V2 = C_SUB_PUB * JACOBIAN
new = forecast(C_SUB_V2)
a = N_LENSES * ETA
say(f"\nC_sub: {C_SUB_PUB:.4f} -> {C_SUB_V2:.4f}  (Paper VI per lens at N = 6: 1.08 -> {1.08 * JACOBIAN:.3f})")
say(f"\nPer-lens count and expected Euclid detections ({N_LENSES} lenses, eta = {ETA}):")
say(f"  {'N':>3s} {'eps':>5s} | {'N_det old':>9s} {'N_det new':>9s} | {'N_Euclid old':>12s} {'N_Euclid new':>12s}")
for iN in range(n_N):
    say(f"  {int(N_grid[iN]):3d} {eps_inf_grid[iN]:5.2f} | {old['N_det_per_lens'][iN]:9.3f} {new['N_det_per_lens'][iN]:9.3f} | "
        f"{old['N_det_per_lens'][iN] * a:12.0f} {new['N_det_per_lens'][iN] * a:12.0f}")
say(f"  text range N = 2..10: {old['N_det_per_lens'][0] * a:,.0f}-{old['N_det_per_lens'][-1] * a:,.0f} -> "
    f"{new['N_det_per_lens'][0] * a:,.0f}-{new['N_det_per_lens'][-1] * a:,.0f}")

say(f"\nISF-true scenario (fiducial N = 6):")
say(f"  expected detections n_exp: {old['n_expected_isf']:.0f} -> {new['n_expected_isf']:.0f}")
say(f"  Poisson sigma sqrt(n):     {np.sqrt(old['n_expected_isf']):.1f} -> {np.sqrt(new['n_expected_isf']):.1f}")
say(f"  sigma(N_det/lens):         {old['sigma_Ndet']:.5f} -> {new['sigma_Ndet']:.5f}")
say(f"  fractional precision:      {old['sigma_Ndet'] / old['N_det_per_lens'][iN_fid] * 100:.1f}% -> "
    f"{new['sigma_Ndet'] / new['N_det_per_lens'][iN_fid] * 100:.1f}%")
say(f"  {'N':>3s} | {'sigma from N=6 old':>18s} {'new':>6s}")
for iN in range(n_N):
    if iN == iN_fid: continue
    so = abs(old['N_det_per_lens'][iN] - old['N_det_per_lens'][iN_fid]) / old['sigma_Ndet']
    sn = abs(new['N_det_per_lens'][iN] - new['N_det_per_lens'][iN_fid]) / new['sigma_Ndet']
    say(f"  {int(N_grid[iN]):3d} | {so:18.1f} {sn:6.1f}")
bo, bn = old['best_idx'], new['best_idx']
say(f"  joint+Euclid best fit: N = {int(N_grid[bo[0]])}, k_t = {k_t_grid[bo[1]]:.4f} -> N = {int(N_grid[bn[0]])}, k_t = {k_t_grid[bn[1]]:.4f}")
say(f"  Dchi2 vs LCDM at best fit: {old['chi2_joint_euclid_isf'][bo] - chi2_lcdm:.2f} -> {new['chi2_joint_euclid_isf'][bn] - chi2_lcdm:.2f}")
say(f"  {'N':>3s} | {'P(N) current':>12s} {'+Euclid old':>12s} {'+Euclid new':>12s}")
for iN in range(n_N):
    say(f"  {int(N_grid[iN]):3d} | {jc['P_N_joint'][iN]:12.6f} {old['P_N_euclid_isf'][iN]:12.6f} {new['P_N_euclid_isf'][iN]:12.6f}")

say(f"\nNull scenario (zero ISF perturbers), exclusion sigma = sqrt(2 n_model):")
say(f"  {'N':>3s} | {'n_model old':>11s} {'new':>7s} | {'sigma old':>9s} {'new':>6s}")
for iN in range(n_N):
    say(f"  {int(N_grid[iN]):3d} | {old['N_det_per_lens'][iN] * a:11.0f} {new['N_det_per_lens'][iN] * a:7.0f} | "
        f"{np.sqrt(old['chi2_euclid_null'][iN, 0]):9.0f} {np.sqrt(new['chi2_euclid_null'][iN, 0]):6.0f}")
say(f"  min Dchi2 vs LCDM over the grid: {old['dchi2_null_vs_lcdm'].min():.1f} -> {new['dchi2_null_vs_lcdm'].min():.1f} "
    f"(>> {np.sqrt(old['dchi2_null_vs_lcdm'].min()):.0f} -> {np.sqrt(new['dchi2_null_vs_lcdm'].min()):.0f} sigma)")

say(f"\nFalsification thresholds (95 percent; observed N_det/lens below this excludes that N):")
say(f"  {'N':>3s} | {'N_det_min old':>13s} {'new':>8s} | {'total old':>9s} {'new':>6s}")
for iN in range(n_N):
    say(f"  {int(N_grid[iN]):3d} | {old['N_det_threshold_95'][iN]:13.4f} {new['N_det_threshold_95'][iN]:8.4f} | "
        f"{old['N_det_threshold_95'][iN] * a:9.0f} {new['N_det_threshold_95'][iN] * a:6.0f}")
say(f"  ALL ISF (N >= 2): fewer than {old['N_det_threshold_95'][0] * a:.0f} -> {new['N_det_threshold_95'][0] * a:.0f} detections")
say(f"  N >= 4:           fewer than {old['N_det_threshold_95'][iN4] * a:.0f} -> {new['N_det_threshold_95'][iN4] * a:.0f} detections")

# CHECK the untouched inputs ──────────────────────────────────────────
say(f"\nUntouched inputs (SHA-256, for the record; git must show them unmodified):")
for p in (JOINT_PATH, SUB_PATH):
    with open(p, 'rb') as f:
        say(f"  {hashlib.sha256(f.read()).hexdigest()}  {os.path.relpath(p, SERIES)}")

# save ────────────────────────────────────────────────────────────────
save = {k: v for k, v in new.items() if k != 'best_idx'}
save['jacobian'] = JACOBIAN; save['C_sub_published'] = C_SUB_PUB
if os.path.exists(OUTPATH):
    # never rewrite a committed product: confirm it is what this run produced
    prev = np.load(OUTPATH, allow_pickle=True)
    same = set(prev.files) == set(save.keys()) and all(
        np.allclose(np.asarray(prev[k], dtype=float), np.asarray(save[k], dtype=float), rtol=1e-12, atol=0) for k in prev.files)
    say(f"\n{OUTPATH} exists and {'matches this run (not rewritten)' if same else 'DIFFERS from this run: archive it and rerun'}")
    if not same:
        raise SystemExit(1)
else:
    np.savez(OUTPATH, **save)
os.makedirs(DATA, exist_ok=True)
with open(os.path.join(DATA, 'paper7_forecast_v2.txt'), 'w') as f:
    f.write("\n".join(_lines) + "\n")
say(f"\nSaved {OUTPATH} and {os.path.join(DATA, 'paper7_forecast_v2.txt')}")
