#!/usr/bin/env python3
"""
Paper VII — Phase 4: Joint Constraint

Steps 4a–4d, 4f of the research plan (§4.4):

  4a. Joint likelihood: chi2_joint(N, k_t) = chi2_WL(N, k_t) + chi2_sub(N)
  4b. Figure preparation (data for contour plot)
  4c. Parameter extraction: best-fit, marginalised constraints, Delta_chi2
  4d. Consistency check: WL vs sub agreement, N=2 disfavoured?
  4f. Sensitivity table

This script first computes the full chi2_WL(N, k_t) on the 6x30 grid
(extending the D2 N=6-only result to all N values), then combines with
the Phase 3 substructure likelihood.

Depends on:
  ../data/kids1000_bandpowers.npz
  ../data/response_operators_tomo.npz
  ../data/lnL_sub_grid.npz          (Phase 3)
  ../data/chi2_wl_grid_v2.npz       (D2, for cross-check)

Output: ../data/joint_constraint.npz
"""

import numpy as np
import os
import time

# ── Paths ─────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA_DIR = os.path.join(REPO_ROOT, 'data')

KIDS_PATH = os.path.join(DATA_DIR, 'kids1000_bandpowers.npz')
TOMO_PATH = os.path.join(DATA_DIR, 'response_operators_tomo.npz')
SUB_PATH = os.path.join(DATA_DIR, 'lnL_sub_grid.npz')
D2_PATH = os.path.join(DATA_DIR, 'chi2_wl_grid_v2.npz')
OUTPATH = os.path.join(DATA_DIR, 'joint_constraint.npz')

for label, path in [('KiDS data', KIDS_PATH), ('Tomo operators', TOMO_PATH),
                     ('Sub grid', SUB_PATH), ('D2 grid', D2_PATH)]:
    if not os.path.isfile(path):
        print(f"ERROR: {label} not found at:\n  {path}")
        raise SystemExit(1)

# ══════════════════════════════════════════════════════════════════
# PART 1: Load data
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("Phase 4: Joint Constraint")
print("=" * 70)

print("\n[1] Loading data...")

dk = np.load(KIDS_PATH, allow_pickle=True)
dr = np.load(TOMO_PATH, allow_pickle=True)
sub = np.load(SUB_PATH, allow_pickle=True)
d2 = np.load(D2_PATH, allow_pickle=True)

bp_data = dk['bp_data']           # (15, 8)
cov_ee = dk['cov_ee']             # (120, 120)
ell_eff = dk['ell_eff']           # (8,)
pair_labels = dk['pair_labels']   # (15, 2)

A_tomo = dr['A_tomo']             # (15, 8, 500)
k_hMpc = dr['k_hMpc']            # (500,)
cl_lcdm_tomo = dr['cl_lcdm_tomo']  # (15, 8)

n_pairs = len(pair_labels)
n_ell = len(ell_eff)
n_data = n_pairs * n_ell  # 120

# Grid definition (matching Phase 3)
N_grid = sub['N_grid']               # [2, 3, 4, 6, 8, 10]
eps_inf_grid = sub['eps_inf_grid']   # [0.00, 0.25, 0.40, 0.58, 0.68, 0.74]
k_t_grid = sub['k_t_grid']          # 30 values, [0.1, 2.0]
alpha = 2

n_N = len(N_grid)
n_kt = len(k_t_grid)

print(f"  Data: {n_pairs} pairs x {n_ell} ell-bins = {n_data} points")
print(f"  Grid: {n_N} N x {n_kt} k_t = {n_N * n_kt} points")
print(f"  k_t range: [{k_t_grid[0]:.3f}, {k_t_grid[-1]:.3f}] h/Mpc")

# Flatten and invert covariance
data_flat = bp_data.flatten()
cov_inv = np.linalg.inv(cov_ee)
Cinv_d = cov_inv @ data_flat
dCd = data_flat @ Cinv_d


# ══════════════════════════════════════════════════════════════════
# PART 2: LCDM baseline
# ══════════════════════════════════════════════════════════════════
print("\n[2] LCDM baseline...")

m_lcdm_flat = cl_lcdm_tomo.flatten()
Cinv_m_lcdm = cov_inv @ m_lcdm_flat
dCm_lcdm = data_flat @ Cinv_m_lcdm
mCm_lcdm = m_lcdm_flat @ Cinv_m_lcdm

A_opt_lcdm = dCm_lcdm / mCm_lcdm
chi2_lcdm = dCd - dCm_lcdm**2 / mCm_lcdm

print(f"  chi^2_LCDM = {chi2_lcdm:.2f} (A_opt = {A_opt_lcdm:.1f})")


# ══════════════════════════════════════════════════════════════════
# PART 3: Compute chi2_WL on full (N, k_t) grid
# ══════════════════════════════════════════════════════════════════
print(f"\n[3] Computing chi^2_WL on {n_N}x{n_kt} grid...")
t0 = time.time()

chi2_wl_grid = np.zeros((n_N, n_kt))
A_opt_wl_grid = np.zeros((n_N, n_kt))
dchi2_wl_grid = np.zeros((n_N, n_kt))

for iN, (N_val, eps_inf) in enumerate(zip(N_grid, eps_inf_grid)):
    for ikt, k_t in enumerate(k_t_grid):
        # ISF transfer function M^2(k)
        eps_k = eps_inf / (1.0 + (k_t / k_hMpc)**alpha)
        M = (1.0 + (N_val - 1) * eps_k) / N_val
        M_sq = M**2

        # Model vector
        cl_isf = np.array([A_tomo[ip] @ M_sq for ip in range(n_pairs)])
        m_flat = cl_isf.flatten()

        # Profiled-amplitude chi^2
        Cinv_m = cov_inv @ m_flat
        dCm = data_flat @ Cinv_m
        mCm = m_flat @ Cinv_m

        if mCm > 0:
            A_opt_wl_grid[iN, ikt] = dCm / mCm
            chi2_wl_grid[iN, ikt] = dCd - dCm**2 / mCm
        else:
            chi2_wl_grid[iN, ikt] = dCd

        dchi2_wl_grid[iN, ikt] = chi2_wl_grid[iN, ikt] - chi2_lcdm

elapsed = time.time() - t0
print(f"  Computed in {elapsed:.3f}s")

# Cross-check N=6 row against D2
iN6 = np.where(N_grid == 6)[0][0]
# D2 used a different k_t grid (0.05-2.0), so check at the best-fit
idx_min_N6 = np.argmin(chi2_wl_grid[iN6])
print(f"\n  Cross-check (N=6):")
print(f"    Best k_t = {k_t_grid[idx_min_N6]:.4f}, "
      f"Dchi^2 = {dchi2_wl_grid[iN6, idx_min_N6]:.2f}")
print(f"    D2 best k_t^app = {float(d2['kt_best_apparent']):.4f}, "
      f"Dchi^2 = {float(d2['chi2_best_apparent']) - float(d2['chi2_lcdm']):.2f}")

# Per-N summary
print(f"\n  Per-N WL best fits:")
print(f"  {'N':>3s}  {'eps_inf':>7s}  {'k_t_best':>8s}  {'Dchi2_WL':>9s}  {'A_opt':>10s}")

kt_best_wl = np.zeros(n_N)
chi2_best_wl = np.zeros(n_N)

for iN in range(n_N):
    idx = np.argmin(chi2_wl_grid[iN])
    kt_best_wl[iN] = k_t_grid[idx]
    chi2_best_wl[iN] = chi2_wl_grid[iN, idx]
    dchi2 = chi2_best_wl[iN] - chi2_lcdm
    print(f"  {N_grid[iN]:3d}  {eps_inf_grid[iN]:7.2f}  "
          f"{kt_best_wl[iN]:8.4f}  {dchi2:+9.2f}  "
          f"{A_opt_wl_grid[iN, idx]:10.1f}")


# ══════════════════════════════════════════════════════════════════
# PART 4: Step 4a — Joint likelihood
# ══════════════════════════════════════════════════════════════════
print(f"\n[4] Step 4a: Joint likelihood...")

chi2_sub_grid = sub['chi2_sub_grid']   # (6, 30)
chi2_joint_grid = chi2_wl_grid + chi2_sub_grid
dchi2_joint_grid = chi2_joint_grid - chi2_lcdm  # sub adds 0 to LCDM

# Best-fit from joint
idx_min_joint = np.unravel_index(np.argmin(chi2_joint_grid),
                                  chi2_joint_grid.shape)
N_best_joint = N_grid[idx_min_joint[0]]
kt_best_joint = k_t_grid[idx_min_joint[1]]
chi2_best_joint = chi2_joint_grid[idx_min_joint]
dchi2_best_joint = chi2_best_joint - chi2_lcdm

print(f"  Joint best-fit: N={N_best_joint}, k_t={kt_best_joint:.4f} h/Mpc")
print(f"  chi^2_joint = {chi2_best_joint:.2f}")
print(f"  Dchi^2 vs LCDM = {dchi2_best_joint:.2f}")

# Per-N joint best fits
print(f"\n  Per-N joint best fits:")
print(f"  {'N':>3s}  {'k_t_best':>8s}  {'Dchi2_WL':>9s}  {'chi2_sub':>8s}  "
      f"{'Dchi2_joint':>11s}")

kt_best_joint_perN = np.zeros(n_N)
for iN in range(n_N):
    idx = np.argmin(chi2_joint_grid[iN])
    kt_best_joint_perN[iN] = k_t_grid[idx]
    dchi2_wl = chi2_wl_grid[iN, idx] - chi2_lcdm
    dchi2_j = chi2_joint_grid[iN, idx] - chi2_lcdm
    print(f"  {N_grid[iN]:3d}  {k_t_grid[idx]:8.4f}  {dchi2_wl:+9.2f}  "
          f"{chi2_sub_grid[iN, 0]:8.3f}  {dchi2_j:+11.2f}")


# ══════════════════════════════════════════════════════════════════
# PART 5: Step 4c — Parameter extraction (marginalisation)
# ══════════════════════════════════════════════════════════════════
print(f"\n[5] Step 4c: Parameter extraction...")

# Convert chi^2 to unnormalised log-likelihood
# L propto exp(-chi2/2), so lnL = -chi2/2
# For marginalisation, we work with relative likelihoods

# --- WL-only ---
dchi2_wl_from_min = chi2_wl_grid - chi2_wl_grid.min()
lnL_wl = -0.5 * dchi2_wl_from_min

# --- Sub-only ---
dchi2_sub_from_min = chi2_sub_grid - chi2_sub_grid.min()
lnL_sub = -0.5 * dchi2_sub_from_min

# --- Joint ---
dchi2_joint_from_min = chi2_joint_grid - chi2_joint_grid.min()
lnL_joint = -0.5 * dchi2_joint_from_min

# Convert to linear likelihood (relative)
L_wl = np.exp(lnL_wl)
L_sub = np.exp(lnL_sub)
L_joint = np.exp(lnL_joint)

# Marginalise over k_t to get P(N)
# Use trapezoidal rule in log(k_t) space
dlog_kt = np.diff(np.log(k_t_grid))
# Midpoint weights for trapezoidal rule
wt = np.zeros(n_kt)
wt[0] = dlog_kt[0] / 2
wt[-1] = dlog_kt[-1] / 2
wt[1:-1] = (dlog_kt[:-1] + dlog_kt[1:]) / 2

P_N_wl = np.array([np.sum(L_wl[iN] * wt) for iN in range(n_N)])
P_N_sub = np.array([np.sum(L_sub[iN] * wt) for iN in range(n_N)])
P_N_joint = np.array([np.sum(L_joint[iN] * wt) for iN in range(n_N)])

# Normalise
P_N_wl /= P_N_wl.sum()
P_N_sub /= P_N_sub.sum()
P_N_joint /= P_N_joint.sum()

print(f"\n  Marginalised P(N):")
print(f"  {'N':>3s}  {'P_WL':>7s}  {'P_sub':>7s}  {'P_joint':>7s}")
for iN in range(n_N):
    print(f"  {N_grid[iN]:3d}  {P_N_wl[iN]:7.4f}  {P_N_sub[iN]:7.4f}  "
          f"{P_N_joint[iN]:7.4f}")

# Marginalise over N to get P(k_t) — use flat prior on discrete N
P_kt_wl = np.sum(L_wl, axis=0)
P_kt_sub = np.sum(L_sub, axis=0)
P_kt_joint = np.sum(L_joint, axis=0)

P_kt_wl /= np.sum(P_kt_wl * wt)
P_kt_sub /= np.sum(P_kt_sub * wt)
P_kt_joint /= np.sum(P_kt_joint * wt)

# Find k_t confidence intervals from marginalised P(k_t)
def find_ci(kt_grid, P_kt, wt, level=0.68):
    """Find shortest interval containing `level` probability."""
    # Normalise
    norm = np.sum(P_kt * wt)
    p = P_kt * wt / norm
    # Sort by decreasing density
    idx_sorted = np.argsort(-P_kt)
    cumul = 0.0
    in_ci = np.zeros(len(kt_grid), dtype=bool)
    for i in idx_sorted:
        in_ci[i] = True
        cumul += p[i]
        if cumul >= level:
            break
    return kt_grid[in_ci].min(), kt_grid[in_ci].max()

kt_wl_68 = find_ci(k_t_grid, P_kt_wl, wt, 0.68)
kt_wl_95 = find_ci(k_t_grid, P_kt_wl, wt, 0.95)
kt_joint_68 = find_ci(k_t_grid, P_kt_joint, wt, 0.68)
kt_joint_95 = find_ci(k_t_grid, P_kt_joint, wt, 0.95)

# Peak of marginalised P(k_t)
kt_peak_wl = k_t_grid[np.argmax(P_kt_wl)]
kt_peak_joint = k_t_grid[np.argmax(P_kt_joint)]

print(f"\n  Marginalised k_t constraints:")
print(f"    WL-only:  peak = {kt_peak_wl:.3f}, "
      f"68% = [{kt_wl_68[0]:.3f}, {kt_wl_68[1]:.3f}], "
      f"95% = [{kt_wl_95[0]:.3f}, {kt_wl_95[1]:.3f}]")
print(f"    Joint:    peak = {kt_peak_joint:.3f}, "
      f"68% = [{kt_joint_68[0]:.3f}, {kt_joint_68[1]:.3f}], "
      f"95% = [{kt_joint_95[0]:.3f}, {kt_joint_95[1]:.3f}]")
print(f"    Sub-only: unconstrained (flat in k_t)")


# ══════════════════════════════════════════════════════════════════
# PART 6: Step 4d — Consistency check
# ══════════════════════════════════════════════════════════════════
print(f"\n[6] Step 4d: Consistency check...")

# Is N=2 disfavoured?
dchi2_N2_vs_best = (chi2_joint_grid[0].min() -
                     chi2_joint_grid.min())
print(f"\n  N=2 vs joint best-fit: Dchi^2 = +{dchi2_N2_vs_best:.2f}")
if dchi2_N2_vs_best > 4:
    print(f"  -> N=2 disfavoured at >2sigma")
elif dchi2_N2_vs_best > 1:
    print(f"  -> N=2 disfavoured at >1sigma")
else:
    print(f"  -> N=2 not significantly disfavoured")

# Is there tension between WL and sub?
# WL prefers: best-fit N from WL-only
idx_best_wl = np.unravel_index(np.argmin(chi2_wl_grid), chi2_wl_grid.shape)
N_best_wl = N_grid[idx_best_wl[0]]
# Sub prefers: best-fit N from sub-only
idx_best_sub_N = np.argmin(sub['chi2_sub'])
N_best_sub = N_grid[idx_best_sub_N]

print(f"\n  WL best-fit N: {N_best_wl}")
print(f"  Sub best-fit N: {N_best_sub}")
if N_best_wl == N_best_sub:
    print(f"  -> Agreement: both prefer N={N_best_wl}")
else:
    # Check overlap
    wl_within_1sig = set()
    for iN in range(n_N):
        if chi2_wl_grid[iN].min() - chi2_wl_grid.min() < 1.0:
            wl_within_1sig.add(N_grid[iN])
    sub_within_1sig = set()
    for iN in range(n_N):
        if sub['chi2_sub'][iN] - sub['chi2_sub'].min() < 1.0:
            sub_within_1sig.add(N_grid[iN])
    overlap = wl_within_1sig & sub_within_1sig
    print(f"  WL 1-sigma N set: {sorted(wl_within_1sig)}")
    print(f"  Sub 1-sigma N set: {sorted(sub_within_1sig)}")
    print(f"  Overlap: {sorted(overlap) if overlap else 'NONE (tension!)'}")

# Paper V consistency
print(f"\n  Paper V fiducial check:")
print(f"    Paper V: N=6, k_t=0.20")
# Find chi^2 at Paper V point (k_t=0.20 is below our grid minimum of 0.1)
# So evaluate it directly
kt_pv = 0.20
iN6 = np.where(N_grid == 6)[0][0]
eps_k_pv = eps_inf_grid[iN6] / (1.0 + (kt_pv / k_hMpc)**alpha)
M_pv = (1.0 + (N_grid[iN6] - 1) * eps_k_pv) / N_grid[iN6]
m_pv = np.array([A_tomo[ip] @ (M_pv**2) for ip in range(n_pairs)]).flatten()
Cinv_m_pv = cov_inv @ m_pv
dCm_pv = data_flat @ Cinv_m_pv
mCm_pv = m_pv @ Cinv_m_pv
chi2_pv = dCd - dCm_pv**2 / mCm_pv
dchi2_pv = chi2_pv - chi2_lcdm

# Paper VII fiducial: N=6, k_t=0.52
kt_p7 = 0.52
eps_k_p7 = eps_inf_grid[iN6] / (1.0 + (kt_p7 / k_hMpc)**alpha)
M_p7 = (1.0 + (N_grid[iN6] - 1) * eps_k_p7) / N_grid[iN6]
m_p7 = np.array([A_tomo[ip] @ (M_p7**2) for ip in range(n_pairs)]).flatten()
Cinv_m_p7 = cov_inv @ m_p7
dCm_p7 = data_flat @ Cinv_m_p7
mCm_p7 = m_p7 @ Cinv_m_p7
chi2_p7 = dCd - dCm_p7**2 / mCm_p7
dchi2_p7 = chi2_p7 - chi2_lcdm

print(f"    At Paper V (N=6, k_t=0.20): Dchi^2_WL = {dchi2_pv:+.2f}")
print(f"    At Paper VII (N=6, k_t=0.52): Dchi^2_WL = {dchi2_p7:+.2f}")
print(f"    At joint best (N={N_best_joint}, k_t={kt_best_joint:.2f}): "
      f"Dchi^2_joint = {dchi2_best_joint:+.2f}")
print(f"    Improvement from 0.20 -> 0.52: {dchi2_pv - dchi2_p7:+.2f}")
print(f"    Improvement from 0.20 -> best: {dchi2_pv - dchi2_best_joint:+.2f}")


# ══════════════════════════════════════════════════════════════════
# PART 7: Step 4f — Sensitivity table
# ══════════════════════════════════════════════════════════════════
print(f"\n[7] Step 4f: Sensitivity table...")

print(f"""
  ┌──────────────────────────┬──────────────┬──────────────┬──────────────┐
  │ Parameter                │   WL-only    │   Sub-only   │    Joint     │
  ├──────────────────────────┼──────────────┼──────────────┼──────────────┤
  │ k_t [h/Mpc] (68%)       │ {kt_wl_68[0]:.2f} - {kt_wl_68[1]:.2f}   │ unconstrained│ {kt_joint_68[0]:.2f} - {kt_joint_68[1]:.2f}   │
  │ k_t [h/Mpc] (95%)       │ {kt_wl_95[0]:.2f} - {kt_wl_95[1]:.2f}   │ unconstrained│ {kt_joint_95[0]:.2f} - {kt_joint_95[1]:.2f}   │
  │ N (best-fit)             │ {N_best_wl:>12d} │ {N_best_sub:>12d} │ {N_best_joint:>12d} │
  │ N=2 Dchi^2 vs best       │ {chi2_wl_grid[0].min()-chi2_wl_grid.min():+12.2f} │ {sub['chi2_sub'][0]-sub['chi2_sub'].min():+12.2f} │ {dchi2_N2_vs_best:+12.2f} │
  │ Dchi^2 vs LCDM (best)   │ {chi2_wl_grid.min()-chi2_lcdm:+12.2f} │     ---      │ {dchi2_best_joint:+12.2f} │
  │ P(N=6) marginalised     │ {P_N_wl[iN6]:12.4f} │ {P_N_sub[iN6]:12.4f} │ {P_N_joint[iN6]:12.4f} │
  └──────────────────────────┴──────────────┴──────────────┴──────────────┘
""")


# ══════════════════════════════════════════════════════════════════
# PART 8: Summary
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
  Joint best-fit: N={N_best_joint}, k_t={kt_best_joint:.4f} h/Mpc
  Dchi^2 vs LCDM: {dchi2_best_joint:.2f}
  Significance:   ~{np.sqrt(abs(dchi2_best_joint)):.1f} sigma

  WL provides k_t constraint:
    Peak: {kt_peak_wl:.3f} h/Mpc
    68%:  [{kt_wl_68[0]:.3f}, {kt_wl_68[1]:.3f}]

  Sub provides N discrimination:
    Best N: {N_best_sub} (soft constraint, max Dchi^2_sub = {sub['chi2_sub'].max():.2f})
    N=2 disfavoured by Dchi^2 = {dchi2_N2_vs_best:.2f} in joint

  Joint marginalised P(N):""")
for iN in range(n_N):
    bar = '#' * int(P_N_joint[iN] * 50)
    print(f"    N={N_grid[iN]:2d}: {P_N_joint[iN]:.4f} {bar}")

print(f"""
  Consistency:
    WL best-fit N = {N_best_wl}, Sub best-fit N = {N_best_sub}
    Paper V  (N=6, k_t=0.20): Dchi^2_WL = {dchi2_pv:+.2f}
    Paper VII (N=6, k_t=0.52): Dchi^2_WL = {dchi2_p7:+.2f}
    WL and sub contours are {'consistent' if N_best_wl >= 4 and N_best_sub >= 4 else 'in tension'}.

  For Phase 5:
    - Euclid forecast should use joint best-fit as fiducial
    - Paper V fiducial at k_t=0.20 is superseded
    - k_t^phys = {kt_peak_joint:.2f} h/Mpc is the primary WL result
""")


# ══════════════════════════════════════════════════════════════════
# PART 9: Save checkpoint
# ══════════════════════════════════════════════════════════════════
print("[9] Saving checkpoint...")

np.savez(
    OUTPATH,
    # Grid axes
    N_grid=N_grid,
    eps_inf_grid=eps_inf_grid,
    k_t_grid=k_t_grid,
    alpha=alpha,
    # WL likelihood (full grid)
    chi2_wl_grid=chi2_wl_grid,         # (6, 30)
    A_opt_wl_grid=A_opt_wl_grid,       # (6, 30)
    dchi2_wl_grid=dchi2_wl_grid,       # (6, 30)
    chi2_lcdm=chi2_lcdm,
    A_opt_lcdm=A_opt_lcdm,
    # Sub likelihood
    chi2_sub_grid=chi2_sub_grid,       # (6, 30)
    f_GGSL=sub['f_GGSL'],
    # Joint
    chi2_joint_grid=chi2_joint_grid,   # (6, 30)
    dchi2_joint_grid=dchi2_joint_grid,  # (6, 30)
    # Best fits
    N_best_joint=N_best_joint,
    kt_best_joint=kt_best_joint,
    chi2_best_joint=chi2_best_joint,
    dchi2_best_joint=dchi2_best_joint,
    # Marginalised posteriors
    P_N_wl=P_N_wl,
    P_N_sub=P_N_sub,
    P_N_joint=P_N_joint,
    P_kt_wl=P_kt_wl,
    P_kt_joint=P_kt_joint,
    # k_t confidence intervals
    kt_wl_68=np.array(kt_wl_68),
    kt_wl_95=np.array(kt_wl_95),
    kt_joint_68=np.array(kt_joint_68),
    kt_joint_95=np.array(kt_joint_95),
    kt_peak_wl=kt_peak_wl,
    kt_peak_joint=kt_peak_joint,
    # Reference points
    chi2_paper_v=chi2_pv,
    dchi2_paper_v=dchi2_pv,
    chi2_paper_vii=chi2_p7,
    dchi2_paper_vii=dchi2_p7,
    # Metadata
    n_data=n_data,
    ell_eff=ell_eff,
)

print(f"  Saved to {OUTPATH}")
print(f"  Grid shapes: chi2_wl{chi2_wl_grid.shape}, "
      f"chi2_joint{chi2_joint_grid.shape}")
print("\nPhase 4 complete.")
