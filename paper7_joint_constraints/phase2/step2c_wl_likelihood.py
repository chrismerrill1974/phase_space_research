#!/usr/bin/env python3
"""
Paper VII - Phase 2, Step 2c
Weak Lensing Likelihood Construction

Computes χ²_WL(N, k_t) on the 6×20 ISF parameter grid by comparing
ISF-predicted tomographic band powers to KiDS-1000 data.

Method: profiled-amplitude χ² (analytically marginalises over an
overall calibration amplitude A):

    χ²_prof = d·C⁻¹·d − (d·C⁻¹·m)² / (m·C⁻¹·m)

where d = data vector (120 band powers), m = ISF model vector,
C = 120×120 covariance.  The best-fit amplitude is:

    A_opt = (d·C⁻¹·m) / (m·C⁻¹·m)

This absorbs:
  • Planck-vs-KiDS cosmology offset (S₈ tension)
  • Band power estimator window function effects
  • Any residual calibration factors

The SHAPE information (ℓ-dependence within and across bin pairs)
constrains k_t, while Δχ² ≡ χ²(ISF) − χ²(ΛCDM) quantifies the
improvement from ISF's scale-dependent suppression.

Depends on:
  data/kids1000_bandpowers.npz  (Step 2a)
  data/response_operators_tomo.npz  (Step 2b)

Output: data/chi2_wl_grid.npz
"""

import numpy as np
import os
import time

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

OUTPATH = os.path.join(DATA_DIR, 'chi2_wl_grid.npz')

# ══════════════════════════════════════════════════════════════════
# PART 1: Load data and response operators
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("Step 2c: Weak Lensing Likelihood on (N, k_t) Grid")
print("=" * 70)

print("\n[1] Loading data...")

dk = np.load(os.path.join(DATA_DIR, 'kids1000_bandpowers.npz'),
             allow_pickle=True)
dr = np.load(os.path.join(DATA_DIR, 'response_operators_tomo.npz'),
             allow_pickle=True)

bp_data = dk['bp_data']           # (15, 8) band power values
ell_eff = dk['ell_eff']           # (8,) effective ℓ
cov_ee = dk['cov_ee']             # (120, 120) covariance
pair_labels = dk['pair_labels']   # (15, 2)

A_tomo = dr['A_tomo']             # (15, 8, 500) response operators
k_hMpc = dr['k_hMpc']            # (500,) k-grid in h/Mpc
cl_lcdm_tomo = dr['cl_lcdm_tomo']  # (15, 8) ΛCDM predictions

n_pairs = len(pair_labels)
n_ell = len(ell_eff)
n_k = len(k_hMpc)
n_data = n_pairs * n_ell  # 120

print(f"  Data: {n_pairs} pairs × {n_ell} ℓ-bins = {n_data} points")
print(f"  k-grid: {n_k} points, [{k_hMpc[0]:.4f}, {k_hMpc[-1]:.1f}] h/Mpc")

# ── Flatten data vector ────────────────────────────────────────────
data_flat = bp_data.flatten()  # (120,)

# ── Invert covariance ─────────────────────────────────────────────
print("\n  Computing C⁻¹...")
cov_inv = np.linalg.inv(cov_ee)

# Verify inversion quality
identity_check = cov_ee @ cov_inv
max_off_diag = np.max(np.abs(
    identity_check - np.eye(n_data)))
print(f"  Inversion check: max|C·C⁻¹ - I| = {max_off_diag:.2e}")

# Precompute d·C⁻¹·d (constant for all models)
Cinv_d = cov_inv @ data_flat
dCd = data_flat @ Cinv_d
print(f"  d·C⁻¹·d = {dCd:.2f}")

# ══════════════════════════════════════════════════════════════════
# PART 2: Define ISF parameter grid
# ══════════════════════════════════════════════════════════════════
print("\n[2] Setting up ISF parameter grid...")

# N grid and corresponding ε∞ (from Paper V S₈ matching)
N_grid = np.array([2, 3, 4, 6, 8, 10])
eps_inf_grid = np.array([0.00, 0.25, 0.40, 0.58, 0.68, 0.74])
alpha = 2  # fixed

# k_t grid: 30 log-spaced values from 0.05 to 2.0 h/Mpc
# Extended beyond the Paper V 0.05-0.50 range after initial scan
# showed the minimum lies near k_t ~ 0.8 h/Mpc.
k_t_grid = np.logspace(np.log10(0.05), np.log10(2.0), 30)

n_N = len(N_grid)
n_kt = len(k_t_grid)

print(f"  N grid: {N_grid}")
print(f"  ε∞ grid: {eps_inf_grid}")
print(f"  k_t grid: {n_kt} points, [{k_t_grid[0]:.3f}, "
      f"{k_t_grid[-1]:.3f}] h/Mpc")
print(f"  Total grid points: {n_N * n_kt}")

# ══════════════════════════════════════════════════════════════════
# PART 3: Compute ΛCDM baseline χ²
# ══════════════════════════════════════════════════════════════════
print("\n[3] Computing ΛCDM baseline...")

m_lcdm_flat = cl_lcdm_tomo.flatten()
Cinv_m_lcdm = cov_inv @ m_lcdm_flat
dCm_lcdm = data_flat @ Cinv_m_lcdm
mCm_lcdm = m_lcdm_flat @ Cinv_m_lcdm

A_opt_lcdm = dCm_lcdm / mCm_lcdm
chi2_lcdm = dCd - dCm_lcdm**2 / mCm_lcdm

print(f"  ΛCDM profiled amplitude: A_opt = {A_opt_lcdm:.1f}")
print(f"  ΛCDM profiled χ² = {chi2_lcdm:.2f}")
print(f"  Ndof = {n_data} - 1 = {n_data - 1}")
print(f"  χ²/Ndof = {chi2_lcdm / (n_data - 1):.3f}")

# ══════════════════════════════════════════════════════════════════
# PART 4: Evaluate χ² on the full (N, k_t) grid
# ══════════════════════════════════════════════════════════════════
print(f"\n[4] Computing χ² on {n_N}×{n_kt} grid...")
t0 = time.time()

# Storage arrays
chi2_grid = np.zeros((n_N, n_kt))
A_opt_grid = np.zeros((n_N, n_kt))
delta_chi2_grid = np.zeros((n_N, n_kt))

for iN, (N_val, eps_inf) in enumerate(zip(N_grid, eps_inf_grid)):
    for ikt, k_t in enumerate(k_t_grid):
        # ── ISF transfer function M²(k) ───────────────────
        eps_k = eps_inf / (1.0 + (k_t / k_hMpc)**alpha)
        M = (1.0 + (N_val - 1) * eps_k) / N_val
        M_sq = M**2

        # ── Model vector: C_ij^ISF(ℓ) for all pairs ──────
        cl_isf = np.array([A_tomo[ip] @ M_sq
                           for ip in range(n_pairs)])
        m_flat = cl_isf.flatten()

        # ── Profiled-amplitude χ² ─────────────────────────
        Cinv_m = cov_inv @ m_flat
        dCm = data_flat @ Cinv_m
        mCm = m_flat @ Cinv_m

        if mCm > 0:
            A_opt = dCm / mCm
            chi2_prof = dCd - dCm**2 / mCm
        else:
            A_opt = 0.0
            chi2_prof = dCd

        chi2_grid[iN, ikt] = chi2_prof
        A_opt_grid[iN, ikt] = A_opt
        delta_chi2_grid[iN, ikt] = chi2_prof - chi2_lcdm

elapsed = time.time() - t0
print(f"  Grid computed in {elapsed:.3f}s "
      f"({elapsed/(n_N*n_kt)*1e6:.0f} μs per model)")

# ── Refinement: fine grid around the minimum for each N ─────────
print("\n  Refining around minimum for each N...")

kt_fine_best = np.zeros(n_N)
chi2_fine_best = np.zeros(n_N)

for iN, (N_val, eps_inf) in enumerate(zip(N_grid, eps_inf_grid)):
    if eps_inf == 0:
        # N=2 with eps=0 is ΛCDM everywhere → skip
        kt_fine_best[iN] = k_t_grid[0]
        chi2_fine_best[iN] = chi2_grid[iN, 0]
        continue

    # Find coarse minimum
    ikt_min = np.argmin(chi2_grid[iN])
    kt_lo = k_t_grid[max(0, ikt_min - 2)]
    kt_hi = k_t_grid[min(n_kt - 1, ikt_min + 2)]

    # Also check if minimum is at boundary
    if ikt_min == n_kt - 1:
        kt_hi = k_t_grid[-1] * 2.0  # extend further
    if ikt_min == 0:
        kt_lo = k_t_grid[0] * 0.5

    kt_fine = np.linspace(kt_lo, kt_hi, 50)
    chi2_fine = np.zeros(50)

    for j, kt in enumerate(kt_fine):
        eps_k = eps_inf / (1.0 + (kt / k_hMpc)**alpha)
        M = (1.0 + (N_val - 1) * eps_k) / N_val
        M_sq = M**2
        cl_isf = np.array([A_tomo[ip] @ M_sq
                           for ip in range(n_pairs)])
        m_flat = cl_isf.flatten()
        Cinv_m = cov_inv @ m_flat
        dCm = data_flat @ Cinv_m
        mCm = m_flat @ Cinv_m
        chi2_fine[j] = dCd - dCm**2 / mCm

    idx = np.argmin(chi2_fine)
    kt_fine_best[iN] = kt_fine[idx]
    chi2_fine_best[iN] = chi2_fine[idx]
    print(f"    N={N_val}: k_t = {kt_fine[idx]:.4f} h/Mpc, "
          f"χ² = {chi2_fine[idx]:.2f}")

# ══════════════════════════════════════════════════════════════════
# PART 5: Find best-fit and report results
# ══════════════════════════════════════════════════════════════════
print("\n[5] Results:")
print("-" * 60)

# Best fit overall (from coarse grid)
idx_min = np.unravel_index(np.argmin(chi2_grid), chi2_grid.shape)
N_best = N_grid[idx_min[0]]
kt_best = kt_fine_best[idx_min[0]]  # use refined value
eps_best = eps_inf_grid[idx_min[0]]
chi2_best = chi2_fine_best[idx_min[0]]
A_best = A_opt_grid[idx_min]
dchi2_best = chi2_best - chi2_lcdm

# Recompute A_opt at refined best fit
eps_k_bf = eps_best / (1.0 + (kt_best / k_hMpc)**alpha)
M_bf = (1.0 + (N_best - 1) * eps_k_bf) / N_best
m_bf = np.array([A_tomo[ip] @ (M_bf**2)
                  for ip in range(n_pairs)]).flatten()
Cinv_m_bf = cov_inv @ m_bf
A_best = (data_flat @ Cinv_m_bf) / (m_bf @ Cinv_m_bf)

print(f"\n  Global best fit (refined):")
print(f"    N = {N_best}, ε∞ = {eps_best:.2f}, "
      f"k_t = {kt_best:.4f} h/Mpc")
print(f"    χ² = {chi2_best:.2f} (Δχ² vs ΛCDM = {dchi2_best:.2f})")
print(f"    A_opt = {A_best:.1f}")
print(f"    χ²/Ndof = {chi2_best/(n_data - 1):.3f}")

# Best fit at fiducial N=6 (refined)
iN6 = np.where(N_grid == 6)[0][0]
kt_best_N6 = kt_fine_best[iN6]
chi2_best_N6 = chi2_fine_best[iN6]
dchi2_best_N6 = chi2_best_N6 - chi2_lcdm

# Recompute A_opt at N=6 refined best fit
eps_k_N6 = 0.58 / (1.0 + (kt_best_N6 / k_hMpc)**alpha)
M_N6 = (1.0 + 5 * eps_k_N6) / 6.0
m_N6 = np.array([A_tomo[ip] @ (M_N6**2)
                  for ip in range(n_pairs)]).flatten()
Cinv_m_N6 = cov_inv @ m_N6
A_best_N6 = (data_flat @ Cinv_m_N6) / (m_N6 @ Cinv_m_N6)

print(f"\n  Best fit at N=6 (refined):")
print(f"    k_t = {kt_best_N6:.4f} h/Mpc")
print(f"    χ² = {chi2_best_N6:.2f} (Δχ² vs ΛCDM = {dchi2_best_N6:.2f})")
print(f"    A_opt = {A_best_N6:.1f}")

# Paper V comparison: k_t = 0.20 h/Mpc
print(f"\n  Paper V reference: k_t = 0.20 h/Mpc")
print(f"  Best-fit k_t shift: {kt_best_N6:.4f} → "
      f"Δk_t/k_t = {(kt_best_N6 - 0.20)/0.20 * 100:+.1f}%")

if abs(kt_best_N6 - 0.20) / 0.20 < 0.20:
    print(f"  ✓ Shift <20% — validates Paper V approximation")
elif abs(kt_best_N6 - 0.20) / 0.20 < 0.50:
    print(f"  ~ Shift 20-50% — moderate, worth noting")
else:
    print(f"  ⚠ Shift >50% — requires investigation (see §7 below)")

# ── Per-N summary table (refined) ─────────────────────────────
print(f"\n  Per-N best fits (refined):")
print(f"  {'N':>3s}  {'ε∞':>5s}  {'k_t_best':>10s}  "
      f"{'χ²':>8s}  {'Δχ²':>8s}")
for iN in range(n_N):
    print(f"  {N_grid[iN]:3d}  {eps_inf_grid[iN]:5.2f}  "
          f"{kt_fine_best[iN]:10.4f}  "
          f"{chi2_fine_best[iN]:8.2f}  "
          f"{chi2_fine_best[iN] - chi2_lcdm:8.2f}")

# ── Δχ² as function of k_t for fiducial N=6 ───────────────────
print(f"\n  Δχ²(k_t) at N=6 (coarse grid):")
print(f"  {'k_t':>8s}  {'χ²':>8s}  {'Δχ²':>8s}  {'A_opt':>10s}")
for ikt in range(n_kt):
    print(f"  {k_t_grid[ikt]:8.4f}  "
          f"{chi2_grid[iN6, ikt]:8.2f}  "
          f"{delta_chi2_grid[iN6, ikt]:8.2f}  "
          f"{A_opt_grid[iN6, ikt]:10.1f}")

# ── Confidence intervals at N=6 ───────────────────────────────
# Use refined minimum. Δχ² < 1 → 1σ, < 4 → 2σ
# Need a dense scan around minimum for proper intervals
kt_ci_scan = np.logspace(np.log10(0.05), np.log10(2.0), 200)
chi2_ci_scan = np.zeros(200)
for j, kt in enumerate(kt_ci_scan):
    eps_k = 0.58 / (1.0 + (kt / k_hMpc)**alpha)
    M = (1.0 + 5 * eps_k) / 6.0
    m = np.array([A_tomo[ip] @ (M**2)
                   for ip in range(n_pairs)]).flatten()
    Cinv_m = cov_inv @ m
    dCm = data_flat @ Cinv_m
    mCm = m @ Cinv_m
    chi2_ci_scan[j] = dCd - dCm**2 / mCm

chi2_min_ci = chi2_ci_scan.min()
mask_1sig = (chi2_ci_scan - chi2_min_ci) < 1.0
mask_2sig = (chi2_ci_scan - chi2_min_ci) < 4.0

print(f"\n  N=6 confidence intervals (from Δχ² profile):")
if np.any(mask_1sig):
    kt_1sig_lo = kt_ci_scan[mask_1sig].min()
    kt_1sig_hi = kt_ci_scan[mask_1sig].max()
    print(f"    1σ: k_t ∈ [{kt_1sig_lo:.3f}, {kt_1sig_hi:.3f}] h/Mpc")
if np.any(mask_2sig):
    kt_2sig_lo = kt_ci_scan[mask_2sig].min()
    kt_2sig_hi = kt_ci_scan[mask_2sig].max()
    print(f"    2σ: k_t ∈ [{kt_2sig_lo:.3f}, {kt_2sig_hi:.3f}] h/Mpc")

# ══════════════════════════════════════════════════════════════════
# PART 6: Diagnostic — ISF/ΛCDM suppression at best fit
# ══════════════════════════════════════════════════════════════════
print(f"\n[6] Diagnostic: ISF suppression at best fit "
      f"(N={N_best}, k_t={kt_best:.4f}):")

eps_k_best = eps_best / (1.0 + (kt_best / k_hMpc)**alpha)
M_best = (1.0 + (N_best - 1) * eps_k_best) / N_best
M_sq_best = M_best**2

cl_isf_best = np.array([A_tomo[ip] @ M_sq_best
                         for ip in range(n_pairs)])
ratio_best = cl_isf_best / cl_lcdm_tomo

print(f"  M²(k→0) = {M_sq_best[0]:.4f} (= 1/N²)")
print(f"  M²(k→∞) = {M_sq_best[-1]:.4f} (= [(1+(N-1)ε∞)/N]²)")
print(f"\n  ISF/ΛCDM C_ℓ ratio at selected pairs and ℓ-bins:")
print(f"  {'Pair':>8s}  {'ℓ=118':>7s}  {'ℓ=327':>7s}  "
      f"{'ℓ=644':>7s}  {'ℓ=1266':>7s}")
for ip in [0, 5, 9, 12, 13, 14]:  # (1,1),(2,2),(3,3),(4,4),(4,5),(5,5)
    b1, b2 = pair_labels[ip]
    print(f"  ({b1},{b2}){'':<3s}  {ratio_best[ip,0]:7.4f}  "
          f"{ratio_best[ip,3]:7.4f}  {ratio_best[ip,5]:7.4f}  "
          f"{ratio_best[ip,7]:7.4f}")

# ══════════════════════════════════════════════════════════════════
# PART 7: Decision gate assessment
# ══════════════════════════════════════════════════════════════════
print(f"\n[7] Decision gate assessment:")
print("-" * 60)

kt_shift = abs(kt_best_N6 - 0.20) / 0.20

print(f"  Paper V value: k_t = 0.20 h/Mpc")
print(f"  Paper VII value (N=6): k_t = {kt_best_N6:.4f} h/Mpc")
print(f"  Fractional shift: {kt_shift*100:.1f}%")

if kt_shift < 0.20:
    gate = "PASS"
    print(f"  Decision: ✓ {gate} — <20% shift, Paper V validated")
elif kt_shift < 0.50:
    gate = "MARGINAL"
    print(f"  Decision: ~ {gate} — 20-50% shift, note in paper")
elif kt_shift < 2.0:
    gate = "INVESTIGATE"
    print(f"  Decision: ⚠ {gate} — >50% shift, investigate cause")
else:
    gate = "FAIL"
    print(f"  Decision: ✗ {gate} — >2× shift, ISF sigmoid "
          f"may not survive proper projection")

print(f"\n  Δχ²(best ISF vs ΛCDM) = {dchi2_best:.2f}")
if dchi2_best < -9:
    print(f"  ISF preferred at >{np.sqrt(abs(dchi2_best)):.1f}σ significance")
elif dchi2_best < -4:
    print(f"  ISF preferred at ~{np.sqrt(abs(dchi2_best)):.1f}σ significance")
elif dchi2_best < -1:
    print(f"  Mild ISF preference (~{np.sqrt(abs(dchi2_best)):.1f}σ)")
else:
    print(f"  No significant ISF preference (Δχ² ≈ {dchi2_best:.1f})")

# ══════════════════════════════════════════════════════════════════
# PART 8: Save checkpoint
# ══════════════════════════════════════════════════════════════════
print(f"\n[8] Saving checkpoint...")

np.savez(
    OUTPATH,
    # Parameter grid
    N_grid=N_grid,
    eps_inf_grid=eps_inf_grid,
    k_t_grid=k_t_grid,
    alpha=alpha,
    # χ² results
    chi2_grid=chi2_grid,               # (6, 20) profiled χ²
    A_opt_grid=A_opt_grid,             # (6, 20) profiled amplitudes
    delta_chi2_grid=delta_chi2_grid,   # (6, 20) Δχ² vs ΛCDM
    chi2_lcdm=chi2_lcdm,              # scalar
    A_opt_lcdm=A_opt_lcdm,            # scalar
    # Best-fit results
    N_best=N_best,
    kt_best=kt_best,
    eps_best=eps_best,
    chi2_best=chi2_best,
    # Best-fit at N=6
    kt_best_N6=kt_best_N6,
    chi2_best_N6=chi2_best_N6,
    # Data info
    n_data=n_data,
    ell_eff=ell_eff,
    pair_labels=pair_labels,
    # Decision gate
    gate=gate,
    kt_shift_frac=kt_shift,
)

print(f"  ✓ Saved to {OUTPATH}")
print(f"  Grid shape: {chi2_grid.shape}")

# ══════════════════════════════════════════════════════════════════
# PART 9: Summary
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
Weak lensing likelihood computed on {n_N}×{n_kt} = {n_N*n_kt} grid.

Method: profiled-amplitude χ² (free overall amplitude A)
  Absorbs S₈ tension + band power calibration.
  Shape information constrains k_t.

ΛCDM baseline: χ² = {chi2_lcdm:.2f}, χ²/Ndof = {chi2_lcdm/(n_data-1):.3f}

Global best fit: N={N_best}, k_t={kt_best:.4f} h/Mpc
  χ² = {chi2_best:.2f}, Δχ² = {dchi2_best:.2f}

Best fit at N=6: k_t = {kt_best_N6:.4f} h/Mpc
  Δk_t/k_t = {(kt_best_N6-0.20)/0.20*100:+.1f}% from Paper V
  Decision gate: {gate}

Next: Step 2d — Diagnostic plot + detailed validation
""")

print("Script complete.")
