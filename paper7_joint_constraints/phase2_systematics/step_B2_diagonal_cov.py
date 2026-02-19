#!/usr/bin/env python3
"""
Phase B, Step B2: Repeat chi^2 grid with diagonal-only covariance

The question: are the off-diagonal covariance elements driving the
k_t preference? If k_t changes substantially with diagonal-only
errors, the off-diagonal terms are suspect.

We re-run the profiled-amplitude chi^2 on the (N, k_t) grid using:
  (a) Full covariance (reproduces Step 2c)
  (b) Diagonal-only covariance
  (c) Block-diagonal covariance (correlations within each pair, but
      not between pairs)

Then compare best-fit k_t, Delta_chi^2, and confidence intervals.

Run: python3 step_B2_diagonal_cov.py

Depends on:
  ../../data/kids1000_bandpowers.npz
  ../../data/response_operators_tomo.npz
"""

import numpy as np
import os
import time

# ── Paths ─────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
DATA_DIR = os.path.join(REPO_ROOT, 'data')

KIDS_PATH = os.path.join(DATA_DIR, 'kids1000_bandpowers.npz')
TOMO_PATH = os.path.join(DATA_DIR, 'response_operators_tomo.npz')

for label, path in [('KiDS data', KIDS_PATH), ('Tomo operators', TOMO_PATH)]:
    if not os.path.isfile(path):
        print(f"ERROR: {label} not found at:\n  {path}")
        raise SystemExit(1)

# ── Load ──────────────────────────────────────────────────────────
print("=" * 70)
print("Step B2: Diagonal covariance test")
print("=" * 70)

dk = np.load(KIDS_PATH, allow_pickle=True)
dr = np.load(TOMO_PATH, allow_pickle=True)

bp_data = dk['bp_data']             # (15, 8)
ell_eff = dk['ell_eff']             # (8,)
cov_ee = dk['cov_ee']               # (120, 120)
pair_labels = dk['pair_labels']     # (15, 2)

A_tomo = dr['A_tomo']               # (15, 8, 500)
k_hMpc = dr['k_hMpc']              # (500,)
cl_lcdm_tomo = dr['cl_lcdm_tomo']  # (15, 8)

n_pairs = len(pair_labels)
n_ell = len(ell_eff)
n_data = n_pairs * n_ell
n_k = len(k_hMpc)

data_flat = bp_data.flatten()

# ── ISF parameter grid (same as Step 2c) ─────────────────────────
N_grid = np.array([2, 3, 4, 6, 8, 10])
eps_inf_grid = np.array([0.00, 0.25, 0.40, 0.58, 0.68, 0.74])
alpha = 2
k_t_grid = np.logspace(np.log10(0.05), np.log10(2.0), 30)
n_N = len(N_grid)
n_kt = len(k_t_grid)

# ── Build three covariance variants ──────────────────────────────
print("\n[1] Building covariance variants...")

# (a) Full covariance
cov_full = cov_ee.copy()

# (b) Diagonal only
cov_diag = np.diag(np.diag(cov_ee))

# (c) Block-diagonal: 15 blocks of 8x8 (one per pair)
cov_block = np.zeros_like(cov_ee)
for ip in range(n_pairs):
    i0 = ip * n_ell
    i1 = (ip + 1) * n_ell
    cov_block[i0:i1, i0:i1] = cov_ee[i0:i1, i0:i1]

# Check properties
for label, cov in [('Full', cov_full), ('Diagonal', cov_diag),
                    ('Block-diagonal', cov_block)]:
    eigvals = np.linalg.eigvalsh(cov)
    n_nonzero = np.count_nonzero(cov)
    print(f"  {label:16s}: non-zero={n_nonzero:6d}/{n_data**2}, "
          f"min_eig={eigvals.min():.2e}, cond={eigvals.max()/eigvals.min():.2e}")

# Invert all three
cov_inv_full = np.linalg.inv(cov_full)
cov_inv_diag = np.linalg.inv(cov_diag)
cov_inv_block = np.linalg.inv(cov_block)


# ══════════════════════════════════════════════════════════════════
# Profiled chi^2 computation function
# ══════════════════════════════════════════════════════════════════
def compute_grid(cov_inv, label):
    """Run the full (N, k_t) grid with a given inverse covariance."""
    Cinv_d = cov_inv @ data_flat
    dCd = data_flat @ Cinv_d

    # LCDM baseline
    m_lcdm = cl_lcdm_tomo.flatten()
    Cinv_m_lcdm = cov_inv @ m_lcdm
    dCm_lcdm = data_flat @ Cinv_m_lcdm
    mCm_lcdm = m_lcdm @ Cinv_m_lcdm
    A_opt_lcdm = dCm_lcdm / mCm_lcdm
    chi2_lcdm = dCd - dCm_lcdm**2 / mCm_lcdm

    # Grid
    chi2_grid = np.zeros((n_N, n_kt))
    A_opt_grid = np.zeros((n_N, n_kt))

    for iN, (N_val, eps_inf) in enumerate(zip(N_grid, eps_inf_grid)):
        for ikt, k_t in enumerate(k_t_grid):
            eps_k = eps_inf / (1.0 + (k_t / k_hMpc)**alpha)
            M = (1.0 + (N_val - 1) * eps_k) / N_val
            M_sq = M**2

            cl_isf = np.array([A_tomo[ip] @ M_sq for ip in range(n_pairs)])
            m_flat = cl_isf.flatten()

            Cinv_m = cov_inv @ m_flat
            dCm = data_flat @ Cinv_m
            mCm = m_flat @ Cinv_m

            if mCm > 0:
                A_opt_grid[iN, ikt] = dCm / mCm
                chi2_grid[iN, ikt] = dCd - dCm**2 / mCm
            else:
                A_opt_grid[iN, ikt] = 0.0
                chi2_grid[iN, ikt] = dCd

    delta_chi2 = chi2_grid - chi2_lcdm

    # Fine-grid refinement for N=6
    iN6 = np.where(N_grid == 6)[0][0]
    ikt_min = np.argmin(chi2_grid[iN6])
    kt_lo = k_t_grid[max(0, ikt_min - 2)]
    kt_hi = k_t_grid[min(n_kt - 1, ikt_min + 2)]
    if ikt_min == n_kt - 1:
        kt_hi = k_t_grid[-1] * 2.0
    if ikt_min == 0:
        kt_lo = k_t_grid[0] * 0.5

    kt_fine = np.linspace(kt_lo, kt_hi, 100)
    chi2_fine = np.zeros(100)
    for j, kt in enumerate(kt_fine):
        eps_k = 0.58 / (1.0 + (kt / k_hMpc)**alpha)
        M = (1.0 + 5 * eps_k) / 6.0
        m = np.array([A_tomo[ip] @ (M**2) for ip in range(n_pairs)]).flatten()
        Cinv_m = cov_inv @ m
        dCm = data_flat @ Cinv_m
        mCm = m @ Cinv_m
        chi2_fine[j] = dCd - dCm**2 / mCm

    idx_fine = np.argmin(chi2_fine)
    kt_best_N6 = kt_fine[idx_fine]
    chi2_best_N6 = chi2_fine[idx_fine]

    # Confidence intervals
    kt_ci = np.logspace(np.log10(0.05), np.log10(2.0), 200)
    chi2_ci = np.zeros(200)
    for j, kt in enumerate(kt_ci):
        eps_k = 0.58 / (1.0 + (kt / k_hMpc)**alpha)
        M = (1.0 + 5 * eps_k) / 6.0
        m = np.array([A_tomo[ip] @ (M**2) for ip in range(n_pairs)]).flatten()
        Cinv_m = cov_inv @ m
        dCm = data_flat @ Cinv_m
        mCm = m @ Cinv_m
        chi2_ci[j] = dCd - dCm**2 / mCm

    chi2_min_ci = chi2_ci.min()
    mask_1sig = (chi2_ci - chi2_min_ci) < 1.0
    mask_2sig = (chi2_ci - chi2_min_ci) < 4.0

    kt_1sig = (kt_ci[mask_1sig].min(), kt_ci[mask_1sig].max()) if np.any(mask_1sig) else (0, 0)
    kt_2sig = (kt_ci[mask_2sig].min(), kt_ci[mask_2sig].max()) if np.any(mask_2sig) else (0, 0)

    # A_opt at best fit
    eps_k_bf = 0.58 / (1.0 + (kt_best_N6 / k_hMpc)**alpha)
    M_bf = (1.0 + 5 * eps_k_bf) / 6.0
    m_bf = np.array([A_tomo[ip] @ (M_bf**2) for ip in range(n_pairs)]).flatten()
    Cinv_m_bf = cov_inv @ m_bf
    A_opt_bf = (data_flat @ Cinv_m_bf) / (m_bf @ Cinv_m_bf)

    return {
        'label': label,
        'chi2_lcdm': chi2_lcdm,
        'A_opt_lcdm': A_opt_lcdm,
        'chi2_grid': chi2_grid,
        'A_opt_grid': A_opt_grid,
        'delta_chi2': delta_chi2,
        'kt_best_N6': kt_best_N6,
        'chi2_best_N6': chi2_best_N6,
        'dchi2_best_N6': chi2_best_N6 - chi2_lcdm,
        'A_opt_bf': A_opt_bf,
        'kt_1sig': kt_1sig,
        'kt_2sig': kt_2sig,
    }


# ══════════════════════════════════════════════════════════════════
# Run all three variants
# ══════════════════════════════════════════════════════════════════
print("\n[2] Running chi^2 grids...")

t0 = time.time()
res_full = compute_grid(cov_inv_full, 'Full')
t1 = time.time()
print(f"  Full covariance: {t1-t0:.2f}s")

res_diag = compute_grid(cov_inv_diag, 'Diagonal')
t2 = time.time()
print(f"  Diagonal covariance: {t2-t1:.2f}s")

res_block = compute_grid(cov_inv_block, 'Block-diagonal')
t3 = time.time()
print(f"  Block-diagonal covariance: {t3-t2:.2f}s")


# ══════════════════════════════════════════════════════════════════
# Compare results
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("COMPARISON: N=6 best-fit k_t across covariance variants")
print("=" * 70)

print(f"\n  {'Covariance':>16s}  {'k_t_best':>10s}  {'Dchi2':>8s}  "
      f"{'A_opt':>10s}  {'chi2_LCDM':>10s}  {'chi2/Ndof':>10s}")

for res in [res_full, res_diag, res_block]:
    chi2_ndof = res['chi2_lcdm'] / (n_data - 1)
    print(f"  {res['label']:>16s}  {res['kt_best_N6']:10.4f}  "
          f"{res['dchi2_best_N6']:8.2f}  {res['A_opt_bf']:10.1f}  "
          f"{res['chi2_lcdm']:10.2f}  {chi2_ndof:10.3f}")


print(f"\n  Confidence intervals (N=6):")
print(f"  {'Covariance':>16s}  {'1-sigma':>22s}  {'2-sigma':>22s}")

for res in [res_full, res_diag, res_block]:
    s1 = f"[{res['kt_1sig'][0]:.3f}, {res['kt_1sig'][1]:.3f}]"
    s2 = f"[{res['kt_2sig'][0]:.3f}, {res['kt_2sig'][1]:.3f}]"
    print(f"  {res['label']:>16s}  {s1:>22s}  {s2:>22s}")


# ── k_t shift comparison ─────────────────────────────────────────
print(f"\n  k_t shift from full to diagonal:")
kt_shift = abs(res_diag['kt_best_N6'] - res_full['kt_best_N6'])
kt_frac = kt_shift / res_full['kt_best_N6'] * 100
print(f"    |k_t(diag) - k_t(full)| = {kt_shift:.4f} h/Mpc ({kt_frac:.1f}%)")

kt_shift_block = abs(res_block['kt_best_N6'] - res_full['kt_best_N6'])
kt_frac_block = kt_shift_block / res_full['kt_best_N6'] * 100
print(f"    |k_t(block) - k_t(full)| = {kt_shift_block:.4f} h/Mpc ({kt_frac_block:.1f}%)")


# ── Per-N comparison ──────────────────────────────────────────────
print(f"\n" + "=" * 70)
print("Per-N best fits (all three covariance variants)")
print("=" * 70)

iN6 = np.where(N_grid == 6)[0][0]

for res in [res_full, res_diag, res_block]:
    print(f"\n  {res['label']} covariance:")
    print(f"  {'N':>3s}  {'eps_inf':>7s}  {'kt_min':>10s}  "
          f"{'chi2_min':>10s}  {'Dchi2':>8s}")
    for iN in range(n_N):
        ikt_min = np.argmin(res['chi2_grid'][iN])
        chi2_min_N = res['chi2_grid'][iN, ikt_min]
        dchi2_N = chi2_min_N - res['chi2_lcdm']
        print(f"  {N_grid[iN]:3d}  {eps_inf_grid[iN]:7.2f}  "
              f"{k_t_grid[ikt_min]:10.4f}  {chi2_min_N:10.2f}  {dchi2_N:8.2f}")


# ── Delta_chi^2 profile at N=6 for all three ─────────────────────
print(f"\n" + "=" * 70)
print("Delta_chi^2(k_t) profile at N=6 (selected k_t values)")
print("=" * 70)

print(f"\n  {'k_t':>8s}  {'Dchi2_full':>12s}  {'Dchi2_diag':>12s}  {'Dchi2_block':>12s}")
for ikt in range(0, n_kt, 3):  # every 3rd point for brevity
    print(f"  {k_t_grid[ikt]:8.4f}  "
          f"{res_full['delta_chi2'][iN6, ikt]:12.2f}  "
          f"{res_diag['delta_chi2'][iN6, ikt]:12.2f}  "
          f"{res_block['delta_chi2'][iN6, ikt]:12.2f}")


# ══════════════════════════════════════════════════════════════════
# Off-diagonal contribution analysis
# ══════════════════════════════════════════════════════════════════
print(f"\n" + "=" * 70)
print("Off-diagonal covariance analysis")
print("=" * 70)

# How much of the covariance is off-diagonal?
diag_frac = np.sum(np.diag(cov_ee)**2) / np.sum(cov_ee**2)
print(f"\n  Frobenius norm fraction (diagonal): {diag_frac:.4f}")
print(f"  Frobenius norm fraction (off-diag): {1-diag_frac:.4f}")

# Correlation matrix stats
diag_inv = 1.0 / np.sqrt(np.diag(cov_ee))
corr_mat = cov_ee * np.outer(diag_inv, diag_inv)
off_diag_corr = corr_mat[np.triu_indices_from(corr_mat, k=1)]

print(f"\n  Off-diagonal correlation statistics:")
print(f"    Mean:   {np.mean(off_diag_corr):.4f}")
print(f"    Std:    {np.std(off_diag_corr):.4f}")
print(f"    Max:    {np.max(off_diag_corr):.4f}")
print(f"    Min:    {np.min(off_diag_corr):.4f}")
print(f"    |r|>0.1: {np.sum(np.abs(off_diag_corr) > 0.1)} / {len(off_diag_corr)}")
print(f"    |r|>0.2: {np.sum(np.abs(off_diag_corr) > 0.2)} / {len(off_diag_corr)}")

# Strongest correlations between which pairs?
print(f"\n  Strongest off-diagonal correlations:")
n_top = 10
flat_upper = []
for i in range(n_data):
    for j in range(i+1, n_data):
        flat_upper.append((i, j, corr_mat[i, j]))
flat_upper.sort(key=lambda x: abs(x[2]), reverse=True)

for rank, (i, j, r) in enumerate(flat_upper[:n_top]):
    ip_i, il_i = divmod(i, n_ell)
    ip_j, il_j = divmod(j, n_ell)
    b1_i, b2_i = pair_labels[ip_i]
    b1_j, b2_j = pair_labels[ip_j]
    print(f"    {rank+1}. ({b1_i},{b2_i}) ell={ell_eff[il_i]:.0f} <-> "
          f"({b1_j},{b2_j}) ell={ell_eff[il_j]:.0f}: r = {r:.3f}")


# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
  B2 Question: Do off-diagonal covariance elements drive k_t?

  Results at N=6:
    Full covariance:       k_t = {res_full['kt_best_N6']:.4f}, Dchi2 = {res_full['dchi2_best_N6']:.2f}
    Diagonal covariance:   k_t = {res_diag['kt_best_N6']:.4f}, Dchi2 = {res_diag['dchi2_best_N6']:.2f}
    Block-diagonal:        k_t = {res_block['kt_best_N6']:.4f}, Dchi2 = {res_block['dchi2_best_N6']:.2f}

  k_t shift (full -> diag):  {kt_frac:.1f}%
  k_t shift (full -> block): {kt_frac_block:.1f}%
""")

if kt_frac < 20:
    print(f"  CONCLUSION: k_t is STABLE (<20% shift). Off-diagonal")
    print(f"  covariance does not drive the k_t preference.")
elif kt_frac < 50:
    print(f"  CONCLUSION: k_t shows MODERATE sensitivity ({kt_frac:.0f}% shift).")
    print(f"  Off-diagonal elements affect k_t but do not fully explain it.")
else:
    print(f"  CONCLUSION: k_t is UNSTABLE ({kt_frac:.0f}% shift). Off-diagonal")
    print(f"  covariance is significantly influencing the k_t result.")

print(f"\nScript complete.")
