#!/usr/bin/env python3
"""
Paper VII - Phase 1, Step 1b
Apply the ISF transfer function to generate P_ISF(k, z).

ISF transfer function:
    M_ISF(k) = [1 + (N-1) ε(k)] / N
    ε(k) = ε∞ / [1 + (k_t/k)^α]

Fiducial: N=6, ε∞=0.58, k_t=0.20 h/Mpc, α=2

Asymptotic checks:
    k >> k_t: P_ISF/P_NL → [(1 + (N-1)ε∞)/N]² = [(1+5×0.58)/6]² = 0.3403
    k << k_t: P_ISF/P_NL → 1/N² = 1/36 = 0.02778

Output: data/pk_isf_fiducial.npz
"""

import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
outpath = os.path.join(DATA_DIR, 'pk_isf_fiducial.npz')

# ── Load ΛCDM baseline from Step 1a ─────────────────────────────
d = np.load(os.path.join(DATA_DIR, 'pknl_camb.npz'), allow_pickle=True)
k = d['k']          # (500,) h/Mpc
z_arr = d['z']      # (7,)
pknl = d['pknl']    # (7, 500) (Mpc/h)³
print(f"Loaded P_NL: k=({k[0]:.1e}, {k[-1]:.1f}), {len(z_arr)} redshifts")

# ── ISF functions ────────────────────────────────────────────────
def epsilon_sigmoid(k, eps_inf, k_t, alpha):
    """Cross-branch correlation function ε(k). Eq. 3 of plan."""
    return eps_inf / (1.0 + (k_t / k)**alpha)

def M_ISF(k, N, eps_inf, k_t, alpha):
    """ISF transfer function. Eq. 2 of plan."""
    eps = epsilon_sigmoid(k, eps_inf, k_t, alpha)
    return (1.0 + (N - 1) * eps) / N

def P_ISF(k, pknl_1d, N, eps_inf, k_t, alpha):
    """ISF-modified power spectrum. Eq. 1 of plan."""
    m = M_ISF(k, N, eps_inf, k_t, alpha)
    return m**2 * pknl_1d

# ── Fiducial parameters ─────────────────────────────────────────
N_fid = 6
eps_inf_fid = 0.58
k_t_fid = 0.20   # h/Mpc
alpha_fid = 2

print(f"\nFiducial ISF: N={N_fid}, ε∞={eps_inf_fid}, "
      f"k_t={k_t_fid} h/Mpc, α={alpha_fid}")

# ── Compute fiducial ISF spectra ─────────────────────────────────
pk_isf_fid = np.zeros_like(pknl)
m_isf_fid = M_ISF(k, N_fid, eps_inf_fid, k_t_fid, alpha_fid)

for i, z in enumerate(z_arr):
    pk_isf_fid[i, :] = m_isf_fid**2 * pknl[i, :]

print(f"M_ISF shape: {m_isf_fid.shape}")
print(f"P_ISF shape: {pk_isf_fid.shape}")

# ── Asymptotic verification ──────────────────────────────────────
print("\n── Asymptotic checks ──")

# High-k limit: ε → ε∞, M → [1 + (N-1)ε∞]/N
M_highk_theory = (1 + (N_fid - 1) * eps_inf_fid) / N_fid
ratio_highk_theory = M_highk_theory**2
k_high_mask = k > 10 * k_t_fid  # k >> k_t
ratio_highk_actual = (m_isf_fid[k_high_mask])**2

print(f"High-k (k >> k_t = {k_t_fid}):")
print(f"  Theory: M² = [{1 + (N_fid-1)*eps_inf_fid}/{N_fid}]² "
      f"= {M_highk_theory:.4f}² = {ratio_highk_theory:.4f}")
print(f"  Actual at k={k[k_high_mask][0]:.2f}: "
      f"M² = {ratio_highk_actual[0]:.4f}")
print(f"  Actual at k={k[k_high_mask][-1]:.1f}: "
      f"M² = {ratio_highk_actual[-1]:.4f}")

err_high = abs(ratio_highk_actual[-1] - ratio_highk_theory) / ratio_highk_theory
assert err_high < 0.001, f"High-k asymptote off by {err_high*100:.2f}%"
print(f"  ✓ High-k asymptote verified ({err_high*100:.3f}% error)")

# Low-k limit: ε → 0, M → 1/N
M_lowk_theory = 1.0 / N_fid
ratio_lowk_theory = M_lowk_theory**2
k_low_mask = k < 0.01 * k_t_fid  # k << k_t
ratio_lowk_actual = (m_isf_fid[k_low_mask])**2

print(f"\nLow-k (k << k_t = {k_t_fid}):")
print(f"  Theory: M² = [1/{N_fid}]² = {ratio_lowk_theory:.5f}")
print(f"  Actual at k={k[k_low_mask][-1]:.1e}: "
      f"M² = {ratio_lowk_actual[-1]:.5f}")
print(f"  Actual at k={k[k_low_mask][0]:.1e}: "
      f"M² = {ratio_lowk_actual[0]:.5f}")

err_low = abs(ratio_lowk_actual[0] - ratio_lowk_theory) / ratio_lowk_theory
assert err_low < 0.01, f"Low-k asymptote off by {err_low*100:.2f}%"
print(f"  ✓ Low-k asymptote verified ({err_low*100:.3f}% error)")

# ── Compute for full N grid (will be needed later) ───────────────
print("\n── N-grid ISF transfer functions ──")

N_grid = np.array([2, 3, 4, 6, 8, 10])
eps_inf_grid = np.array([0.00, 0.25, 0.40, 0.58, 0.68, 0.74])

# Verify (N-1)(1-ε∞) values match Table 1 of plan
for N, eps in zip(N_grid, eps_inf_grid):
    excess = (N - 1) * (1 - eps)
    M_high = (1 + (N - 1) * eps) / N
    M_low = 1.0 / N
    print(f"  N={N:2d}, ε∞={eps:.2f}: (N-1)(1-ε)={excess:.2f}, "
          f"M²_high={M_high**2:.4f}, M²_low={M_low**2:.5f}")

# ── Save checkpoint ──────────────────────────────────────────────
np.savez(
    outpath,
    k=k,
    z=z_arr,
    pk_isf_fiducial=pk_isf_fid,   # (7, 500)
    m_isf_fiducial=m_isf_fid,     # (500,)
    pknl=pknl,                     # (7, 500) for reference
    N_fid=N_fid,
    eps_inf_fid=eps_inf_fid,
    k_t_fid=k_t_fid,
    alpha_fid=alpha_fid,
    N_grid=N_grid,
    eps_inf_grid=eps_inf_grid,
)
print(f"\n✓ Saved to {outpath}")
print("\n── Step 1b complete ──")
