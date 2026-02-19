"""
compute_isf_grid_v2.py
======================
Revised ISF model grid with ε-floor and S₈ constraint.

Key insight: the S₈ tension constrains ε at the σ₈ scale (~0.8 for N=6).
We use this to SOLVE for ε_∞ as a function of (N, k_t, α), producing
a physically motivated 1-parameter family for each N.

Modified sigmoid:
  ε(k) = ε_∞ + (1 - ε_∞) / [1 + (k/k_t)^α]

where ε_∞ = asymptotic small-scale floor.
"""
import numpy as np
from scipy.optimize import brentq
import csv

# Load CAMB baseline
data = np.load('/home/claude/paper5_wl/camb_pk_planck2018.npz')
k = data['k']
# Use z=0 for σ₈ (that's the standard definition)
Pk_lin_z0 = data['Pk_lin_z0']
sigma8_lcdm = float(data['sigma8'][0])

# ── Model functions ────────────────────────────────────────────
def epsilon_sigmoid(k, k_t, alpha, eps_floor=0.0):
    """ε(k) = ε_∞ + (1 - ε_∞) / [1 + (k/k_t)^α]"""
    return eps_floor + (1.0 - eps_floor) / (1.0 + (k / k_t)**alpha)

def isf_mod_factor(k, N, k_t, alpha, eps_floor=0.0):
    """[1 + (N-1)ε(k)] / N"""
    eps = epsilon_sigmoid(k, k_t, alpha, eps_floor)
    return (1.0 + (N - 1) * eps) / N

# Top-hat window for σ₈
R8 = 8.0  # Mpc/h
def tophat_W(k, R):
    x = k * R
    return np.where(x < 1e-10, 1.0,
                    3.0 * (np.sin(x) - x * np.cos(x)) / x**3)

def compute_sigma8_ratio(N, k_t, alpha, eps_floor):
    """Compute σ₈(ISF)/σ₈(ΛCDM) for given parameters."""
    mod = isf_mod_factor(k, N, k_t, alpha, eps_floor)
    integrand_isf = k**3 * Pk_lin_z0 * mod * tophat_W(k, R8)**2 / (2*np.pi**2)
    integrand_lcdm = k**3 * Pk_lin_z0 * tophat_W(k, R8)**2 / (2*np.pi**2)
    ratio_sq = np.trapezoid(integrand_isf, np.log(k)) / np.trapezoid(integrand_lcdm, np.log(k))
    return np.sqrt(ratio_sq)

# Verify ΛCDM σ₈
sig8_check = compute_sigma8_ratio(6, 0.01, 2, 1.0)  # eps_floor=1 → no ISF
print(f"σ₈ check (eps_floor=1, should be 1.0): {sig8_check:.6f}")

# ── Solve for ε_∞ given S₈ target ─────────────────────────────
def solve_eps_floor(N, k_t, alpha, target_s8_ratio):
    """
    Find ε_∞ such that σ₈(ISF)/σ₈(ΛCDM) = target_s8_ratio.
    Returns None if no solution exists.
    """
    # Check bounds
    ratio_at_0 = compute_sigma8_ratio(N, k_t, alpha, 0.0)
    ratio_at_1 = compute_sigma8_ratio(N, k_t, alpha, 1.0)

    if target_s8_ratio < ratio_at_0 or target_s8_ratio > ratio_at_1:
        return None  # No solution

    def objective(eps_f):
        return compute_sigma8_ratio(N, k_t, alpha, eps_f) - target_s8_ratio

    return brentq(objective, 0.0, 1.0, xtol=1e-6)

# ── S₈ tension parameters ─────────────────────────────────────
# KiDS-1000: S₈ = 0.759 ± 0.024
# DES Y3:    S₈ = 0.776 ± 0.017
# Planck:    S₈ = 0.834 ± 0.016
# Tension:   Δ ≈ 0.06-0.08, ratio ≈ 0.91-0.93

s8_central = 0.77 / 0.834  # 0.923
s8_lo = 0.74 / 0.85        # conservative lower
s8_hi = 0.80 / 0.82        # conservative upper

print(f"\nS₈ tension: ratio = {s8_central:.4f} [{s8_lo:.4f}, {s8_hi:.4f}]")

# ── Compute grid ───────────────────────────────────────────────
N_values = [2, 3, 4, 5, 6, 8, 10]
kt_values = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5]
alpha_values = [1, 2, 4]

print(f"\nGrid: {len(N_values)} N × {len(kt_values)} k_t × {len(alpha_values)} α"
      f" = {len(N_values)*len(kt_values)*len(alpha_values)} models")

results = []
for N in N_values:
    for kt in kt_values:
        for alpha in alpha_values:
            eps_f = solve_eps_floor(N, kt, alpha, s8_central)
            if eps_f is None:
                continue
            if eps_f < 0 or eps_f > 1:
                continue

            # Compute the modification at key scales
            mod = isf_mod_factor(k, N, kt, alpha, eps_f)

            # Effective ε at key k values
            eps_001 = epsilon_sigmoid(0.01, kt, alpha, eps_f)
            eps_01 = epsilon_sigmoid(0.1, kt, alpha, eps_f)
            eps_1 = epsilon_sigmoid(1.0, kt, alpha, eps_f)

            # Modification at key k values
            mod_001 = (1 + (N-1)*eps_001) / N
            mod_01 = (1 + (N-1)*eps_01) / N
            mod_1 = (1 + (N-1)*eps_1) / N

            # Max deviation from ΛCDM (minimum of modification factor)
            min_mod = np.min(mod)

            results.append({
                'N': N, 'k_t': kt, 'alpha': alpha, 'eps_floor': eps_f,
                'mod_k001': mod_001, 'mod_k01': mod_01, 'mod_k1': mod_1,
                'min_mod': min_mod,
                'eps_k001': eps_001, 'eps_k01': eps_01, 'eps_k1': eps_1,
            })

print(f"Valid models: {len(results)}")

# ── Save CSV ───────────────────────────────────────────────────
outfile = '/home/claude/paper5_wl/isf_s8_constrained_grid.csv'
fields = ['N', 'k_t', 'alpha', 'eps_floor',
          'mod_k001', 'mod_k01', 'mod_k1', 'min_mod',
          'eps_k001', 'eps_k01', 'eps_k1']

with open(outfile, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in results:
        row = {kk: (f"{v:.6f}" if isinstance(v, float) else v)
               for kk, v in r.items()}
        w.writerow(row)
print(f"Saved {outfile}")

# ── Print N=6 models ──────────────────────────────────────────
print(f"\n=== N=6 S₈-constrained models ===")
print(f"{'k_t':>6s} {'α':>3s} {'ε_∞':>8s} {'mod(0.01)':>10s}"
      f" {'mod(0.1)':>10s} {'mod(1.0)':>10s} {'min(mod)':>10s}")
print("-" * 65)
for r in results:
    if r['N'] == 6:
        print(f"{r['k_t']:6.3f} {r['alpha']:3d} {r['eps_floor']:8.4f}"
              f" {r['mod_k001']:10.4f} {r['mod_k01']:10.4f}"
              f" {r['mod_k1']:10.4f} {r['min_mod']:10.4f}")

# ── Save modification curves for N=6 models ───────────────────
pk_dict = {'k': k}
for r in results:
    if r['N'] == 6:
        mod = isf_mod_factor(k, 6, r['k_t'], r['alpha'], r['eps_floor'])
        key = f"mod_kt{r['k_t']}_a{r['alpha']}"
        pk_dict[key] = mod
np.savez('/home/claude/paper5_wl/isf_s8_constrained_N6.npz', **pk_dict)
print(f"\nSaved N=6 modification curves")
