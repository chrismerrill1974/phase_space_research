#!/usr/bin/env python3
"""
Paper VII - Phase 1, Step 1e diagnostic plot.
Left: A @ 1 vs direct Limber (ΛCDM validation).
Right: ISF C_κ(ℓ) suppression for all N values via response operator.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')

d = np.load(os.path.join(DATA_DIR, 'response_operator.npz'), allow_pickle=True)
A = d['A']
ell = d['ell']
k_hMpc = d['k_hMpc']
cl_from_A = d['cl_lcdm_from_A']
cl_direct = d['cl_lcdm_direct']
h_val = float(d['h'])

# ISF parameters
N_grid = np.array([2, 3, 4, 6, 8, 10])
eps_grid = np.array([0.00, 0.25, 0.40, 0.58, 0.68, 0.74])
k_t = 0.20  # h/Mpc
alpha = 2

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# ── Left: Validation ─────────────────────────────────────────────
ax = axes[0]

lcl_A = ell * (ell + 1) * cl_from_A / (2 * np.pi)
lcl_direct = ell * (ell + 1) * cl_direct / (2 * np.pi)

ax.loglog(ell, lcl_direct, 'b-', lw=2.5, label='Direct Limber (Step 1c)')
ax.loglog(ell, lcl_A, 'r--', lw=1.5, label=r'$A \cdot \mathbf{1}$ (response op.)')

ax.set_xlabel(r'Multipole $\ell$', fontsize=12)
ax.set_ylabel(r'$\ell(\ell+1)\,C_\kappa/(2\pi)$', fontsize=12)
ax.set_title('Response Operator Validation', fontsize=13)
ax.legend(fontsize=10)

# Inset: ratio
ax_in = ax.inset_axes([0.4, 0.12, 0.55, 0.35])
ratio = cl_from_A / cl_direct
ax_in.semilogx(ell, ratio, 'k-', lw=1.5)
ax_in.axhline(1.0, color='blue', ls='--', lw=0.8)
ax_in.axhspan(0.999, 1.001, alpha=0.2, color='blue')
ax_in.set_ylim(0.996, 1.004)
ax_in.set_xlim(10, 10000)
ax_in.set_ylabel('Ratio', fontsize=9)
ax_in.set_xlabel(r'$\ell$', fontsize=9)
ax_in.tick_params(labelsize=8)
ax_in.text(0.05, 0.85, f'Max diff: {np.abs(ratio-1).max()*100:.2f}%',
           transform=ax_in.transAxes, fontsize=8)

# ── Right: ISF suppression for all N ─────────────────────────────
ax = axes[1]
colors = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3', '#a65628']

for i, (N, eps, c) in enumerate(zip(N_grid, eps_grid, colors)):
    eps_k = eps / (1.0 + (k_t / k_hMpc)**alpha)
    M = (1.0 + (N - 1) * eps_k) / N
    M_sq = M**2

    cl_isf = A @ M_sq
    ratio_isf = cl_isf / cl_from_A

    ax.semilogx(ell, ratio_isf, color=c, lw=2,
                label=f'N={N}, ε∞={eps:.2f}')

ax.axhline(1.0, color='black', ls=':', lw=0.8)
ax.set_xlabel(r'Multipole $\ell$', fontsize=12)
ax.set_ylabel(r'$C_\kappa^{\mathrm{ISF}} / C_\kappa^{\Lambda\mathrm{CDM}}$',
              fontsize=12)
ax.set_title(r'ISF Convergence Power Suppression ($k_t=0.20\,h$/Mpc)',
             fontsize=13)
ax.set_xlim(10, 10000)
ax.set_ylim(0, 0.7)
ax.legend(fontsize=9, loc='right')

# Annotate k_t transition
# k_t = 0.20 h/Mpc. At what ℓ does this correspond?
# Roughly ℓ ~ k_t × χ_eff × h, where χ_eff ~ 2000 Mpc
ell_kt = 0.20 * 2000  # ≈ 400
ax.axvline(ell_kt, color='grey', ls='--', lw=1, alpha=0.5)
ax.text(ell_kt * 1.1, 0.65, r'$\ell \sim k_t \chi_{\rm eff}$',
        fontsize=9, color='grey')

plt.tight_layout()
outpath = os.path.join(FIG_DIR, 'step1e_response_operator_diagnostic.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Saved: {outpath}")
