#!/usr/bin/env python3
"""
Paper VII - Phase 1, Step 1b diagnostic plot
Visualise M_ISF(k) and P_ISF/P_NL for all N values.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')

d = np.load(os.path.join(DATA_DIR, 'pk_isf_fiducial.npz'), allow_pickle=True)
k = d['k']
N_grid = d['N_grid']
eps_grid = d['eps_inf_grid']
k_t = float(d['k_t_fid'])
alpha = float(d['alpha_fid'])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3', '#a65628']

# ── Left panel: M_ISF²(k) for each N ────────────────────────────
ax = axes[0]
for i, (N, eps, c) in enumerate(zip(N_grid, eps_grid, colors)):
    eps_k = eps / (1.0 + (k_t / k)**alpha)
    m = (1.0 + (N - 1) * eps_k) / N
    ax.semilogx(k, m**2, color=c, lw=2, label=f'N={N}, ε∞={eps:.2f}')

ax.axvline(k_t, color='grey', ls='--', lw=1, alpha=0.7, label=f'k_t={k_t}')
ax.axhline(1.0, color='black', ls=':', lw=0.8)
ax.set_xlabel(r'$k$ [$h$/Mpc]', fontsize=12)
ax.set_ylabel(r'$M_{\mathrm{ISF}}^2(k) = P_{\mathrm{ISF}}/P_{\Lambda\mathrm{CDM}}$',
              fontsize=11)
ax.set_title('ISF Power Spectrum Suppression', fontsize=13)
ax.set_xlim(1e-4, 50)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9, loc='right')

# ── Right panel: P_ISF(k) at z=0 vs ΛCDM ────────────────────────
ax = axes[1]
pknl = d['pknl']
ax.loglog(k, pknl[0], 'k-', lw=2.5, label=r'$\Lambda$CDM', zorder=10)

for i, (N, eps, c) in enumerate(zip(N_grid, eps_grid, colors)):
    eps_k = eps / (1.0 + (k_t / k)**alpha)
    m = (1.0 + (N - 1) * eps_k) / N
    pk_isf = m**2 * pknl[0]
    ax.loglog(k, pk_isf, color=c, lw=1.5, alpha=0.8,
              label=f'ISF N={N}')

ax.set_xlabel(r'$k$ [$h$/Mpc]', fontsize=12)
ax.set_ylabel(r'$P(k, z=0)$ [(Mpc/$h$)$^3$]', fontsize=12)
ax.set_title(r'Matter Power Spectrum at $z=0$', fontsize=13)
ax.set_xlim(1e-4, 50)
ax.set_ylim(1e-2, 1e5)
ax.legend(fontsize=9, loc='lower left', ncol=2)

plt.tight_layout()
outpath = os.path.join(FIG_DIR, 'step1b_isf_transfer_diagnostic.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Saved: {outpath}")
