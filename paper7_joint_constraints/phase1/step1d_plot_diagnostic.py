#!/usr/bin/env python3
"""
Paper VII - Phase 1, Step 1d diagnostic plot.
Left: ξ±(θ) cosmic shear correlation functions.
Right: ΔΣ(R) convergence profile.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')

d = np.load(os.path.join(DATA_DIR, 'dsigma_lcdm.npz'), allow_pickle=True)
theta = d['theta_arcmin']
xi_p = d['xi_plus']
xi_m = d['xi_minus']
R = d['R_hMpc']
dsigma = d['dsigma_hMsun_pc2']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ── Left: ξ±(θ) ──────────────────────────────────────────────────
ax = axes[0]
ax.loglog(theta, xi_p, 'b-', lw=2, label=r'$\xi_+(\theta)$')
ax.loglog(theta, np.abs(xi_m), 'r--', lw=2, label=r'$|\xi_-(\theta)|$')

# Shade typical KiDS/DES range
ax.axvspan(0.5, 300, alpha=0.05, color='green')
ax.axvspan(2, 200, alpha=0.08, color='green', label='Typical KiDS range')

ax.set_xlabel(r'$\theta$ [arcmin]', fontsize=12)
ax.set_ylabel(r'$\xi_\pm(\theta)$', fontsize=12)
ax.set_title(r'$\Lambda$CDM Cosmic Shear Correlation Functions', fontsize=13)
ax.set_xlim(0.5, 300)
ax.set_ylim(1e-7, 1e-3)
ax.legend(fontsize=11, loc='upper right')
ax.text(0.03, 0.05, 'KiDS-like source\n'
        r'$n(z) \propto z^2 e^{-(z/0.7)^{1.5}}$',
        transform=ax.transAxes, fontsize=9, va='bottom',
        bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))

# ── Right: ΔΣ(R) ─────────────────────────────────────────────────
ax = axes[1]
ax.loglog(R, dsigma, 'k-o', lw=2, ms=4)

ax.set_xlabel(r'$R$ [$h^{-1}$Mpc]', fontsize=12)
ax.set_ylabel(r'$\Delta\Sigma(R)$ [$h\,M_\odot$/pc$^2$]', fontsize=12)
ax.set_title(r'Convergence Profile ($z_L=0.27$)', fontsize=13)
ax.set_xlim(0.1, 100)

ax.text(0.03, 0.05,
        'From convergence auto-spectrum\n'
        '(for ΛCDM vs ISF comparison,\n not absolute GGL calibration)',
        transform=ax.transAxes, fontsize=8, va='bottom',
        bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.7))

plt.tight_layout()
outpath = os.path.join(FIG_DIR, 'step1d_bessel_diagnostic.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Saved: {outpath}")
