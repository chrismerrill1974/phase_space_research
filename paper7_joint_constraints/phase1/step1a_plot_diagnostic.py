#!/usr/bin/env python3
"""
Paper VII - Phase 1, Step 1a diagnostic plot
Visualise P_NL(k, z) and the Halofit enhancement.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Load checkpoint
d = np.load(os.path.join(DATA_DIR, 'pknl_camb.npz'), allow_pickle=True)
k = d['k']
z_arr = d['z']
pknl = d['pknl']
pklin = d['pklin']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ── Left panel: P_NL(k) at multiple z ───────────────────────────
ax = axes[0]
colors = plt.cm.viridis(np.linspace(0, 0.85, len(z_arr)))
for i, z in enumerate(z_arr):
    ax.loglog(k, pknl[i], color=colors[i], lw=1.5, label=f'z={z:.1f}')
    ax.loglog(k, pklin[i], color=colors[i], lw=0.7, ls='--', alpha=0.5)

ax.set_xlabel(r'$k$ [$h$/Mpc]', fontsize=12)
ax.set_ylabel(r'$P(k)$ [(Mpc/$h$)$^3$]', fontsize=12)
ax.set_title(r'$P_{\mathrm{NL}}(k,z)$ — Planck 2018 + Halofit', fontsize=13)
ax.set_xlim(1e-4, 50)
ax.set_ylim(1e-2, 1e5)
ax.legend(fontsize=9, ncol=2, loc='lower left')
ax.text(0.02, 0.02, 'Solid: nonlinear (Halofit)\nDashed: linear',
        transform=ax.transAxes, fontsize=8, va='bottom',
        bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))

# ── Right panel: Halofit boost ratio ─────────────────────────────
ax = axes[1]
for i, z in enumerate(z_arr):
    ratio = pknl[i] / pklin[i]
    ax.semilogx(k, ratio, color=colors[i], lw=1.5, label=f'z={z:.1f}')

ax.axhline(1, color='grey', ls=':', lw=0.8)
ax.set_xlabel(r'$k$ [$h$/Mpc]', fontsize=12)
ax.set_ylabel(r'$P_{\mathrm{NL}}/P_{\mathrm{lin}}$', fontsize=12)
ax.set_title('Halofit Enhancement Factor', fontsize=13)
ax.set_xlim(1e-4, 50)
ax.set_ylim(0.5, 100)
ax.set_yscale('log')
ax.legend(fontsize=9, ncol=2, loc='upper left')

plt.tight_layout()
outpath = os.path.join(FIG_DIR, 'step1a_pknl_diagnostic.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Saved: {outpath}")
