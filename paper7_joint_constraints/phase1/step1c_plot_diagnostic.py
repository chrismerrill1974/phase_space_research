#!/usr/bin/env python3
"""
Paper VII - Phase 1, Step 1c diagnostic plot.
Top: C_κ(ℓ) comparison (ours vs CAMB).
Bottom left: Residual ratio. Bottom right: Lensing kernel W(χ) and n(z).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')

d = np.load(os.path.join(DATA_DIR, 'cl_kappa_lcdm.npz'), allow_pickle=True)
ell = d['ell']
cl_ours = d['cl_kappa_lcdm']
ell_camb = d['ell_camb']
cl_camb_raw = d['cl_kappa_camb']
chi_W = d['chi_W']
W_arr = d['W_arr']
z_nz = d['z_nz']
nz_vals = d['nz_vals']
h = float(d['h'])

# Reconstruct z from chi (approximate — use linear interp)
# We need z_of_chi. Rough version from the data:
# chi_W ranges from ~10 to ~6450 Mpc. Use the ratio approach.
# Actually, let's just compute ℓ(ℓ+1)C/2π directly.

lcl_ours = ell * (ell + 1) * cl_ours / (2 * np.pi)

# CAMB was stored as raw Cℓ (already converted in step1c)
lcl_camb = ell_camb * (ell_camb + 1) * cl_camb_raw / (2 * np.pi)

fig = plt.figure(figsize=(13, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.3, wspace=0.3)

# ── Top: C_κ(ℓ) comparison ──────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.loglog(ell_camb, lcl_camb, 'b-', lw=2, alpha=0.7, label='CAMB (reference)')
ax1.loglog(ell, lcl_ours, 'r--', lw=2, label='Our Limber integral')

ax1.axvspan(100, 3000, alpha=0.1, color='green', label='Validation range')
ax1.set_xlabel(r'Multipole $\ell$', fontsize=12)
ax1.set_ylabel(r'$\ell(\ell+1)\,C_\kappa(\ell)/(2\pi)$', fontsize=12)
ax1.set_title(r'$\Lambda$CDM Convergence Power Spectrum — Limber Validation',
              fontsize=14)
ax1.set_xlim(10, 10000)
ax1.legend(fontsize=11, loc='lower right')
ax1.text(0.02, 0.95, 'KiDS-like source distribution\n'
         r'$n(z) \propto z^2 \exp[-(z/0.7)^{1.5}]$',
         transform=ax1.transAxes, fontsize=10, va='top',
         bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))

# ── Bottom left: Residual ratio ──────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
cl_camb_interp = interp1d(ell_camb, cl_camb_raw, kind='cubic')
mask = (ell >= 20) & (ell <= 8000)
ratio = cl_ours[mask] / cl_camb_interp(ell[mask])

ax2.semilogx(ell[mask], ratio, 'k-', lw=1.5)
ax2.axhline(1.0, color='blue', ls='--', lw=1)
ax2.axhspan(0.95, 1.05, alpha=0.15, color='green', label='±5% target')
ax2.axhspan(0.99, 1.01, alpha=0.2, color='blue', label='±1%')
ax2.axvspan(100, 3000, alpha=0.08, color='green')

ax2.set_xlabel(r'Multipole $\ell$', fontsize=12)
ax2.set_ylabel(r'$C_\kappa^{\mathrm{ours}} / C_\kappa^{\mathrm{CAMB}}$', fontsize=12)
ax2.set_title('Ratio (Ours / CAMB)', fontsize=13)
ax2.set_xlim(20, 8000)
ax2.set_ylim(0.96, 1.04)
ax2.legend(fontsize=9, loc='lower right')

# Annotate max deviation
max_dev = np.max(np.abs(ratio - 1.0))
ax2.text(0.02, 0.05, f'Max |deviation| = {max_dev*100:.2f}%\nin [100, 3000]',
         transform=ax2.transAxes, fontsize=10, va='bottom',
         bbox=dict(boxstyle='round', fc='lightyellow'))

# ── Bottom right: Lensing kernel and n(z) ────────────────────────
ax3 = fig.add_subplot(gs[1, 1])

# We need z(χ). Approximate: use a simple mapping.
# From the CAMB data, χ(z=0)=0, χ(z=3)≈6500, roughly linear at low z.
# Better: reconstruct from the W(χ) grid
# For plotting purposes, use linear interp from known points
from scipy.interpolate import interp1d as i1d

# Approximate z(χ) from the fact that W peaks at z~0.5, χ~1969
# and goes to z~3 at χ~6500
chi_z_approx = np.array([0, 440, 860, 1260, 1970, 2590, 3400, 4700, 5700, 6500])
z_approx = np.array([0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
z_of_chi_approx = i1d(chi_z_approx, z_approx, kind='cubic',
                       bounds_error=False, fill_value=(0, 3))

z_W = z_of_chi_approx(chi_W)

ax3_twin = ax3.twinx()

# Plot W(χ) vs z
ax3.plot(z_W, W_arr * 1e5, 'b-', lw=2, label=r'$W(\chi) \times 10^5$')
ax3.set_xlabel('Redshift $z$', fontsize=12)
ax3.set_ylabel(r'$W(\chi) \times 10^5$ [1/Mpc]', fontsize=12, color='blue')
ax3.tick_params(axis='y', labelcolor='blue')

# Plot n(z)
ax3_twin.plot(z_nz, nz_vals, 'r-', lw=2, alpha=0.7, label='$n(z)$')
ax3_twin.set_ylabel('$n(z)$ [normalised]', fontsize=12, color='red')
ax3_twin.tick_params(axis='y', labelcolor='red')

ax3.set_title('Lensing Kernel & Source Distribution', fontsize=13)
ax3.set_xlim(0, 3.0)

# Combined legend
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper right')

plt.savefig(os.path.join(FIG_DIR, 'step1c_limber_validation.png'),
            dpi=150, bbox_inches='tight')
print("Saved: figures/step1c_limber_validation.png")
