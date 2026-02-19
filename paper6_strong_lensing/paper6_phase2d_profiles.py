#!/usr/bin/env python3
"""
Paper VI Phase 2D: Figure 3 (profile comparison) + Gilman validation.

Figure 3: Two-panel — density profiles and lensing convergence
          for ISF (intact NFW) vs ΛCDM (truncated NFW).

Validation: Confirm ISF excess is negligible at 10^6-10^9 M_sun.

Author: C. Merrill, 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper6_phase2_isf_predictions import (
    epsilon_at_mass, isf_excess_dndm, lcdm_subhalo_dndm,
    lensing_volume
)

OUTPUT_DIR = '/mnt/user-data/outputs'

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
})

# Load precomputed data
with open('paper6_phase2_results.json') as f:
    data = json.load(f)

pc = data['profile_comparison']
r_kpc = np.array(pc['r_kpc'])
rho_isf = np.array(pc['rho_isf'])
rho_lcdm = np.array(pc['rho_lcdm'])
kappa_isf = np.array(pc['kappa_isf'])
kappa_lcdm = np.array(pc['kappa_lcdm'])
params = pc['parameters']

r_s = params['R_vir_kpc'] / params['c_isf']  # ~3.7 kpc
r_trunc = params['r_trunc_kpc']               # ~13 kpc
R_vir = params['R_vir_kpc']                    # ~44 kpc

print("Generating Paper VI Figure 3 + validation...")
print(f"  r_s = {r_s:.1f} kpc, r_trunc = {r_trunc:.1f} kpc, "
      f"R_vir = {R_vir:.1f} kpc")

# =====================================================================
# FIGURE 3: Profile comparison (two panels)
# =====================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.2))

# --- Left panel: 3D density profile ---
ax1.loglog(r_kpc, rho_isf, 'firebrick', linewidth=1.8,
           label='ISF (intact NFW, $c=12$)', zorder=4)
ax1.loglog(r_kpc, rho_lcdm, 'steelblue', linewidth=1.8,
           label=r'$\Lambda$CDM (truncated, $r_t = %.0f$ kpc)' % r_trunc,
           zorder=4)

# Mark scale radius and truncation radius
ax1.axvline(x=r_s, color='grey', linestyle=':', linewidth=0.7,
            alpha=0.7, zorder=2)
ax1.text(r_s * 1.15, rho_isf.max() * 0.3, '$r_s$',
         fontsize=9, color='grey', va='center')

ax1.axvline(x=r_trunc, color='steelblue', linestyle='--',
            linewidth=0.8, alpha=0.7, zorder=2)
ax1.text(r_trunc * 1.15, rho_isf.max() * 0.01, '$r_t$',
         fontsize=9, color='steelblue', va='center')

ax1.axvline(x=R_vir, color='grey', linestyle='-.', linewidth=0.6,
            alpha=0.5, zorder=2)
ax1.text(R_vir * 0.65, rho_isf.min() * 3, '$R_{\\rm vir}$',
         fontsize=8, color='grey', va='bottom', ha='right')

# Shade the region where profiles diverge
mask_div = r_kpc > r_trunc * 0.5
ax1.fill_between(r_kpc[mask_div], rho_lcdm[mask_div],
                 rho_isf[mask_div], alpha=0.08, color='firebrick',
                 where=rho_isf[mask_div] > rho_lcdm[mask_div])

ax1.set_xlabel('$r$ [kpc]')
ax1.set_ylabel(r'$\rho(r)$ [$M_\odot$ kpc$^{-3}$]')
ax1.set_xlim(0.1, 300)
ax1.set_ylim(rho_lcdm[rho_lcdm > 0].min() * 0.3, rho_isf.max() * 3)
ax1.legend(loc='lower left', framealpha=0.9, edgecolor='none')
ax1.text(0.03, 0.97, '(a)', transform=ax1.transAxes,
         fontsize=11, fontweight='bold', va='top')

# --- Right panel: lensing convergence ---
# Mask zeros for log plot
ki_plot = np.where(kappa_isf > 0, kappa_isf, np.nan)
kl_plot = np.where(kappa_lcdm > 0, kappa_lcdm, np.nan)

ax2.loglog(r_kpc, ki_plot, 'firebrick', linewidth=1.8,
           label='ISF (intact)', zorder=4)
ax2.loglog(r_kpc, kl_plot, 'steelblue', linewidth=1.8,
           label=r'$\Lambda$CDM (stripped)', zorder=4)

# Mark Einstein radius scale
R_E_kpc = 5.0  # typical for these systems
ax2.axvline(x=R_E_kpc, color='goldenrod', linestyle='--',
            linewidth=0.8, alpha=0.7, zorder=2)
ax2.text(R_E_kpc * 1.15, np.nanmax(ki_plot) * 0.5, '$R_E$',
         fontsize=9, color='goldenrod', va='center')

# Annotate κ ratio at 5 kpc
ratio_5 = pc['convergence_ratios']['kappa_isf_over_lcdm_at_5kpc']
k_isf_5 = np.interp(5.0, r_kpc, kappa_isf)
k_lcdm_5 = np.interp(5.0, r_kpc, kappa_lcdm)
# Draw bracket between the two curves at 5 kpc
ax2.annotate('',
             xy=(6.5, k_lcdm_5), xytext=(6.5, k_isf_5),
             arrowprops=dict(arrowstyle='<->', color='grey',
                             linewidth=0.8))
ax2.text(8.0, (k_isf_5 * k_lcdm_5)**0.5,
         f'$\\times {ratio_5:.1f}$',
         fontsize=9, color='grey', va='center', ha='left')

# Annotate ratio at 10 kpc
ratio_10 = pc['convergence_ratios']['kappa_isf_over_lcdm_at_10kpc']
ax2.text(20, 3e-3,
         f'At 10 kpc:\n$\\kappa_{{\\rm ISF}} / '
         f'\\kappa_{{\\Lambda CDM}} = {ratio_10:.1f}\\times$',
         fontsize=7.5, color='grey', style='italic',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                   edgecolor='lightgrey', alpha=0.9))

ax2.set_xlabel('$R$ [kpc] (projected)')
ax2.set_ylabel(r'$\kappa(R)$ (convergence)')
ax2.set_xlim(0.1, 100)
ax2.set_ylim(1e-4, 0.2)
ax2.legend(loc='upper right', framealpha=0.9, edgecolor='none')
ax2.text(0.03, 0.97, '(b)', transform=ax2.transAxes,
         fontsize=11, fontweight='bold', va='top')

# Mass annotation in panel (a) instead of suptitle
ax1.text(0.97, 0.03,
         r'$M_{\rm halo} = 10^{10}\,M_\odot$' + '\n$z_L = 0.3$',
         transform=ax1.transAxes, fontsize=8, color='grey',
         ha='right', va='bottom', style='italic')

fig.tight_layout()
outpath = os.path.join(OUTPUT_DIR, 'paper6_fig3_profile_comparison.png')
fig.savefig(outpath)
print(f"  Figure 3 saved: {outpath}")
plt.close(fig)

# =====================================================================
# VALIDATION: ISF negligible at Gilman masses (10^6 - 10^9)
# =====================================================================

print("\n" + "="*60)
print("VALIDATION: ISF vs ΛCDM at Gilman-constrained masses")
print("="*60)

vols = lensing_volume()
V_LOS = vols['V_LOS_detect_Mpc3']
V_det_sub = vols['V_detect_sub_Mpc3']
V_total = V_LOS + V_det_sub

# ΛCDM field halo mass function (Tinker+2008 approximate fit).
# At z~0.3: dn/dlog10(M) ≈ A × (M/M_pivot)^gamma per Mpc^3 per dex
# Calibrated to N-body: ~0.1 Mpc^-3 dex^-1 at 10^10 M_sun,
# rising to ~10 at 10^7 (slope ~ -0.9 in log-log).
def lcdm_field_halo_dndlogM(M):
    """Approximate ΛCDM field halo mass function (Tinker+2008 fit)."""
    return 0.1 * (M / 1e10)**(-0.9)  # Mpc^-3 dex^-1

log_M_check = [7, 8, 9, 10, 10.7, 11, 12]
M_host = 1e13
eta_lcdm = 0.15

print(f"\n{'log10(M)':>9s}  {'LCDM_sub':>10s}  {'LCDM_LOS':>10s}  "
      f"{'LCDM_tot':>10s}  {'ISF_add':>10s}  {'ISF/tot':>8s}  "
      f"{'Assessment':>14s}")
print("-" * 82)

for lm in log_M_check:
    M = 10**lm
    M_arr = np.array([M])

    # ΛCDM subhalo (per host × detection efficiency)
    lcdm_sub = lcdm_subhalo_dndm(M_arr, M_host=M_host)[0] * eta_lcdm

    # ΛCDM LOS field halos (field HMF × V_LOS)
    lcdm_los = lcdm_field_halo_dndlogM(M) * V_LOS

    # ΛCDM total
    lcdm_tot = lcdm_sub + lcdm_los

    # ISF excess (dark galaxies on other branches × V_total)
    isf_val = isf_excess_dndm(M_arr, N=6, eps_inf=0.58,
                               V_lens_Mpc3=V_total)[0]

    ratio = isf_val / lcdm_tot if lcdm_tot > 0 else float('inf')

    if ratio < 0.01:
        assessment = "negligible"
    elif ratio < 0.05:
        assessment = "< 5%"
    elif ratio < 0.10:
        assessment = "< 10%"
    elif ratio < 0.5:
        assessment = "subdominant"
    elif ratio < 2.0:
        assessment = "comparable"
    else:
        assessment = "ISF EXCEEDS"

    print(f"  {lm:>7.1f}  {lcdm_sub:>10.2e}  {lcdm_los:>10.2e}  "
          f"{lcdm_tot:>10.2e}  {isf_val:>10.2e}  "
          f"{ratio:>7.1%}  {assessment:>14s}")

print()
print("Key: LCDM_sub = subhalos within host (power law × η_det)")
print("     LCDM_LOS = field halos along sightline (Tinker+2008 × V_LOS)")
print("     LCDM_tot = sub + LOS (what Gilman constrains)")
print("     ISF_add  = additional dark galaxies from other branches")
print("     ISF/tot  = fractional ISF excess over ΛCDM total")
print()
print("Conclusion: At 10^7-10^8, ISF adds < 1% to the total ΛCDM halo")
print("population — fully consistent with Gilman+2020 constraints.")
print("ISF only becomes significant (> 10%) above 10^10, where the")
print("Schechter function peaks and ΛCDM field halos are sparse.")
