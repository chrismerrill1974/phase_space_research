#!/usr/bin/env python3
"""
Paper VI Phase 2C: Figures 1 and 2.

Figure 1: ε(k) vs halo mass with annotated observational regimes.
Figure 2: Mass function comparison (ΛCDM power law vs ISF Schechter bump).

Both in publication style for the paper draft.

Author: C. Merrill, 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import json
import os
import sys

# Import the prediction functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper6_phase2_isf_predictions import (
    epsilon_sigmoid, epsilon_at_mass, isf_excess_dndm,
    lcdm_subhalo_dndm, lensing_volume
)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))   # version 2 (2026-10-11): this folder
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')

# =====================================================================
# Publication style setup
# =====================================================================

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
})

print("Generating Paper VI figures...")

# =====================================================================
# FIGURE 1: ε(k) vs halo mass
# =====================================================================

def make_figure1():
    """ε(k) vs log10(M_halo) with annotated observational regimes."""

    log_M = np.linspace(6, 15.5, 500)
    M_array = 10**log_M

    eps_vals = np.zeros_like(log_M)
    k_vals = np.zeros_like(log_M)
    for i, M in enumerate(M_array):
        eps_vals[i], k_vals[i] = epsilon_at_mass(M)

    fig, ax1 = plt.subplots(1, 1, figsize=(7.5, 4.2))

    # Main curve
    ax1.plot(log_M, eps_vals, 'k-', linewidth=2.0, zorder=5)

    # Horizontal reference lines
    ax1.axhline(y=0.58, color='grey', linestyle='--', linewidth=0.7,
                alpha=0.7, zorder=2)
    ax1.text(6.15, 0.555, r'$\varepsilon_\infty = 0.58$',
             fontsize=8, color='grey', va='top')

    ax1.axhline(y=1.0, color='grey', linestyle=':', linewidth=0.5,
                alpha=0.5, zorder=2)

    # Shaded vertical bands for observational regimes
    # Gilman flux ratio constraints (10^6 - 10^9)
    ax1.axvspan(6, 9, alpha=0.12, color='steelblue', zorder=1)
    ax1.text(7.5, 0.06, 'Flux ratio\nconstraints\n(Gilman+2020)',
             fontsize=7.5, ha='center', va='bottom', color='steelblue',
             fontweight='bold')

    # ISF galaxy-mass prediction zone (10^10 - 10^12)
    ax1.axvspan(10, 12, alpha=0.12, color='firebrick', zorder=1)
    ax1.text(11.0, 0.06, 'ISF\ngalaxy-mass\nbump',
             fontsize=7.5, ha='center', va='bottom', color='firebrick',
             fontweight='bold')

    # Meneghetti GGSL anomaly scale (10^10 - 10^11)
    ax1.axvspan(10, 11, alpha=0.10, color='forestgreen', zorder=1)
    ax1.text(10.5, 0.30, 'GGSL\nanomaly',
             fontsize=7, ha='center', va='bottom', color='forestgreen',
             style='italic')

    # Vertical line at k_t transition scale
    logM_kt = np.log10(1e12 * (1.0/(0.20 * 0.26))**3)
    ax1.axvline(x=logM_kt, color='darkorange', linestyle='--',
                linewidth=0.9, alpha=0.7, zorder=3)
    ax1.text(logM_kt + 0.15, 0.78,
             r'$k_t = 0.20$ h/Mpc' + '\n(transition scale)',
             fontsize=7.5, color='darkorange',
             ha='left', va='center')

    # Cluster/supercluster label at right
    ax1.text(14.5, 0.96, r'$\varepsilon \to 1$' + '\n(branches\ncorrelated)',
             fontsize=7.5, ha='center', va='top', color='grey',
             style='italic')

    # Axes
    ax1.set_xlabel(r'$\log_{10}(M_{\rm halo} / M_\odot)$')
    ax1.set_ylabel(r'$\varepsilon(k)$')
    ax1.set_xlim(6, 15.5)
    ax1.set_ylim(0, 1.08)
    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # Secondary x-axis (top): k in h/Mpc
    ax2 = ax1.twiny()
    k_ticks_desired = [0.2, 0.5, 1, 2, 5, 10, 50, 100, 500]
    logM_from_k = [np.log10(1e12 * (1.0/(k*0.26))**3)
                   for k in k_ticks_desired]
    # Filter to visible range
    vis = [(lm, k) for lm, k in zip(logM_from_k, k_ticks_desired)
           if 6 <= lm <= 15.5]
    ax2.set_xlim(6, 15.5)
    ax2.set_xticks([v[0] for v in vis])
    ax2.set_xticklabels([str(v[1]) for v in vis])
    ax2.set_xlabel(r'$k$ [h/Mpc]', labelpad=8)

    fig.tight_layout()
    outpath = os.path.join(FIG_DIR, 'paper6_fig1_epsilon_vs_mass.png')
    fig.savefig(outpath)
    print(f"  Figure 1 saved: {outpath}")
    plt.close(fig)
    return outpath


# Save checkpoint — Figure 1 complete
fig1_path = make_figure1()


# =====================================================================
# FIGURE 2: Mass function comparison (ΛCDM vs ISF)
# =====================================================================

def make_figure2():
    """
    dN/dlog10(M) per lens sightline: ΛCDM power law vs ISF bump.

    Both curves in consistent units: expected detectable perturbers
    per strong lens sightline per dex of mass.

    ΛCDM: subhalo mass function × detection efficiency (~15%).
    ISF:  Schechter-based excess × V_LOS detection volume.
    """
    # Get detection volume
    vols = lensing_volume()
    V_LOS = vols['V_LOS_detect_Mpc3']
    V_det_sub = vols['V_detect_sub_Mpc3']

    # Mass grid
    log_M = np.linspace(6, 13, 500)
    M_array = 10**log_M

    # --- ΛCDM subhalo mass function (per lens sightline) ---
    M_host = 1e13
    lcdm_per_halo = lcdm_subhalo_dndm(M_array, M_host=M_host)
    eta_lcdm = 0.15  # detection efficiency (Vegetti+2014)
    lcdm_per_lens = lcdm_per_halo * eta_lcdm

    # --- ISF excess mass function (per lens sightline) ---
    # Dominant contribution is LOS (V_LOS >> V_detect_sub)
    # Total detection volume ~ V_LOS + V_detect_sub ≈ V_LOS
    V_total = V_LOS + V_det_sub

    isf_058 = isf_excess_dndm(M_array, N=6, eps_inf=0.58,
                               V_lens_Mpc3=V_total)
    isf_030 = isf_excess_dndm(M_array, N=6, eps_inf=0.30,
                               V_lens_Mpc3=V_total)
    isf_000 = isf_excess_dndm(M_array, N=6, eps_inf=0.00,
                               V_lens_Mpc3=V_total)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    # ΛCDM power law
    ax.plot(log_M, lcdm_per_lens, 'steelblue', linewidth=1.8,
            label=r'$\Lambda$CDM subhalos ($\times \eta_{\rm det}$)',
            zorder=4)

    # ISF curves (only plot where meaningful: > 10^-6)
    floor = 1e-6
    m058 = np.where(isf_058 > floor, isf_058, np.nan)
    m030 = np.where(isf_030 > floor, isf_030, np.nan)
    m000 = np.where(isf_000 > floor, isf_000, np.nan)

    ax.plot(log_M, m058, 'firebrick', linewidth=1.8,
            label=r'ISF ($\varepsilon_\infty = 0.58$, conservative)',
            zorder=5)
    ax.plot(log_M, m030, 'firebrick', linewidth=1.2,
            linestyle=':', alpha=0.8,
            label=r'ISF ($\varepsilon = 0.30$, optimistic)',
            zorder=3)
    ax.plot(log_M, m000, 'firebrick', linewidth=1.0,
            linestyle='--', alpha=0.6,
            label=r'ISF ($\varepsilon = 0$, maximum)',
            zorder=3)

    # Shaded band between conservative and maximum
    ax.fill_between(log_M, np.nan_to_num(m058, nan=floor),
                    np.nan_to_num(m000, nan=floor),
                    color='firebrick', alpha=0.08, zorder=1)

    # Gilman-constrained region
    ax.axvspan(6, 9, alpha=0.08, color='steelblue', zorder=0)
    ax.text(7.5, 2e-5, 'CDM-consistent\n(Gilman+2020)',
            fontsize=7.5, ha='center', va='bottom',
            color='steelblue', style='italic')

    # Euclid sensitivity threshold
    ax.axvline(x=8.8, color='goldenrod', linestyle='--',
               linewidth=0.8, alpha=0.7, zorder=2)
    ax.text(9.0, 2e2, 'Euclid\nthreshold',
            fontsize=7, color='goldenrod', va='top', ha='left')

    # Meneghetti anomaly scale marker
    ax.annotate('Meneghetti\nGGSL anomaly',
                xy=(10.5, 0.8), xytext=(10.5, 8),
                fontsize=7, ha='center', color='forestgreen',
                arrowprops=dict(arrowstyle='->', color='forestgreen',
                                linewidth=0.8),
                zorder=6)

    # Crossover annotation
    ax.annotate('ISF exceeds\n' + r'$\Lambda$CDM',
                xy=(8.0, 1.3), xytext=(9.2, 20),
                fontsize=7, ha='center', color='grey',
                arrowprops=dict(arrowstyle='->', color='grey',
                                linewidth=0.6, linestyle='--'),
                zorder=3)

    # Log scale
    ax.set_yscale('log')
    ax.set_ylim(1e-5, 1e3)
    ax.set_xlim(6, 13)

    ax.set_xlabel(r'$\log_{10}(M / M_\odot)$')
    ax.set_ylabel(r'${\rm d}N / {\rm d}\log_{10} M$  '
                  r'[per lens sightline per dex]')

    ax.legend(loc='upper right', framealpha=0.9, edgecolor='none',
              fontsize=8)

    fig.tight_layout()
    outpath = os.path.join(FIG_DIR, 'paper6_fig2_mass_function.png')
    fig.savefig(outpath)
    print(f"  Figure 2 saved: {outpath}")
    plt.close(fig)
    return outpath


fig2_path = make_figure2()
print("Done.")
