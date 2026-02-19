#!/usr/bin/env python3
"""
phaseB_step1_fdm_proxy.py  (v2 — fixed K-corrections & IMF)
=============================================================
Phase B, Step 1: Compute dark matter fraction proxies for BOSS LOWZ galaxies.

v2 fixes:
  - K-corrections for E/S0 galaxies using Chilingarian & Zolotukhin (2010)
    polynomials. The v1 linear approximations underestimated the g-band
    K-correction by ~5x at z=0.3, causing 82% of galaxies to fail the
    rest-frame color cut.
  - Shifted to Chabrier IMF (-0.15 dex from Bell+2003 diet Salpeter),
    which was overestimating stellar masses by ~40%.
  - Added dual stellar mass estimation (color-based + constant M/L_i for
    LRGs) as internal consistency check.
  - Extensive diagnostics at every stage.

Inputs (in DATA_DIR):
  - lowz_vdisp_masses.csv
  - sdss_preprocessed.csv  (optional, for cross-check)

Outputs (in DATA_DIR):
  - phaseB_merged.csv
  - phaseB_step1_diagnostics.png
  - phaseB_step1_summary.txt

Usage:
  python3 phaseB_step1_fdm_proxy.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import quad

# ── Configuration ──────────────────────────────────────────────────
DATA_DIR = "/Users/chris/Documents/research_data/cosmic_structure"

# Cosmology (Planck 2018, flat LCDM)
H0 = 67.4          # km/s/Mpc
OMEGA_M = 0.315
C_LIGHT = 299792.458  # km/s

# Bell et al. (2003) Table 7: log10(M*/L_r) = a + b*(g-r)
# Original: diet Salpeter IMF → a=-0.306, b=1.097
# Chabrier IMF: shift by -0.15 dex (Bell+2003 Section 4.2)
BELL_A_R = -0.306 - 0.15   # = -0.456
BELL_B_R = 1.097

# Solar absolute magnitudes (AB system, Blanton & Roweis 2007)
MSUN_R = 4.65
MSUN_I = 4.53

# K-corrections for E/S0 galaxies (Chilingarian & Zolotukhin 2010, Table A1)
# K(z) = a1*z + a2*z^2, valid for z < 0.5
# These are much larger than the linear approximations used in v1
KCORR_COEFFS = {
    'u': (4.51, 19.6),
    'g': (2.45, 13.8),
    'r': (0.77,  5.6),
    'i': (0.35,  3.0),
    'z': (0.17,  1.3),
}

# Constant M*/L_i for LRGs (Chabrier IMF, Maraston+2013)
# Backup method — avoids color-mass relation entirely
LOG_ML_I_CONST = 0.35  # log10(M*/L_i) for old stellar pops, Chabrier IMF


# ── K-correction function ─────────────────────────────────────────
def kcorr(z, band):
    """K-correction for E/S0 galaxies.
    
    From Chilingarian & Zolotukhin (2010), polynomial fit.
    Valid for z < 0.5.
    """
    a1, a2 = KCORR_COEFFS[band]
    return a1 * z + a2 * z**2


# ── Cosmological distances ────────────────────────────────────────
def comoving_distance(z_arr):
    """Comoving distance in Mpc for flat LCDM."""
    def integrand(zp):
        return 1.0 / np.sqrt(OMEGA_M * (1 + zp)**3 + (1 - OMEGA_M))
    
    z_arr = np.atleast_1d(z_arr)
    result = np.zeros_like(z_arr, dtype=float)
    for i, z in enumerate(z_arr):
        val, _ = quad(integrand, 0, z)
        result[i] = (C_LIGHT / H0) * val
    return result


def luminosity_distance(z):
    """Luminosity distance in Mpc."""
    return comoving_distance(z) * (1 + z)


# ── Stellar mass functions ────────────────────────────────────────
def stellar_mass_color(Mr, gmr_rest):
    """Stellar mass via Bell+2003 color-mass relation (Chabrier IMF).
    
    Parameters
    ----------
    Mr : absolute r-band magnitude (K-corrected)
    gmr_rest : rest-frame (g-r) color
    
    Returns
    -------
    log10(M*/Msun)
    """
    log_Lr = -0.4 * (Mr - MSUN_R)       # log10(L_r / L_sun)
    log_ML = BELL_A_R + BELL_B_R * gmr_rest  # log10(M*/L_r)
    return log_ML + log_Lr


def stellar_mass_constant_ml(Mi):
    """Stellar mass via constant M/L_i for LRGs (Chabrier IMF).
    
    For passive early-type galaxies, M*/L_i is nearly constant.
    This avoids color and K-correction uncertainties in g-band.
    
    Parameters
    ----------
    Mi : absolute i-band magnitude (K-corrected)
    
    Returns
    -------
    log10(M*/Msun)
    """
    log_Li = -0.4 * (Mi - MSUN_I)
    return LOG_ML_I_CONST + log_Li


# ── Dynamical mass ────────────────────────────────────────────────
def dynamical_mass(sigma_v, R_eff_kpc):
    """Dynamical mass from velocity dispersion and effective radius.
    
    M_dyn = beta * sigma^2 * R_eff / G
    beta = 5.0 (Cappellari+2006 for early-type galaxies)
    """
    G_pc = 4.3009e-3   # pc (km/s)^2 / Msun
    R_eff_pc = R_eff_kpc * 1000
    beta = 5.0
    M_dyn = beta * sigma_v**2 * R_eff_pc / G_pc
    return np.log10(M_dyn)


# ── Main ──────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("Phase B Step 1 (v2): f_DM proxies for LOWZ galaxies")
    print("  K-corrections: Chilingarian & Zolotukhin (2010) for E/S0")
    print("  Stellar mass: Bell+2003 (Chabrier IMF, -0.15 dex shift)")
    print("=" * 65)

    # ── Load ──────────────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    vdisp_file = os.path.join(DATA_DIR, "lowz_vdisp_masses.csv")
    if not os.path.exists(vdisp_file):
        print(f"  ERROR: {vdisp_file} not found")
        sys.exit(1)
    
    df = pd.read_csv(vdisp_file)
    print(f"  Loaded {len(df)} rows")
    
    # Force numeric
    num_cols = ['ra', 'dec', 'z', 'zerr', 'veldisp', 'veldisperr',
                'dered_u', 'dered_g', 'dered_r', 'dered_i', 'dered_z',
                'petroR50_r', 'petroR90_r']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # ── Quality cuts ──────────────────────────────────────────────
    print("\n[2/6] Quality cuts...")
    n0 = len(df)
    
    mask_z = (df['z'] > 0.1) & (df['z'] < 0.5)
    mask_vd = (df['veldisp'] > 50) & (df['veldisp'] < 500) & (df['veldisperr'] < 100)
    mask_phot = ((df['dered_r'] > 10) & (df['dered_r'] < 25) &
                 (df['dered_g'] > 10) & (df['dered_g'] < 25) &
                 (df['dered_i'] > 10) & (df['dered_i'] < 25))
    mask_size = (df['petroR50_r'] > 0.5) & (df['petroR50_r'] < 60)
    
    mask_all = mask_z & mask_vd & mask_phot & mask_size
    
    print(f"  Redshift 0.1-0.5:    {mask_z.sum():>7,} pass")
    print(f"  Good veldisp:        {mask_vd.sum():>7,} pass")
    print(f"  Valid photometry:    {mask_phot.sum():>7,} pass")
    print(f"  Valid angular size:  {mask_size.sum():>7,} pass")
    print(f"  Combined:            {mask_all.sum():>7,} ({mask_all.sum()/n0*100:.1f}%)")
    
    df = df[mask_all].copy()
    df.reset_index(drop=True, inplace=True)

    # ── Diagnostic: raw observed colors ───────────────────────────
    print("\n[3/6] K-corrections & colors...")
    
    gmr_obs = df['dered_g'] - df['dered_r']
    rmi_obs = df['dered_r'] - df['dered_i']
    
    print(f"\n  OBSERVED colors (before K-correction):")
    print(f"    (g-r)_obs: median={gmr_obs.median():.3f}, "
          f"mean={gmr_obs.mean():.3f}, std={gmr_obs.std():.3f}")
    print(f"    (r-i)_obs: median={rmi_obs.median():.3f}, "
          f"mean={rmi_obs.mean():.3f}, std={rmi_obs.std():.3f}")
    print(f"    (g-r)_obs range: [{gmr_obs.min():.2f}, {gmr_obs.max():.2f}]")
    
    # K-corrections
    z = df['z'].values
    Kg = kcorr(z, 'g')
    Kr = kcorr(z, 'r')
    Ki = kcorr(z, 'i')
    
    print(f"\n  K-corrections at median z={np.median(z):.3f}:")
    print(f"    K_g = {kcorr(np.median(z), 'g'):.3f}")
    print(f"    K_r = {kcorr(np.median(z), 'r'):.3f}")
    print(f"    K_i = {kcorr(np.median(z), 'i'):.3f}")
    print(f"    K_g - K_r = {kcorr(np.median(z), 'g') - kcorr(np.median(z), 'r'):.3f}")
    print(f"  K-corrections at z=0.30:")
    print(f"    K_g = {kcorr(0.3, 'g'):.3f}")
    print(f"    K_r = {kcorr(0.3, 'r'):.3f}")
    print(f"    K_g - K_r = {kcorr(0.3, 'g') - kcorr(0.3, 'r'):.3f}")
    
    # Compare to v1:
    print(f"\n  v1 K-corrections at z=0.30 (for reference):")
    print(f"    K_g = {1.6*0.3:.3f}  (v1: 1.6z)")
    print(f"    K_r = {1.2*0.3:.3f}  (v1: 1.2z)")
    print(f"    K_g - K_r = {0.4*0.3:.3f}")
    print(f"    -> v2 differential is {(kcorr(0.3,'g')-kcorr(0.3,'r'))/(0.4*0.3):.1f}x larger!")
    
    # Absolute magnitudes (K-corrected)
    d_L = luminosity_distance(z)
    dist_mod = 5 * np.log10(d_L * 1e6 / 10)  # distance modulus
    
    df['Mr'] = df['dered_r'].values - dist_mod - Kr
    df['Mg'] = df['dered_g'].values - dist_mod - Kg
    df['Mi'] = df['dered_i'].values - dist_mod - Ki
    
    # Rest-frame colors
    df['gmr_rest'] = df['Mg'] - df['Mr']
    df['rmi_rest'] = df['Mr'] - df['Mi']
    
    print(f"\n  REST-FRAME colors (after K-correction):")
    print(f"    (g-r)_rest: median={df['gmr_rest'].median():.3f}, "
          f"mean={df['gmr_rest'].mean():.3f}, std={df['gmr_rest'].std():.3f}")
    print(f"    (g-r)_rest range: [{df['gmr_rest'].min():.2f}, {df['gmr_rest'].max():.2f}]")
    
    # Color cut (should now retain most galaxies)
    color_mask = (df['gmr_rest'] > 0.2) & (df['gmr_rest'] < 1.2)
    n_pass = color_mask.sum()
    n_fail = (~color_mask).sum()
    print(f"\n  Color cut 0.2 < (g-r)_rest < 1.2:")
    print(f"    Pass: {n_pass:,} ({n_pass/len(df)*100:.1f}%)")
    print(f"    Fail: {n_fail:,} ({n_fail/len(df)*100:.1f}%)")
    
    df = df[color_mask].copy()
    df.reset_index(drop=True, inplace=True)

    # ── Stellar masses ────────────────────────────────────────────
    print("\n[4/6] Computing masses...")
    
    # Method A: Color-mass relation (Bell+2003, Chabrier)
    df['log_Mstar_A'] = stellar_mass_color(df['Mr'].values, df['gmr_rest'].values)
    
    # Method B: Constant M/L_i for LRGs
    df['log_Mstar_B'] = stellar_mass_constant_ml(df['Mi'].values)
    
    # Primary: use Method A (color-based), Method B as cross-check
    df['log_Mstar'] = df['log_Mstar_A']
    
    print(f"  Method A (color-mass, Chabrier):")
    print(f"    log(M*/Msun): median={df['log_Mstar_A'].median():.2f}, "
          f"mean={df['log_Mstar_A'].mean():.2f}")
    print(f"  Method B (constant M/L_i):")
    print(f"    log(M*/Msun): median={df['log_Mstar_B'].median():.2f}, "
          f"mean={df['log_Mstar_B'].mean():.2f}")
    print(f"  Difference A-B: {(df['log_Mstar_A'] - df['log_Mstar_B']).median():.3f} dex "
          f"(median), {(df['log_Mstar_A'] - df['log_Mstar_B']).std():.3f} dex (scatter)")
    
    # Physical sizes
    d_c = comoving_distance(df['z'].values)
    d_A = d_c / (1 + df['z'].values)  # angular diameter distance, Mpc
    df['R50_kpc'] = df['petroR50_r'].values * d_A * 1000 / 206265
    
    size_mask = (df['R50_kpc'] > 0.5) & (df['R50_kpc'] < 50)
    print(f"\n  Physical size 0.5-50 kpc: {size_mask.sum():,} pass "
          f"(dropping {(~size_mask).sum():,})")
    df = df[size_mask].copy()
    df.reset_index(drop=True, inplace=True)
    
    print(f"  R_eff: median={df['R50_kpc'].median():.1f} kpc")
    
    # Dynamical mass
    df['log_Mdyn'] = dynamical_mass(df['veldisp'].values, df['R50_kpc'].values)
    
    print(f"\n  Dynamical mass:")
    print(f"    log(Mdyn/Msun): median={df['log_Mdyn'].median():.2f}")
    
    # ── f_DM proxy ────────────────────────────────────────────────
    print("\n[5/6] Computing f_DM...")
    
    df['fDM'] = 1.0 - 10**(df['log_Mstar'] - df['log_Mdyn'])
    
    # Also compute from Method B for comparison
    df['fDM_B'] = 1.0 - 10**(df['log_Mstar_B'] - df['log_Mdyn'])
    
    # Diagnostic: mass comparison
    mass_diff = df['log_Mstar'] - df['log_Mdyn']
    print(f"\n  log(M*/Mdyn): median={mass_diff.median():.3f}, "
          f"mean={mass_diff.mean():.3f}")
    print(f"    Fraction with M* > Mdyn (unphysical): "
          f"{(mass_diff > 0).sum():,} ({(mass_diff > 0).mean()*100:.1f}%)")
    
    # Keep galaxies with reasonable f_DM
    fdm_mask = (df['fDM'] > -0.5) & (df['fDM'] < 1.0)
    n_pass_fdm = fdm_mask.sum()
    n_fail_fdm = (~fdm_mask).sum()
    print(f"\n  f_DM cut [-0.5, 1.0]:")
    print(f"    Pass: {n_pass_fdm:,} ({n_pass_fdm/len(df)*100:.1f}%)")
    print(f"    Fail: {n_fail_fdm:,} ({n_fail_fdm/len(df)*100:.1f}%)")
    
    df = df[fdm_mask].copy()
    df.reset_index(drop=True, inplace=True)
    
    print(f"\n  *** Final sample: {len(df):,} galaxies ***")
    print(f"  f_DM (Method A): mean={df['fDM'].mean():.3f}, "
          f"median={df['fDM'].median():.3f}, std={df['fDM'].std():.3f}")
    print(f"  f_DM (Method B): mean={df['fDM_B'].mean():.3f}, "
          f"median={df['fDM_B'].median():.3f}, std={df['fDM_B'].std():.3f}")
    
    fdm_mean = df['fDM'].mean()
    if 0 < fdm_mean < 1:
        print(f"  Implied N (from mean f_DM): {1/(1 - fdm_mean):.1f}")
    else:
        print(f"  Implied N: f_DM outside valid range for N estimate")
    
    # ── 3D positions ──────────────────────────────────────────────
    print("\n[6/6] Computing 3D positions & saving...")
    
    d_c = comoving_distance(df['z'].values)
    ra_rad = np.radians(df['ra'].values)
    dec_rad = np.radians(df['dec'].values)
    
    df['d_comoving'] = d_c
    df['x'] = d_c * np.cos(dec_rad) * np.cos(ra_rad)
    df['y'] = d_c * np.cos(dec_rad) * np.sin(ra_rad)
    df['z_cart'] = d_c * np.sin(dec_rad)
    
    print(f"  Distance range: {d_c.min():.0f} - {d_c.max():.0f} Mpc")
    
    # Save
    out_csv = os.path.join(DATA_DIR, "phaseB_merged.csv")
    df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv} ({len(df):,} rows, "
          f"{os.path.getsize(out_csv)/1e6:.1f} MB)")
    
    # ── Diagnostics figure ────────────────────────────────────────
    print("\n  Generating diagnostic figure...")
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.38, wspace=0.35)
    
    # Panel 1: Redshift distribution
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(df['z'], bins=80, color='steelblue', edgecolor='navy', alpha=0.7)
    ax.set_xlabel('Redshift')
    ax.set_ylabel('Count')
    ax.set_title(f'Redshift (N={len(df):,})')
    ax.axvline(df['z'].median(), color='red', ls='--',
               label=f'med={df["z"].median():.3f}')
    ax.legend(fontsize=8)
    
    # Panel 2: K-correction magnitude
    ax = fig.add_subplot(gs[0, 1])
    z_grid = np.linspace(0.1, 0.5, 100)
    for band, color in [('g', 'green'), ('r', 'red'), ('i', 'orange')]:
        ax.plot(z_grid, kcorr(z_grid, band), color=color, lw=2, label=f'$K_{band}$')
    ax.plot(z_grid, kcorr(z_grid, 'g') - kcorr(z_grid, 'r'),
            color='purple', lw=2, ls='--', label='$K_g - K_r$')
    ax.plot(z_grid, 1.6*z_grid, color='green', lw=1, ls=':', alpha=0.5, label='v1 $K_g$')
    ax.plot(z_grid, 1.2*z_grid, color='red', lw=1, ls=':', alpha=0.5, label='v1 $K_r$')
    ax.set_xlabel('Redshift')
    ax.set_ylabel('K-correction (mag)')
    ax.set_title('K-corrections: v2 (solid) vs v1 (dotted)')
    ax.legend(fontsize=7, ncol=2)
    
    # Panel 3: Rest-frame color distribution
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(df['gmr_rest'], bins=80, color='salmon', edgecolor='darkred',
            alpha=0.7, range=(0.2, 1.1))
    ax.set_xlabel('$(g-r)_0$ (rest-frame)')
    ax.set_ylabel('Count')
    ax.set_title('Rest-frame color')
    ax.axvline(df['gmr_rest'].median(), color='navy', ls='--',
               label=f'med={df["gmr_rest"].median():.3f}')
    ax.legend(fontsize=8)
    
    # Panel 4: Color-magnitude diagram
    ax = fig.add_subplot(gs[1, 0])
    ax.hexbin(df['Mr'], df['gmr_rest'], gridsize=80, cmap='inferno',
              mincnt=1, extent=[-24, -19, 0.3, 1.1])
    ax.set_xlabel('$M_r$ (absolute)')
    ax.set_ylabel('$(g-r)_0$')
    ax.set_title('Color-magnitude')
    ax.invert_xaxis()
    
    # Panel 5: Stellar mass comparison (Method A vs B)
    ax = fig.add_subplot(gs[1, 1])
    ax.hexbin(df['log_Mstar_A'], df['log_Mstar_B'], gridsize=80,
              cmap='inferno', mincnt=1, extent=[10, 12.5, 10, 12.5])
    ax.plot([10, 12.5], [10, 12.5], 'w--', lw=1, alpha=0.7)
    ax.set_xlabel('$\\log M_*$ (Method A: color)')
    ax.set_ylabel('$\\log M_*$ (Method B: const $M/L_i$)')
    ax.set_title('Stellar mass cross-check')
    
    # Panel 6: Stellar vs dynamical mass
    ax = fig.add_subplot(gs[1, 2])
    ax.hexbin(df['log_Mdyn'], df['log_Mstar'], gridsize=80, cmap='inferno',
              mincnt=1, extent=[10, 13, 10, 13])
    ax.plot([10, 13], [10, 13], 'w--', lw=1, alpha=0.7, label='$M_* = M_{dyn}$')
    ax.set_xlabel('$\\log M_{dyn}$')
    ax.set_ylabel('$\\log M_*$ (Method A)')
    ax.set_title('Stellar vs dynamical mass')
    ax.legend(fontsize=8, loc='upper left')
    
    # Panel 7: f_DM distribution
    ax = fig.add_subplot(gs[2, 0])
    ax.hist(df['fDM'], bins=100, color='seagreen', edgecolor='darkgreen',
            alpha=0.7, range=(-0.5, 1.0))
    ax.axvline(df['fDM'].mean(), color='red', ls='--',
               label=f'mean={df["fDM"].mean():.3f}')
    ax.axvline(df['fDM'].median(), color='navy', ls=':',
               label=f'med={df["fDM"].median():.3f}')
    ax.axvline(5/6, color='orange', ls='-', lw=2, alpha=0.7,
               label='$f_{DM}=(N{-}1)/N$, N=6')
    ax.set_xlabel('$f_{DM}$ proxy')
    ax.set_ylabel('Count')
    ax.set_title('Dark matter fraction (Method A)')
    ax.legend(fontsize=7)
    
    # Panel 8: f_DM vs stellar mass
    ax = fig.add_subplot(gs[2, 1])
    ax.hexbin(df['log_Mstar'], df['fDM'], gridsize=80, cmap='inferno',
              mincnt=1, extent=[10, 12.5, -0.3, 1.0])
    ax.axhline(5/6, color='cyan', ls='--', alpha=0.7, label='N=6')
    ax.axhline(0, color='white', ls=':', alpha=0.5)
    ax.set_xlabel('$\\log M_*$')
    ax.set_ylabel('$f_{DM}$')
    ax.set_title('$f_{DM}$ vs stellar mass')
    ax.legend(fontsize=8)
    
    # Panel 9: f_DM vs redshift (check for systematics)
    ax = fig.add_subplot(gs[2, 2])
    ax.hexbin(df['z'], df['fDM'], gridsize=80, cmap='inferno',
              mincnt=1, extent=[0.1, 0.5, -0.3, 1.0])
    ax.axhline(0, color='white', ls=':', alpha=0.5)
    # Running median
    z_bins = np.linspace(0.1, 0.48, 20)
    z_mids = 0.5 * (z_bins[:-1] + z_bins[1:])
    fdm_medians = []
    for i in range(len(z_bins)-1):
        mask_bin = (df['z'] >= z_bins[i]) & (df['z'] < z_bins[i+1])
        if mask_bin.sum() > 10:
            fdm_medians.append(df.loc[mask_bin, 'fDM'].median())
        else:
            fdm_medians.append(np.nan)
    ax.plot(z_mids, fdm_medians, 'c-o', ms=3, lw=2, label='running median')
    ax.set_xlabel('Redshift')
    ax.set_ylabel('$f_{DM}$')
    ax.set_title('$f_{DM}$ vs redshift (systematics check)')
    ax.legend(fontsize=8)
    
    fig.suptitle('Phase B Step 1 (v2): BOSS LOWZ $f_{DM}$ Proxy Diagnostics',
                 fontsize=14, fontweight='bold', y=0.99)
    
    out_fig = os.path.join(DATA_DIR, "phaseB_step1_diagnostics.png")
    plt.savefig(out_fig, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_fig}")
    
    # ── Summary ───────────────────────────────────────────────────
    lines = []
    lines.append("Phase B Step 1 (v2): Summary")
    lines.append("=" * 55)
    lines.append(f"Final sample: {len(df):,} galaxies")
    lines.append(f"Redshift: {df['z'].min():.3f} - {df['z'].max():.3f} "
                 f"(median {df['z'].median():.3f})")
    lines.append(f"Velocity dispersion median: {df['veldisp'].median():.0f} km/s")
    lines.append(f"R_eff median: {df['R50_kpc'].median():.1f} kpc")
    lines.append(f"")
    lines.append(f"Stellar mass (Method A, color-mass, Chabrier):")
    lines.append(f"  log(M*/Msun) median: {df['log_Mstar_A'].median():.2f}")
    lines.append(f"Stellar mass (Method B, const M/L_i):")
    lines.append(f"  log(M*/Msun) median: {df['log_Mstar_B'].median():.2f}")
    lines.append(f"Dynamical mass:")
    lines.append(f"  log(Mdyn/Msun) median: {df['log_Mdyn'].median():.2f}")
    lines.append(f"")
    lines.append(f"f_DM (Method A):")
    lines.append(f"  mean:   {df['fDM'].mean():.4f}")
    lines.append(f"  median: {df['fDM'].median():.4f}")
    lines.append(f"  std:    {df['fDM'].std():.4f}")
    if 0 < df['fDM'].mean() < 1:
        lines.append(f"  Implied N: {1/(1 - df['fDM'].mean()):.2f}")
    lines.append(f"")
    lines.append(f"f_DM percentiles:")
    for p in [5, 25, 50, 75, 95]:
        lines.append(f"  {p:>3}th: {np.percentile(df['fDM'], p):.4f}")
    lines.append(f"")
    lines.append(f"Fraction with f_DM > 0:    {(df['fDM'] > 0).mean():.4f}")
    lines.append(f"Fraction with f_DM > 0.5:  {(df['fDM'] > 0.5).mean():.4f}")
    lines.append(f"Fraction with f_DM > 0.83: {(df['fDM'] > 0.83).mean():.4f}")
    
    summary = "\n".join(lines)
    print(f"\n{summary}")
    
    out_txt = os.path.join(DATA_DIR, "phaseB_step1_summary.txt")
    with open(out_txt, 'w') as f:
        f.write(summary)
    print(f"\n  Saved: {out_txt}")
    
    print("\n" + "=" * 65)
    print("Step 1 complete. Next: Step 2 (regression + residuals)")
    print("=" * 65)


if __name__ == "__main__":
    main()
