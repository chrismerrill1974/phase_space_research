#!/usr/bin/env python3
"""
Step 4: Generate Smooth Null Profiles & Compute Residuals
==========================================================
For each qualifying galaxy, fit three smooth null models to f_DM(R):
  Null A: NFW halo  -> smooth f_DM,NFW(R)
  Null B: Pseudo-isothermal halo (cored profile)
  Null C: LOWESS smooth (model-agnostic)

Then compute residuals: Δf(R) = f_DM,obs(R) - f_DM,null(R)

Output:
  - sparc_radial_nulls.csv: All points with null model values + residuals
  - step4_null_models.png: Multi-panel diagnostic
  - step4_null_examples.png: Example fits for selected galaxies

Author: C. K. Merrill
Date: 2026-02-10
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import UnivariateSpline
import csv
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────
PROFILES_CSV = "sparc_radial_profiles.csv"
CATALOG_CSV  = "sparc_dm_fractions.csv"
UPSILON      = 0.5

OUTPUT_CSV   = "sparc_radial_nulls.csv"
OUTPUT_MAIN  = "step4_null_models.png"
OUTPUT_EXAMPLES = "step4_null_examples.png"

MIN_PTS      = 10
MIN_RMAX_RD  = 3.0


# ── Load data ─────────────────────────────────────────────────────
def load_profiles(filepath):
    data = defaultdict(list)
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gname = row['Galaxy']
            data[gname].append({
                'R_kpc': float(row['R_kpc']),
                'R_Rd': float(row['R_Rd']),
                'fDM': float(row['fDM']),
                'e_fDM': float(row['e_fDM']),
                'Vobs': float(row['Vobs']),
                'eVobs': float(row['eVobs']),
                'Q': int(row['Q']),
                'Vflat': float(row['Vflat']),
                'T': int(row['T']),
                'Rdisk': float(row['Rdisk']),
                'L36': float(row['L36']),
                'Vgas': float(row['Vgas']),
                'Vdisk_raw': float(row['Vdisk_raw']),
                'Vbul_raw': float(row['Vbul_raw']),
            })
    return data


def load_catalog(filepath):
    meta = {}
    with open(filepath, 'r') as f:
        for row in csv.DictReader(f):
            meta[row['Galaxy']] = {
                'Vflat': float(row['Vflat_kms']),
                'Q': int(row['Q']),
                'Rdisk': float(row['Rdisk_kpc']),
                'Rmax': float(row['Rmax_kpc']),
                'T': int(row['T']),
                'L36': float(row['L36_1e9Lsun']),
                'Npts': int(row['Npts']),
            }
    return meta


def select_galaxies(data, meta):
    selected = []
    for gname, pts in data.items():
        if gname not in meta:
            continue
        m = meta[gname]
        if m['Q'] > 1 or m['Vflat'] <= 0:
            continue
        if len(pts) < MIN_PTS:
            continue
        Rmax = max(p['R_kpc'] for p in pts)
        Rd = m['Rdisk']
        if Rd <= 0 or Rmax / Rd < MIN_RMAX_RD:
            continue
        selected.append(gname)
    return sorted(selected)


def classify(meta, gname):
    vf = meta[gname]['Vflat']
    if vf < 80: return 'dwarf'
    elif vf < 150: return 'intermediate'
    else: return 'massive'


# ── Null Model A: NFW Halo ────────────────────────────────────────
def nfw_vdm2(R_kpc, Mvir, c):
    """NFW dark matter V^2 contribution at radius R.
    Mvir in 1e12 Msun, c = concentration.
    Returns V_DM^2 in (km/s)^2.
    """
    G = 4.302e-3  # pc (km/s)^2 / Msun
    Mvir_msun = Mvir * 1e12
    # Virial radius from Mvir (assuming rho_crit * 200)
    # Rvir ~ 206 * (Mvir/1e12)^(1/3) kpc  (approximate)
    Rvir = 206.0 * (Mvir)**(1.0/3.0)  # kpc
    Rs = Rvir / c
    x = R_kpc / Rs
    # Avoid x=0
    x = np.maximum(x, 1e-6)
    gc = np.log(1 + c) - c / (1 + c)
    gx = np.log(1 + x) - x / (1 + x)
    # V^2 = G * Mvir * gx / (R * gc)
    # Convert: G in pc, R in kpc -> need R in pc
    R_pc = R_kpc * 1e3
    V2 = G * Mvir_msun * gx / (R_pc * gc)
    return V2


def fit_nfw_to_galaxy(pts, upsilon=0.5):
    """Fit NFW halo to observed V_DM(R) for a galaxy.
    Returns (Mvir, c, V2_nfw_at_each_R) or None on failure.
    """
    R = np.array([p['R_kpc'] for p in pts])
    Vobs = np.array([p['Vobs'] for p in pts])
    Vgas = np.array([p['Vgas'] for p in pts])
    Vdisk = np.array([p['Vdisk_raw'] for p in pts])
    Vbul = np.array([p['Vbul_raw'] for p in pts])
    eVobs = np.array([p['eVobs'] for p in pts])

    Vbar2 = Vgas**2 + upsilon * Vdisk**2 + upsilon * Vbul**2
    VDM2_obs = Vobs**2 - Vbar2

    # Only fit where VDM2 > 0 and R > 0
    mask = (VDM2_obs > 0) & (R > 0)
    if mask.sum() < 4:
        return None

    R_fit = R[mask]
    VDM2_fit = VDM2_obs[mask]
    # Weight by 1/sigma^2; sigma_VDM2 ~ 2*Vobs*eVobs
    sigma = np.maximum(2 * Vobs[mask] * eVobs[mask], 1.0)

    def residuals(params):
        Mvir, c = params
        if Mvir <= 0 or c <= 1 or c > 100:
            return 1e20
        model = nfw_vdm2(R_fit, Mvir, c)
        return np.sum(((VDM2_fit - model) / sigma)**2)

    # Grid search for initial guess
    best_chi2 = 1e30
    best_p = (0.1, 10)
    for logM in np.linspace(-3, 2, 15):
        for c in [5, 8, 12, 18, 25, 40]:
            chi2 = residuals((10**logM, c))
            if chi2 < best_chi2:
                best_chi2 = chi2
                best_p = (10**logM, c)

    try:
        result = minimize(residuals, best_p, method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-8})
        Mvir, c = result.x
        if Mvir <= 0 or c <= 1 or c > 100:
            return None
    except:
        return None

    # Compute NFW V^2 at all radii
    V2_nfw = nfw_vdm2(R, Mvir, c)
    # f_DM,NFW = V2_nfw / (V2_nfw + Vbar2)
    Vtot2_nfw = V2_nfw + Vbar2
    Vtot2_nfw = np.maximum(Vtot2_nfw, 1e-10)
    fDM_nfw = V2_nfw / Vtot2_nfw

    return Mvir, c, fDM_nfw


# ── Null Model B: Pseudo-Isothermal Halo ──────────────────────────
def iso_vdm2(R_kpc, rho0, Rc):
    """Pseudo-isothermal halo V^2 at radius R.
    rho0 in Msun/pc^3, Rc in kpc.
    V^2 = 4*pi*G*rho0*Rc^2 * [1 - (Rc/R)*atan(R/Rc)]
    """
    G = 4.302e-3  # pc (km/s)^2 / Msun
    Rc_pc = Rc * 1e3
    R_pc = R_kpc * 1e3
    x = R_pc / Rc_pc
    x = np.maximum(x, 1e-6)
    V2 = 4 * np.pi * G * rho0 * Rc_pc**2 * (1 - np.arctan(x) / x)
    return V2


def fit_iso_to_galaxy(pts, upsilon=0.5):
    """Fit pseudo-isothermal halo to observed V_DM(R)."""
    R = np.array([p['R_kpc'] for p in pts])
    Vobs = np.array([p['Vobs'] for p in pts])
    Vgas = np.array([p['Vgas'] for p in pts])
    Vdisk = np.array([p['Vdisk_raw'] for p in pts])
    Vbul = np.array([p['Vbul_raw'] for p in pts])
    eVobs = np.array([p['eVobs'] for p in pts])

    Vbar2 = Vgas**2 + upsilon * Vdisk**2 + upsilon * Vbul**2
    VDM2_obs = Vobs**2 - Vbar2

    mask = (VDM2_obs > 0) & (R > 0)
    if mask.sum() < 4:
        return None

    R_fit = R[mask]
    VDM2_fit = VDM2_obs[mask]
    sigma = np.maximum(2 * Vobs[mask] * eVobs[mask], 1.0)

    def residuals(params):
        log_rho0, log_Rc = params
        rho0 = 10**log_rho0
        Rc = 10**log_Rc
        if Rc < 0.01 or Rc > 100:
            return 1e20
        model = iso_vdm2(R_fit, rho0, Rc)
        return np.sum(((VDM2_fit - model) / sigma)**2)

    best_chi2 = 1e30
    best_p = (-2, 0)
    for lr in np.linspace(-4, 0, 12):
        for lc in np.linspace(-1.5, 2, 12):
            chi2 = residuals((lr, lc))
            if chi2 < best_chi2:
                best_chi2 = chi2
                best_p = (lr, lc)

    try:
        result = minimize(residuals, best_p, method='Nelder-Mead',
                         options={'maxiter': 5000})
        rho0 = 10**result.x[0]
        Rc = 10**result.x[1]
        if Rc < 0.01 or Rc > 100:
            return None
    except:
        return None

    V2_iso = iso_vdm2(R, rho0, Rc)
    Vtot2_iso = V2_iso + Vbar2
    Vtot2_iso = np.maximum(Vtot2_iso, 1e-10)
    fDM_iso = V2_iso / Vtot2_iso

    return rho0, Rc, fDM_iso


# ── Null Model C: LOWESS / Smoothing Spline ───────────────────────
def fit_smooth_spline(pts):
    """Fit a smoothing spline to f_DM(R/Rd) as model-agnostic smooth null."""
    sorted_pts = sorted(pts, key=lambda p: p['R_Rd'])
    R = np.array([p['R_Rd'] for p in sorted_pts])
    f = np.array([p['fDM'] for p in sorted_pts])
    e = np.array([p['e_fDM'] for p in sorted_pts])

    # Weights: 1/sigma, with floor
    w = 1.0 / np.maximum(e, 0.01)

    if len(R) < 5:
        return None

    try:
        # Smoothing parameter: len(R) gives GCV-like behavior
        s = len(R) * 1.0  # moderate smoothing
        spline = UnivariateSpline(R, f, w=w, s=s, k=3)
        f_smooth = spline(R)
        return f_smooth
    except:
        return None


# ── Main ──────────────────────────────────────────────────────────
def main():
    print("Step 4: Generate Smooth Null Profiles")
    print("=" * 60)

    data = load_profiles(PROFILES_CSV)
    meta = load_catalog(CATALOG_CSV)
    selected = select_galaxies(data, meta)

    print(f"Fitting null models for {len(selected)} galaxies...")

    # Results storage
    all_rows = []
    fit_stats = {'nfw_ok': 0, 'iso_ok': 0, 'spline_ok': 0,
                 'nfw_fail': 0, 'iso_fail': 0, 'spline_fail': 0}
    galaxy_fits = {}

    for gi, gname in enumerate(selected):
        pts = sorted(data[gname], key=lambda p: p['R_kpc'])
        n = len(pts)

        # Observed
        R_kpc = np.array([p['R_kpc'] for p in pts])
        R_Rd = np.array([p['R_Rd'] for p in pts])
        fDM_obs = np.array([p['fDM'] for p in pts])
        e_fDM = np.array([p['e_fDM'] for p in pts])

        # Fit NFW
        nfw_result = fit_nfw_to_galaxy(pts, UPSILON)
        if nfw_result is not None:
            Mvir, c, fDM_nfw = nfw_result
            fit_stats['nfw_ok'] += 1
        else:
            fDM_nfw = np.full(n, np.nan)
            Mvir, c = np.nan, np.nan
            fit_stats['nfw_fail'] += 1

        # Fit ISO
        iso_result = fit_iso_to_galaxy(pts, UPSILON)
        if iso_result is not None:
            rho0, Rc, fDM_iso = iso_result
            fit_stats['iso_ok'] += 1
        else:
            fDM_iso = np.full(n, np.nan)
            rho0, Rc = np.nan, np.nan
            fit_stats['iso_fail'] += 1

        # Fit spline
        spline_result = fit_smooth_spline(pts)
        if spline_result is not None:
            fDM_spline = spline_result
            fit_stats['spline_ok'] += 1
        else:
            fDM_spline = np.full(n, np.nan)
            fit_stats['spline_fail'] += 1

        # Residuals
        res_nfw = fDM_obs - fDM_nfw
        res_iso = fDM_obs - fDM_iso
        res_spline = fDM_obs - fDM_spline

        # Store fits for this galaxy
        galaxy_fits[gname] = {
            'R_Rd': R_Rd, 'R_kpc': R_kpc,
            'fDM_obs': fDM_obs, 'e_fDM': e_fDM,
            'fDM_nfw': fDM_nfw, 'fDM_iso': fDM_iso, 'fDM_spline': fDM_spline,
            'res_nfw': res_nfw, 'res_iso': res_iso, 'res_spline': res_spline,
            'Mvir': Mvir, 'c': c, 'rho0': rho0 if iso_result else np.nan,
            'Rc': Rc if iso_result else np.nan,
        }

        # Build output rows
        for i in range(n):
            all_rows.append({
                'Galaxy': gname,
                'R_kpc': f"{R_kpc[i]:.4f}",
                'R_Rd': f"{R_Rd[i]:.4f}",
                'fDM_obs': f"{fDM_obs[i]:.6f}",
                'e_fDM': f"{e_fDM[i]:.6f}",
                'fDM_nfw': f"{fDM_nfw[i]:.6f}" if np.isfinite(fDM_nfw[i]) else "",
                'fDM_iso': f"{fDM_iso[i]:.6f}" if np.isfinite(fDM_iso[i]) else "",
                'fDM_spline': f"{fDM_spline[i]:.6f}" if np.isfinite(fDM_spline[i]) else "",
                'res_nfw': f"{res_nfw[i]:.6f}" if np.isfinite(res_nfw[i]) else "",
                'res_iso': f"{res_iso[i]:.6f}" if np.isfinite(res_iso[i]) else "",
                'res_spline': f"{res_spline[i]:.6f}" if np.isfinite(res_spline[i]) else "",
            })

        if (gi + 1) % 20 == 0:
            print(f"  Fitted {gi+1}/{len(selected)} galaxies...")

    print(f"\nFit statistics:")
    print(f"  NFW:    {fit_stats['nfw_ok']} OK, {fit_stats['nfw_fail']} failed")
    print(f"  ISO:    {fit_stats['iso_ok']} OK, {fit_stats['iso_fail']} failed")
    print(f"  Spline: {fit_stats['spline_ok']} OK, {fit_stats['spline_fail']} failed")

    # ── Write CSV ─────────────────────────────────────────────────
    print(f"\nWriting {OUTPUT_CSV}...")
    fields = ['Galaxy', 'R_kpc', 'R_Rd', 'fDM_obs', 'e_fDM',
              'fDM_nfw', 'fDM_iso', 'fDM_spline',
              'res_nfw', 'res_iso', 'res_spline']
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"  Wrote {len(all_rows)} rows")

    # ── Collect all residuals ─────────────────────────────────────
    all_res_nfw = []
    all_res_iso = []
    all_res_spline = []
    all_fDM_for_res = []

    for gname in selected:
        gf = galaxy_fits[gname]
        for i in range(len(gf['R_Rd'])):
            if gf['R_Rd'][i] > 1.0:  # Skip innermost baryon-dominated
                if np.isfinite(gf['res_nfw'][i]):
                    all_res_nfw.append(gf['res_nfw'][i])
                if np.isfinite(gf['res_iso'][i]):
                    all_res_iso.append(gf['res_iso'][i])
                if np.isfinite(gf['res_spline'][i]):
                    all_res_spline.append(gf['res_spline'][i])
                    all_fDM_for_res.append(gf['fDM_obs'][i])

    all_res_nfw = np.array(all_res_nfw)
    all_res_iso = np.array(all_res_iso)
    all_res_spline = np.array(all_res_spline)
    all_fDM_for_res = np.array(all_fDM_for_res)

    # RMS residuals
    rms_nfw = np.sqrt(np.mean(all_res_nfw**2))
    rms_iso = np.sqrt(np.mean(all_res_iso**2))
    rms_spline = np.sqrt(np.mean(all_res_spline**2))
    print(f"\nRMS residuals (R/Rd > 1):")
    print(f"  NFW:    {rms_nfw:.4f}")
    print(f"  ISO:    {rms_iso:.4f}")
    print(f"  Spline: {rms_spline:.4f}")

    # ══════════════════════════════════════════════════════════════
    # FIGURE 1: Example fits for 12 representative galaxies
    # ══════════════════════════════════════════════════════════════
    print(f"\nGenerating {OUTPUT_EXAMPLES}...")

    # Pick 4 from each mass class
    dwarfs = [g for g in selected if classify(meta, g) == 'dwarf']
    intermediate = [g for g in selected if classify(meta, g) == 'intermediate']
    massive_list = [g for g in selected if classify(meta, g) == 'massive']

    # Pick galaxies with most points for best visual
    def pick_best(glist, n=4):
        return sorted(glist, key=lambda g: len(data[g]), reverse=True)[:n]

    examples = pick_best(massive_list, 4) + pick_best(intermediate, 4) + pick_best(dwarfs, 4)

    fig, axes = plt.subplots(4, 3, figsize=(16, 16))
    fig.suptitle('Step 4: Null Model Fits — Representative Galaxies',
                 fontsize=14, fontweight='bold')

    for idx, gname in enumerate(examples):
        row, col = divmod(idx, 3)
        ax = axes[row][col]
        gf = galaxy_fits[gname]

        ax.errorbar(gf['R_Rd'], gf['fDM_obs'], yerr=gf['e_fDM'],
                    fmt='o', ms=3, color='black', capsize=1.5,
                    elinewidth=0.5, label='Observed', zorder=5)
        if np.any(np.isfinite(gf['fDM_nfw'])):
            ax.plot(gf['R_Rd'], gf['fDM_nfw'], '-', color='red',
                    lw=1.5, label='NFW', alpha=0.8)
        if np.any(np.isfinite(gf['fDM_iso'])):
            ax.plot(gf['R_Rd'], gf['fDM_iso'], '-', color='blue',
                    lw=1.5, label='ISO', alpha=0.8)
        if np.any(np.isfinite(gf['fDM_spline'])):
            ax.plot(gf['R_Rd'], gf['fDM_spline'], '-', color='green',
                    lw=1.5, label='Spline', alpha=0.8)

        for N in range(3, 9):
            ax.axhline((N-1)/N, color='orange', ls='--', alpha=0.25, lw=0.5)

        vf = meta[gname]['Vflat']
        cls = classify(meta, gname)
        ax.set_title(f'{gname} (Vf={vf:.0f}, {cls})', fontsize=9)
        ax.set_ylim(-0.5, 1.1)
        ax.set_xlabel('R / R$_d$', fontsize=8)
        ax.set_ylabel('f$_{DM}$', fontsize=8)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=7, loc='lower right')

    plt.tight_layout()
    plt.savefig(OUTPUT_EXAMPLES, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {OUTPUT_EXAMPLES}")

    # ══════════════════════════════════════════════════════════════
    # FIGURE 2: Main diagnostic (9 panels)
    # ══════════════════════════════════════════════════════════════
    print(f"\nGenerating {OUTPUT_MAIN}...")

    fig = plt.figure(figsize=(18, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle('Step 4: Smooth Null Models — Residual Analysis',
                 fontsize=15, fontweight='bold', y=0.98)

    # Panel (0,0): Residual histograms for all three nulls
    ax = fig.add_subplot(gs[0, 0])
    bins = np.arange(-0.15, 0.155, 0.005)
    ax.hist(all_res_nfw, bins=bins, alpha=0.5, color='red',
            label=f'NFW (RMS={rms_nfw:.4f})', density=True)
    ax.hist(all_res_iso, bins=bins, alpha=0.5, color='blue',
            label=f'ISO (RMS={rms_iso:.4f})', density=True)
    ax.hist(all_res_spline, bins=bins, alpha=0.5, color='green',
            label=f'Spline (RMS={rms_spline:.4f})', density=True)
    ax.axvline(0, color='black', ls='-', lw=0.5)
    ax.set_xlabel('Residual: f$_{DM,obs}$ − f$_{DM,null}$')
    ax.set_ylabel('Density')
    ax.set_title('Residual distributions (R/R$_d$ > 1)')
    ax.legend(fontsize=7)

    # Panel (0,1): Residuals vs f_DM (check for structure at predicted levels)
    ax = fig.add_subplot(gs[0, 1])
    # Use spline residuals as most model-agnostic
    ax.scatter(all_fDM_for_res, all_res_spline, s=2, alpha=0.2,
               color='green')
    for N in range(3, 9):
        ax.axvline((N-1)/N, color='red', ls='--', alpha=0.4, lw=0.8)
    ax.axhline(0, color='black', ls='-', lw=0.5)
    ax.set_xlabel('f$_{DM,obs}$')
    ax.set_ylabel('Spline residual')
    ax.set_title('Residuals vs observed f$_{DM}$')
    ax.set_xlim(-0.2, 1.05)
    ax.set_ylim(-0.15, 0.15)

    # Panel (0,2): Stacked residuals vs R/Rd
    ax = fig.add_subplot(gs[0, 2])
    R_bins = np.arange(0, 15.5, 0.5)
    R_centers = 0.5 * (R_bins[:-1] + R_bins[1:])

    for null_name, color in [('nfw', 'red'), ('iso', 'blue'), ('spline', 'green')]:
        medians = []
        q25 = []
        q75 = []
        for j in range(len(R_centers)):
            vals = []
            for gname in selected:
                gf = galaxy_fits[gname]
                key = f'res_{null_name}'
                for i in range(len(gf['R_Rd'])):
                    if R_bins[j] <= gf['R_Rd'][i] < R_bins[j+1]:
                        if np.isfinite(gf[key][i]):
                            vals.append(gf[key][i])
            if len(vals) >= 3:
                medians.append(np.median(vals))
                q25.append(np.percentile(vals, 25))
                q75.append(np.percentile(vals, 75))
            else:
                medians.append(np.nan)
                q25.append(np.nan)
                q75.append(np.nan)
        medians = np.array(medians)
        q25 = np.array(q25)
        q75 = np.array(q75)
        m = ~np.isnan(medians)
        ax.plot(R_centers[m], medians[m], 'o-', color=color, ms=3, lw=1,
                label=null_name.upper())
        ax.fill_between(R_centers[m], q25[m], q75[m], alpha=0.1, color=color)
    ax.axhline(0, color='black', ls='-', lw=0.5)
    ax.set_xlabel('R / R$_d$')
    ax.set_ylabel('Median residual')
    ax.set_title('Stacked residuals vs radius')
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.08, 0.08)
    ax.legend(fontsize=7)

    # Panel (1,0): Stacked residuals at predicted (N-1)/N values
    ax = fig.add_subplot(gs[1, 0])
    # Bin residuals by their observed f_DM, check if residuals are
    # systematically different near predicted levels
    fDM_bins = np.arange(0.0, 1.01, 0.02)
    fDM_centers = 0.5 * (fDM_bins[:-1] + fDM_bins[1:])
    med_res = []
    cnt_res = []
    for j in range(len(fDM_centers)):
        mask = (all_fDM_for_res >= fDM_bins[j]) & (all_fDM_for_res < fDM_bins[j+1])
        vals = all_res_spline[mask]
        if len(vals) >= 3:
            med_res.append(np.median(vals))
            cnt_res.append(len(vals))
        else:
            med_res.append(np.nan)
            cnt_res.append(0)
    med_res = np.array(med_res)
    cnt_res = np.array(cnt_res)
    m = ~np.isnan(med_res)
    ax.bar(fDM_centers[m], med_res[m], width=0.018, color='steelblue', alpha=0.7)
    for N in range(3, 9):
        ax.axvline((N-1)/N, color='red', ls='--', alpha=0.5, lw=1)
        ax.text((N-1)/N, 0.012, f'N={N}', fontsize=7, color='red',
                ha='center')
    ax.axhline(0, color='black', ls='-', lw=0.5)
    ax.set_xlabel('f$_{DM,obs}$')
    ax.set_ylabel('Median spline residual')
    ax.set_title('Residual structure at predicted levels')
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(-0.015, 0.015)

    # Panel (1,1): Per-galaxy RMS residuals (NFW vs ISO vs Spline)
    ax = fig.add_subplot(gs[1, 1])
    rms_per_gal_nfw = []
    rms_per_gal_iso = []
    rms_per_gal_spline = []
    for gname in selected:
        gf = galaxy_fits[gname]
        mask = gf['R_Rd'] > 1.0
        rn = gf['res_nfw'][mask]
        ri = gf['res_iso'][mask]
        rs = gf['res_spline'][mask]
        if np.any(np.isfinite(rn)) and np.any(np.isfinite(ri)):
            rms_per_gal_nfw.append(np.sqrt(np.nanmean(rn**2)))
            rms_per_gal_iso.append(np.sqrt(np.nanmean(ri**2)))
        if np.any(np.isfinite(rs)):
            rms_per_gal_spline.append(np.sqrt(np.nanmean(rs**2)))

    ax.hist(rms_per_gal_nfw, bins=20, alpha=0.5, color='red', label='NFW')
    ax.hist(rms_per_gal_iso, bins=20, alpha=0.5, color='blue', label='ISO')
    ax.hist(rms_per_gal_spline, bins=20, alpha=0.5, color='green', label='Spline')
    ax.set_xlabel('Per-galaxy RMS residual')
    ax.set_ylabel('Count')
    ax.set_title('Fit quality distribution')
    ax.legend(fontsize=7)

    # Panel (1,2): NFW vs ISO comparison (which fits better?)
    ax = fig.add_subplot(gs[1, 2])
    rms_n = np.array(rms_per_gal_nfw)
    rms_i = np.array(rms_per_gal_iso[:len(rms_n)])
    ax.scatter(rms_n, rms_i, s=15, alpha=0.6, color='purple')
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], 'k--', lw=0.5, alpha=0.5)
    nfw_better = np.sum(rms_n < rms_i)
    iso_better = np.sum(rms_i < rms_n)
    ax.set_xlabel('NFW RMS residual')
    ax.set_ylabel('ISO RMS residual')
    ax.set_title(f'NFW vs ISO (NFW better: {nfw_better}, ISO: {iso_better})')
    ax.set_aspect('equal')

    # Panel (2,0): Stacked residuals by mass class (spline)
    ax = fig.add_subplot(gs[2, 0])
    for label, gnames, color in [
            ('Dwarfs', dwarfs, 'blue'),
            ('Interm.', intermediate, 'green'),
            ('Massive', massive_list, 'red')]:
        medians = []
        for j in range(len(R_centers)):
            vals = []
            for gname in gnames:
                if gname not in galaxy_fits:
                    continue
                gf = galaxy_fits[gname]
                for i in range(len(gf['R_Rd'])):
                    if R_bins[j] <= gf['R_Rd'][i] < R_bins[j+1]:
                        if np.isfinite(gf['res_spline'][i]):
                            vals.append(gf['res_spline'][i])
            if len(vals) >= 3:
                medians.append(np.median(vals))
            else:
                medians.append(np.nan)
        medians = np.array(medians)
        m = ~np.isnan(medians)
        ax.plot(R_centers[m], medians[m], 'o-', color=color, ms=3, lw=1.5,
                label=label)
    ax.axhline(0, color='black', ls='-', lw=0.5)
    ax.set_xlabel('R / R$_d$')
    ax.set_ylabel('Median spline residual')
    ax.set_title('Spline residuals by mass class')
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.06, 0.06)
    ax.legend(fontsize=7)

    # Panel (2,1): Autocorrelation of residuals (test for step-like structure)
    ax = fig.add_subplot(gs[2, 1])
    # For each galaxy, compute lag-1 autocorrelation of spline residuals
    autocorrs = []
    for gname in selected:
        gf = galaxy_fits[gname]
        res = gf['res_spline']
        if len(res) < 5 or not np.all(np.isfinite(res)):
            continue
        # Lag-1 autocorrelation
        res_centered = res - np.mean(res)
        var = np.var(res_centered)
        if var > 0:
            ac1 = np.mean(res_centered[:-1] * res_centered[1:]) / var
            autocorrs.append(ac1)

    autocorrs = np.array(autocorrs)
    ax.hist(autocorrs, bins=25, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='black', ls='-', lw=0.5)
    ax.axvline(np.median(autocorrs), color='red', ls='--',
               label=f'Median = {np.median(autocorrs):.3f}')
    ax.set_xlabel('Lag-1 autocorrelation of spline residuals')
    ax.set_ylabel('Count')
    ax.set_title('Residual autocorrelation (positive → correlated structure)')
    ax.legend(fontsize=8)

    # Panel (2,2): Summary
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')

    summary = (
        f"STEP 4 SUMMARY\n"
        f"{'='*42}\n\n"
        f"Galaxies fitted: {len(selected)}\n\n"
        f"Fit success rates:\n"
        f"  NFW:    {fit_stats['nfw_ok']}/{len(selected)}\n"
        f"  ISO:    {fit_stats['iso_ok']}/{len(selected)}\n"
        f"  Spline: {fit_stats['spline_ok']}/{len(selected)}\n\n"
        f"RMS residuals (R/Rd > 1):\n"
        f"  NFW:    {rms_nfw:.4f}\n"
        f"  ISO:    {rms_iso:.4f}\n"
        f"  Spline: {rms_spline:.4f}\n\n"
        f"NFW better than ISO: {nfw_better}/{len(rms_n)}\n"
        f"ISO better than NFW: {iso_better}/{len(rms_n)}\n\n"
        f"Median lag-1 autocorrelation:\n"
        f"  {np.median(autocorrs):.3f}\n"
        f"  (>0 = correlated structure;\n"
        f"   plateaus would give positive)\n\n"
        f"Predicted step Δf(N=4→5): 0.050\n"
        f"Predicted step Δf(N=5→6): 0.033\n"
        f"Spline RMS:               {rms_spline:.4f}\n"
        f"  → Steps at 2× spline RMS\n"
        f"    would be detectable"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=9, fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.savefig(OUTPUT_MAIN, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {OUTPUT_MAIN}")

    print("\nStep 4 complete.")


if __name__ == '__main__':
    main()
