#!/usr/bin/env python3
"""
Step 3: Exploratory Data Analysis — Radial f_DM(R) Profiles
=============================================================
Examine the shape of f_DM(R) profiles within individual galaxies
and across stacked samples. This is the first look at profile
morphology after predictions were sealed in Step 2.

Analyses:
  - Individual profiles for all qualifying galaxies, grouped by mass
  - Stacked median profiles by mass class with IQR
  - Error budget: can we resolve predicted steps?
  - Derivative df_DM/dR analysis
  - Histogram of f_DM values at all radii (pooled)
  - Upsilon_star sensitivity
  - Bulge vs no-bulge comparison

Output:
  - step3_radial_eda.png: 12-panel diagnostic figure
  - step3_individual_profiles.png: All individual profiles by mass class

Author: C. K. Merrill
Date: 2026-02-10
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import csv
from collections import defaultdict

# ── Configuration ──────────────────────────────────────────────────
PROFILES_CSV = "sparc_radial_profiles.csv"
CATALOG_CSV  = "sparc_dm_fractions.csv"
MASS_MODELS  = "MassModels_Lelli2016c.mrt"
UPSILON      = 0.5

OUTPUT_MAIN  = "step3_radial_eda.png"
OUTPUT_INDIV = "step3_individual_profiles.png"

# Selection from Step 1
MIN_PTS     = 10
MIN_RMAX_RD = 3.0


# ── Load data ─────────────────────────────────────────────────────
def load_profiles(filepath):
    """Load the radial profiles CSV from Step 1."""
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
    """Load Paper 1 catalog for metadata."""
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
                'SBeff': float(row['SBeff']),
            }
    return meta


def select_galaxies(data, meta):
    """Apply selection cuts, return list of qualifying names."""
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
    """Return mass class string."""
    vf = meta[gname]['Vflat']
    if vf < 80:
        return 'dwarf'
    elif vf < 150:
        return 'intermediate'
    else:
        return 'massive'


def has_bulge(data, gname):
    """Check if galaxy has a bulge component."""
    return any(abs(p['Vbul_raw']) > 0.1 for p in data[gname])


def compute_fdm_at_upsilon(pts, upsilon):
    """Recompute f_DM for a different Upsilon_star."""
    results = []
    for p in pts:
        Vobs2 = p['Vobs']**2
        if Vobs2 <= 0:
            continue
        Vbar2 = p['Vgas']**2 + upsilon * p['Vdisk_raw']**2 + upsilon * p['Vbul_raw']**2
        fDM = (Vobs2 - Vbar2) / Vobs2
        results.append((p['R_Rd'], fDM))
    return results


# ── Stacking helper ───────────────────────────────────────────────
def stack_profiles(data, gnames, R_bins):
    """Compute stacked median, 25th, 75th percentile in R/Rd bins."""
    R_centers = 0.5 * (R_bins[:-1] + R_bins[1:])
    medians = np.full(len(R_centers), np.nan)
    q25 = np.full(len(R_centers), np.nan)
    q75 = np.full(len(R_centers), np.nan)
    counts = np.zeros(len(R_centers), dtype=int)

    for j in range(len(R_centers)):
        vals = []
        for gname in gnames:
            for p in data[gname]:
                if R_bins[j] <= p['R_Rd'] < R_bins[j+1]:
                    vals.append(p['fDM'])
        if len(vals) >= 3:
            medians[j] = np.median(vals)
            q25[j] = np.percentile(vals, 25)
            q75[j] = np.percentile(vals, 75)
            counts[j] = len(vals)

    return R_centers, medians, q25, q75, counts


def compute_derivatives(pts):
    """Compute numerical derivative df_DM/d(R/Rd) using central differences."""
    # Sort by R_Rd
    sorted_pts = sorted(pts, key=lambda p: p['R_Rd'])
    R = np.array([p['R_Rd'] for p in sorted_pts])
    f = np.array([p['fDM'] for p in sorted_pts])

    if len(R) < 3:
        return np.array([]), np.array([])

    # Central differences for interior, forward/backward at edges
    dfdR = np.zeros(len(R))
    for i in range(1, len(R)-1):
        dR = R[i+1] - R[i-1]
        if dR > 0:
            dfdR[i] = (f[i+1] - f[i-1]) / dR
    # Edges
    if R[1] - R[0] > 0:
        dfdR[0] = (f[1] - f[0]) / (R[1] - R[0])
    if R[-1] - R[-2] > 0:
        dfdR[-1] = (f[-1] - f[-2]) / (R[-1] - R[-2])

    return R, dfdR


# ── Main ──────────────────────────────────────────────────────────
def main():
    print("Step 3: Exploratory Data Analysis — Radial Profiles")
    print("=" * 60)

    data = load_profiles(PROFILES_CSV)
    meta = load_catalog(CATALOG_CSV)
    selected = select_galaxies(data, meta)

    dwarfs = [g for g in selected if classify(meta, g) == 'dwarf']
    intermediate = [g for g in selected if classify(meta, g) == 'intermediate']
    massive = [g for g in selected if classify(meta, g) == 'massive']

    print(f"Selected: {len(selected)} galaxies")
    print(f"  Dwarfs: {len(dwarfs)}, Intermediate: {len(intermediate)}, "
          f"Massive: {len(massive)}")

    bulge_gals = [g for g in selected if has_bulge(data, g)]
    nobulge_gals = [g for g in selected if not has_bulge(data, g)]
    print(f"  With bulge: {len(bulge_gals)}, No bulge: {len(nobulge_gals)}")

    R_bins = np.arange(0, 16.0, 0.5)

    # ══════════════════════════════════════════════════════════════
    # FIGURE 1: Individual profiles for all galaxies
    # ══════════════════════════════════════════════════════════════
    print(f"\nGenerating {OUTPUT_INDIV}...")

    n_cols = 6
    for class_name, gnames, fname_suffix in [
            ('Dwarfs', dwarfs, 'dwarfs'),
            ('Intermediate', intermediate, 'intermediate'),
            ('Massive', massive, 'massive')]:

        n_gals = len(gnames)
        if n_gals == 0:
            continue
        n_rows = int(np.ceil(n_gals / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(3.2*n_cols, 2.8*n_rows),
                                  squeeze=False)
        fig.suptitle(f'Individual f$_{{DM}}$(R) Profiles — {class_name} '
                     f'(N={n_gals})', fontsize=13, fontweight='bold')

        # Sort by Vflat
        gnames_sorted = sorted(gnames, key=lambda g: meta[g]['Vflat'])

        for idx, gname in enumerate(gnames_sorted):
            row, col = divmod(idx, n_cols)
            ax = axes[row][col]
            pts = data[gname]
            R_Rd = [p['R_Rd'] for p in pts]
            fDM = [p['fDM'] for p in pts]
            e_fDM = [p['e_fDM'] for p in pts]

            ax.errorbar(R_Rd, fDM, yerr=e_fDM, fmt='o-', ms=2.5, lw=0.8,
                        color='steelblue', capsize=1.5, elinewidth=0.5)
            for N in range(3, 9):
                ax.axhline((N-1)/N, color='red', ls='--', alpha=0.25, lw=0.5)
            ax.axhline(0, color='black', ls='-', alpha=0.2, lw=0.5)
            ax.set_ylim(-0.6, 1.1)
            ax.set_xlim(0, max(R_Rd)*1.1 if R_Rd else 10)
            vf = meta[gname]['Vflat']
            ax.set_title(f'{gname} (Vf={vf:.0f})', fontsize=7)
            ax.tick_params(labelsize=6)
            if col == 0:
                ax.set_ylabel('f$_{DM}$', fontsize=7)
            if row == n_rows - 1:
                ax.set_xlabel('R/R$_d$', fontsize=7)

        # Hide unused axes
        for idx in range(n_gals, n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row][col].axis('off')

        plt.tight_layout()
        outname = f'step3_profiles_{fname_suffix}.png'
        plt.savefig(outname, dpi=130, bbox_inches='tight')
        plt.close()
        print(f"  Saved {outname}")

    # ══════════════════════════════════════════════════════════════
    # FIGURE 2: Main EDA figure (12 panels)
    # ══════════════════════════════════════════════════════════════
    print(f"\nGenerating {OUTPUT_MAIN}...")

    fig = plt.figure(figsize=(18, 20))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle('Step 3: Radial f$_{DM}$(R) — Exploratory Data Analysis',
                 fontsize=15, fontweight='bold', y=0.98)

    # ── Panel (0,0): Stacked median profiles by mass class ────────
    ax = fig.add_subplot(gs[0, 0])
    colors = {'dwarf': 'blue', 'intermediate': 'green', 'massive': 'red'}
    for label, gnames, color in [
            (f'Dwarfs (N={len(dwarfs)})', dwarfs, 'blue'),
            (f'Interm. (N={len(intermediate)})', intermediate, 'green'),
            (f'Massive (N={len(massive)})', massive, 'red')]:
        Rc, med, q2, q7, cnt = stack_profiles(data, gnames, R_bins)
        mask = ~np.isnan(med)
        ax.plot(Rc[mask], med[mask], 'o-', color=color, ms=3, lw=1.5,
                label=label)
        ax.fill_between(Rc[mask], q2[mask], q7[mask], alpha=0.12, color=color)
    for N in range(3, 9):
        ax.axhline((N-1)/N, color='gray', ls='--', alpha=0.3, lw=0.5)
        ax.text(14.5, (N-1)/N, f'{N}', fontsize=7, color='gray',
                va='center')
    ax.set_xlabel('R / R$_{disk}$')
    ax.set_ylabel('Median f$_{DM}$(R)')
    ax.set_title('Stacked profiles by mass class')
    ax.set_xlim(0, 15)
    ax.set_ylim(-0.3, 1.05)
    ax.legend(fontsize=7, loc='lower right')

    # ── Panel (0,1): Stacked profiles zoomed to f_DM > 0.5 ───────
    ax = fig.add_subplot(gs[0, 1])
    for label, gnames, color in [
            ('Dwarfs', dwarfs, 'blue'),
            ('Interm.', intermediate, 'green'),
            ('Massive', massive, 'red')]:
        Rc, med, q2, q7, cnt = stack_profiles(data, gnames, R_bins)
        mask = ~np.isnan(med)
        ax.plot(Rc[mask], med[mask], 'o-', color=color, ms=3, lw=1.5,
                label=label)
        ax.fill_between(Rc[mask], q2[mask], q7[mask], alpha=0.12, color=color)
    for N in range(3, 9):
        ax.axhline((N-1)/N, color='red', ls='--', alpha=0.35, lw=0.8)
        ax.text(0.3, (N-1)/N + 0.003, f'N={N}', fontsize=7, color='red',
                alpha=0.7)
    ax.set_xlabel('R / R$_{disk}$')
    ax.set_ylabel('Median f$_{DM}$(R)')
    ax.set_title('Zoomed: predicted (N-1)/N levels')
    ax.set_xlim(0, 12)
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=7)

    # ── Panel (0,2): Count per R/Rd bin ───────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    for label, gnames, color in [
            ('Dwarfs', dwarfs, 'blue'),
            ('Interm.', intermediate, 'green'),
            ('Massive', massive, 'red')]:
        Rc, med, q2, q7, cnt = stack_profiles(data, gnames, R_bins)
        ax.bar(Rc, cnt, width=0.45, alpha=0.5, color=color, label=label)
    ax.set_xlabel('R / R$_{disk}$')
    ax.set_ylabel('Points per bin')
    ax.set_title('Radial sampling density')
    ax.set_xlim(0, 15)
    ax.legend(fontsize=7)

    # ── Panel (1,0): Pooled histogram of all f_DM values ─────────
    ax = fig.add_subplot(gs[1, 0])
    all_fdm = [p['fDM'] for g in selected for p in data[g]
               if p['R_Rd'] > 1.0]  # Exclude innermost baryon-dominated
    ax.hist(all_fdm, bins=np.arange(-0.5, 1.05, 0.02), color='steelblue',
            edgecolor='white', alpha=0.8)
    for N in range(3, 9):
        ax.axvline((N-1)/N, color='red', ls='--', alpha=0.4, lw=0.8)
    ax.set_xlabel('f$_{DM}$')
    ax.set_ylabel('Count (all radial points, R/R$_d$ > 1)')
    ax.set_title('Pooled f$_{DM}$ distribution (all radii)')
    ax.set_xlim(-0.3, 1.05)

    # ── Panel (1,1): Pooled by mass class ─────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    for label, gnames, color in [
            ('Dwarfs', dwarfs, 'blue'),
            ('Interm.', intermediate, 'green'),
            ('Massive', massive, 'red')]:
        vals = [p['fDM'] for g in gnames for p in data[g] if p['R_Rd'] > 1.0]
        ax.hist(vals, bins=np.arange(-0.3, 1.05, 0.03), alpha=0.4,
                color=color, label=label, density=True)
    for N in range(3, 9):
        ax.axvline((N-1)/N, color='red', ls='--', alpha=0.3, lw=0.5)
    ax.set_xlabel('f$_{DM}$')
    ax.set_ylabel('Density')
    ax.set_title('Pooled f$_{DM}$ by mass class')
    ax.set_xlim(-0.1, 1.05)
    ax.legend(fontsize=7)

    # ── Panel (1,2): Derivative distribution ──────────────────────
    ax = fig.add_subplot(gs[1, 2])
    all_derivs = []
    for gname in selected:
        R, dfdR = compute_derivatives(data[gname])
        # Only use points at R/Rd > 1.5 (avoid noisy inner region)
        pts_sorted = sorted(data[gname], key=lambda p: p['R_Rd'])
        R_arr = np.array([p['R_Rd'] for p in pts_sorted])
        mask = R_arr > 1.5
        if len(R) > 0 and len(dfdR) > 0:
            all_derivs.extend(dfdR[mask[:len(dfdR)]])

    all_derivs = np.array(all_derivs)
    all_derivs = all_derivs[np.isfinite(all_derivs)]
    # Clip outliers for display
    clip = np.percentile(np.abs(all_derivs), 99)
    all_derivs_clip = all_derivs[np.abs(all_derivs) < clip]

    ax.hist(all_derivs_clip, bins=80, color='steelblue',
            edgecolor='white', alpha=0.8, density=True)
    ax.axvline(0, color='black', ls='-', lw=0.5)
    ax.set_xlabel('df$_{DM}$/d(R/R$_d$)')
    ax.set_ylabel('Density')
    ax.set_title('Derivative distribution (R/R$_d$ > 1.5)')

    # If staircase: expect peak near 0 (plateaus) + tails (transitions)
    # If smooth: expect single smooth distribution
    med_deriv = np.median(all_derivs_clip)
    frac_near_zero = np.mean(np.abs(all_derivs_clip) < 0.01)
    ax.text(0.95, 0.95,
            f'Median: {med_deriv:.4f}\n'
            f'|df/dR| < 0.01: {frac_near_zero:.1%}',
            transform=ax.transAxes, fontsize=8, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # ── Panel (2,0): Upsilon sensitivity — stacked massive ────────
    ax = fig.add_subplot(gs[2, 0])
    for ups, ls, alpha in [(0.3, '-', 0.9), (0.5, '--', 0.9), (0.7, ':', 0.9)]:
        # Recompute f_DM for massive galaxies at this upsilon
        R_all = []
        f_all = []
        for gname in massive:
            pts = compute_fdm_at_upsilon(data[gname], ups)
            for r, f in pts:
                R_all.append(r)
                f_all.append(f)
        # Bin
        R_all = np.array(R_all)
        f_all = np.array(f_all)
        R_centers = 0.5 * (R_bins[:-1] + R_bins[1:])
        medians = []
        for j in range(len(R_centers)):
            mask = (R_all >= R_bins[j]) & (R_all < R_bins[j+1])
            if mask.sum() >= 3:
                medians.append(np.median(f_all[mask]))
            else:
                medians.append(np.nan)
        medians = np.array(medians)
        m = ~np.isnan(medians)
        ax.plot(R_centers[m], medians[m], ls, lw=1.5, alpha=alpha,
                label=f'Υ★={ups}')
    for N in range(3, 9):
        ax.axhline((N-1)/N, color='gray', ls='--', alpha=0.3, lw=0.5)
    ax.set_xlabel('R / R$_{disk}$')
    ax.set_ylabel('Median f$_{DM}$(R)')
    ax.set_title('Υ★ sensitivity (massive galaxies)')
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.1, 1.05)
    ax.legend(fontsize=8)

    # ── Panel (2,1): Upsilon sensitivity — stacked dwarfs ─────────
    ax = fig.add_subplot(gs[2, 1])
    for ups, ls, alpha in [(0.3, '-', 0.9), (0.5, '--', 0.9), (0.7, ':', 0.9)]:
        R_all = []
        f_all = []
        for gname in dwarfs:
            pts = compute_fdm_at_upsilon(data[gname], ups)
            for r, f in pts:
                R_all.append(r)
                f_all.append(f)
        R_all = np.array(R_all)
        f_all = np.array(f_all)
        R_centers = 0.5 * (R_bins[:-1] + R_bins[1:])
        medians = []
        for j in range(len(R_centers)):
            mask = (R_all >= R_bins[j]) & (R_all < R_bins[j+1])
            if mask.sum() >= 3:
                medians.append(np.median(f_all[mask]))
            else:
                medians.append(np.nan)
        medians = np.array(medians)
        m = ~np.isnan(medians)
        ax.plot(R_centers[m], medians[m], ls, lw=1.5, alpha=alpha,
                label=f'Υ★={ups}')
    for N in range(3, 9):
        ax.axhline((N-1)/N, color='gray', ls='--', alpha=0.3, lw=0.5)
    ax.set_xlabel('R / R$_{disk}$')
    ax.set_ylabel('Median f$_{DM}$(R)')
    ax.set_title('Υ★ sensitivity (dwarfs)')
    ax.set_xlim(0, 12)
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=8)

    # ── Panel (2,2): Bulge vs no-bulge ────────────────────────────
    ax = fig.add_subplot(gs[2, 2])
    for label, gnames, color in [
            (f'With bulge (N={len(bulge_gals)})', bulge_gals, 'darkred'),
            (f'No bulge (N={len(nobulge_gals)})', nobulge_gals, 'steelblue')]:
        Rc, med, q2, q7, cnt = stack_profiles(data, gnames, R_bins)
        m = ~np.isnan(med)
        ax.plot(Rc[m], med[m], 'o-', color=color, ms=3, lw=1.5,
                label=label)
        ax.fill_between(Rc[m], q2[m], q7[m], alpha=0.12, color=color)
    for N in range(3, 9):
        ax.axhline((N-1)/N, color='gray', ls='--', alpha=0.3, lw=0.5)
    ax.set_xlabel('R / R$_{disk}$')
    ax.set_ylabel('Median f$_{DM}$(R)')
    ax.set_title('Bulge vs no-bulge galaxies')
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.2, 1.05)
    ax.legend(fontsize=7)

    # ── Panel (3,0): f_DM at specific R/Rd slices vs Vflat ───────
    ax = fig.add_subplot(gs[3, 0])
    for R_target, color, marker in [(2.0, 'blue', 'o'), (5.0, 'green', 's'),
                                     (8.0, 'red', '^')]:
        vflats = []
        fdms = []
        for gname in selected:
            pts = data[gname]
            # Find point closest to R_target in R/Rd
            best = min(pts, key=lambda p: abs(p['R_Rd'] - R_target))
            if abs(best['R_Rd'] - R_target) < 1.0:
                vflats.append(meta[gname]['Vflat'])
                fdms.append(best['fDM'])
        ax.scatter(vflats, fdms, s=12, c=color, marker=marker, alpha=0.6,
                   label=f'R/R$_d$ ≈ {R_target}')
    ax.set_xlabel('V$_{flat}$ (km/s)')
    ax.set_ylabel('f$_{DM}$')
    ax.set_title('f$_{DM}$ at fixed R/R$_d$ vs V$_{flat}$')
    ax.legend(fontsize=7)
    ax.set_ylim(-0.5, 1.1)

    # ── Panel (3,1): Error bars vs radius for individual galaxies ─
    ax = fig.add_subplot(gs[3, 1])
    for gname in selected[::3]:  # Every 3rd galaxy to avoid clutter
        pts = sorted(data[gname], key=lambda p: p['R_Rd'])
        R_arr = [p['R_Rd'] for p in pts]
        e_arr = [p['e_fDM'] for p in pts]
        vf = meta[gname]['Vflat']
        color = plt.cm.viridis(np.clip((vf - 30) / 270, 0, 1))
        ax.plot(R_arr, e_arr, '-', lw=0.5, alpha=0.4, color=color)
    # Overlay predicted step heights
    for N in [4, 5, 6, 7]:
        step = 1.0 / (N * (N + 1))
        ax.axhline(step, color='red', ls='--', alpha=0.5, lw=1)
        ax.text(14, step + 0.001, f'Δf(N={N})', fontsize=7, color='red')
    ax.set_xlabel('R / R$_{disk}$')
    ax.set_ylabel('σ(f$_{DM}$)')
    ax.set_title('Per-point uncertainty vs radius')
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 0.20)

    # ── Panel (3,2): Summary text ─────────────────────────────────
    ax = fig.add_subplot(gs[3, 2])
    ax.axis('off')

    # Compute some summary stats
    all_fdm_outer = [p['fDM'] for g in selected for p in data[g]
                     if p['R_Rd'] > 5.0]
    all_fdm_inner = [p['fDM'] for g in selected for p in data[g]
                     if 1.0 < p['R_Rd'] < 3.0]
    all_errs = [p['e_fDM'] for g in selected for p in data[g]
                if p['R_Rd'] > 1.0]

    frac_near_zero_str = f"{frac_near_zero:.1%}"

    summary = (
        f"STEP 3 EDA SUMMARY\n"
        f"{'='*40}\n\n"
        f"Sample: {len(selected)} galaxies, "
        f"{sum(len(data[g]) for g in selected)} pts\n\n"
        f"f_DM at inner radii (1<R/Rd<3):\n"
        f"  Median: {np.median(all_fdm_inner):.3f}\n"
        f"  IQR: [{np.percentile(all_fdm_inner,25):.3f}, "
        f"{np.percentile(all_fdm_inner,75):.3f}]\n\n"
        f"f_DM at outer radii (R/Rd>5):\n"
        f"  Median: {np.median(all_fdm_outer):.3f}\n"
        f"  IQR: [{np.percentile(all_fdm_outer,25):.3f}, "
        f"{np.percentile(all_fdm_outer,75):.3f}]\n\n"
        f"Per-point σ(f_DM) at R/Rd>1:\n"
        f"  Median: {np.median(all_errs):.4f}\n\n"
        f"Derivative df/d(R/Rd):\n"
        f"  Median: {med_deriv:.4f}\n"
        f"  |df/dR| < 0.01: {frac_near_zero_str}\n\n"
        f"With bulge: {len(bulge_gals)}\n"
        f"No bulge: {len(nobulge_gals)}\n\n"
        f"Prediction B check:\n"
        f"  Dwarfs > Interm > Massive\n"
        f"  at all R/Rd?  (see panel 0,0)"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=8.5, fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.savefig(OUTPUT_MAIN, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {OUTPUT_MAIN}")

    print("\nStep 3 complete.")


if __name__ == '__main__':
    main()
