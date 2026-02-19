#!/usr/bin/env python3
"""
Step 1: Compute Radial f_DM(R) Profiles for All SPARC Galaxies
================================================================
For each galaxy, compute f_DM at every measured radius using the
MassModels table. Merge with galaxy metadata from Paper 1 catalog.
Apply selection cuts for the radial profile analysis.

Output:
  - sparc_radial_profiles.csv: All radial points with f_DM(R)
  - step1_radial_diagnostic.png: Multi-panel diagnostic figure

Author: C. K. Merrill
Date: 2026-02-10
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import os

# ── Configuration ──────────────────────────────────────────────────
MASS_MODELS = "MassModels_Lelli2016c.mrt"
CATALOG     = "sparc_dm_fractions.csv"
UPSILON     = 0.5   # Fiducial M/L at [3.6]

OUTPUT_CSV  = "sparc_radial_profiles.csv"
OUTPUT_PLOT = "step1_radial_diagnostic.png"

# Selection cuts for radial analysis
MIN_PTS     = 10    # Minimum radial points
MIN_RMAX_RD = 3.0   # Minimum Rmax / Rdisk


# ── Parse MassModels MRT ──────────────────────────────────────────
def parse_mass_models(filepath):
    """Parse MassModels_Lelli2016c.mrt into per-galaxy radial data.
    Returns dict: galaxy_name -> list of dicts with R, Vobs, eVobs, etc.
    Uses fixed-width column positions from the MRT header.
    """
    galaxies = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find data start: after last '---' separator
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('---'):
            data_start = i + 1

    for line in lines[data_start:]:
        line = line.rstrip('\r\n')
        if len(line) < 60:
            continue
        try:
            gal_id = line[0:11].strip()
            D      = float(line[12:18])
            R      = float(line[19:25])
            Vobs   = float(line[26:32])
            eVobs  = float(line[33:38])
            Vgas   = float(line[39:45])
            Vdisk  = float(line[46:52])
            Vbul   = float(line[53:59])
            SBdisk = float(line[60:67])
            SBbul  = float(line[68:76])
        except (ValueError, IndexError):
            continue

        if gal_id not in galaxies:
            galaxies[gal_id] = []

        galaxies[gal_id].append({
            'R': R, 'Vobs': Vobs, 'eVobs': eVobs,
            'Vgas': Vgas, 'Vdisk': Vdisk, 'Vbul': Vbul,
            'SBdisk': SBdisk, 'SBbul': SBbul, 'D': D
        })

    return galaxies


# ── Parse Paper 1 catalog for metadata ────────────────────────────
def parse_catalog(filepath):
    """Load the Paper 1 CSV catalog for galaxy metadata."""
    meta = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['Galaxy'].strip()
            meta[name] = {
                'T': int(row['T']),
                'D_Mpc': float(row['D_Mpc']),
                'Rdisk_kpc': float(row['Rdisk_kpc']),
                'Vflat_kms': float(row['Vflat_kms']),
                'L36': float(row['L36_1e9Lsun']),
                'Q': int(row['Q']),
                'Npts': int(row['Npts']),
                'Rmax_kpc': float(row['Rmax_kpc']),
                'SBeff': float(row['SBeff']),
            }
    return meta


# ── Compute f_DM(R) ──────────────────────────────────────────────
def compute_fdm_profiles(galaxies, meta, upsilon=0.5):
    """Compute f_DM(R) for every radial point in every galaxy.
    Returns list of dicts (one per radial point) and per-galaxy summary.
    """
    all_points = []
    galaxy_summary = {}

    for gal_name, points in galaxies.items():
        if gal_name not in meta:
            continue

        m = meta[gal_name]
        Rdisk = m['Rdisk_kpc']
        Q = m['Q']
        Vflat = m['Vflat_kms']

        gal_points = []
        for pt in points:
            R = pt['R']
            Vobs = pt['Vobs']
            eVobs = pt['eVobs']
            Vgas = pt['Vgas']
            Vdisk = pt['Vdisk']
            Vbul = pt['Vbul']

            if Vobs <= 0:
                continue

            # V_bar^2 = V_gas^2 + Upsilon * V_disk^2 + Upsilon * V_bul^2
            Vbar2 = Vgas**2 + upsilon * Vdisk**2 + upsilon * Vbul**2
            Vobs2 = Vobs**2
            VDM2 = Vobs2 - Vbar2
            fDM = VDM2 / Vobs2

            # Propagate uncertainty from eVobs
            if Vobs > 0 and eVobs > 0:
                e_fDM = 2.0 * Vbar2 / (Vobs**3) * eVobs
            else:
                e_fDM = 0.0

            R_Rd = R / Rdisk if Rdisk > 0 else np.nan

            row = {
                'Galaxy': gal_name,
                'Q': Q,
                'Vflat': Vflat,
                'T': m['T'],
                'Rdisk': Rdisk,
                'L36': m['L36'],
                'R_kpc': R,
                'R_Rd': R_Rd,
                'Vobs': Vobs,
                'eVobs': eVobs,
                'Vgas': Vgas,
                'Vdisk_raw': Vdisk,
                'Vbul_raw': Vbul,
                'Vbar2': Vbar2,
                'VDM2': VDM2,
                'fDM': fDM,
                'e_fDM': abs(e_fDM),
            }
            gal_points.append(row)
            all_points.append(row)

        if len(gal_points) > 0:
            Rmax = max(p['R_kpc'] for p in gal_points)
            Rmax_Rd = Rmax / Rdisk if Rdisk > 0 else 0
            n_neg = sum(1 for p in gal_points if p['VDM2'] < 0)
            galaxy_summary[gal_name] = {
                'Npts': len(gal_points),
                'Rmax_kpc': Rmax,
                'Rmax_Rd': Rmax_Rd,
                'Q': Q,
                'Vflat': Vflat,
                'T': m['T'],
                'Rdisk': Rdisk,
                'L36': m['L36'],
                'n_neg_fDM': n_neg,
                'frac_neg': n_neg / len(gal_points),
                'fDM_outer': gal_points[-1]['fDM'],
                'median_e_fDM': np.median([p['e_fDM'] for p in gal_points]),
            }

    return all_points, galaxy_summary


# ── Selection ─────────────────────────────────────────────────────
def select_galaxies(galaxy_summary, min_pts=10, min_rmax_rd=3.0, max_q=1):
    """Apply selection cuts and return list of qualifying galaxy names."""
    selected = []
    for name, s in galaxy_summary.items():
        if s['Q'] > max_q:
            continue
        if s['Npts'] < min_pts:
            continue
        if s['Rmax_Rd'] < min_rmax_rd:
            continue
        if s['Vflat'] <= 0:
            continue
        selected.append(name)
    return sorted(selected)


# ── Main ──────────────────────────────────────────────────────────
def main():
    print("Step 1: Computing radial f_DM(R) profiles")
    print("=" * 60)

    # Parse data
    print("Parsing MassModels...")
    galaxies = parse_mass_models(MASS_MODELS)
    print(f"  Found {len(galaxies)} galaxies in MassModels file")

    print("Parsing Paper 1 catalog...")
    meta = parse_catalog(CATALOG)
    print(f"  Found {len(meta)} galaxies in catalog")

    # Compute profiles
    print(f"Computing f_DM(R) at Upsilon_star = {UPSILON}...")
    all_points, galaxy_summary = compute_fdm_profiles(galaxies, meta, UPSILON)
    print(f"  Total radial points: {len(all_points)}")
    print(f"  Galaxies with data: {len(galaxy_summary)}")

    # Selection
    selected = select_galaxies(galaxy_summary, MIN_PTS, MIN_RMAX_RD, max_q=1)
    print(f"\nSelection cuts: Q=1, Npts>={MIN_PTS}, Rmax/Rd>={MIN_RMAX_RD}, Vflat>0")
    print(f"  Qualifying galaxies: {len(selected)}")

    # Also compute relaxed selection for comparison
    selected_q2 = select_galaxies(galaxy_summary, MIN_PTS, MIN_RMAX_RD, max_q=2)
    print(f"  With Q<=2: {len(selected_q2)}")

    # Stats on qualifying galaxies
    npts_list = [galaxy_summary[g]['Npts'] for g in selected]
    med_err = [galaxy_summary[g]['median_e_fDM'] for g in selected]
    rmax_rd = [galaxy_summary[g]['Rmax_Rd'] for g in selected]
    vflat_list = [galaxy_summary[g]['Vflat'] for g in selected]

    print(f"\nQualifying galaxy statistics:")
    print(f"  Radial points: median={np.median(npts_list):.0f}, "
          f"range=[{min(npts_list)}, {max(npts_list)}]")
    print(f"  Rmax/Rd: median={np.median(rmax_rd):.1f}, "
          f"range=[{min(rmax_rd):.1f}, {max(rmax_rd):.1f}]")
    print(f"  Median per-point e_fDM: {np.median(med_err):.4f}")
    print(f"  Vflat range: [{min(vflat_list):.0f}, {max(vflat_list):.0f}] km/s")

    # Total radial points in selected galaxies
    sel_points = [p for p in all_points if p['Galaxy'] in selected]
    print(f"  Total radial points in selected sample: {len(sel_points)}")

    # Predicted step heights for reference
    print(f"\nPredicted step heights for reference:")
    for N in range(4, 9):
        step = 1.0 / (N * (N + 1))
        fN = (N - 1) / N
        print(f"  N={N}->N+1={N+1}: f_DM = {fN:.3f} -> {N/(N+1):.3f}, "
              f"step = {step:.4f}")
    print(f"  Median per-point uncertainty: {np.median(med_err):.4f}")

    # ── Write CSV ─────────────────────────────────────────────────
    print(f"\nWriting {OUTPUT_CSV}...")
    fields = ['Galaxy', 'Q', 'Vflat', 'T', 'Rdisk', 'L36',
              'R_kpc', 'R_Rd', 'Vobs', 'eVobs', 'Vgas',
              'Vdisk_raw', 'Vbul_raw', 'Vbar2', 'VDM2', 'fDM', 'e_fDM']
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for p in all_points:
            row = {k: (f"{p[k]:.6f}" if isinstance(p[k], float) else p[k])
                   for k in fields}
            writer.writerow(row)
    print(f"  Wrote {len(all_points)} rows")

    # ── Diagnostic Figure ─────────────────────────────────────────
    print(f"\nGenerating {OUTPUT_PLOT}...")
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.suptitle('Step 1: Radial f$_{DM}$(R) Profiles — Diagnostic Overview',
                 fontsize=14, fontweight='bold')

    # Panel 1: Representative profiles (dwarfs: Vflat < 80)
    ax = axes[0, 0]
    dwarfs = [g for g in selected if galaxy_summary[g]['Vflat'] < 80]
    for gname in dwarfs[:5]:
        pts = [(p['R_Rd'], p['fDM'], p['e_fDM'])
               for p in all_points if p['Galaxy'] == gname and p['R_Rd'] > 0]
        if not pts:
            continue
        rr, ff, ee = zip(*pts)
        ax.errorbar(rr, ff, yerr=ee, fmt='o-', ms=3, lw=1,
                    label=gname, alpha=0.8, capsize=2)
    for N in range(4, 9):
        ax.axhline((N-1)/N, color='gray', ls='--', alpha=0.3, lw=0.5)
    ax.set_xlabel('R / R$_{disk}$')
    ax.set_ylabel('f$_{DM}$(R)')
    ax.set_title(f'Dwarfs (V$_{{flat}}$ < 80, N={len(dwarfs)})')
    ax.set_ylim(-0.5, 1.1)
    ax.legend(fontsize=6, loc='lower right')

    # Panel 2: Representative profiles (intermediate: 80-150)
    ax = axes[0, 1]
    mid = [g for g in selected if 80 <= galaxy_summary[g]['Vflat'] < 150]
    for gname in mid[:5]:
        pts = [(p['R_Rd'], p['fDM'], p['e_fDM'])
               for p in all_points if p['Galaxy'] == gname and p['R_Rd'] > 0]
        if not pts:
            continue
        rr, ff, ee = zip(*pts)
        ax.errorbar(rr, ff, yerr=ee, fmt='o-', ms=3, lw=1,
                    label=gname, alpha=0.8, capsize=2)
    for N in range(4, 9):
        ax.axhline((N-1)/N, color='gray', ls='--', alpha=0.3, lw=0.5)
    ax.set_xlabel('R / R$_{disk}$')
    ax.set_ylabel('f$_{DM}$(R)')
    ax.set_title(f'Intermediate (80 ≤ V$_{{flat}}$ < 150, N={len(mid)})')
    ax.set_ylim(-0.5, 1.1)
    ax.legend(fontsize=6, loc='lower right')

    # Panel 3: Representative profiles (massive: Vflat >= 150)
    ax = axes[0, 2]
    massive = [g for g in selected if galaxy_summary[g]['Vflat'] >= 150]
    for gname in massive[:5]:
        pts = [(p['R_Rd'], p['fDM'], p['e_fDM'])
               for p in all_points if p['Galaxy'] == gname and p['R_Rd'] > 0]
        if not pts:
            continue
        rr, ff, ee = zip(*pts)
        ax.errorbar(rr, ff, yerr=ee, fmt='o-', ms=3, lw=1,
                    label=gname, alpha=0.8, capsize=2)
    for N in range(4, 9):
        ax.axhline((N-1)/N, color='gray', ls='--', alpha=0.3, lw=0.5)
    ax.set_xlabel('R / R$_{disk}$')
    ax.set_ylabel('f$_{DM}$(R)')
    ax.set_title(f'Massive (V$_{{flat}}$ ≥ 150, N={len(massive)})')
    ax.set_ylim(-0.5, 1.1)
    ax.legend(fontsize=6, loc='lower right')

    # Panel 4: All profiles overlaid (selected sample), color by Vflat
    ax = axes[1, 0]
    for gname in selected:
        pts = [(p['R_Rd'], p['fDM'])
               for p in all_points if p['Galaxy'] == gname and p['R_Rd'] > 0]
        if not pts:
            continue
        rr, ff = zip(*pts)
        vf = galaxy_summary[gname]['Vflat']
        color = plt.cm.viridis(np.clip((vf - 30) / 270, 0, 1))
        ax.plot(rr, ff, '-', lw=0.5, alpha=0.4, color=color)
    for N in range(4, 9):
        ax.axhline((N-1)/N, color='red', ls='--', alpha=0.4, lw=0.8)
        ax.text(0.3, (N-1)/N + 0.005, f'N={N}', fontsize=7, color='red', alpha=0.6)
    ax.set_xlabel('R / R$_{disk}$')
    ax.set_ylabel('f$_{DM}$(R)')
    ax.set_title(f'All Q=1 profiles (N={len(selected)}, color=V$_{{flat}}$)')
    ax.set_ylim(-0.5, 1.1)
    ax.set_xlim(0, 15)
    sm = plt.cm.ScalarMappable(cmap='viridis',
                                norm=plt.Normalize(30, 300))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='V$_{flat}$ (km/s)', shrink=0.8)

    # Panel 5: Stacked median profile in R/Rd bins
    ax = axes[1, 1]
    R_bins = np.arange(0, 15.5, 0.5)
    R_centers = 0.5 * (R_bins[:-1] + R_bins[1:])

    for label, gnames, color in [
            (f'Dwarfs (N={len(dwarfs)})', dwarfs, 'blue'),
            (f'Intermediate (N={len(mid)})', mid, 'green'),
            (f'Massive (N={len(massive)})', massive, 'red')]:
        medians = []
        q25 = []
        q75 = []
        counts = []
        for j in range(len(R_centers)):
            vals = [p['fDM'] for p in all_points
                    if p['Galaxy'] in gnames
                    and R_bins[j] <= p['R_Rd'] < R_bins[j+1]]
            if len(vals) >= 3:
                medians.append(np.median(vals))
                q25.append(np.percentile(vals, 25))
                q75.append(np.percentile(vals, 75))
                counts.append(len(vals))
            else:
                medians.append(np.nan)
                q25.append(np.nan)
                q75.append(np.nan)
                counts.append(0)
        medians = np.array(medians)
        q25 = np.array(q25)
        q75 = np.array(q75)
        mask = ~np.isnan(medians)
        ax.plot(R_centers[mask], medians[mask], 'o-', color=color,
                label=label, ms=3, lw=1.5)
        ax.fill_between(R_centers[mask], q25[mask], q75[mask],
                        alpha=0.15, color=color)
    for N in range(4, 9):
        ax.axhline((N-1)/N, color='gray', ls='--', alpha=0.3, lw=0.5)
    ax.set_xlabel('R / R$_{disk}$')
    ax.set_ylabel('Median f$_{DM}$(R)')
    ax.set_title('Stacked median profiles by mass class')
    ax.set_ylim(-0.2, 1.05)
    ax.set_xlim(0, 12)
    ax.legend(fontsize=7, loc='lower right')

    # Panel 6: Error budget — median e_fDM vs R/Rd
    ax = axes[1, 2]
    for label, gnames, color in [
            ('Dwarfs', dwarfs, 'blue'),
            ('Intermediate', mid, 'green'),
            ('Massive', massive, 'red')]:
        err_medians = []
        for j in range(len(R_centers)):
            vals = [p['e_fDM'] for p in all_points
                    if p['Galaxy'] in gnames
                    and R_bins[j] <= p['R_Rd'] < R_bins[j+1]]
            if len(vals) >= 3:
                err_medians.append(np.median(vals))
            else:
                err_medians.append(np.nan)
        err_medians = np.array(err_medians)
        mask = ~np.isnan(err_medians)
        ax.plot(R_centers[mask], err_medians[mask], 'o-', color=color,
                label=label, ms=3, lw=1.5)
    # Overlay predicted step heights
    for N, ls in [(4, '-'), (5, '--'), (6, ':'), (7, '-.')]:
        step = 1.0 / (N * (N + 1))
        ax.axhline(step, color='orange', ls=ls, alpha=0.5, lw=1)
        ax.text(11.5, step + 0.001, f'Δf(N={N}→{N+1})', fontsize=6,
                color='orange', alpha=0.7)
    ax.set_xlabel('R / R$_{disk}$')
    ax.set_ylabel('Median σ(f$_{DM}$)')
    ax.set_title('Error budget vs predicted step heights')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 0.15)
    ax.legend(fontsize=7)

    # Panel 7: Histogram of Npts per galaxy
    ax = axes[2, 0]
    ax.hist(npts_list, bins=range(10, max(npts_list)+2), color='steelblue',
            edgecolor='white', alpha=0.8)
    ax.axvline(np.median(npts_list), color='red', ls='--',
               label=f'Median = {np.median(npts_list):.0f}')
    ax.set_xlabel('Number of radial points')
    ax.set_ylabel('Count')
    ax.set_title(f'Radial sampling (N={len(selected)} galaxies)')
    ax.legend(fontsize=8)

    # Panel 8: Histogram of Rmax/Rd
    ax = axes[2, 1]
    ax.hist(rmax_rd, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(np.median(rmax_rd), color='red', ls='--',
               label=f'Median = {np.median(rmax_rd):.1f}')
    ax.set_xlabel('R$_{max}$ / R$_{disk}$')
    ax.set_ylabel('Count')
    ax.set_title('Radial extent of qualifying galaxies')
    ax.legend(fontsize=8)

    # Panel 9: Selection summary text
    ax = axes[2, 2]
    ax.axis('off')
    summary_text = (
        f"STEP 1 SUMMARY\n"
        f"{'='*40}\n\n"
        f"Total galaxies in MassModels: {len(galaxies)}\n"
        f"Matched with catalog: {len(galaxy_summary)}\n\n"
        f"Selection: Q=1, Npts≥{MIN_PTS},\n"
        f"  Rmax/Rd≥{MIN_RMAX_RD}, Vflat>0\n\n"
        f"Qualifying galaxies: {len(selected)}\n"
        f"  Dwarfs (Vf<80): {len(dwarfs)}\n"
        f"  Intermediate (80≤Vf<150): {len(mid)}\n"
        f"  Massive (Vf≥150): {len(massive)}\n\n"
        f"Total radial points: {len(sel_points)}\n"
        f"Points per galaxy: {np.median(npts_list):.0f} "
        f"[{min(npts_list)}-{max(npts_list)}]\n"
        f"Rmax/Rd: {np.median(rmax_rd):.1f} "
        f"[{min(rmax_rd):.1f}-{max(rmax_rd):.1f}]\n\n"
        f"Median σ(f_DM): {np.median(med_err):.4f}\n"
        f"Predicted Δf (N=6→7): 0.0238\n"
        f"Predicted Δf (N=5→6): 0.0333\n"
        f"Predicted Δf (N=4→5): 0.0500\n\n"
        f"Υ★ = {UPSILON} M☉/L☉"
    )
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=9, fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches='tight')
    print(f"  Saved {OUTPUT_PLOT}")

    print("\nStep 1 complete.")


if __name__ == '__main__':
    main()
