#!/usr/bin/env python3
"""
Phase A Follow-Up: Confound Checks
====================================
The initial Phase A results show significant spatial autocorrelation
of f_DM residuals (Moran's I = 0.67, p=0.006). Before claiming this
supports the ISF hypothesis, we must rule out:

1. GALAXY GROUPS: Many SPARC galaxies are in the Ursa Major cluster
   or other groups. Group members share environment, distance errors,
   and systematic biases. This could produce spurious spatial
   correlation.

2. DISTANCE-DEPENDENT SYSTEMATICS: f_DM depends on distance through
   Upsilon_star and V_obs scaling. Galaxies at similar distances
   might share distance method biases.

3. ENVIRONMENT: Dense environments have different f_DM (ram pressure,
   tidal stripping). This is a LCDM effect, not ISF.

4. DIRECTION ON SKY: SPARC galaxies cluster on the sky (northern
   hemisphere bias). Angular proximity ≠ physical proximity.

5. MULTIPLE TESTING: We tested 6 separation bins. Need correction.

Author: C. K. Merrill
Date: 2026-02-11
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr
import csv
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

RESULTS_CSV = "phaseA_results.csv"
CATALOG_CSV = "sparc_dm_fractions.csv"

OUTPUT_MAIN = "phaseA_confounds.png"


def angular_to_3d(ra_deg, dec_deg, d_mpc):
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    x = d_mpc * np.cos(dec) * np.cos(ra)
    y = d_mpc * np.cos(dec) * np.sin(ra)
    z = d_mpc * np.sin(dec)
    return x, y, z


def angular_sep(ra1, dec1, ra2, dec2):
    """Angular separation in degrees."""
    ra1, dec1, ra2, dec2 = map(np.radians, [ra1, dec1, ra2, dec2])
    cos_sep = (np.sin(dec1)*np.sin(dec2) +
               np.cos(dec1)*np.cos(dec2)*np.cos(ra1-ra2))
    return np.degrees(np.arccos(np.clip(cos_sep, -1, 1)))


def main():
    print("Phase A Follow-Up: Confound Checks")
    print("=" * 60)

    # Load Phase A results
    galaxies = []
    with open(RESULTS_CSV) as f:
        for row in csv.DictReader(f):
            galaxies.append({
                'name': row['Galaxy'],
                'RA': float(row['RA']),
                'Dec': float(row['Dec']),
                'D_Mpc': float(row['D_Mpc']),
                'Vflat': float(row['Vflat']),
                'fDM': float(row['fDM']),
                'e_fDM': float(row['e_fDM']),
                'fDM_pred': float(row['fDM_pred']),
                'residual': float(row['residual']),
            })

    # Add 3D positions
    for g in galaxies:
        g['x'], g['y'], g['z'] = angular_to_3d(g['RA'], g['Dec'], g['D_Mpc'])

    N = len(galaxies)
    residuals = np.array([g['residual'] for g in galaxies])
    distances = np.array([g['D_Mpc'] for g in galaxies])

    # ── Check 1: Identify Galaxy Groups ───────────────────────────
    print("\n  CHECK 1: Galaxy Groups / Clusters")
    print("  " + "-" * 50)

    # Identify groups by angular + distance proximity
    # Ursa Major cluster: RA ~ 170-185, Dec ~ 43-55, D ~ 15-20 Mpc
    uma_members = []
    for i, g in enumerate(galaxies):
        if (165 < g['RA'] < 190 and 40 < g['Dec'] < 58
                and 12 < g['D_Mpc'] < 25):
            uma_members.append(i)

    print(f"    Ursa Major candidates: {len(uma_members)}")
    if uma_members:
        uma_names = [galaxies[i]['name'] for i in uma_members]
        uma_res = [residuals[i] for i in uma_members]
        print(f"    UMa members: {', '.join(uma_names[:10])}...")
        print(f"    UMa mean residual: {np.mean(uma_res):.4f}")
        print(f"    UMa std residual:  {np.std(uma_res):.4f}")
        print(f"    Sample mean residual: {np.mean(residuals):.4f}")
        print(f"    Sample std residual:  {np.std(residuals):.4f}")

    # Find all close groups (< 3 Mpc separation)
    groups = []  # list of lists
    assigned = set()
    for i in range(N):
        if i in assigned:
            continue
        group = [i]
        for j in range(i+1, N):
            if j in assigned:
                continue
            dx = galaxies[i]['x'] - galaxies[j]['x']
            dy = galaxies[i]['y'] - galaxies[j]['y']
            dz = galaxies[i]['z'] - galaxies[j]['z']
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            if r < 3.0:
                group.append(j)
        if len(group) > 1:
            groups.append(group)
            assigned.update(group)

    print(f"\n    Groups (< 3 Mpc): {len(groups)}")
    for grp in groups:
        names = [galaxies[i]['name'] for i in grp]
        mean_d = np.mean([galaxies[i]['D_Mpc'] for i in grp])
        mean_res = np.mean([residuals[i] for i in grp])
        print(f"      {len(grp)} members at ~{mean_d:.1f} Mpc, "
              f"mean_res={mean_res:+.4f}: {', '.join(names[:5])}")

    group_members = set()
    for grp in groups:
        group_members.update(grp)
    field_idx = [i for i in range(N) if i not in group_members]
    print(f"\n    Field galaxies: {len(field_idx)}")
    print(f"    Group galaxies: {len(group_members)}")

    # ── Check 2: Rerun Moran's I excluding group members ─────────
    print("\n  CHECK 2: Moran's I Excluding Group Members")
    print("  " + "-" * 50)

    R_CUT = 20.0
    n_shuffle = 10000
    np.random.seed(42)

    def compute_morans_I(idx_list):
        n = len(idx_list)
        res = np.array([residuals[i] for i in idx_list])
        gals = [galaxies[i] for i in idx_list]
        W_sum = 0.0
        WR_sum = 0.0
        pairs_used = 0
        for ii in range(n):
            for jj in range(ii+1, n):
                dx = gals[ii]['x'] - gals[jj]['x']
                dy = gals[ii]['y'] - gals[jj]['y']
                dz = gals[ii]['z'] - gals[jj]['z']
                r = np.sqrt(dx**2 + dy**2 + dz**2)
                if 0 < r < R_CUT:
                    w = 1.0 / r
                    W_sum += w
                    WR_sum += w * res[ii] * res[jj]
                    pairs_used += 1
        var_res = np.var(res)
        if W_sum > 0 and var_res > 0:
            I = (n / W_sum) * (WR_sum / (n * var_res))
        else:
            I = 0.0

        # Null
        null_I = np.zeros(n_shuffle)
        for s in range(n_shuffle):
            perm = np.random.permutation(res)
            wr = 0.0
            for ii in range(n):
                for jj in range(ii+1, n):
                    dx = gals[ii]['x'] - gals[jj]['x']
                    dy = gals[ii]['y'] - gals[jj]['y']
                    dz = gals[ii]['z'] - gals[jj]['z']
                    r = np.sqrt(dx**2 + dy**2 + dz**2)
                    if 0 < r < R_CUT:
                        wr += (1.0/r) * perm[ii] * perm[jj]
            if W_sum > 0:
                null_I[s] = (n / W_sum) * (wr / (n * var_res))

        p = np.mean(null_I >= I)
        return I, p, pairs_used, null_I

    # Full sample
    I_full, p_full, pairs_full, null_full = compute_morans_I(list(range(N)))
    print(f"    Full sample:  I={I_full:.4f}, p={p_full:.4f}, pairs={pairs_full}")

    # Excluding group members
    I_field, p_field, pairs_field, null_field = compute_morans_I(field_idx)
    print(f"    Field only:   I={I_field:.4f}, p={p_field:.4f}, pairs={pairs_field}")

    # Excluding Ursa Major only
    non_uma = [i for i in range(N) if i not in uma_members]
    I_nouma, p_nouma, pairs_nouma, null_nouma = compute_morans_I(non_uma)
    print(f"    Excl. UMa:    I={I_nouma:.4f}, p={p_nouma:.4f}, pairs={pairs_nouma}")

    # ── Check 3: Distance-dependent systematics ───────────────────
    print("\n  CHECK 3: Distance-Dependent Systematics")
    print("  " + "-" * 50)

    r_dist, p_dist = spearmanr(distances, residuals)
    print(f"    Residual vs Distance: rho={r_dist:.3f}, p={p_dist:.3f}")

    # Check if nearby galaxies (D < 10 Mpc) drive the signal
    near_idx = [i for i in range(N) if distances[i] < 10]
    far_idx = [i for i in range(N) if distances[i] >= 10]
    print(f"    D < 10 Mpc: {len(near_idx)} galaxies, "
          f"mean res = {np.mean(residuals[near_idx]):.4f}")
    print(f"    D >= 10 Mpc: {len(far_idx)} galaxies, "
          f"mean res = {np.mean(residuals[far_idx]):.4f}")

    # ── Check 4: Angular vs 3D separation ─────────────────────────
    print("\n  CHECK 4: Angular vs 3D Separation")
    print("  " + "-" * 50)

    # Compute angular two-point correlation
    ang_bins = [0, 5, 10, 20, 40, 90, 180]
    xi_ang = []
    xi_ang_n = []
    for k in range(len(ang_bins)-1):
        products = []
        for i in range(N):
            for j in range(i+1, N):
                ang = angular_sep(galaxies[i]['RA'], galaxies[i]['Dec'],
                                   galaxies[j]['RA'], galaxies[j]['Dec'])
                if ang_bins[k] <= ang < ang_bins[k+1]:
                    products.append(residuals[i] * residuals[j])
        if products:
            xi_ang.append(np.mean(products))
            xi_ang_n.append(len(products))
        else:
            xi_ang.append(np.nan)
            xi_ang_n.append(0)
        print(f"    {ang_bins[k]:3d}-{ang_bins[k+1]:3d} deg: "
              f"xi={xi_ang[-1]:.6f}, N={xi_ang_n[-1]}")

    # ── Check 5: Multiple testing correction ──────────────────────
    print("\n  CHECK 5: Multiple Testing Correction")
    print("  " + "-" * 50)

    # Original p-values from Phase A bins
    orig_p = [0.0088, 0.4949, 0.0211, 0.4541, 0.9986, 0.1927]
    n_tests = len(orig_p)
    bonf_p = [min(p * n_tests, 1.0) for p in orig_p]
    print(f"    Original p-values: {[f'{p:.4f}' for p in orig_p]}")
    print(f"    Bonferroni corr:   {[f'{p:.4f}' for p in bonf_p]}")
    print(f"    Bins still significant (Bonf < 0.05): "
          f"{sum(1 for p in bonf_p if p < 0.05)}")

    # Benjamini-Hochberg
    sorted_p = sorted(enumerate(orig_p), key=lambda x: x[1])
    bh_sig = []
    for rank, (idx, p) in enumerate(sorted_p, 1):
        threshold = 0.05 * rank / n_tests
        if p <= threshold:
            bh_sig.append(idx)
    print(f"    BH significant bins: {len(bh_sig)}")

    # ── Check 6: Use one galaxy per group ─────────────────────────
    print("\n  CHECK 6: One Galaxy Per Group")
    print("  " + "-" * 50)

    # Keep the galaxy with smallest e_fDM per group
    keep_idx = set(range(N))
    for grp in groups:
        best = min(grp, key=lambda i: galaxies[i]['e_fDM'])
        for i in grp:
            if i != best:
                keep_idx.discard(i)

    dedupe_idx = sorted(keep_idx)
    I_dedupe, p_dedupe, pairs_dedupe, null_dedupe = compute_morans_I(dedupe_idx)
    print(f"    Deduplicated: {len(dedupe_idx)} galaxies")
    print(f"    Moran's I: {I_dedupe:.4f}, p={p_dedupe:.4f}, pairs={pairs_dedupe}")

    # ══════════════════════════════════════════════════════════════
    # FIGURE
    # ══════════════════════════════════════════════════════════════
    print(f"\nGenerating {OUTPUT_MAIN}...")

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
    fig.suptitle('Phase A Confound Checks',
                 fontsize=14, fontweight='bold', y=0.98)

    # (0,0): Sky map with groups highlighted
    ax = fig.add_subplot(gs[0, 0])
    ra = [g['RA'] for g in galaxies]
    dec = [g['Dec'] for g in galaxies]
    ax.scatter(ra, dec, c='gray', s=15, alpha=0.5, label='Field')
    for grp in groups:
        g_ra = [galaxies[i]['RA'] for i in grp]
        g_dec = [galaxies[i]['Dec'] for i in grp]
        ax.scatter(g_ra, g_dec, s=30, zorder=5)
    if uma_members:
        u_ra = [galaxies[i]['RA'] for i in uma_members]
        u_dec = [galaxies[i]['Dec'] for i in uma_members]
        ax.scatter(u_ra, u_dec, s=40, marker='s', edgecolors='red',
                   facecolors='none', linewidths=1.5, label='UMa candidates', zorder=6)
    ax.set_xlabel('RA (deg)')
    ax.set_ylabel('Dec (deg)')
    ax.set_title(f'Groups identified ({len(groups)} groups, '
                 f'{len(group_members)} members)')
    ax.legend(fontsize=7)

    # (0,1): Moran's I comparison
    ax = fig.add_subplot(gs[0, 1])
    labels = ['Full\nsample', 'Field\nonly', 'Excl.\nUMa', 'One per\ngroup']
    I_vals = [I_full, I_field, I_nouma, I_dedupe]
    p_vals_moran = [p_full, p_field, p_nouma, p_dedupe]
    colors = ['green' if p < 0.05 else 'salmon' for p in p_vals_moran]
    bars = ax.bar(labels, I_vals, color=colors, alpha=0.8, edgecolor='black')
    for bar, p in zip(bars, p_vals_moran):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'p={p:.3f}', ha='center', fontsize=8)
    ax.set_ylabel("Moran's I")
    ax.set_title("Moran's I under different cuts")
    ax.axhline(0, color='gray', ls='-', lw=0.5)

    # (0,2): Residual vs distance
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(distances, residuals, s=15, alpha=0.5, color='steelblue')
    ax.axhline(0, color='gray', ls='-', lw=0.5)
    ax.set_xlabel('Distance (Mpc)')
    ax.set_ylabel('$\\Delta f_{DM}$ residual')
    ax.set_title(f'Distance dependence (ρ={r_dist:.3f}, p={p_dist:.3f})')

    # (1,0): Null distributions comparison
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(null_full, bins=40, alpha=0.5, color='blue', density=True,
            label='Full sample null')
    ax.hist(null_field, bins=40, alpha=0.5, color='orange', density=True,
            label='Field only null')
    ax.axvline(I_full, color='blue', ls='-', lw=2, label=f'Full I={I_full:.3f}')
    ax.axvline(I_field, color='orange', ls='-', lw=2, label=f'Field I={I_field:.3f}')
    ax.set_xlabel("Moran's I")
    ax.set_ylabel('Density')
    ax.set_title('Null distributions')
    ax.legend(fontsize=7)

    # (1,1): Angular correlation
    ax = fig.add_subplot(gs[1, 1])
    ang_centers = [0.5*(ang_bins[k]+ang_bins[k+1]) for k in range(len(ang_bins)-1)]
    valid = [k for k in range(len(ang_centers)) if xi_ang_n[k] > 0]
    ax.plot([ang_centers[k] for k in valid],
            [xi_ang[k] for k in valid], 'ko-', ms=5)
    ax.axhline(0, color='gray', ls='-', lw=0.5)
    ax.set_xlabel('Angular separation (deg)')
    ax.set_ylabel('$\\xi_{\\Delta f}(\\theta)$')
    ax.set_title('Angular autocorrelation')

    # (1,2): Summary
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    summary = (
        f"CONFOUND CHECK RESULTS\n"
        f"{'='*40}\n\n"
        f"Groups found: {len(groups)}\n"
        f"  ({len(group_members)} galaxies in groups)\n"
        f"  UMa candidates: {len(uma_members)}\n\n"
        f"Moran's I:\n"
        f"  Full:         {I_full:.4f} (p={p_full:.4f})\n"
        f"  Field only:   {I_field:.4f} (p={p_field:.4f})\n"
        f"  Excl. UMa:    {I_nouma:.4f} (p={p_nouma:.4f})\n"
        f"  One/group:    {I_dedupe:.4f} (p={p_dedupe:.4f})\n\n"
        f"Distance correlation:\n"
        f"  rho={r_dist:.3f}, p={p_dist:.3f}\n\n"
        f"Multiple testing:\n"
        f"  Bonferroni survivors: "
        f"{sum(1 for p in bonf_p if p < 0.05)}/6\n"
        f"  BH survivors: {len(bh_sig)}/6\n\n"
        f"ASSESSMENT: See discussion"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=9, fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.savefig(OUTPUT_MAIN, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {OUTPUT_MAIN}")
    print("\nConfound checks complete.")


if __name__ == '__main__':
    main()
