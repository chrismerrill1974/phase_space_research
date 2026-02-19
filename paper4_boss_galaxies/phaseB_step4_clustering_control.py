#!/usr/bin/env python3
"""
phaseB_step4_clustering_control.py
===================================
Phase B, Step 4: Galaxy clustering control test.

The critical question: is the spatial autocorrelation of f_DM residuals
(detected in Steps 3/3b) simply reflecting the galaxy two-point
correlation function ξ(r), or is there genuine mark-position coupling?

Method:
  1. Compute ξ(r) via Landy-Szalay estimator using BOSS random catalogs
  2. Compute normalized mark correlation M(r) = w(r) / ξ(r)
  3. If M(r) = constant → residuals passively trace clustering (no ISF)
  4. If M(r) varies with scale → excess structure beyond clustering

Also tests:
  - Mark randomization: shuffle marks, recompute M(r) → should be flat
  - Scale-dependent test: is M(r) elevated specifically at ISF scales?
  - Within z-slices for consistency with Step 3b

Inputs (in DATA_DIR):
  - phaseB_residuals.csv                      (from Step 2)
  - phaseB_step3_correlation.csv              (w(r) from Step 3)
  - phaseB_step3b_stacked.csv                 (w(r) from Step 3b)
  - galaxy_DR12v5_LOWZ_North.fits.gz          (galaxy catalog)
  - galaxy_DR12v5_LOWZ_South.fits.gz
  - random0_DR12v5_LOWZ_North.fits.gz         (random catalog)
  - random0_DR12v5_LOWZ_South.fits.gz

Outputs (in DATA_DIR):
  - phaseB_step4_clustering.csv               (ξ(r), M(r) per bin)
  - phaseB_step4_diagnostics.png              (diagnostic figure)
  - phaseB_step4_summary.txt                  (text summary)

Usage:
  python3 -u phaseB_step4_clustering_control.py 2>&1 | tee phaseB_step4_output.txt
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial import cKDTree
from scipy import stats
from astropy.io import fits

# ── Configuration ──────────────────────────────────────────────────
DATA_DIR = "/Users/chris/Documents/research_data/cosmic_structure"

# Cosmology (must match Step 1)
H0 = 67.4          # km/s/Mpc
OMEGA_M = 0.315
C_LIGHT = 299792.458  # km/s

# Random catalog subsampling
N_RANDOMS = 100000   # subsample randoms for tractability

# Distance bins — match Step 3
R_MAX_STEP3 = 200.0
N_BINS_STEP3 = 20

# For M(r) computation, also use Step 3b bins
R_MAX_3B = 150.0
N_BINS_3B = 15

# Mark shuffle null
N_MARK_SHUFFLES = 200

SEED = 42


# ── Cosmology functions (must match Step 1) ───────────────────────
def comoving_distance(z, n_steps=1000):
    """Comoving distance in Mpc via numerical integration (flat ΛCDM)."""
    z_arr = np.atleast_1d(z)
    d = np.zeros_like(z_arr, dtype=np.float64)
    for i, zi in enumerate(z_arr):
        zz = np.linspace(0, zi, n_steps)
        dz = zz[1] - zz[0] if n_steps > 1 else zi
        E = np.sqrt(OMEGA_M * (1 + zz)**3 + (1 - OMEGA_M))
        d[i] = (C_LIGHT / H0) * np.trapz(1.0 / E, zz)
    return d if len(d) > 1 else d[0]


def radec_z_to_cartesian(ra, dec, z):
    """Convert (RA, Dec, z) to comoving Cartesian (x, y, z_cart) in Mpc."""
    d_c = comoving_distance(z)
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    x = d_c * np.cos(dec_rad) * np.cos(ra_rad)
    y = d_c * np.cos(dec_rad) * np.sin(ra_rad)
    z_cart = d_c * np.sin(dec_rad)
    return x, y, z_cart


# ── Pair counting ─────────────────────────────────────────────────
def count_pairs_binned(tree1, tree2, bin_edges):
    """Count pairs between two KD-trees in distance bins.

    Uses cumulative count_neighbors for efficiency — no pair enumeration.
    Returns array of pair counts per bin.
    """
    n_bins = len(bin_edges) - 1
    # count_neighbors returns cumulative counts at each radius
    cumulative = tree1.count_neighbors(tree2, bin_edges)
    # Difference gives per-bin counts
    counts = np.diff(cumulative).astype(np.int64)
    return counts


def count_pairs_marked(positions, residuals, center_idx, bin_edges, r_max):
    """Count pairs AND sum residual products per bin (for direct M(r)).

    Uses subsampled centers like Step 3 for marked correlation.
    Returns w(r) and pair counts.
    """
    n_bins = len(bin_edges) - 1
    tree = cKDTree(positions)

    sum_products = np.zeros(n_bins, dtype=np.float64)
    n_pairs = np.zeros(n_bins, dtype=np.int64)

    for ci in center_idx:
        neighbors = tree.query_ball_point(positions[ci], r_max)
        neighbors = np.array(neighbors, dtype=np.int32)
        neighbors = neighbors[neighbors != ci]

        if len(neighbors) == 0:
            continue

        dists = np.linalg.norm(positions[neighbors] - positions[ci], axis=1)
        bin_idx = np.digitize(dists, bin_edges) - 1
        valid = (bin_idx >= 0) & (bin_idx < n_bins)

        if valid.sum() == 0:
            continue

        products = residuals[ci] * residuals[neighbors[valid]]
        np.add.at(sum_products, bin_idx[valid], products)
        np.add.at(n_pairs, bin_idx[valid], 1)

    w = np.where(n_pairs > 0, sum_products / n_pairs, np.nan)
    return w, n_pairs


# ── Main ──────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("Phase B Step 4: Galaxy clustering control test")
    print("  Is the w(r) signal just tracing ξ(r)?")
    print("=" * 65)

    rng = np.random.RandomState(SEED)

    # ── Load data ─────────────────────────────────────────────────
    print("\n[1/6] Loading galaxy data...")
    df = pd.read_csv(os.path.join(DATA_DIR, "phaseB_residuals.csv"))
    print(f"  Galaxies: {len(df):,}")

    pos_data = df[['x', 'y', 'z_cart']].values.astype(np.float64)
    resid_A = df['resid_rf_ext'].values.astype(np.float64)
    resid_B = df['resid_rf_ext_B'].values.astype(np.float64)
    resid_A -= resid_A.mean()
    resid_B -= resid_B.mean()

    N_D = len(df)

    # ── Load Step 3 results ───────────────────────────────────────
    step3_file = os.path.join(DATA_DIR, "phaseB_step3_correlation.csv")
    step3b_file = os.path.join(DATA_DIR, "phaseB_step3b_stacked.csv")

    if os.path.exists(step3_file):
        step3 = pd.read_csv(step3_file)
        print(f"  Step 3 results loaded ({len(step3)} bins)")
    else:
        print("  WARNING: Step 3 results not found — will recompute w(r)")
        step3 = None

    if os.path.exists(step3b_file):
        step3b = pd.read_csv(step3b_file)
        print(f"  Step 3b stacked results loaded ({len(step3b)} bins)")
    else:
        step3b = None

    # ── Load random catalogs ──────────────────────────────────────
    print("\n[2/6] Loading BOSS random catalogs...")
    t0 = time.time()

    randoms_list = []
    for fname in ['random0_DR12v5_LOWZ_North.fits.gz',
                   'random0_DR12v5_LOWZ_South.fits.gz']:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  WARNING: {fname} not found — skipping")
            continue
        with fits.open(fpath) as hdul:
            rdata = hdul[1].data
            randoms_list.append({
                'ra': rdata['RA'].astype(np.float64),
                'dec': rdata['DEC'].astype(np.float64),
                'z': rdata['Z'].astype(np.float64),
            })
            print(f"  {fname}: {len(rdata):,} randoms")

    if len(randoms_list) == 0:
        print("  ERROR: No random catalogs found. Cannot compute ξ(r).")
        sys.exit(1)

    ra_rand = np.concatenate([r['ra'] for r in randoms_list])
    dec_rand = np.concatenate([r['dec'] for r in randoms_list])
    z_rand = np.concatenate([r['z'] for r in randoms_list])
    print(f"  Total randoms: {len(ra_rand):,}")

    # Apply same redshift cut as data
    z_mask = (z_rand >= df['z'].min() - 0.01) & (z_rand <= df['z'].max() + 0.01)
    ra_rand = ra_rand[z_mask]
    dec_rand = dec_rand[z_mask]
    z_rand = z_rand[z_mask]
    print(f"  After redshift cut: {len(ra_rand):,}")

    # Subsample randoms
    if len(ra_rand) > N_RANDOMS:
        idx_r = rng.choice(len(ra_rand), size=N_RANDOMS, replace=False)
        ra_rand = ra_rand[idx_r]
        dec_rand = dec_rand[idx_r]
        z_rand = z_rand[idx_r]
        print(f"  Subsampled to: {N_RANDOMS:,}")

    N_R = len(ra_rand)

    # Convert randoms to 3D comoving
    print(f"  Computing comoving distances for randoms...")
    t1 = time.time()

    # Vectorized comoving distance (batch for speed)
    batch_size = 10000
    x_rand = np.zeros(N_R)
    y_rand = np.zeros(N_R)
    z_cart_rand = np.zeros(N_R)

    for i in range(0, N_R, batch_size):
        j = min(i + batch_size, N_R)
        x_rand[i:j], y_rand[i:j], z_cart_rand[i:j] = radec_z_to_cartesian(
            ra_rand[i:j], dec_rand[i:j], z_rand[i:j])
        if (i // batch_size) % 5 == 0:
            print(f"\r    {j:,}/{N_R:,}", end='', flush=True)

    pos_rand = np.column_stack([x_rand, y_rand, z_cart_rand])
    print(f"\n  Random positions computed in {time.time()-t1:.0f}s")
    print(f"  Total loading + processing: {time.time()-t0:.0f}s")

    # ── Build KD-trees ────────────────────────────────────────────
    print("\n[3/6] Building KD-trees and counting pairs...")
    t0 = time.time()

    tree_D = cKDTree(pos_data)
    tree_R = cKDTree(pos_rand)
    print(f"  Trees built in {time.time()-t0:.1f}s")

    # ── Step 3 bins (match exactly) ───────────────────────────────
    bins_s3 = np.logspace(np.log10(1.0), np.log10(R_MAX_STEP3), N_BINS_STEP3 + 1)
    bc_s3 = np.sqrt(bins_s3[:-1] * bins_s3[1:])

    print(f"\n  Counting DD pairs (Step 3 bins)...")
    t0 = time.time()
    DD = count_pairs_binned(tree_D, tree_D, bins_s3)
    # Remove self-pairs and double-counting
    DD = DD // 2  # each pair counted twice in self-cross
    print(f"    DD: {time.time()-t0:.1f}s, total={DD.sum():,}")

    print(f"  Counting RR pairs...")
    t0 = time.time()
    RR = count_pairs_binned(tree_R, tree_R, bins_s3)
    RR = RR // 2
    print(f"    RR: {time.time()-t0:.1f}s, total={RR.sum():,}")

    print(f"  Counting DR pairs...")
    t0 = time.time()
    DR = count_pairs_binned(tree_D, tree_R, bins_s3)
    print(f"    DR: {time.time()-t0:.1f}s, total={DR.sum():,}")

    # ── Landy-Szalay estimator ────────────────────────────────────
    print(f"\n[4/6] Computing ξ(r) via Landy-Szalay...")

    # Normalization factors
    f_DD = 2.0 / (N_D * (N_D - 1))
    f_RR = 2.0 / (N_R * (N_R - 1))
    f_DR = 1.0 / (N_D * N_R)

    DD_norm = DD.astype(np.float64) * f_DD
    RR_norm = RR.astype(np.float64) * f_RR
    DR_norm = DR.astype(np.float64) * f_DR

    # Landy-Szalay: ξ(r) = (DD - 2DR + RR) / RR
    xi = np.where(RR_norm > 0,
                   (DD_norm - 2 * DR_norm + RR_norm) / RR_norm,
                   np.nan)

    print(f"\n  ξ(r) results (Step 3 bins):")
    print(f"  {'r_cen':>7s}  {'DD':>10s} {'RR':>10s} {'DR':>10s}  {'ξ(r)':>10s}")
    print(f"  {'-'*7:>7s}  {'-'*10:>10s} {'-'*10:>10s} {'-'*10:>10s}  {'-'*10:>10s}")
    for i in range(N_BINS_STEP3):
        print(f"  {bc_s3[i]:7.1f}  {DD[i]:10,} {RR[i]:10,} {DR[i]:10,}  {xi[i]:10.4f}")

    # ── Compute M(r) = w(r) / ξ(r) ───────────────────────────────
    print(f"\n[5/6] Computing normalized mark correlation M(r)...")

    # Use Step 3 w(r) values
    if step3 is not None:
        w_A_s3 = step3['w_A'].values
        w_B_s3 = step3['w_B'].values

        # M(r) = w(r) / ξ(r)
        # Only meaningful where ξ(r) > 0 (clustered regime)
        M_A_s3 = np.where((xi > 0) & ~np.isnan(w_A_s3),
                           w_A_s3 / xi, np.nan)
        M_B_s3 = np.where((xi > 0) & ~np.isnan(w_B_s3),
                           w_B_s3 / xi, np.nan)

        print(f"\n  M(r) = w(r)/ξ(r) [Step 3 bins]:")
        print(f"  {'r_cen':>7s}  {'ξ(r)':>10s}  {'w_A':>10s} {'M_A':>10s}  "
              f"{'w_B':>10s} {'M_B':>10s}")
        print(f"  {'-'*7:>7s}  {'-'*10:>10s}  {'-'*10:>10s} {'-'*10:>10s}  "
              f"{'-'*10:>10s} {'-'*10:>10s}")
        for i in range(N_BINS_STEP3):
            flag = " *" if (bc_s3[i] >= 10 and bc_s3[i] <= 50) else "  "
            ma = f"{M_A_s3[i]:10.6f}" if not np.isnan(M_A_s3[i]) else "       nan"
            mb = f"{M_B_s3[i]:10.6f}" if not np.isnan(M_B_s3[i]) else "       nan"
            print(f"  {bc_s3[i]:7.1f}  {xi[i]:10.4f}  {w_A_s3[i]:10.6f} {ma}  "
                  f"{w_B_s3[i]:10.6f} {mb}{flag}")

    # ── Also compute for Step 3b bins ─────────────────────────────
    bins_3b = np.logspace(np.log10(1.0), np.log10(R_MAX_3B), N_BINS_3B + 1)
    bc_3b = np.sqrt(bins_3b[:-1] * bins_3b[1:])

    DD_3b = count_pairs_binned(tree_D, tree_D, bins_3b) // 2
    RR_3b = count_pairs_binned(tree_R, tree_R, bins_3b) // 2
    DR_3b = count_pairs_binned(tree_D, tree_R, bins_3b)

    DD_3b_norm = DD_3b.astype(np.float64) * f_DD
    RR_3b_norm = RR_3b.astype(np.float64) * f_RR
    DR_3b_norm = DR_3b.astype(np.float64) * f_DR

    xi_3b = np.where(RR_3b_norm > 0,
                      (DD_3b_norm - 2 * DR_3b_norm + RR_3b_norm) / RR_3b_norm,
                      np.nan)

    if step3b is not None:
        w_A_3b = step3b['stacked_w_A'].values
        w_B_3b = step3b['stacked_w_B'].values

        M_A_3b = np.where((xi_3b > 0) & ~np.isnan(w_A_3b),
                           w_A_3b / xi_3b, np.nan)
        M_B_3b = np.where((xi_3b > 0) & ~np.isnan(w_B_3b),
                           w_B_3b / xi_3b, np.nan)

        print(f"\n  M(r) from z-slice stacked w(r) [Step 3b bins]:")
        print(f"  {'r_cen':>7s}  {'ξ(r)':>10s}  {'w_A':>10s} {'M_A':>10s}  "
              f"{'w_B':>10s} {'M_B':>10s}")
        print(f"  {'-'*7:>7s}  {'-'*10:>10s}  {'-'*10:>10s} {'-'*10:>10s}  "
              f"{'-'*10:>10s} {'-'*10:>10s}")
        for i in range(N_BINS_3B):
            flag = " *" if (bc_3b[i] >= 10 and bc_3b[i] <= 50) else "  "
            ma = f"{M_A_3b[i]:10.6f}" if not np.isnan(M_A_3b[i]) else "       nan"
            mb = f"{M_B_3b[i]:10.6f}" if not np.isnan(M_B_3b[i]) else "       nan"
            print(f"  {bc_3b[i]:7.1f}  {xi_3b[i]:10.4f}  {w_A_3b[i]:10.6f} {ma}  "
                  f"{w_B_3b[i]:10.6f} {mb}{flag}")

    # ── Statistical tests on M(r) ─────────────────────────────────
    print(f"\n  Testing whether M(r) is constant (marks trace clustering)...")

    # ISF range mask for Step 3 bins
    isf_s3 = (bc_s3 >= 10) & (bc_s3 <= 50)
    # All bins where ξ > 0 (meaningful regime)
    valid_s3 = (xi > 0) & ~np.isnan(xi)

    if step3 is not None:
        # For M(r) constancy test: use bins where ξ(r) > 0.001 (avoid noise)
        good = valid_s3 & (xi > 0.001) & ~np.isnan(M_A_s3)
        if good.sum() > 2:
            M_A_good = M_A_s3[good]
            M_A_mean = np.nanmean(M_A_good)
            M_A_std = np.nanstd(M_A_good)
            # Coefficient of variation
            M_A_cv = M_A_std / abs(M_A_mean) if M_A_mean != 0 else np.inf

            print(f"\n  M_A(r) statistics (ξ > 0.001):")
            print(f"    Mean: {M_A_mean:.6f}")
            print(f"    Std:  {M_A_std:.6f}")
            print(f"    CV:   {M_A_cv:.3f}")

            # Is M(r) in ISF range different from other scales?
            isf_good = good & isf_s3
            non_isf_good = good & ~isf_s3
            if isf_good.sum() > 0 and non_isf_good.sum() > 0:
                M_isf = M_A_s3[isf_good]
                M_non = M_A_s3[non_isf_good]
                t_stat, p_diff = stats.mannwhitneyu(
                    M_isf, M_non, alternative='two-sided')
                print(f"    ISF range M_A mean: {M_isf.mean():.6f}")
                print(f"    Non-ISF M_A mean:   {M_non.mean():.6f}")
                print(f"    Mann-Whitney p:     {p_diff:.4f}")

        good_B = valid_s3 & (xi > 0.001) & ~np.isnan(M_B_s3)
        if good_B.sum() > 2:
            M_B_good = M_B_s3[good_B]
            M_B_mean = np.nanmean(M_B_good)
            M_B_std = np.nanstd(M_B_good)
            M_B_cv = M_B_std / abs(M_B_mean) if M_B_mean != 0 else np.inf

            print(f"\n  M_B(r) statistics (ξ > 0.001):")
            print(f"    Mean: {M_B_mean:.6f}")
            print(f"    Std:  {M_B_std:.6f}")
            print(f"    CV:   {M_B_cv:.3f}")

    # ── Proportionality test: w(r) = α × ξ(r)? ──────────────────
    print(f"\n  Proportionality test: w(r) = α × ξ(r)?")
    if step3 is not None:
        fit_mask = valid_s3 & ~np.isnan(w_A_s3) & (xi > 0)
        if fit_mask.sum() > 2:
            # Fit w = α × ξ (no intercept)
            from numpy.linalg import lstsq
            xi_fit = xi[fit_mask].reshape(-1, 1)
            w_fit_A = w_A_s3[fit_mask]
            w_fit_B = w_B_s3[fit_mask]

            alpha_A = lstsq(xi_fit, w_fit_A, rcond=None)[0][0]
            alpha_B = lstsq(xi_fit, w_fit_B, rcond=None)[0][0]

            resid_fit_A = w_fit_A - alpha_A * xi_fit.ravel()
            resid_fit_B = w_fit_B - alpha_B * xi_fit.ravel()

            ss_res_A = np.sum(resid_fit_A**2)
            ss_tot_A = np.sum((w_fit_A - w_fit_A.mean())**2)
            r2_prop_A = 1 - ss_res_A / ss_tot_A if ss_tot_A > 0 else np.nan

            ss_res_B = np.sum(resid_fit_B**2)
            ss_tot_B = np.sum((w_fit_B - w_fit_B.mean())**2)
            r2_prop_B = 1 - ss_res_B / ss_tot_B if ss_tot_B > 0 else np.nan

            print(f"    w_A = {alpha_A:.6f} × ξ  (R² = {r2_prop_A:.4f})")
            print(f"    w_B = {alpha_B:.6f} × ξ  (R² = {r2_prop_B:.4f})")

            # Residuals from proportional fit — is there ISF-scale excess?
            isf_fit = fit_mask & isf_s3
            if isf_fit.sum() > 0:
                resid_isf_A = w_A_s3[isf_fit] - alpha_A * xi[isf_fit]
                print(f"    ISF-range residual from fit (A): "
                      f"mean={resid_isf_A.mean():.7f}, "
                      f"max|res|={np.max(np.abs(resid_isf_A)):.7f}")

    # ── Mark shuffle control ──────────────────────────────────────
    print(f"\n[6/6] Mark shuffle control ({N_MARK_SHUFFLES} shuffles)...")
    print(f"  Computing w(r) with shuffled marks to verify M(r) method...")

    # Subsample centers for marked correlation (same approach as Step 3)
    n_centers_ctrl = 10000
    center_idx = rng.choice(N_D, size=min(n_centers_ctrl, N_D), replace=False)
    center_idx.sort()

    # Data w(r) with these centers
    w_A_ctrl, npairs_ctrl = count_pairs_marked(
        pos_data, resid_A, center_idx, bins_s3, R_MAX_STEP3)

    # Shuffled w(r) distribution
    null_w = np.zeros((N_MARK_SHUFFLES, N_BINS_STEP3))
    t0 = time.time()

    for s in range(N_MARK_SHUFFLES):
        perm = rng.permutation(N_D)
        null_w[s], _ = count_pairs_marked(
            pos_data, resid_A[perm], center_idx, bins_s3, R_MAX_STEP3)
        if (s + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (s + 1) * (N_MARK_SHUFFLES - s - 1)
            print(f"\r    Shuffle {s+1}/{N_MARK_SHUFFLES} "
                  f"({elapsed:.0f}s, ~{eta:.0f}s remaining)",
                  end='', flush=True)

    print(f"\n    Done in {time.time()-t0:.0f}s")

    null_mean = null_w.mean(axis=0)
    null_std = null_w.std(axis=0)

    # M(r) for shuffled marks — should be ~0 since <shuffled marks> are uncorrelated
    M_null = np.where(xi > 0, null_mean / xi, np.nan)

    print(f"\n  Shuffled M(r) mean: {np.nanmean(M_null):.8f} (should be ~0)")
    print(f"  Shuffled M(r) std:  {np.nanstd(M_null):.8f}")

    # ── Save results ──────────────────────────────────────────────
    result_df = pd.DataFrame({
        'r_center': bc_s3,
        'r_min': bins_s3[:-1],
        'r_max': bins_s3[1:],
        'DD': DD,
        'RR': RR,
        'DR': DR,
        'xi': xi,
        'w_A': w_A_s3 if step3 is not None else np.nan,
        'w_B': w_B_s3 if step3 is not None else np.nan,
        'M_A': M_A_s3 if step3 is not None else np.nan,
        'M_B': M_B_s3 if step3 is not None else np.nan,
        'w_ctrl': w_A_ctrl,
        'w_null_mean': null_mean,
        'w_null_std': null_std,
    })

    out_csv = os.path.join(DATA_DIR, "phaseB_step4_clustering.csv")
    result_df.to_csv(out_csv, index=False)
    print(f"\n  Saved: {out_csv}")

    # ── Diagnostic figure ─────────────────────────────────────────
    print(f"\n  Generating diagnostic figure...")

    fig = plt.figure(figsize=(18, 15))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Panel 1: ξ(r) — the galaxy two-point function
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(bc_s3, xi, 'ko-', ms=4, lw=1.5)
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.axvspan(10, 50, alpha=0.08, color='blue', label='ISF range')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Comoving separation (Mpc)')
    ax.set_ylabel('$\\xi(r)$')
    ax.set_title('Galaxy two-point correlation function')
    ax.set_ylim(bottom=1e-4)
    ax.legend(fontsize=8)

    # Panel 2: w(r) and α×ξ(r) overlay
    ax = fig.add_subplot(gs[0, 1])
    if step3 is not None:
        ax.plot(bc_s3, w_A_s3, 'o-', color='darkred', ms=4, lw=1.5,
                label='$w_A(r)$ (Step 3)')
        if 'alpha_A' in dir():
            ax.plot(bc_s3[valid_s3], alpha_A * xi[valid_s3], '--',
                    color='gray', lw=2,
                    label=f'$\\alpha \\xi(r)$, α={alpha_A:.5f}')
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.axvspan(10, 50, alpha=0.08, color='blue')
    ax.set_xscale('log')
    ax.set_xlabel('Comoving separation (Mpc)')
    ax.set_ylabel('$w(r)$')
    ax.set_title('$w(r)$ vs proportional $\\xi(r)$ fit')
    ax.legend(fontsize=8)

    # Panel 3: M(r) = w(r)/ξ(r) — THE KEY PLOT
    ax = fig.add_subplot(gs[0, 2])
    if step3 is not None:
        valid_plot = valid_s3 & ~np.isnan(M_A_s3) & (xi > 0.001)
        ax.plot(bc_s3[valid_plot], M_A_s3[valid_plot], 'o-', color='darkred',
                ms=5, lw=1.5, label='Method A')
        valid_plot_B = valid_s3 & ~np.isnan(M_B_s3) & (xi > 0.001)
        ax.plot(bc_s3[valid_plot_B], M_B_s3[valid_plot_B], 's-', color='navy',
                ms=5, lw=1.5, label='Method B')
        if 'M_A_mean' in dir():
            ax.axhline(M_A_mean, color='darkred', ls='--', alpha=0.3,
                       label=f'Mean M_A={M_A_mean:.5f}')
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.axvspan(10, 50, alpha=0.08, color='blue', label='ISF range')
    ax.set_xscale('log')
    ax.set_xlabel('Comoving separation (Mpc)')
    ax.set_ylabel('$M(r) = w(r) / \\xi(r)$')
    ax.set_title('Normalized mark correlation (KEY TEST)')
    ax.legend(fontsize=7)

    # Panel 4: w(r) vs ξ(r) scatter (proportionality check)
    ax = fig.add_subplot(gs[1, 0])
    if step3 is not None:
        valid_sc = valid_s3 & ~np.isnan(w_A_s3)
        sc = ax.scatter(xi[valid_sc], w_A_s3[valid_sc], s=40,
                        c=np.log10(bc_s3[valid_sc]), cmap='viridis',
                        edgecolors='black', lw=0.5, zorder=3)
        plt.colorbar(sc, ax=ax, label='log(r/Mpc)')
        if 'alpha_A' in dir():
            xi_line = np.linspace(0, xi[valid_sc].max() * 1.1, 50)
            ax.plot(xi_line, alpha_A * xi_line, 'r--', lw=1.5,
                    label=f'α={alpha_A:.5f}')
    ax.set_xlabel('$\\xi(r)$')
    ax.set_ylabel('$w(r)$ Method A')
    ax.set_title('Proportionality: $w$ vs $\\xi$')
    ax.legend(fontsize=8)

    # Panel 5: Step 3b M(r) (z-slice stacked)
    ax = fig.add_subplot(gs[1, 1])
    if step3b is not None:
        valid_3b = (xi_3b > 0.001) & ~np.isnan(M_A_3b)
        if valid_3b.sum() > 0:
            ax.plot(bc_3b[valid_3b], M_A_3b[valid_3b], 'o-', color='darkred',
                    ms=5, lw=1.5, label='Method A (z-sliced)')
        valid_3b_B = (xi_3b > 0.001) & ~np.isnan(M_B_3b)
        if valid_3b_B.sum() > 0:
            ax.plot(bc_3b[valid_3b_B], M_B_3b[valid_3b_B], 's-', color='navy',
                    ms=5, lw=1.5, label='Method B (z-sliced)')
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.axvspan(10, 50, alpha=0.08, color='blue')
    ax.set_xscale('log')
    ax.set_xlabel('Comoving separation (Mpc)')
    ax.set_ylabel('$M(r)$ (z-slice stacked)')
    ax.set_title('Mark correlation: z-slice stacked')
    ax.legend(fontsize=8)

    # Panel 6: ξ(r) × r² (emphasizes BAO scale)
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(bc_s3, xi * bc_s3**2, 'ko-', ms=4, lw=1.5)
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.axvspan(10, 50, alpha=0.08, color='blue', label='ISF range')
    ax.axvspan(90, 110, alpha=0.08, color='green', label='BAO scale')
    ax.set_xscale('log')
    ax.set_xlabel('Comoving separation (Mpc)')
    ax.set_ylabel('$r^2 \\xi(r)$')
    ax.set_title('$r^2 \\xi(r)$ (BAO feature check)')
    ax.legend(fontsize=8)

    # Panel 7: Mark shuffle null — M(r) for shuffled marks
    ax = fig.add_subplot(gs[2, 0])
    M_ctrl = np.where(xi > 0, w_A_ctrl / xi, np.nan)
    valid_ctrl = (xi > 0.001) & ~np.isnan(M_ctrl)
    if valid_ctrl.sum() > 0:
        ax.plot(bc_s3[valid_ctrl], M_ctrl[valid_ctrl], 'o-', color='darkred',
                ms=4, lw=1.5, label='Data marks')

    # Null envelope
    null_M_lo = np.percentile(null_w, 2.5, axis=0)
    null_M_hi = np.percentile(null_w, 97.5, axis=0)
    M_null_lo = np.where(xi > 0, null_M_lo / xi, np.nan)
    M_null_hi = np.where(xi > 0, null_M_hi / xi, np.nan)
    if valid_ctrl.sum() > 0:
        ax.fill_between(bc_s3[valid_ctrl],
                         M_null_lo[valid_ctrl], M_null_hi[valid_ctrl],
                         color='gray', alpha=0.3, label='95% null envelope')
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.axvspan(10, 50, alpha=0.08, color='blue')
    ax.set_xscale('log')
    ax.set_xlabel('Comoving separation (Mpc)')
    ax.set_ylabel('$M(r)$')
    ax.set_title('M(r): Data vs shuffled marks')
    ax.legend(fontsize=8)

    # Panel 8: Ratio w(r)/ξ(r) vs scale — log-log
    ax = fig.add_subplot(gs[2, 1])
    if step3 is not None:
        # Show absolute values on log scale
        pos_mask = valid_s3 & (w_A_s3 > 0) & (xi > 0)
        neg_mask = valid_s3 & (w_A_s3 < 0) & (xi > 0)
        if pos_mask.sum() > 0:
            ax.plot(bc_s3[pos_mask], np.abs(M_A_s3[pos_mask]), 'o',
                    color='darkred', ms=5, label='M_A > 0')
        if neg_mask.sum() > 0:
            ax.plot(bc_s3[neg_mask], np.abs(M_A_s3[neg_mask]), 'x',
                    color='salmon', ms=8, mew=2, label='M_A < 0')
    ax.axvspan(10, 50, alpha=0.08, color='blue')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Comoving separation (Mpc)')
    ax.set_ylabel('$|M(r)|$')
    ax.set_title('$|M(r)|$ log-log scale')
    ax.legend(fontsize=8)

    # Panel 9: Summary verdict
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    verdict_lines = [
        "Step 4 Summary",
        "─" * 30,
        f"N_galaxies: {N_D:,}",
        f"N_randoms:  {N_R:,}",
        "",
    ]
    if step3 is not None and 'alpha_A' in dir():
        verdict_lines += [
            f"w_A = {alpha_A:.6f} × ξ(r)",
            f"Proportionality R² = {r2_prop_A:.4f}",
            "",
        ]
    if step3 is not None and 'M_A_mean' in dir():
        verdict_lines += [
            f"M_A mean = {M_A_mean:.6f}",
            f"M_A CV   = {M_A_cv:.3f}",
            "",
        ]

    # Verdict
    if step3 is not None and 'r2_prop_A' in dir():
        if r2_prop_A > 0.9:
            verdict_lines.append("VERDICT: w(r) ∝ ξ(r)")
            verdict_lines.append("Signal is galaxy clustering,")
            verdict_lines.append("NOT ISF-specific structure.")
        elif r2_prop_A > 0.7:
            verdict_lines.append("VERDICT: Mostly clustering")
            verdict_lines.append("with possible deviations.")
        else:
            verdict_lines.append("VERDICT: w(r) NOT ∝ ξ(r)")
            verdict_lines.append("Possible ISF-scale excess.")

    ax.text(0.05, 0.95, "\n".join(verdict_lines),
            transform=ax.transAxes, fontsize=10, va='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle('Phase B Step 4: Galaxy Clustering Control Test',
                 fontsize=14, fontweight='bold', y=0.99)

    out_fig = os.path.join(DATA_DIR, "phaseB_step4_diagnostics.png")
    plt.savefig(out_fig, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {out_fig}")

    # ── Summary ───────────────────────────────────────────────────
    lines = []
    lines.append("Phase B Step 4: Galaxy Clustering Control Summary")
    lines.append("=" * 60)
    lines.append(f"Sample: {N_D:,} galaxies, {N_R:,} randoms")
    lines.append(f"")
    lines.append(f"Galaxy two-point function ξ(r):")
    lines.append(f"  ξ(10 Mpc) ≈ {xi[np.argmin(np.abs(bc_s3-10))]:.4f}")
    lines.append(f"  ξ(50 Mpc) ≈ {xi[np.argmin(np.abs(bc_s3-50))]:.4f}")
    lines.append(f"  ξ(100 Mpc) ≈ {xi[np.argmin(np.abs(bc_s3-100))]:.4f}")

    if step3 is not None and 'alpha_A' in dir():
        lines.append(f"")
        lines.append(f"Proportionality test: w(r) = α × ξ(r)")
        lines.append(f"  Method A: α = {alpha_A:.6f}, R² = {r2_prop_A:.4f}")
        lines.append(f"  Method B: α = {alpha_B:.6f}, R² = {r2_prop_B:.4f}")

    if step3 is not None and 'M_A_mean' in dir():
        lines.append(f"")
        lines.append(f"Normalized mark correlation M(r) = w(r)/ξ(r):")
        lines.append(f"  M_A: mean={M_A_mean:.6f}, std={M_A_std:.6f}, CV={M_A_cv:.3f}")
        if 'M_B_mean' in dir():
            lines.append(f"  M_B: mean={M_B_mean:.6f}, std={M_B_std:.6f}, CV={M_B_cv:.3f}")

    lines.append(f"")
    lines.append(f"Interpretation:")

    if step3 is not None and 'r2_prop_A' in dir():
        if r2_prop_A > 0.9:
            lines.append(f"  w(r) is well-described as proportional to ξ(r) (R²={r2_prop_A:.3f}).")
            lines.append(f"  The spatial autocorrelation of f_DM residuals is explained")
            lines.append(f"  by galaxy clustering — galaxies that are spatially close")
            lines.append(f"  share similar large-scale environments that the local")
            lines.append(f"  5th-neighbor density failed to capture.")
            lines.append(f"  No evidence for ISF-specific scale-dependent structure.")
            lines.append(f"  This is a CLEAN NULL RESULT for the ISF hypothesis.")
        elif r2_prop_A > 0.7:
            lines.append(f"  w(r) is mostly proportional to ξ(r) (R²={r2_prop_A:.3f}).")
            lines.append(f"  Clustering explains most of the signal, with possible")
            lines.append(f"  scale-dependent residuals that warrant investigation.")
        else:
            lines.append(f"  w(r) is NOT well-described as proportional to ξ(r).")
            lines.append(f"  M(r) = w(r)/ξ(r) shows significant scale dependence,")
            lines.append(f"  which could indicate ISF-type structure or other")
            lines.append(f"  physics beyond simple clustering.")
    else:
        lines.append(f"  Could not compute proportionality test (missing Step 3 data).")

    summary = "\n".join(lines)
    print(f"\n{summary}")

    out_txt = os.path.join(DATA_DIR, "phaseB_step4_summary.txt")
    with open(out_txt, 'w') as f:
        f.write(summary)
    print(f"\n  Saved: {out_txt}")

    print("\n" + "=" * 65)
    print("Step 4 complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()
