#!/usr/bin/env python3
"""
phaseB_step3b_zslice_control.py
================================
Control test: repeat Step 3 spatial autocorrelation within narrow
redshift slices to diagnose whether the signal is genuine spatial
structure or redshift-dependent systematic leakage.

Logic:
  - Within a thin redshift shell (Δz ≈ 0.03), all galaxies are at
    nearly the same distance, eliminating the radial z-gradient
  - If w(r) signal survives within slices → real spatial structure
  - If w(r) signal vanishes within slices → redshift leakage artifact

Also computes:
  - Stacked (inverse-variance weighted) w(r) across all slices
  - Per-slice z-scores for each distance bin

Inputs (in DATA_DIR):
  - phaseB_residuals.csv  (from Step 2)

Outputs (in DATA_DIR):
  - phaseB_step3b_zslice.csv         (per-bin results for each slice)
  - phaseB_step3b_stacked.csv        (stacked w(r) across slices)
  - phaseB_step3b_diagnostics.png    (diagnostic figure)
  - phaseB_step3b_summary.txt        (text summary)

Usage:
  python3 -u phaseB_step3b_zslice_control.py 2>&1 | tee phaseB_step3b_output.txt
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

# ── Configuration ──────────────────────────────────────────────────
DATA_DIR = "/Users/chris/Documents/research_data/cosmic_structure"

# Redshift slices
Z_MIN = 0.10
Z_MAX = 0.45
Z_WIDTH = 0.03         # slice width — thin enough to kill z-gradient

# Pair counting
N_CENTERS_PER_SLICE = 5000   # centers per slice (adapt if slice is small)
R_MAX = 150.0                # Mpc — reduced since slices are thinner
N_BINS = 15                  # distance bins
N_SHUFFLES = 200             # null iterations per slice
SEED = 42
MIN_GALAXIES = 2000          # skip slices with fewer galaxies


# ── Pair logic (reused from Step 3) ───────────────────────────────
def precompute_pairs(positions, center_idx, bins, r_max):
    """Pre-compute (center, neighbor, bin) tuples using KD-tree."""
    n_bins = len(bins) - 1
    tree = cKDTree(positions)

    all_centers = []
    all_neighbors = []
    all_binidx = []
    total_pairs = 0

    for ci in center_idx:
        neighbors = tree.query_ball_point(positions[ci], r_max)
        neighbors = np.array(neighbors, dtype=np.int32)
        neighbors = neighbors[neighbors != ci]

        if len(neighbors) == 0:
            continue

        dists = np.linalg.norm(positions[neighbors] - positions[ci], axis=1)
        bin_idx = np.digitize(dists, bins) - 1
        valid = (bin_idx >= 0) & (bin_idx < n_bins)

        if valid.sum() == 0:
            continue

        all_centers.append(np.full(valid.sum(), ci, dtype=np.int32))
        all_neighbors.append(neighbors[valid])
        all_binidx.append(bin_idx[valid].astype(np.int16))
        total_pairs += valid.sum()

    if total_pairs == 0:
        return (np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int16))

    return (np.concatenate(all_centers),
            np.concatenate(all_neighbors),
            np.concatenate(all_binidx))


def compute_w(residuals, pair_c, pair_n, pair_b, n_bins):
    """Compute w(r) = <δ_i δ_j> per bin."""
    if len(pair_c) == 0:
        return np.full(n_bins, np.nan), np.zeros(n_bins, dtype=np.int64)

    products = residuals[pair_c] * residuals[pair_n]
    sum_prod = np.zeros(n_bins, dtype=np.float64)
    n_pairs = np.zeros(n_bins, dtype=np.int64)
    np.add.at(sum_prod, pair_b, products)
    np.add.at(n_pairs, pair_b, 1)
    w = np.where(n_pairs > 0, sum_prod / n_pairs, np.nan)
    return w, n_pairs


def run_slice_analysis(positions, resid_A, resid_B, n_centers, bins, r_max,
                        n_shuffles, rng):
    """Run full marked correlation + null for one redshift slice."""
    n_bins = len(bins) - 1
    n_gal = len(positions)

    nc = min(n_centers, n_gal)
    center_idx = rng.choice(n_gal, size=nc, replace=False)
    center_idx.sort()

    pair_c, pair_n, pair_b = precompute_pairs(positions, center_idx, bins, r_max)

    if len(pair_c) == 0:
        return None

    # Data
    w_A, npairs = compute_w(resid_A, pair_c, pair_n, pair_b, n_bins)
    w_B, _ = compute_w(resid_B, pair_c, pair_n, pair_b, n_bins)

    # Null distribution
    null_A = np.zeros((n_shuffles, n_bins))
    null_B = np.zeros((n_shuffles, n_bins))

    for s in range(n_shuffles):
        perm = rng.permutation(n_gal)
        null_A[s], _ = compute_w(resid_A[perm], pair_c, pair_n, pair_b, n_bins)
        null_B[s], _ = compute_w(resid_B[perm], pair_c, pair_n, pair_b, n_bins)

    null_mean_A = null_A.mean(axis=0)
    null_std_A = null_A.std(axis=0)
    null_mean_B = null_B.mean(axis=0)
    null_std_B = null_B.std(axis=0)

    z_A = np.where(null_std_A > 0, (w_A - null_mean_A) / null_std_A, 0)
    z_B = np.where(null_std_B > 0, (w_B - null_mean_B) / null_std_B, 0)

    return {
        'w_A': w_A, 'w_B': w_B,
        'z_A': z_A, 'z_B': z_B,
        'npairs': npairs,
        'n_centers': nc,
        'n_pairs_total': len(pair_c),
        'null_std_A': null_std_A,
        'null_std_B': null_std_B,
    }


# ── Main ──────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("Phase B Step 3b: Redshift-slice control test")
    print(f"  Testing whether Step 3 signal survives within narrow")
    print(f"  z-slices (Δz = {Z_WIDTH})")
    print("=" * 65)

    # ── Load ──────────────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    infile = os.path.join(DATA_DIR, "phaseB_residuals.csv")
    if not os.path.exists(infile):
        print(f"  ERROR: {infile} not found. Run Step 2 first.")
        sys.exit(1)

    df = pd.read_csv(infile)
    print(f"  Loaded {len(df):,} galaxies")

    positions = df[['x', 'y', 'z_cart']].values.astype(np.float64)
    resid_A = df['resid_rf_ext'].values.astype(np.float64)
    resid_B = df['resid_rf_ext_B'].values.astype(np.float64)
    redshifts = df['z'].values

    resid_A -= resid_A.mean()
    resid_B -= resid_B.mean()

    # Distance bins
    bins = np.logspace(np.log10(1.0), np.log10(R_MAX), N_BINS + 1)
    bin_centers = np.sqrt(bins[:-1] * bins[1:])

    # ── Define slices ─────────────────────────────────────────────
    z_edges = np.arange(Z_MIN, Z_MAX + Z_WIDTH, Z_WIDTH)
    n_slices = len(z_edges) - 1

    print(f"\n  Slices: {n_slices} from z={Z_MIN:.2f} to z={Z_MAX:.2f}")
    print(f"  Slice width: Δz = {Z_WIDTH}")
    print(f"  Distance bins: {N_BINS} log-spaced, {bins[0]:.1f} - {bins[-1]:.0f} Mpc")

    # ── Run per-slice analysis ────────────────────────────────────
    print(f"\n[2/4] Running per-slice autocorrelation...")

    rng = np.random.RandomState(SEED)
    slice_results = []

    t0_total = time.time()

    for s in range(n_slices):
        z_lo = z_edges[s]
        z_hi = z_edges[s + 1]
        z_mid = 0.5 * (z_lo + z_hi)

        mask = (redshifts >= z_lo) & (redshifts < z_hi)
        n_gal = mask.sum()

        if n_gal < MIN_GALAXIES:
            print(f"  Slice {s+1:2d}/{n_slices}: z=[{z_lo:.3f}, {z_hi:.3f}) "
                  f"N={n_gal:,} -- SKIPPED (< {MIN_GALAXIES})")
            continue

        t0 = time.time()

        # Extract slice data — re-index to 0..n_gal-1
        pos_slice = positions[mask]
        rA_slice = resid_A[mask].copy()
        rB_slice = resid_B[mask].copy()
        rA_slice -= rA_slice.mean()
        rB_slice -= rB_slice.mean()

        result = run_slice_analysis(
            pos_slice, rA_slice, rB_slice,
            N_CENTERS_PER_SLICE, bins, R_MAX,
            N_SHUFFLES, rng
        )

        elapsed = time.time() - t0

        if result is None:
            print(f"  Slice {s+1:2d}/{n_slices}: z=[{z_lo:.3f}, {z_hi:.3f}) "
                  f"N={n_gal:,} -- NO PAIRS")
            continue

        result['z_lo'] = z_lo
        result['z_hi'] = z_hi
        result['z_mid'] = z_mid
        result['n_galaxies'] = n_gal
        slice_results.append(result)

        # Summary for this slice
        isf_mask = (bin_centers >= 10) & (bin_centers <= 50)
        if isf_mask.sum() > 0:
            max_zA = np.nanmax(np.abs(result['z_A'][isf_mask]))
            max_zB = np.nanmax(np.abs(result['z_B'][isf_mask]))
        else:
            max_zA = max_zB = 0

        print(f"  Slice {s+1:2d}/{n_slices}: z=[{z_lo:.3f}, {z_hi:.3f}) "
              f"N={n_gal:>6,}  pairs={result['n_pairs_total']:>8,}  "
              f"max|z_A|={max_zA:.1f}  max|z_B|={max_zB:.1f}  "
              f"({elapsed:.0f}s)")

    total_elapsed = time.time() - t0_total
    print(f"\n  Total: {len(slice_results)} slices processed in "
          f"{total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

    if len(slice_results) == 0:
        print("  ERROR: No valid slices. Check data.")
        sys.exit(1)

    # ── Stack results across slices ───────────────────────────────
    print(f"\n[3/4] Stacking results across slices...")

    # Inverse-variance weighted stack of w(r)
    # Weight by 1/null_std² — gives more weight to slices with tighter null
    stack_w_A = np.zeros(N_BINS)
    stack_w_B = np.zeros(N_BINS)
    stack_weight_A = np.zeros(N_BINS)
    stack_weight_B = np.zeros(N_BINS)
    stack_npairs = np.zeros(N_BINS, dtype=np.int64)

    # Also collect per-slice z-scores for meta-analysis
    all_zA = []
    all_zB = []

    for res in slice_results:
        for i in range(N_BINS):
            if not np.isnan(res['w_A'][i]) and res['null_std_A'][i] > 0:
                wt = 1.0 / res['null_std_A'][i]**2
                stack_w_A[i] += res['w_A'][i] * wt
                stack_weight_A[i] += wt

            if not np.isnan(res['w_B'][i]) and res['null_std_B'][i] > 0:
                wt = 1.0 / res['null_std_B'][i]**2
                stack_w_B[i] += res['w_B'][i] * wt
                stack_weight_B[i] += wt

            stack_npairs[i] += res['npairs'][i]

        all_zA.append(res['z_A'])
        all_zB.append(res['z_B'])

    # Weighted mean
    stacked_w_A = np.where(stack_weight_A > 0, stack_w_A / stack_weight_A, np.nan)
    stacked_w_B = np.where(stack_weight_B > 0, stack_w_B / stack_weight_B, np.nan)

    # Stacked uncertainty
    stacked_err_A = np.where(stack_weight_A > 0, 1.0 / np.sqrt(stack_weight_A), np.nan)
    stacked_err_B = np.where(stack_weight_B > 0, 1.0 / np.sqrt(stack_weight_B), np.nan)

    # Stacked z-scores
    stacked_z_A = np.where(stacked_err_A > 0, stacked_w_A / stacked_err_A, 0)
    stacked_z_B = np.where(stacked_err_B > 0, stacked_w_B / stacked_err_B, 0)

    # Meta-analysis: Stouffer's method (sum of z-scores / sqrt(n))
    all_zA = np.array(all_zA)  # shape (n_slices, n_bins)
    all_zB = np.array(all_zB)
    n_valid_slices = len(slice_results)

    stouffer_z_A = np.nansum(all_zA, axis=0) / np.sqrt(
        np.sum(~np.isnan(all_zA), axis=0).clip(1))
    stouffer_z_B = np.nansum(all_zB, axis=0) / np.sqrt(
        np.sum(~np.isnan(all_zB), axis=0).clip(1))

    print(f"\n  Stacked results ({n_valid_slices} slices):")
    print(f"\n  {'r_cen':>7s}  {'w_A_stack':>10s} {'z_A':>6s}  "
          f"{'w_B_stack':>10s} {'z_B':>6s}  "
          f"{'Stouf_A':>7s} {'Stouf_B':>7s}  {'N_pairs':>10s}")
    print(f"  {'-'*7:>7s}  {'-'*10:>10s} {'-'*6:>6s}  "
          f"{'-'*10:>10s} {'-'*6:>6s}  "
          f"{'-'*7:>7s} {'-'*7:>7s}  {'-'*10:>10s}")

    for i in range(N_BINS):
        flag = " *" if (bin_centers[i] >= 10 and bin_centers[i] <= 50) else "  "
        print(f"  {bin_centers[i]:7.1f}  "
              f"{stacked_w_A[i]:10.6f} {stacked_z_A[i]:6.2f}  "
              f"{stacked_w_B[i]:10.6f} {stacked_z_B[i]:6.2f}  "
              f"{stouffer_z_A[i]:7.2f} {stouffer_z_B[i]:7.2f}  "
              f"{stack_npairs[i]:10,}{flag}")

    # ISF range significance
    isf_mask = (bin_centers >= 10) & (bin_centers <= 50)
    if isf_mask.sum() > 0:
        max_stacked_zA = np.nanmax(np.abs(stacked_z_A[isf_mask]))
        max_stacked_zB = np.nanmax(np.abs(stacked_z_B[isf_mask]))
        max_stouffer_zA = np.nanmax(np.abs(stouffer_z_A[isf_mask]))
        max_stouffer_zB = np.nanmax(np.abs(stouffer_z_B[isf_mask]))

        print(f"\n  ISF range (10-50 Mpc):")
        print(f"    Stacked max |z|: A={max_stacked_zA:.2f}, B={max_stacked_zB:.2f}")
        print(f"    Stouffer max |z|: A={max_stouffer_zA:.2f}, B={max_stouffer_zB:.2f}")

    # Compare with Step 3 full-sample result
    print(f"\n  Comparison with full-sample Step 3:")
    print(f"    If stacked z-scores are MUCH smaller than Step 3 z-scores,")
    print(f"    the signal was driven by redshift-dependent systematics.")
    print(f"    If comparable, the signal represents genuine spatial structure.")

    # ── Save results ──────────────────────────────────────────────
    # Per-slice detail
    rows = []
    for res in slice_results:
        for i in range(N_BINS):
            rows.append({
                'z_lo': res['z_lo'],
                'z_hi': res['z_hi'],
                'z_mid': res['z_mid'],
                'n_galaxies': res['n_galaxies'],
                'r_center': bin_centers[i],
                'r_min': bins[i],
                'r_max': bins[i + 1],
                'w_A': res['w_A'][i],
                'w_B': res['w_B'][i],
                'z_score_A': res['z_A'][i],
                'z_score_B': res['z_B'][i],
                'npairs': res['npairs'][i],
            })

    detail_df = pd.DataFrame(rows)
    out_detail = os.path.join(DATA_DIR, "phaseB_step3b_zslice.csv")
    detail_df.to_csv(out_detail, index=False)
    print(f"\n  Saved: {out_detail}")

    # Stacked results
    stacked_df = pd.DataFrame({
        'r_center': bin_centers,
        'r_min': bins[:-1],
        'r_max': bins[1:],
        'stacked_w_A': stacked_w_A,
        'stacked_err_A': stacked_err_A,
        'stacked_z_A': stacked_z_A,
        'stouffer_z_A': stouffer_z_A,
        'stacked_w_B': stacked_w_B,
        'stacked_err_B': stacked_err_B,
        'stacked_z_B': stacked_z_B,
        'stouffer_z_B': stouffer_z_B,
        'total_npairs': stack_npairs,
    })
    out_stacked = os.path.join(DATA_DIR, "phaseB_step3b_stacked.csv")
    stacked_df.to_csv(out_stacked, index=False)
    print(f"  Saved: {out_stacked}")

    # ── Diagnostic figure ─────────────────────────────────────────
    print(f"\n[4/4] Generating diagnostic figure...")

    fig = plt.figure(figsize=(18, 15))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Panel 1: Stacked w(r) Method A with errors
    ax = fig.add_subplot(gs[0, 0])
    ax.errorbar(bin_centers, stacked_w_A, yerr=stacked_err_A,
                fmt='o-', color='darkred', ms=4, lw=1.5, capsize=3,
                label='Stacked (z-slices)')
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.axvspan(10, 50, alpha=0.08, color='blue', label='ISF range')
    ax.set_xscale('log')
    ax.set_xlabel('Comoving separation (Mpc)')
    ax.set_ylabel('$w(r)$')
    ax.set_title('Method A: Stacked across z-slices')
    ax.legend(fontsize=8)

    # Panel 2: Stacked w(r) Method B with errors
    ax = fig.add_subplot(gs[0, 1])
    ax.errorbar(bin_centers, stacked_w_B, yerr=stacked_err_B,
                fmt='o-', color='navy', ms=4, lw=1.5, capsize=3,
                label='Stacked (z-slices)')
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.axvspan(10, 50, alpha=0.08, color='blue', label='ISF range')
    ax.set_xscale('log')
    ax.set_xlabel('Comoving separation (Mpc)')
    ax.set_ylabel('$w(r)$')
    ax.set_title('Method B: Stacked across z-slices')
    ax.legend(fontsize=8)

    # Panel 3: Stouffer z-scores (the key diagnostic)
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(bin_centers, stouffer_z_A, 'o-', color='darkred', ms=4, lw=1.5,
            label='Method A')
    ax.plot(bin_centers, stouffer_z_B, 's-', color='navy', ms=4, lw=1.5,
            label='Method B')
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.axhline(2, color='orange', ls='--', alpha=0.5, label='2σ')
    ax.axhline(-2, color='orange', ls='--', alpha=0.5)
    ax.axhline(3, color='red', ls='--', alpha=0.5, label='3σ')
    ax.axhline(-3, color='red', ls='--', alpha=0.5)
    ax.axvspan(10, 50, alpha=0.08, color='blue')
    ax.set_xscale('log')
    ax.set_xlabel('Comoving separation (Mpc)')
    ax.set_ylabel('Stouffer combined z-score')
    ax.set_title('Combined significance across z-slices')
    ax.legend(fontsize=7)

    # Panel 4: Per-slice z-scores as heatmap (Method A)
    ax = fig.add_subplot(gs[1, 0])
    z_mids = [res['z_mid'] for res in slice_results]
    im = ax.imshow(all_zA, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3,
                    extent=[np.log10(bins[0]), np.log10(bins[-1]),
                            len(slice_results) - 0.5, -0.5])
    ax.set_xlabel('log(r/Mpc)')
    ax.set_ylabel('Slice index')
    ax.set_title('Per-slice z-scores (Method A)')
    plt.colorbar(im, ax=ax, label='z-score')
    # Mark ISF range
    ax.axvline(np.log10(10), color='blue', ls='--', alpha=0.5)
    ax.axvline(np.log10(50), color='blue', ls='--', alpha=0.5)

    # Panel 5: Per-slice z-scores as heatmap (Method B)
    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(all_zB, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3,
                    extent=[np.log10(bins[0]), np.log10(bins[-1]),
                            len(slice_results) - 0.5, -0.5])
    ax.set_xlabel('log(r/Mpc)')
    ax.set_ylabel('Slice index')
    ax.set_title('Per-slice z-scores (Method B)')
    plt.colorbar(im, ax=ax, label='z-score')
    ax.axvline(np.log10(10), color='blue', ls='--', alpha=0.5)
    ax.axvline(np.log10(50), color='blue', ls='--', alpha=0.5)

    # Panel 6: Galaxies per slice
    ax = fig.add_subplot(gs[1, 2])
    slice_z_mids = [res['z_mid'] for res in slice_results]
    slice_ngal = [res['n_galaxies'] for res in slice_results]
    ax.bar(slice_z_mids, slice_ngal, width=Z_WIDTH * 0.8,
           color='steelblue', edgecolor='navy', alpha=0.7)
    ax.set_xlabel('Redshift (slice midpoint)')
    ax.set_ylabel('Galaxies per slice')
    ax.set_title(f'{len(slice_results)} slices, Δz={Z_WIDTH}')

    # Panel 7: Distribution of per-slice z-scores in ISF range
    ax = fig.add_subplot(gs[2, 0])
    if isf_mask.sum() > 0:
        isf_zA = all_zA[:, isf_mask].ravel()
        isf_zB = all_zB[:, isf_mask].ravel()
        isf_zA = isf_zA[~np.isnan(isf_zA)]
        isf_zB = isf_zB[~np.isnan(isf_zB)]

        ax.hist(isf_zA, bins=30, color='salmon', edgecolor='darkred',
                alpha=0.7, density=True, label=f'Method A (N={len(isf_zA)})')
        ax.hist(isf_zB, bins=30, color='lightblue', edgecolor='navy',
                alpha=0.5, density=True, label=f'Method B (N={len(isf_zB)})')

        # Standard normal for reference
        x_grid = np.linspace(-4, 4, 100)
        ax.plot(x_grid, stats.norm.pdf(x_grid), 'k--', lw=1.5,
                label='N(0,1) expected')

        ax.set_xlabel('z-score')
        ax.set_ylabel('Density')
        ax.set_title('Per-slice z-scores in ISF range (10-50 Mpc)')
        ax.legend(fontsize=7)

        # One-sample t-test: are mean z-scores > 0?
        t_A, p_tA = stats.ttest_1samp(isf_zA, 0)
        t_B, p_tB = stats.ttest_1samp(isf_zB, 0)
        ax.text(0.05, 0.85,
                f'Mean z_A={isf_zA.mean():.2f} (t={t_A:.2f}, p={p_tA:.3f})\n'
                f'Mean z_B={isf_zB.mean():.2f} (t={t_B:.2f}, p={p_tB:.3f})',
                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel 8: Stacked z-score comparison with full-sample Step 3
    ax = fig.add_subplot(gs[2, 1])
    # Load Step 3 results if available
    step3_file = os.path.join(DATA_DIR, "phaseB_step3_correlation.csv")
    if os.path.exists(step3_file):
        step3 = pd.read_csv(step3_file)
        ax.plot(step3['r_center'], step3['z_score_A'], 'o--', color='darkred',
                ms=3, lw=1, alpha=0.5, label='Step 3 full-sample A')
        ax.plot(step3['r_center'], step3['z_score_B'], 's--', color='navy',
                ms=3, lw=1, alpha=0.5, label='Step 3 full-sample B')

    ax.plot(bin_centers, stouffer_z_A, 'o-', color='darkred', ms=5, lw=2,
            label='z-slice stacked A')
    ax.plot(bin_centers, stouffer_z_B, 's-', color='navy', ms=5, lw=2,
            label='z-slice stacked B')
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.axhline(2, color='orange', ls='--', alpha=0.3)
    ax.axhline(-2, color='orange', ls='--', alpha=0.3)
    ax.axvspan(10, 50, alpha=0.08, color='blue')
    ax.set_xscale('log')
    ax.set_xlabel('Comoving separation (Mpc)')
    ax.set_ylabel('z-score')
    ax.set_title('Full-sample vs z-slice stacked')
    ax.legend(fontsize=6, ncol=2)

    # Panel 9: Signal-to-noise ratio (z-slice / full-sample)
    ax = fig.add_subplot(gs[2, 2])
    if os.path.exists(step3_file):
        # Interpolate Step 3 to our bin centers
        from scipy.interpolate import interp1d
        interp_zA = interp1d(step3['r_center'], step3['z_score_A'],
                              bounds_error=False, fill_value=np.nan)
        interp_zB = interp1d(step3['r_center'], step3['z_score_B'],
                              bounds_error=False, fill_value=np.nan)

        ratio_A = stouffer_z_A / interp_zA(bin_centers)
        ratio_B = stouffer_z_B / interp_zB(bin_centers)

        ax.plot(bin_centers, ratio_A, 'o-', color='darkred', ms=4, lw=1.5,
                label='Method A')
        ax.plot(bin_centers, ratio_B, 's-', color='navy', ms=4, lw=1.5,
                label='Method B')
        ax.axhline(1, color='gray', ls='--', alpha=0.5, label='Unchanged')
        ax.axhline(0, color='gray', ls=':', alpha=0.5, label='Signal killed')
        ax.axvspan(10, 50, alpha=0.08, color='blue')
        ax.set_xscale('log')
        ax.set_xlabel('Comoving separation (Mpc)')
        ax.set_ylabel('z-score ratio (sliced / full)')
        ax.set_title('Signal retention after z-slicing')
        ax.legend(fontsize=7)
        ax.set_ylim(-0.5, 2.0)
    else:
        ax.text(0.5, 0.5, 'Step 3 results not found\nfor comparison',
                transform=ax.transAxes, ha='center', va='center')

    fig.suptitle('Phase B Step 3b: Redshift-Slice Control Test',
                 fontsize=14, fontweight='bold', y=0.99)

    out_fig = os.path.join(DATA_DIR, "phaseB_step3b_diagnostics.png")
    plt.savefig(out_fig, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_fig}")

    # ── Summary ───────────────────────────────────────────────────
    lines = []
    lines.append("Phase B Step 3b: Redshift-Slice Control Test Summary")
    lines.append("=" * 60)
    lines.append(f"Slices: {n_valid_slices} (Δz={Z_WIDTH}, z=[{Z_MIN:.2f}, {Z_MAX:.2f}])")
    lines.append(f"Galaxies per slice: "
                 f"{min(slice_ngal):,} - {max(slice_ngal):,} "
                 f"(median {int(np.median(slice_ngal)):,})")
    lines.append(f"Distance bins: {N_BINS}, {bins[0]:.1f} - {bins[-1]:.0f} Mpc")
    lines.append(f"Null shuffles per slice: {N_SHUFFLES}")
    lines.append(f"Total runtime: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    lines.append(f"")
    lines.append(f"ISF range (10-50 Mpc):")

    if isf_mask.sum() > 0:
        lines.append(f"  Stouffer combined z-scores:")
        for i in np.where(isf_mask)[0]:
            lines.append(f"    r={bin_centers[i]:6.1f} Mpc: "
                         f"z_A={stouffer_z_A[i]:+.2f}, z_B={stouffer_z_B[i]:+.2f}")

        lines.append(f"  Max Stouffer |z|: A={max_stouffer_zA:.2f}, B={max_stouffer_zB:.2f}")

        if isf_mask.sum() > 0 and 'isf_zA' in dir():
            lines.append(f"")
            lines.append(f"  Per-slice z-score distribution (ISF bins):")
            lines.append(f"    Method A: mean={isf_zA.mean():.3f}, "
                         f"std={isf_zA.std():.3f}, "
                         f"t={t_A:.2f}, p={p_tA:.4f}")
            lines.append(f"    Method B: mean={isf_zB.mean():.3f}, "
                         f"std={isf_zB.std():.3f}, "
                         f"t={t_B:.2f}, p={p_tB:.4f}")

    lines.append(f"")
    lines.append(f"Verdict:")
    # Determine outcome
    if isf_mask.sum() > 0:
        signal_survives_A = max_stouffer_zA > 2
        signal_survives_B = max_stouffer_zB > 2

        if signal_survives_A and signal_survives_B:
            lines.append(f"  Signal SURVIVES z-slicing in both methods.")
            lines.append(f"  This suggests genuine spatial structure in f_DM residuals,")
            lines.append(f"  not merely a redshift-dependent systematic.")
        elif signal_survives_A or signal_survives_B:
            lines.append(f"  Signal survives in one method only after z-slicing.")
            lines.append(f"  Ambiguous — could be partial systematic contamination.")
        else:
            lines.append(f"  Signal VANISHES after z-slicing.")
            lines.append(f"  The Step 3 detection was driven by redshift-dependent")
            lines.append(f"  systematics (K-correction residuals, size calibration,")
            lines.append(f"  or Malmquist-like selection effects).")
            lines.append(f"  This is a clean null result for the ISF hypothesis.")

    summary = "\n".join(lines)
    print(f"\n{summary}")

    out_txt = os.path.join(DATA_DIR, "phaseB_step3b_summary.txt")
    with open(out_txt, 'w') as f:
        f.write(summary)
    print(f"\n  Saved: {out_txt}")

    print("\n" + "=" * 65)
    print("Step 3b complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()