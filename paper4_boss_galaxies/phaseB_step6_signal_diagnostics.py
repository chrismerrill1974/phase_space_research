#!/usr/bin/env python3
"""
phaseB_step6_signal_diagnostics.py
===================================
Phase B, Step 6: Investigating the origin of the spatial signal.

Three diagnostic axes:
  1. Angular vs radial decomposition — split pair separations into
     transverse (sky-plane) and line-of-sight components within z-slices.
     Survey systematics are angular; real 3D structure is isotropic.
  2. SDSS plate-level analysis — compute mean f_DM residual per plate.
     If inter-plate variance exceeds shot noise → plate calibration.
  3. Feature-by-feature spatial autocorrelation — which input variable
     (color, R_eff, σ_v) carries the spatial signal?

Inputs (in DATA_DIR):
  - phaseB_residuals.csv       (from Step 2)
  - phaseB_merged.csv          (from Step 1, for plate info)

Outputs (in DATA_DIR):
  - phaseB_step6_diagnostics.png
  - phaseB_step6_aniso.csv
  - phaseB_step6_plates.csv
  - phaseB_step6_features.csv
  - phaseB_step6_summary.txt

Usage:
  python3 -u phaseB_step6_signal_diagnostics.py 2>&1 | tee phaseB_step6_output.txt
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

# Cosmology (must match Step 1)
H0 = 67.4
OMEGA_M = 0.315
C_LIGHT = 299792.458

# Anisotropy test
N_CENTERS_ANISO = 10000
R_MAX_ANISO = 150.0
N_BINS_ANISO = 12
N_SHUFFLES = 200

# Z-slice for anisotropy (pick densest slice)
Z_SLICE_LO = 0.22
Z_SLICE_HI = 0.31     # two slices combined for statistics

# Feature autocorrelation
N_CENTERS_FEAT = 10000
R_MAX_FEAT = 200.0
N_BINS_FEAT = 15

SEED = 42

# ══════════════════════════════════════════════════════════════════
# Diagnostic 1: Angular vs radial (anisotropy) decomposition
# ══════════════════════════════════════════════════════════════════

def decompose_pairs_aniso(positions, center_idx, r_max, observer=None):
    """For each pair, compute transverse and radial separations.

    The line-of-sight direction is defined per-pair as the mean
    position vector from the observer (origin by default).

    Returns
    -------
    centers, neighbors : int arrays — pair indices
    r_perp : float array — transverse (angular) separation in Mpc
    r_para : float array — radial (LOS) separation in Mpc
    r_total : float array — total 3D separation in Mpc
    """
    if observer is None:
        observer = np.zeros(3)

    tree = cKDTree(positions)

    all_c = []
    all_n = []
    all_rperp = []
    all_rpara = []
    all_rtot = []

    for ci in center_idx:
        neighbors = tree.query_ball_point(positions[ci], r_max)
        neighbors = np.array(neighbors, dtype=np.int32)
        neighbors = neighbors[neighbors != ci]

        if len(neighbors) == 0:
            continue

        # Separation vectors
        sep = positions[neighbors] - positions[ci]
        r_tot = np.linalg.norm(sep, axis=1)

        # LOS direction: mean of the two position vectors
        mid = 0.5 * (positions[ci] + positions[neighbors]) - observer
        mid_norm = np.linalg.norm(mid, axis=1, keepdims=True)
        mid_norm = np.clip(mid_norm, 1e-10, None)
        los_hat = mid / mid_norm

        # Radial (LOS) component: π = |sep · los_hat|
        r_para = np.abs(np.sum(sep * los_hat, axis=1))

        # Transverse component: σ = sqrt(r² - π²)
        r_perp = np.sqrt(np.clip(r_tot**2 - r_para**2, 0, None))

        all_c.append(np.full(len(neighbors), ci, dtype=np.int32))
        all_n.append(neighbors)
        all_rperp.append(r_perp.astype(np.float32))
        all_rpara.append(r_para.astype(np.float32))
        all_rtot.append(r_tot.astype(np.float32))

    return (np.concatenate(all_c),
            np.concatenate(all_n),
            np.concatenate(all_rperp),
            np.concatenate(all_rpara),
            np.concatenate(all_rtot))


def compute_w_binned(residuals, pair_c, pair_n, separations, bin_edges):
    """Compute w(r) = <δ_i δ_j> binned by arbitrary separation measure."""
    n_bins = len(bin_edges) - 1
    bin_idx = np.digitize(separations, bin_edges) - 1
    valid = (bin_idx >= 0) & (bin_idx < n_bins)

    products = residuals[pair_c] * residuals[pair_n]

    sum_prod = np.zeros(n_bins, dtype=np.float64)
    n_pairs = np.zeros(n_bins, dtype=np.int64)
    np.add.at(sum_prod, bin_idx[valid], products[valid])
    np.add.at(n_pairs, bin_idx[valid], 1)

    w = np.where(n_pairs > 0, sum_prod / n_pairs, np.nan)
    return w, n_pairs

# ══════════════════════════════════════════════════════════════════
# Standard pair counting (reused from Step 3)
# ══════════════════════════════════════════════════════════════════

def precompute_pairs(positions, center_idx, bins, r_max):
    """Pre-compute (center, neighbor, bin) tuples using KD-tree."""
    n_bins = len(bins) - 1
    tree = cKDTree(positions)

    all_centers = []
    all_neighbors = []
    all_binidx = []

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

    if len(all_centers) == 0:
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


# ══════════════════════════════════════════════════════════════════
# Diagnostic 2: Plate-level analysis
# ══════════════════════════════════════════════════════════════════

def plate_analysis(df, resid_col, plate_col='PLATE'):
    """Compute per-plate mean residual and test for excess variance.

    Returns
    -------
    plate_df : DataFrame with per-plate stats
    F_stat : F-statistic from one-way ANOVA
    p_anova : ANOVA p-value
    excess_var : fraction of total variance that is inter-plate
    """
    if plate_col not in df.columns:
        return None, np.nan, np.nan, np.nan

    grouped = df.groupby(plate_col)[resid_col]
    plate_df = grouped.agg(['mean', 'std', 'count']).reset_index()
    plate_df.columns = [plate_col, 'mean_resid', 'std_resid', 'n_gal']

    # Only plates with enough galaxies
    plate_df = plate_df[plate_df['n_gal'] >= 20].copy()

    if len(plate_df) < 5:
        return plate_df, np.nan, np.nan, np.nan

    # One-way ANOVA: is inter-plate variance significant?
    groups = [g[resid_col].values for _, g in df.groupby(plate_col)
              if len(g) >= 20]
    if len(groups) >= 2:
        F_stat, p_anova = stats.f_oneway(*groups)
    else:
        F_stat, p_anova = np.nan, np.nan

    # Excess variance: var(plate_means) vs expected from shot noise
    plate_means = plate_df['mean_resid'].values
    plate_counts = plate_df['n_gal'].values
    overall_var = df[resid_col].var()

    observed_var = np.var(plate_means)
    expected_var = overall_var / plate_counts.mean()  # shot noise
    excess_var = max(0, (observed_var - expected_var) / overall_var)

    return plate_df, F_stat, p_anova, excess_var

# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("Phase B Step 6: Signal origin diagnostics")
    print("  1. Angular vs radial (anisotropy)")
    print("  2. SDSS plate-level variance")
    print("  3. Feature-by-feature autocorrelation")
    print("=" * 65)

    rng = np.random.RandomState(SEED)

    # ── Load data ─────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    df = pd.read_csv(os.path.join(DATA_DIR, "phaseB_residuals.csv"))
    print(f"  Residuals: {len(df):,} galaxies")

    positions = df[['x', 'y', 'z_cart']].values.astype(np.float64)
    resid_A = df['resid_rf_ext'].values.astype(np.float64)
    resid_B = df['resid_rf_ext_B'].values.astype(np.float64)
    resid_A -= resid_A.mean()
    resid_B -= resid_B.mean()
    redshifts = df['z'].values

    # Try loading merged file for plate info and raw features
    merged_file = os.path.join(DATA_DIR, "phaseB_merged.csv")
    has_merged = os.path.exists(merged_file)
    if has_merged:
        df_m = pd.read_csv(merged_file)
        print(f"  Merged catalog: {len(df_m):,} galaxies")
        # Align by index (both should be same sample in same order)
        if len(df_m) == len(df):
            for col in df_m.columns:
                if col not in df.columns:
                    df[col] = df_m[col].values
        else:
            print(f"  WARNING: merged file length mismatch, skipping")
            has_merged = False

    has_plates = 'PLATE' in df.columns or 'plate' in df.columns
    plate_col = 'PLATE' if 'PLATE' in df.columns else (
        'plate' if 'plate' in df.columns else None)
    print(f"  Has plate info: {has_plates}")

    # ── Diagnostic 1: Anisotropy ──────────────────────────────────
    print(f"\n[2/5] Diagnostic 1: Angular vs radial decomposition")
    print(f"  Z-slice: [{Z_SLICE_LO:.2f}, {Z_SLICE_HI:.2f})")

    z_mask = (redshifts >= Z_SLICE_LO) & (redshifts < Z_SLICE_HI)
    n_slice = z_mask.sum()
    print(f"  Galaxies in slice: {n_slice:,}")

    pos_slice = positions[z_mask]
    rA_slice = resid_A[z_mask].copy()
    rA_slice -= rA_slice.mean()
    rB_slice = resid_B[z_mask].copy()
    rB_slice -= rB_slice.mean()

    # Subsample centers
    nc = min(N_CENTERS_ANISO, n_slice)
    center_idx = rng.choice(n_slice, size=nc, replace=False)
    center_idx.sort()

    print(f"  Decomposing {nc:,} centers into transverse/radial...")
    t0 = time.time()
    pc, pn, r_perp, r_para, r_tot = decompose_pairs_aniso(
        pos_slice, center_idx, R_MAX_ANISO)
    print(f"  {len(pc):,} pairs in {time.time()-t0:.1f}s")

    # Bin edges for transverse and radial separately
    bins_aniso = np.logspace(np.log10(1.0), np.log10(R_MAX_ANISO),
                              N_BINS_ANISO + 1)
    bc_aniso = np.sqrt(bins_aniso[:-1] * bins_aniso[1:])

    # w(r_perp) — angular signal
    w_perp_A, np_perp = compute_w_binned(
        rA_slice, pc, pn, r_perp, bins_aniso)
    w_perp_B, _ = compute_w_binned(
        rB_slice, pc, pn, r_perp, bins_aniso)

    # w(r_para) — radial signal
    w_para_A, np_para = compute_w_binned(
        rA_slice, pc, pn, r_para, bins_aniso)
    w_para_B, _ = compute_w_binned(
        rB_slice, pc, pn, r_para, bins_aniso)

    # w(r_tot) — isotropic (for reference)
    w_tot_A, np_tot = compute_w_binned(
        rA_slice, pc, pn, r_tot, bins_aniso)

    # Null distributions for z-scores
    print(f"  Running {N_SHUFFLES} null shuffles for anisotropy...")
    t0 = time.time()
    null_perp_A = np.zeros((N_SHUFFLES, N_BINS_ANISO))
    null_para_A = np.zeros((N_SHUFFLES, N_BINS_ANISO))
    null_perp_B = np.zeros((N_SHUFFLES, N_BINS_ANISO))
    null_para_B = np.zeros((N_SHUFFLES, N_BINS_ANISO))

    for s in range(N_SHUFFLES):
        perm = rng.permutation(n_slice)
        null_perp_A[s], _ = compute_w_binned(
            rA_slice[perm], pc, pn, r_perp, bins_aniso)
        null_para_A[s], _ = compute_w_binned(
            rA_slice[perm], pc, pn, r_para, bins_aniso)
        null_perp_B[s], _ = compute_w_binned(
            rB_slice[perm], pc, pn, r_perp, bins_aniso)
        null_para_B[s], _ = compute_w_binned(
            rB_slice[perm], pc, pn, r_para, bins_aniso)

        if (s + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (s + 1) * (N_SHUFFLES - s - 1)
            print(f"\r    Shuffle {s+1}/{N_SHUFFLES} "
                  f"({elapsed:.0f}s, ~{eta:.0f}s remaining)",
                  end='', flush=True)

    print(f"\n    Done in {time.time()-t0:.0f}s")

    def zscore(data, null):
        m = null.mean(axis=0)
        s = null.std(axis=0)
        return np.where(s > 0, (data - m) / s, 0)

    z_perp_A = zscore(w_perp_A, null_perp_A)
    z_para_A = zscore(w_para_A, null_para_A)
    z_perp_B = zscore(w_perp_B, null_perp_B)
    z_para_B = zscore(w_para_B, null_para_B)

    print(f"\n  Anisotropy results (Method A):")
    print(f"  {'r_cen':>7s}  {'w_perp':>10s} {'z_perp':>6s}  "
          f"{'w_para':>10s} {'z_para':>6s}  "
          f"{'N_perp':>8s} {'N_para':>8s}")
    print(f"  {'-'*7:>7s}  {'-'*10:>10s} {'-'*6:>6s}  "
          f"{'-'*10:>10s} {'-'*6:>6s}  "
          f"{'-'*8:>8s} {'-'*8:>8s}")
    for i in range(N_BINS_ANISO):
        flag = " *" if (bc_aniso[i] >= 10 and bc_aniso[i] <= 50) else "  "
        print(f"  {bc_aniso[i]:7.1f}  "
              f"{w_perp_A[i]:10.6f} {z_perp_A[i]:6.2f}  "
              f"{w_para_A[i]:10.6f} {z_para_A[i]:6.2f}  "
              f"{np_perp[i]:8,} {np_para[i]:8,}{flag}")

    # Save anisotropy results
    aniso_df = pd.DataFrame({
        'r_center': bc_aniso,
        'w_perp_A': w_perp_A, 'z_perp_A': z_perp_A,
        'w_para_A': w_para_A, 'z_para_A': z_para_A,
        'w_perp_B': w_perp_B, 'z_perp_B': z_perp_B,
        'w_para_B': w_para_B, 'z_para_B': z_para_B,
        'npairs_perp': np_perp, 'npairs_para': np_para,
    })
    aniso_df.to_csv(os.path.join(DATA_DIR, "phaseB_step6_aniso.csv"),
                     index=False)
    print(f"  Saved: phaseB_step6_aniso.csv")

    # ── Diagnostic 2: Plate-level analysis ──────────────────────────
    print(f"\n[3/5] Diagnostic 2: SDSS plate-level analysis")

    if plate_col is not None:
        plate_df, F_stat, p_anova, excess_var = plate_analysis(
            df, 'resid_rf_ext', plate_col)

        if plate_df is not None:
            n_plates = len(plate_df)
            print(f"  Plates with ≥20 galaxies: {n_plates}")
            print(f"  ANOVA F-statistic: {F_stat:.2f}")
            print(f"  ANOVA p-value:     {p_anova:.4e}")
            print(f"  Excess inter-plate variance: {excess_var*100:.2f}%")

            # Top 10 most extreme plates
            plate_df['abs_mean'] = plate_df['mean_resid'].abs()
            top_plates = plate_df.nlargest(10, 'abs_mean')
            print(f"\n  Top 10 most extreme plates:")
            print(f"  {'Plate':>8s}  {'N_gal':>6s}  {'Mean_resid':>10s}")
            for _, row in top_plates.iterrows():
                print(f"  {int(row[plate_col]):>8d}  "
                      f"{int(row['n_gal']):>6d}  "
                      f"{row['mean_resid']:>10.5f}")

            plate_df.to_csv(
                os.path.join(DATA_DIR, "phaseB_step6_plates.csv"),
                index=False)
            print(f"  Saved: phaseB_step6_plates.csv")

            # Also do Method B
            _, F_B, p_B, excess_B = plate_analysis(
                df, 'resid_rf_ext_B', plate_col)
            print(f"\n  Method B: F={F_B:.2f}, p={p_B:.4e}, "
                  f"excess={excess_B*100:.2f}%")
        else:
            print(f"  Not enough plates with ≥20 galaxies")
            n_plates = 0
            F_stat = p_anova = excess_var = np.nan
            F_B = p_B = excess_B = np.nan
    else:
        print(f"  No plate column found — skipping")
        print(f"  (Columns available: {list(df.columns[:20])}...)")
        n_plates = 0
        plate_df = None
        F_stat = p_anova = excess_var = np.nan
        F_B = p_B = excess_B = np.nan

    # ── Diagnostic 3: Feature-by-feature autocorrelation ──────────
    print(f"\n[4/5] Diagnostic 3: Feature-by-feature spatial autocorrelation")
    print(f"  Which input variable carries the spatial signal?")

    # Features to test: each galaxy property's residual
    # after regressing out the OTHER properties
    feature_candidates = []
    for col in ['veldisp', 'R50_kpc', 'gmr_rest', 'log_Mstar_A', 'z']:
        if col in df.columns:
            feature_candidates.append(col)

    print(f"  Features to test: {feature_candidates}")

    # For each feature, compute its residual after regressing on
    # the other features, then measure spatial autocorrelation
    from sklearn.ensemble import RandomForestRegressor

    bins_feat = np.logspace(np.log10(1.0), np.log10(R_MAX_FEAT),
                             N_BINS_FEAT + 1)
    bc_feat = np.sqrt(bins_feat[:-1] * bins_feat[1:])

    # Pre-compute pairs once for full sample
    nc_feat = min(N_CENTERS_FEAT, len(positions))
    center_feat = rng.choice(len(positions), size=nc_feat, replace=False)
    center_feat.sort()

    print(f"  Pre-computing pairs ({nc_feat:,} centers)...")
    t0 = time.time()
    pc_f, pn_f, pb_f = precompute_pairs(
        positions, center_feat, bins_feat, R_MAX_FEAT)
    print(f"  {len(pc_f):,} pairs in {time.time()-t0:.1f}s")

    # Null distribution (one shared set for efficiency)
    print(f"  Computing shared null ({N_SHUFFLES} shuffles)...")
    t0 = time.time()
    # We'll compute z-scores from shuffled version of each feature resid
    # But to be efficient, precompute pair structure, shuffle per-feature

    feature_results = {}

    for feat in feature_candidates:
        t1 = time.time()
        y = df[feat].values.astype(np.float64)

        # Regress feat ~ other features
        other_feats = [f for f in feature_candidates if f != feat]
        X = df[other_feats].values

        rf = RandomForestRegressor(
            n_estimators=100, max_depth=6,
            min_samples_leaf=50, n_jobs=1, random_state=42)
        rf.fit(X, y)
        feat_resid = y - rf.predict(X)
        feat_resid -= feat_resid.mean()

        # w(r) for this feature's residual
        w_feat, np_feat = compute_w(
            feat_resid, pc_f, pn_f, pb_f, N_BINS_FEAT)

        # Quick null (50 shuffles for speed)
        null_feat = np.zeros((50, N_BINS_FEAT))
        for s in range(50):
            perm = rng.permutation(len(feat_resid))
            null_feat[s], _ = compute_w(
                feat_resid[perm], pc_f, pn_f, pb_f, N_BINS_FEAT)

        z_feat = zscore(w_feat, null_feat)

        # ISF range max
        isf_mask = (bc_feat >= 10) & (bc_feat <= 50)
        max_z_isf = np.nanmax(np.abs(z_feat[isf_mask])) if isf_mask.sum() > 0 else 0
        max_z_all = np.nanmax(np.abs(z_feat))

        feature_results[feat] = {
            'w': w_feat,
            'z': z_feat,
            'max_z_isf': max_z_isf,
            'max_z_all': max_z_all,
            'resid_std': feat_resid.std(),
        }

        print(f"    {feat:15s}: max|z| ISF={max_z_isf:.1f}, "
              f"all={max_z_all:.1f}  ({time.time()-t1:.1f}s)")

    # Also compute for the f_DM residuals themselves for comparison
    w_fDM_A, _ = compute_w(resid_A, pc_f, pn_f, pb_f, N_BINS_FEAT)
    null_fDM = np.zeros((50, N_BINS_FEAT))
    for s in range(50):
        perm = rng.permutation(len(resid_A))
        null_fDM[s], _ = compute_w(resid_A[perm], pc_f, pn_f, pb_f, N_BINS_FEAT)
    z_fDM_A = zscore(w_fDM_A, null_fDM)

    isf_mask_f = (bc_feat >= 10) & (bc_feat <= 50)
    max_z_fDM = np.nanmax(np.abs(z_fDM_A[isf_mask_f]))
    print(f"    {'f_DM resid A':15s}: max|z| ISF={max_z_fDM:.1f}, "
          f"all={np.nanmax(np.abs(z_fDM_A)):.1f}")

    # Save feature results
    feat_df_rows = []
    for i in range(N_BINS_FEAT):
        row = {'r_center': bc_feat[i]}
        for feat in feature_candidates:
            row[f'w_{feat}'] = feature_results[feat]['w'][i]
            row[f'z_{feat}'] = feature_results[feat]['z'][i]
        row['w_fDM_A'] = w_fDM_A[i]
        row['z_fDM_A'] = z_fDM_A[i]
        feat_df_rows.append(row)

    pd.DataFrame(feat_df_rows).to_csv(
        os.path.join(DATA_DIR, "phaseB_step6_features.csv"), index=False)
    print(f"  Saved: phaseB_step6_features.csv")

    # ── Figure ─────────────────────────────────────────────────────
    print(f"\n[5/5] Generating diagnostic figure...")

    fig = plt.figure(figsize=(18, 18))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Panel 1: w(r_perp) vs w(r_para) — Method A
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(bc_aniso, z_perp_A, 'o-', color='darkred', ms=5, lw=2,
            label='Transverse (angular)')
    ax.plot(bc_aniso, z_para_A, 's-', color='navy', ms=5, lw=2,
            label='Radial (LOS)')
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.axhline(2, color='orange', ls='--', alpha=0.3)
    ax.axhline(-2, color='orange', ls='--', alpha=0.3)
    ax.axvspan(10, 50, alpha=0.08, color='blue')
    ax.set_xscale('log')
    ax.set_xlabel('Separation (Mpc)')
    ax.set_ylabel('z-score')
    ax.set_title('Anisotropy: Method A (KEY)')
    ax.legend(fontsize=8)

    # Panel 2: w(r_perp) vs w(r_para) — Method B
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(bc_aniso, z_perp_B, 'o-', color='darkred', ms=5, lw=2,
            label='Transverse (angular)')
    ax.plot(bc_aniso, z_para_B, 's-', color='navy', ms=5, lw=2,
            label='Radial (LOS)')
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.axhline(2, color='orange', ls='--', alpha=0.3)
    ax.axhline(-2, color='orange', ls='--', alpha=0.3)
    ax.axvspan(10, 50, alpha=0.08, color='blue')
    ax.set_xscale('log')
    ax.set_xlabel('Separation (Mpc)')
    ax.set_ylabel('z-score')
    ax.set_title('Anisotropy: Method B')
    ax.legend(fontsize=8)

    # Panel 3: Anisotropy ratio
    ax = fig.add_subplot(gs[0, 2])
    # Ratio of transverse to total signal
    ratio_A = np.where(
        (np.abs(z_perp_A) + np.abs(z_para_A)) > 0.5,
        np.abs(z_perp_A) / (np.abs(z_perp_A) + np.abs(z_para_A)),
        np.nan)
    valid_r = ~np.isnan(ratio_A)
    if valid_r.sum() > 0:
        ax.plot(bc_aniso[valid_r], ratio_A[valid_r],
                'o-', color='darkred', ms=5, lw=1.5, label='Method A')
    ratio_B = np.where(
        (np.abs(z_perp_B) + np.abs(z_para_B)) > 0.5,
        np.abs(z_perp_B) / (np.abs(z_perp_B) + np.abs(z_para_B)),
        np.nan)
    valid_rB = ~np.isnan(ratio_B)
    if valid_rB.sum() > 0:
        ax.plot(bc_aniso[valid_rB], ratio_B[valid_rB],
                's-', color='navy', ms=5, lw=1.5, label='Method B')
    ax.axhline(0.5, color='gray', ls='--', alpha=0.5,
               label='Isotropic (0.5)')
    ax.axhline(1.0, color='red', ls=':', alpha=0.3,
               label='Pure angular (1.0)')
    ax.axvspan(10, 50, alpha=0.08, color='blue')
    ax.set_xscale('log')
    ax.set_xlabel('Separation (Mpc)')
    ax.set_ylabel('|z_perp| / (|z_perp| + |z_para|)')
    ax.set_title('Angular fraction of signal')
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=7)

    # Panel 4: Plate-level mean residual distribution
    ax = fig.add_subplot(gs[1, 0])
    if plate_df is not None and len(plate_df) > 5:
        ax.hist(plate_df['mean_resid'].values, bins=50,
                color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', ls='--', lw=1.5)
        ax.set_xlabel('Mean plate residual')
        ax.set_ylabel('Number of plates')
        ax.set_title(f'Plate residuals (F={F_stat:.1f}, p={p_anova:.2e})')
    else:
        ax.text(0.5, 0.5, 'No plate data available',
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Plate analysis: N/A')

    # Panel 5: Feature autocorrelation z-scores
    ax = fig.add_subplot(gs[1, 1])
    colors_feat = plt.cm.Set1(np.linspace(0, 1, len(feature_candidates)))
    for feat, color in zip(feature_candidates, colors_feat):
        z_f = feature_results[feat]['z']
        ax.plot(bc_feat, z_f, 'o-', color=color, ms=3, lw=1.5,
                label=f'{feat} ({feature_results[feat]["max_z_isf"]:.1f}σ)',
                alpha=0.8)
    # Overlay f_DM for comparison
    ax.plot(bc_feat, z_fDM_A, 'k^-', ms=5, lw=2,
            label=f'f_DM resid ({max_z_fDM:.1f}σ)', zorder=10)
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.axhline(2, color='orange', ls='--', alpha=0.3)
    ax.axvspan(10, 50, alpha=0.08, color='blue')
    ax.set_xscale('log')
    ax.set_xlabel('Comoving separation (Mpc)')
    ax.set_ylabel('z-score')
    ax.set_title('Feature-by-feature autocorrelation (KEY)')
    ax.legend(fontsize=6, loc='upper left')

    # Panel 6: Feature max |z| bar chart
    ax = fig.add_subplot(gs[1, 2])
    feat_names = feature_candidates + ['f_DM_A']
    feat_maxz = [feature_results[f]['max_z_isf'] for f in feature_candidates]
    feat_maxz.append(max_z_fDM)
    bar_colors = list(colors_feat) + ['black']
    bars = ax.barh(feat_names, feat_maxz, color=bar_colors,
                    edgecolor='black', alpha=0.7)
    ax.axvline(2, color='orange', ls='--', alpha=0.5, label='2σ')
    ax.axvline(3, color='red', ls='--', alpha=0.5, label='3σ')
    ax.set_xlabel('Max |z-score| in ISF range (10-50 Mpc)')
    ax.set_title('Which feature carries the signal?')
    ax.legend(fontsize=8)

    # Panel 7: w(r) raw values — perp vs para
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(bc_aniso, w_perp_A, 'o-', color='darkred', ms=5, lw=1.5,
            label='w(r_perp) A')
    ax.plot(bc_aniso, w_para_A, 's-', color='navy', ms=5, lw=1.5,
            label='w(r_para) A')
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.axvspan(10, 50, alpha=0.08, color='blue')
    ax.set_xscale('log')
    ax.set_xlabel('Separation (Mpc)')
    ax.set_ylabel('$w(r)$')
    ax.set_title('Raw w(r): transverse vs radial')
    ax.legend(fontsize=8)

    # Panel 8: Feature z-score shapes normalized
    ax = fig.add_subplot(gs[2, 1])
    # Show which features have similar SHAPE to f_DM signal
    z_fDM_norm = z_fDM_A / (np.nanmax(np.abs(z_fDM_A)) + 1e-10)
    ax.plot(bc_feat, z_fDM_norm, 'k-', lw=3, alpha=0.5,
            label='f_DM (shape)')
    for feat, color in zip(feature_candidates, colors_feat):
        z_f = feature_results[feat]['z']
        z_f_norm = z_f / (np.nanmax(np.abs(z_f)) + 1e-10)
        ax.plot(bc_feat, z_f_norm, '-', color=color, lw=1.5,
                label=feat, alpha=0.7)
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.axvspan(10, 50, alpha=0.08, color='blue')
    ax.set_xscale('log')
    ax.set_xlabel('Separation (Mpc)')
    ax.set_ylabel('Normalized z-score')
    ax.set_title('Shape comparison (normalized)')
    ax.legend(fontsize=6)

    # Panel 9: Summary text
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')

    # Determine key findings
    aniso_verdict = "UNKNOWN"
    max_perp = np.nanmax(np.abs(z_perp_A))
    max_para = np.nanmax(np.abs(z_para_A))
    if max_perp > 3 and max_para < 2:
        aniso_verdict = "ANGULAR-DOMINATED → systematic"
    elif max_para > 3 and max_perp < 2:
        aniso_verdict = "RADIAL-DOMINATED → redshift effect"
    elif max_perp > 2 and max_para > 2:
        aniso_verdict = "ISOTROPIC → real 3D structure"
    else:
        aniso_verdict = "WEAK SIGNAL in this z-slice"

    # Which feature dominates?
    feat_ranking = sorted(feature_candidates,
                           key=lambda f: feature_results[f]['max_z_isf'],
                           reverse=True)
    top_feat = feat_ranking[0]
    top_feat_z = feature_results[top_feat]['max_z_isf']

    summary_lines = [
        "Step 6 Diagnostic Summary",
        "─" * 30,
        "",
        "1. ANISOTROPY:",
        f"   max|z_perp| = {max_perp:.1f}",
        f"   max|z_para| = {max_para:.1f}",
        f"   → {aniso_verdict}",
        "",
        "2. PLATES:",
    ]
    if plate_df is not None and len(plate_df) > 5:
        summary_lines.extend([
            f"   F={F_stat:.1f}, p={p_anova:.2e}",
            f"   Excess var: {excess_var*100:.1f}%",
        ])
    else:
        summary_lines.append("   No plate data")

    summary_lines.extend([
        "",
        "3. FEATURE SIGNAL:",
        f"   Top: {top_feat} ({top_feat_z:.1f}σ)",
    ])
    for f in feat_ranking:
        z_val = feature_results[f]['max_z_isf']
        summary_lines.append(f"   {f:15s}: {z_val:.1f}σ")
    summary_lines.append(f"   {'f_DM resid':15s}: {max_z_fDM:.1f}σ")

    ax.text(0.05, 0.95, "\n".join(summary_lines),
            transform=ax.transAxes, fontsize=9, va='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle('Phase B Step 6: Signal Origin Diagnostics',
                 fontsize=14, fontweight='bold', y=0.99)

    out_fig = os.path.join(DATA_DIR, "phaseB_step6_diagnostics.png")
    plt.savefig(out_fig, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_fig}")

    # ── Text summary ────────────────────────────────────────────────
    lines = []
    lines.append("Phase B Step 6: Signal Origin Diagnostics Summary")
    lines.append("=" * 60)
    lines.append(f"Sample: {len(df):,} galaxies")
    lines.append(f"")
    lines.append(f"1. ANISOTROPY TEST (z-slice [{Z_SLICE_LO:.2f}, {Z_SLICE_HI:.2f}))")
    lines.append(f"   N galaxies in slice: {n_slice:,}")
    lines.append(f"   N centers: {nc:,}, N pairs: {len(pc):,}")
    lines.append(f"   Max |z_perp| (angular):  {max_perp:.2f}")
    lines.append(f"   Max |z_para| (radial):   {max_para:.2f}")
    lines.append(f"   Verdict: {aniso_verdict}")
    lines.append(f"")
    lines.append(f"   Per-bin details (Method A):")
    for i in range(N_BINS_ANISO):
        flag = " *" if (bc_aniso[i] >= 10 and bc_aniso[i] <= 50) else ""
        lines.append(f"     r={bc_aniso[i]:6.1f}: "
                      f"z_perp={z_perp_A[i]:+5.2f}, "
                      f"z_para={z_para_A[i]:+5.2f}{flag}")

    lines.append(f"")
    lines.append(f"2. PLATE-LEVEL ANALYSIS")
    if plate_df is not None and len(plate_df) > 5:
        lines.append(f"   N plates (≥20 gal): {n_plates}")
        lines.append(f"   ANOVA F: {F_stat:.2f}, p={p_anova:.4e}")
        lines.append(f"   Excess inter-plate variance A: {excess_var*100:.2f}%")
        lines.append(f"   Excess inter-plate variance B: {excess_B*100:.2f}%")
        if p_anova < 0.001:
            lines.append(f"   → SIGNIFICANT plate-level variation")
        else:
            lines.append(f"   → No significant plate-level variation")
    else:
        lines.append(f"   No plate data available")

    lines.append(f"")
    lines.append(f"3. FEATURE-BY-FEATURE AUTOCORRELATION")
    lines.append(f"   Max |z| in ISF range (10-50 Mpc):")
    for f in feat_ranking:
        z_val = feature_results[f]['max_z_isf']
        lines.append(f"     {f:15s}: {z_val:.1f}σ")
    lines.append(f"     {'f_DM resid A':15s}: {max_z_fDM:.1f}σ")
    lines.append(f"")
    if top_feat_z > max_z_fDM * 0.5:
        lines.append(f"   → Feature '{top_feat}' carries strong spatial signal")
        lines.append(f"     ({top_feat_z:.1f}σ vs f_DM {max_z_fDM:.1f}σ)")
        lines.append(f"     This feature likely drives the f_DM signal")
    else:
        lines.append(f"   → No single feature dominates the spatial signal")
        lines.append(f"     Signal may arise from combination or from")
        lines.append(f"     f_DM-specific physics")

    lines.append(f"")
    lines.append(f"OVERALL INTERPRETATION:")
    # Build verdict from all three diagnostics
    is_angular = (max_perp > 3 and max_para < 2)
    is_plate = (plate_df is not None and len(plate_df) > 5
                and p_anova < 0.001)
    feat_explains = (top_feat_z > max_z_fDM * 0.5)

    if is_angular:
        lines.append(f"  Signal is ANGULAR-DOMINATED: consistent with")
        lines.append(f"  survey systematics (seeing, dust, plate effects)")
        lines.append(f"  rather than 3D physical structure.")
    elif is_plate and feat_explains:
        lines.append(f"  Signal correlates with plate structure and")
        lines.append(f"  is carried by '{top_feat}'. Likely calibration")
        lines.append(f"  artifact rather than astrophysical signal.")
    elif feat_explains:
        lines.append(f"  Signal is primarily carried by '{top_feat}'.")
        lines.append(f"  This may reflect real astrophysical variation")
        lines.append(f"  or systematic calibration effects.")
    else:
        lines.append(f"  Signal origin unclear from these diagnostics.")
        lines.append(f"  Not obviously angular-dominated, not obviously")
        lines.append(f"  driven by a single feature. Could be genuine")
        lines.append(f"  astrophysical signal or subtle systematic.")

    summary = "\n".join(lines)
    print(f"\n{summary}")

    out_txt = os.path.join(DATA_DIR, "phaseB_step6_summary.txt")
    with open(out_txt, 'w') as f:
        f.write(summary)
    print(f"\n  Saved: {out_txt}")

    print("\n" + "=" * 65)
    print("Step 6 complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()

