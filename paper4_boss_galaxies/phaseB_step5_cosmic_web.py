#!/usr/bin/env python3
"""
phaseB_step5_cosmic_web.py
===========================
Phase B, Step 5: Cosmic web classification control test.

Distinguishing test between assembly bias and ISF:
  - Reconstruct the galaxy density field from LOWZ sample
  - Classify each galaxy as void/wall/filament/node via tidal tensor
  - Add cosmic web type to Step 2 regression
  - If Step 3 signal vanishes → assembly bias
  - If Step 3 signal persists → ISF remains viable

Pipeline:
  Stage A: Grid the galaxy density field, smooth with Gaussian kernel
  Stage B: Compute tidal tensor (Hessian of potential) at each galaxy
  Stage C: Classify via eigenvalue signs (Hahn+2007 / Forero-Romero+2009)
  Stage D: Re-run regression with web type, extract new residuals
  Stage E: Re-run spatial autocorrelation on new residuals
  Stage F: Compare with Step 3 — does signal survive?

Inputs (in DATA_DIR):
  - phaseB_residuals.csv  (from Step 2)

Outputs (in DATA_DIR):
  - phaseB_step5_webclass.csv          (catalog with web types)
  - phaseB_step5_correlation.csv       (w(r) with web-corrected residuals)
  - phaseB_step5_diagnostics.png       (diagnostic figure)
  - phaseB_step5_summary.txt           (text summary)

Usage:
  python3 -u phaseB_step5_cosmic_web.py 2>&1 | tee phaseB_step5_output.txt
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
from scipy.ndimage import gaussian_filter
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# ── Configuration ──────────────────────────────────────────────────
DATA_DIR = "/Users/chris/Documents/research_data/cosmic_structure"

# Density field gridding
GRID_CELL = 5.0       # Mpc per grid cell
SMOOTH_SCALE = 10.0   # Gaussian smoothing σ in Mpc (cosmic web scale)

# Tidal tensor eigenvalue threshold (Hahn+2007)
LAMBDA_TH = 0.0       # eigenvalue threshold for classification

# Regression
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 8,
    'min_samples_leaf': 50,
    'n_jobs': 1,
    'random_state': 42,
}

# Spatial autocorrelation (matching Step 3)
N_CENTERS = 15000
R_MAX = 200.0
N_BINS = 20
N_SHUFFLES = 200
CHUNK_SIZE = 500

SEED = 42


# ══════════════════════════════════════════════════════════════════
# Stage A: Density field reconstruction
# ══════════════════════════════════════════════════════════════════

def grid_density_field(positions, cell_size, smooth_sigma):
    """Grid galaxy positions onto a 3D density field and smooth.

    Parameters
    ----------
    positions : (N, 3) array — comoving Cartesian positions in Mpc
    cell_size : float — grid cell size in Mpc
    smooth_sigma : float — Gaussian smoothing scale in Mpc

    Returns
    -------
    delta : 3D array — overdensity field δ = (ρ - ρ̄) / ρ̄
    origin : (3,) array — grid origin (min corner) in Mpc
    cell_size : float — actual cell size used
    """
    # Grid bounds with padding
    pos_min = positions.min(axis=0) - 2 * smooth_sigma
    pos_max = positions.max(axis=0) + 2 * smooth_sigma

    # Grid dimensions
    n_cells = np.ceil((pos_max - pos_min) / cell_size).astype(int)
    origin = pos_min

    print(f"    Grid: {n_cells[0]} × {n_cells[1]} × {n_cells[2]} = "
          f"{np.prod(n_cells):,} cells")
    print(f"    Cell size: {cell_size} Mpc, extent: "
          f"{n_cells[0]*cell_size:.0f} × {n_cells[1]*cell_size:.0f} × "
          f"{n_cells[2]*cell_size:.0f} Mpc")

    # Assign galaxies to grid cells using CIC (Cloud-in-Cell)
    # Fractional cell indices
    frac_idx = (positions - origin) / cell_size

    # Integer indices (floor)
    idx0 = np.floor(frac_idx).astype(int)
    # Fractional offsets for CIC weighting
    dx = frac_idx - idx0

    # CIC: distribute each galaxy to 8 neighboring cells
    density = np.zeros(n_cells, dtype=np.float64)

    for ix in [0, 1]:
        for iy in [0, 1]:
            for iz in [0, 1]:
                # Weight for this corner
                wx = (1 - dx[:, 0]) if ix == 0 else dx[:, 0]
                wy = (1 - dx[:, 1]) if iy == 0 else dx[:, 1]
                wz = (1 - dx[:, 2]) if iz == 0 else dx[:, 2]
                w = wx * wy * wz

                # Target cell indices
                ci = idx0[:, 0] + ix
                cj = idx0[:, 1] + iy
                ck = idx0[:, 2] + iz

                # Bounds check
                valid = ((ci >= 0) & (ci < n_cells[0]) &
                         (cj >= 0) & (cj < n_cells[1]) &
                         (ck >= 0) & (ck < n_cells[2]))

                np.add.at(density, (ci[valid], cj[valid], ck[valid]), w[valid])

    print(f"    Galaxies assigned (CIC): sum={density.sum():.0f}")

    # Smooth with Gaussian
    sigma_cells = smooth_sigma / cell_size
    print(f"    Smoothing: σ = {smooth_sigma} Mpc = {sigma_cells:.1f} cells")
    density_smooth = gaussian_filter(density, sigma=sigma_cells, mode='constant')

    # Convert to overdensity
    mean_density = density_smooth[density_smooth > 0].mean()
    if mean_density > 0:
        delta = (density_smooth - mean_density) / mean_density
    else:
        delta = np.zeros_like(density_smooth)

    print(f"    δ range: [{delta.min():.2f}, {delta.max():.2f}], "
          f"mean={delta.mean():.4f}")

    return delta, origin, cell_size


# ══════════════════════════════════════════════════════════════════
# Stage B: Tidal tensor and cosmic web classification
# ══════════════════════════════════════════════════════════════════

def compute_tidal_tensor_at_galaxies(delta, origin, cell_size, positions):
    """Compute the tidal tensor T_ij = ∂²Φ/∂x_i∂x_j at each galaxy.

    The gravitational potential Φ is obtained by solving Poisson's
    equation ∇²Φ = δ in Fourier space: Φ̃(k) = -δ̃(k) / k².

    The tidal tensor is the Hessian of Φ. We compute it on the grid
    via finite differences, then interpolate to galaxy positions.

    Parameters
    ----------
    delta : 3D array — overdensity field
    origin : (3,) — grid origin in Mpc
    cell_size : float — grid cell size in Mpc
    positions : (N, 3) — galaxy positions in Mpc

    Returns
    -------
    eigenvalues : (N, 3) array — sorted eigenvalues (λ₁ ≥ λ₂ ≥ λ₃)
    """
    nx, ny, nz = delta.shape
    print(f"    Solving Poisson equation in Fourier space...")

    # FFT of density field
    delta_k = np.fft.rfftn(delta)

    # Wave numbers
    kx = np.fft.fftfreq(nx, d=cell_size) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=cell_size) * 2 * np.pi
    kz = np.fft.rfftfreq(nz, d=cell_size) * 2 * np.pi

    kx3d, ky3d, kz3d = np.meshgrid(kx, ky, kz, indexing='ij')
    k2 = kx3d**2 + ky3d**2 + kz3d**2
    k2[0, 0, 0] = 1.0  # avoid division by zero (DC mode)

    # Potential in Fourier space: Φ̃ = -δ̃ / k²
    phi_k = -delta_k / k2
    phi_k[0, 0, 0] = 0.0  # zero mean potential

    # Tidal tensor components: T_ij = ∂²Φ/∂x_i∂x_j
    # In Fourier space: T̃_ij = -k_i k_j Φ̃ = k_i k_j δ̃ / k²
    print(f"    Computing 6 tidal tensor components...")
    k_components = [kx3d, ky3d, kz3d]
    labels = ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']

    # We need T_xx, T_yy, T_zz, T_xy, T_xz, T_yz on the grid
    T_grid = {}
    ij_pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]

    for (i, j), label in zip(ij_pairs, labels):
        T_k = k_components[i] * k_components[j] * phi_k
        T_grid[label] = np.fft.irfftn(T_k, s=delta.shape)

    # Interpolate tidal tensor to galaxy positions (trilinear)
    print(f"    Interpolating to {len(positions):,} galaxy positions...")
    n_gal = len(positions)

    # Fractional grid indices
    frac_idx = (positions - origin) / cell_size
    idx0 = np.floor(frac_idx).astype(int)
    dx = frac_idx - idx0

    # Clip to grid bounds
    idx0 = np.clip(idx0, 0, np.array([nx-2, ny-2, nz-2]))

    # Vectorized trilinear interpolation for all galaxies at once
    # Pre-compute 8-corner weights: shape (n_gal,)
    wx0 = 1 - dx[:, 0]; wx1 = dx[:, 0]
    wy0 = 1 - dx[:, 1]; wy1 = dx[:, 1]
    wz0 = 1 - dx[:, 2]; wz1 = dx[:, 2]

    ix = idx0[:, 0]; iy = idx0[:, 1]; iz = idx0[:, 2]

    # For each tidal component, trilinear interpolate to all galaxies
    T_at_gal = {}  # label → (n_gal,) array
    for label in labels:
        grid = T_grid[label]
        val = (wx0 * wy0 * wz0 * grid[ix,   iy,   iz  ] +
               wx1 * wy0 * wz0 * grid[ix+1, iy,   iz  ] +
               wx0 * wy1 * wz0 * grid[ix,   iy+1, iz  ] +
               wx0 * wy0 * wz1 * grid[ix,   iy,   iz+1] +
               wx1 * wy1 * wz0 * grid[ix+1, iy+1, iz  ] +
               wx1 * wy0 * wz1 * grid[ix+1, iy,   iz+1] +
               wx0 * wy1 * wz1 * grid[ix,   iy+1, iz+1] +
               wx1 * wy1 * wz1 * grid[ix+1, iy+1, iz+1])
        T_at_gal[label] = val

    print(f"      Interpolation done, computing eigenvalues...")

    # Assemble symmetric 3x3 matrices: shape (n_gal, 3, 3)
    T_matrices = np.zeros((n_gal, 3, 3), dtype=np.float64)
    for (i, j), label in zip(ij_pairs, labels):
        T_matrices[:, i, j] = T_at_gal[label]
        T_matrices[:, j, i] = T_at_gal[label]

    # Batch eigendecomposition
    all_eigs = np.linalg.eigvalsh(T_matrices)  # (n_gal, 3), ascending
    eigenvalues = all_eigs[:, ::-1]  # descending: λ₁ ≥ λ₂ ≥ λ₃

    print(f"      {n_gal:,} galaxies — done")

    return eigenvalues


def classify_cosmic_web(eigenvalues, lambda_th=0.0):
    """Classify cosmic web type from tidal tensor eigenvalues.

    Hahn+2007 / Forero-Romero+2009 classification:
      - Node:     λ₁ > th AND λ₂ > th AND λ₃ > th  (3 axes collapsing)
      - Filament: λ₁ > th AND λ₂ > th AND λ₃ ≤ th  (2 axes collapsing)
      - Wall:     λ₁ > th AND λ₂ ≤ th AND λ₃ ≤ th  (1 axis collapsing)
      - Void:     λ₁ ≤ th AND λ₂ ≤ th AND λ₃ ≤ th  (0 axes collapsing)

    Parameters
    ----------
    eigenvalues : (N, 3) — sorted λ₁ ≥ λ₂ ≥ λ₃
    lambda_th : float — threshold

    Returns
    -------
    web_type : (N,) int — 0=void, 1=wall, 2=filament, 3=node
    web_label : (N,) str — string labels
    """
    n_above = np.sum(eigenvalues > lambda_th, axis=1)

    web_type = np.zeros(len(eigenvalues), dtype=int)
    web_type[n_above == 0] = 0  # void
    web_type[n_above == 1] = 1  # wall
    web_type[n_above == 2] = 2  # filament
    web_type[n_above == 3] = 3  # node

    label_map = {0: 'void', 1: 'wall', 2: 'filament', 3: 'node'}
    web_label = np.array([label_map[w] for w in web_type])

    return web_type, web_label


# ══════════════════════════════════════════════════════════════════
# Stage C: Spatial autocorrelation (reused from Step 3)
# ══════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("Phase B Step 5: Cosmic web classification control test")
    print("  Does cosmic web type explain the spatial signal?")
    print("=" * 65)

    rng = np.random.RandomState(SEED)

    # ── Load data ─────────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    df = pd.read_csv(os.path.join(DATA_DIR, "phaseB_residuals.csv"))
    print(f"  Loaded {len(df):,} galaxies")

    positions = df[['x', 'y', 'z_cart']].values.astype(np.float64)
    resid_A_orig = df['resid_rf_ext'].values.astype(np.float64)
    resid_B_orig = df['resid_rf_ext_B'].values.astype(np.float64)

    # ── Stage A: Density field ────────────────────────────────────
    print(f"\n[2/6] Stage A: Gridding galaxy density field...")
    print(f"  Cell size: {GRID_CELL} Mpc, smoothing: {SMOOTH_SCALE} Mpc")
    t0 = time.time()

    delta, origin, cs = grid_density_field(positions, GRID_CELL, SMOOTH_SCALE)

    print(f"  Density field computed in {time.time()-t0:.1f}s")

    # ── Stage B: Tidal tensor ─────────────────────────────────────
    print(f"\n[3/6] Stage B: Computing tidal tensor at galaxy positions...")
    t0 = time.time()

    eigenvalues = compute_tidal_tensor_at_galaxies(
        delta, origin, cs, positions)

    web_type, web_label = classify_cosmic_web(eigenvalues, LAMBDA_TH)

    elapsed = time.time() - t0
    print(f"  Tidal tensor + classification in {elapsed:.0f}s")

    # Census
    has_fDM = 'f_DM_A' in df.columns
    for wt, name in [(0, 'Void'), (1, 'Wall'), (2, 'Filament'), (3, 'Node')]:
        n = (web_type == wt).sum()
        frac = n / len(web_type) * 100
        if has_fDM and n > 0:
            mean_fDM = df['f_DM_A'].values[web_type == wt].mean()
            print(f"    {name:10s}: {n:>7,} ({frac:5.1f}%)  "
                  f"mean f_DM_A = {mean_fDM:.4f}")
        else:
            mean_resid = resid_A_orig[web_type == wt].mean() if n > 0 else 0
            print(f"    {name:10s}: {n:>7,} ({frac:5.1f}%)  "
                  f"mean resid_A = {mean_resid:.6f}")

    # Add to dataframe
    df['web_type'] = web_type
    df['web_label'] = web_label
    df['lambda1'] = eigenvalues[:, 0]
    df['lambda2'] = eigenvalues[:, 1]
    df['lambda3'] = eigenvalues[:, 2]
    df['delta_smooth'] = 0.0  # placeholder — interpolate below

    # Interpolate smoothed overdensity to galaxy positions
    frac_idx = (positions - origin) / cs
    idx0 = np.clip(np.floor(frac_idx).astype(int), 0,
                    np.array(delta.shape) - 2)
    for g in range(len(positions)):
        ix, iy, iz = idx0[g]
        df.at[g, 'delta_smooth'] = delta[ix, iy, iz]  # nearest-cell approx

    # ── Stage C: Regression with cosmic web type ──────────────────
    print(f"\n[4/6] Stage C: Re-running regression with web classification...")

    # Features: original Step 2 features + web type (one-hot) + eigenvalues
    feature_cols_basic = ['log_Mstar_A', 'veldisp', 'R50_kpc', 'gmr_rest', 'z']

    # Check which columns exist
    if 'log_rho5' in df.columns:
        feature_cols_basic.append('log_rho5')

    # Extended features with web info
    feature_cols_web = feature_cols_basic + ['lambda1', 'lambda2', 'lambda3']

    # One-hot encode web type
    for wt in [1, 2, 3]:  # void=0 is reference
        df[f'web_{wt}'] = (df['web_type'] == wt).astype(float)
        feature_cols_web.append(f'web_{wt}')

    # Method A target
    if has_fDM:
        use_residual_regression = False
    else:
        print("  NOTE: f_DM_A not in residuals file.")
        print("  Using residual regression against web features.")
        target_A_resid = resid_A_orig.copy()
        target_B_resid = resid_B_orig.copy()
        use_residual_regression = True

    if not use_residual_regression:
        # Full regression: f_DM ~ properties + web
        X_basic = df[feature_cols_basic].values
        X_web = df[feature_cols_web].values
        y_A = df['f_DM_A'].values
        y_B = df['f_DM_B'].values if 'f_DM_B' in df.columns else None

        # RF with basic features (reproduce Step 2)
        print(f"\n  RF (basic features, {len(feature_cols_basic)} cols)...")
        rf_basic = RandomForestRegressor(**RF_PARAMS)
        cv_basic = cross_val_score(rf_basic, X_basic, y_A, cv=5,
                                    scoring='r2', n_jobs=1)
        print(f"    R² (5-fold CV): {cv_basic.mean():.4f} ± {cv_basic.std():.4f}")

        # RF with web features
        print(f"  RF (web features, {len(feature_cols_web)} cols)...")
        rf_web = RandomForestRegressor(**RF_PARAMS)
        cv_web = cross_val_score(rf_web, X_web, y_A, cv=5,
                                  scoring='r2', n_jobs=1)
        print(f"    R² (5-fold CV): {cv_web.mean():.4f} ± {cv_web.std():.4f}")

        r2_gain = cv_web.mean() - cv_basic.mean()
        print(f"    R² gain from web info: {r2_gain:+.4f}")

        # Fit full model and extract residuals
        rf_web.fit(X_web, y_A)
        pred_web_A = rf_web.predict(X_web)
        resid_web_A = y_A - pred_web_A

        # Feature importances
        importances = rf_web.feature_importances_
        print(f"\n  Feature importances (RF with web):")
        for name, imp in sorted(zip(feature_cols_web, importances),
                                 key=lambda x: -x[1]):
            print(f"    {name:15s}: {imp:.4f}")

        # Method B if available
        if y_B is not None:
            rf_web_B = RandomForestRegressor(**RF_PARAMS)
            rf_web_B.fit(X_web, y_B)
            pred_web_B = rf_web_B.predict(X_web)
            resid_web_B = y_B - pred_web_B
        else:
            resid_web_B = resid_B_orig.copy()

    else:
        # Partial test: regress old residuals against web features
        web_features = df[['lambda1', 'lambda2', 'lambda3',
                           'web_1', 'web_2', 'web_3']].values

        rf_resid = RandomForestRegressor(**RF_PARAMS)
        cv_resid = cross_val_score(rf_resid, web_features, target_A_resid,
                                    cv=5, scoring='r2', n_jobs=1)
        print(f"  RF (web → old residuals): R² = {cv_resid.mean():.4f}")

        rf_resid.fit(web_features, target_A_resid)
        pred_resid = rf_resid.predict(web_features)
        resid_web_A = target_A_resid - pred_resid

        rf_resid_B = RandomForestRegressor(**RF_PARAMS)
        rf_resid_B.fit(web_features, target_B_resid)
        resid_web_B = target_B_resid - rf_resid_B.predict(web_features)

        r2_gain = cv_resid.mean()
        cv_basic = np.array([0.0])
        cv_web = cv_resid

    # Zero-mean residuals
    resid_web_A -= resid_web_A.mean()
    resid_web_B -= resid_web_B.mean()

    # Compare residual variance
    var_orig_A = np.var(resid_A_orig)
    var_web_A = np.var(resid_web_A)
    var_reduction = 1 - var_web_A / var_orig_A

    print(f"\n  Residual variance:")
    print(f"    Original (Step 2): {var_orig_A:.6f}")
    print(f"    After web:         {var_web_A:.6f}")
    print(f"    Variance reduction: {var_reduction*100:.1f}%")

    # Save web-classified catalog
    df['resid_web_A'] = resid_web_A
    df['resid_web_B'] = resid_web_B

    out_cat = os.path.join(DATA_DIR, "phaseB_step5_webclass.csv")
    # Save only key columns to avoid huge file
    save_cols = ['x', 'y', 'z_cart', 'z', 'web_type', 'web_label',
                 'lambda1', 'lambda2', 'lambda3', 'delta_smooth',
                 'resid_rf_ext', 'resid_rf_ext_B',
                 'resid_web_A', 'resid_web_B']
    for c in ['f_DM_A', 'f_DM_B']:
        if c in df.columns:
            save_cols.append(c)
    # Only save columns that exist
    save_cols = [c for c in save_cols if c in df.columns]

    df[save_cols].to_csv(out_cat, index=False)
    print(f"  Saved: {out_cat}")

    # ── Stage D: Spatial autocorrelation on new residuals ─────────
    print(f"\n[5/6] Stage D: Spatial autocorrelation of web-corrected residuals...")

    bins = np.logspace(np.log10(1.0), np.log10(R_MAX), N_BINS + 1)
    bin_centers = np.sqrt(bins[:-1] * bins[1:])

    # Subsample centers
    n_c = min(N_CENTERS, len(positions))
    center_idx = rng.choice(len(positions), size=n_c, replace=False)
    center_idx.sort()

    print(f"  Pre-computing pairs ({n_c:,} centers, r_max={R_MAX} Mpc)...")
    t0 = time.time()
    pair_c, pair_n, pair_b = precompute_pairs(
        positions, center_idx, bins, R_MAX)
    print(f"  {len(pair_c):,} pairs in {time.time()-t0:.1f}s")

    if len(pair_c) == 0:
        print("  ERROR: No pairs found. Check data.")
        sys.exit(1)

    # w(r) for web-corrected residuals
    w_web_A, npairs = compute_w(resid_web_A, pair_c, pair_n, pair_b, N_BINS)
    w_web_B, _ = compute_w(resid_web_B, pair_c, pair_n, pair_b, N_BINS)

    # w(r) for original residuals (same pairs for fair comparison)
    resid_A_zm = resid_A_orig - resid_A_orig.mean()
    resid_B_zm = resid_B_orig - resid_B_orig.mean()
    w_orig_A, _ = compute_w(resid_A_zm, pair_c, pair_n, pair_b, N_BINS)
    w_orig_B, _ = compute_w(resid_B_zm, pair_c, pair_n, pair_b, N_BINS)

    # Null distribution (shuffled web-corrected residuals)
    print(f"  Running {N_SHUFFLES} null shuffles...")
    t0 = time.time()
    null_web_A = np.zeros((N_SHUFFLES, N_BINS))
    null_web_B = np.zeros((N_SHUFFLES, N_BINS))
    null_orig_A = np.zeros((N_SHUFFLES, N_BINS))

    n_gal = len(positions)
    for s in range(N_SHUFFLES):
        perm = rng.permutation(n_gal)
        null_web_A[s], _ = compute_w(
            resid_web_A[perm], pair_c, pair_n, pair_b, N_BINS)
        null_web_B[s], _ = compute_w(
            resid_web_B[perm], pair_c, pair_n, pair_b, N_BINS)
        null_orig_A[s], _ = compute_w(
            resid_A_zm[perm], pair_c, pair_n, pair_b, N_BINS)

        if (s + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (s + 1) * (N_SHUFFLES - s - 1)
            print(f"\r    Shuffle {s+1}/{N_SHUFFLES} "
                  f"({elapsed:.0f}s, ~{eta:.0f}s remaining)",
                  end='', flush=True)

    print(f"\n    Done in {time.time()-t0:.0f}s")

    # z-scores
    def zscore(data, null):
        m = null.mean(axis=0)
        s = null.std(axis=0)
        return np.where(s > 0, (data - m) / s, 0)

    z_web_A = zscore(w_web_A, null_web_A)
    z_web_B = zscore(w_web_B, null_web_B)
    z_orig_A = zscore(w_orig_A, null_orig_A)

    # ISF range
    isf_mask = (bin_centers >= 10) & (bin_centers <= 50)

    print(f"\n  Results (web-corrected vs original):")
    print(f"  {'r_cen':>7s}  {'w_orig_A':>10s} {'z_orig':>6s}  "
          f"{'w_web_A':>10s} {'z_web':>6s}  "
          f"{'w_web_B':>10s} {'z_wB':>6s}  {'N_pairs':>10s}")
    print(f"  {'-'*7:>7s}  {'-'*10:>10s} {'-'*6:>6s}  "
          f"{'-'*10:>10s} {'-'*6:>6s}  "
          f"{'-'*10:>10s} {'-'*6:>6s}  {'-'*10:>10s}")

    for i in range(N_BINS):
        flag = " *" if isf_mask[i] else "  "
        print(f"  {bin_centers[i]:7.1f}  "
              f"{w_orig_A[i]:10.6f} {z_orig_A[i]:6.2f}  "
              f"{w_web_A[i]:10.6f} {z_web_A[i]:6.2f}  "
              f"{w_web_B[i]:10.6f} {z_web_B[i]:6.2f}  "
              f"{npairs[i]:10,}{flag}")

    # Signal reduction
    if isf_mask.sum() > 0:
        max_z_orig = np.nanmax(np.abs(z_orig_A[isf_mask]))
        max_z_web_A = np.nanmax(np.abs(z_web_A[isf_mask]))
        max_z_web_B = np.nanmax(np.abs(z_web_B[isf_mask]))

        print(f"\n  ISF range (10-50 Mpc):")
        print(f"    Original max |z|: {max_z_orig:.2f}")
        print(f"    Web-corrected max |z| A: {max_z_web_A:.2f}")
        print(f"    Web-corrected max |z| B: {max_z_web_B:.2f}")
        print(f"    Signal reduction: {(1 - max_z_web_A/max_z_orig)*100:.0f}%"
              if max_z_orig > 0 else "")

    # Save correlation results
    corr_df = pd.DataFrame({
        'r_center': bin_centers,
        'r_min': bins[:-1],
        'r_max': bins[1:],
        'w_orig_A': w_orig_A,
        'z_orig_A': z_orig_A,
        'w_web_A': w_web_A,
        'z_web_A': z_web_A,
        'w_web_B': w_web_B,
        'z_web_B': z_web_B,
        'npairs': npairs,
    })
    out_corr = os.path.join(DATA_DIR, "phaseB_step5_correlation.csv")
    corr_df.to_csv(out_corr, index=False)
    print(f"  Saved: {out_corr}")

    # ── Stage E: Diagnostic figure ──────────────────────────────────
    print(f"\n[6/6] Stage E: Generating diagnostic figure...")

    fig = plt.figure(figsize=(18, 15))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Panel 1: Cosmic web census
    ax = fig.add_subplot(gs[0, 0])
    labels_cw = ['Void', 'Wall', 'Filament', 'Node']
    counts_cw = [(web_type == i).sum() for i in range(4)]
    colors_cw = ['#2166ac', '#92c5de', '#f4a582', '#b2182b']
    bars = ax.bar(labels_cw, counts_cw, color=colors_cw, edgecolor='black')
    for bar, ct in zip(bars, counts_cw):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f'{ct:,}\n({ct/len(web_type)*100:.0f}%)',
                ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Number of galaxies')
    ax.set_title(f'Cosmic web classification (σ={SMOOTH_SCALE} Mpc)')

    # Panel 2: f_DM by web type (if available)
    ax = fig.add_subplot(gs[0, 1])
    if has_fDM:
        bp_data = [df.loc[df['web_type'] == i, 'f_DM_A'].values
                    for i in range(4)]
        bp = ax.boxplot(bp_data, labels=labels_cw, patch_artist=True,
                        showfliers=False, widths=0.6)
        for patch, color in zip(bp['boxes'], colors_cw):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel('$f_{DM}$ (Method A)')
        ax.set_title('Dark matter fraction by web type')
    else:
        # Show original residuals by web type
        bp_data = [resid_A_orig[web_type == i] for i in range(4)]
        bp = ax.boxplot(bp_data, labels=labels_cw, patch_artist=True,
                        showfliers=False, widths=0.6)
        for patch, color in zip(bp['boxes'], colors_cw):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel('Step 2 residual (Method A)')
        ax.set_title('Residuals by web type')

    # Panel 3: Eigenvalue distributions
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(eigenvalues[:, 0], bins=80, alpha=0.6, color='red',
            density=True, label='$\\lambda_1$')
    ax.hist(eigenvalues[:, 1], bins=80, alpha=0.6, color='green',
            density=True, label='$\\lambda_2$')
    ax.hist(eigenvalues[:, 2], bins=80, alpha=0.6, color='blue',
            density=True, label='$\\lambda_3$')
    ax.axvline(LAMBDA_TH, color='black', ls='--', label=f'threshold={LAMBDA_TH}')
    ax.set_xlabel('Eigenvalue')
    ax.set_ylabel('Density')
    ax.set_title('Tidal tensor eigenvalue distributions')
    ax.legend(fontsize=8)
    ax.set_xlim(-3, 5)

    # Panel 4: THE KEY PLOT — z-scores original vs web-corrected
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(bin_centers, z_orig_A, 'o--', color='gray', ms=4, lw=1,
            alpha=0.7, label='Original (Step 2)')
    ax.plot(bin_centers, z_web_A, 'o-', color='darkred', ms=5, lw=2,
            label='Web-corrected (A)')
    ax.plot(bin_centers, z_web_B, 's-', color='navy', ms=5, lw=2,
            label='Web-corrected (B)')
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.axhline(2, color='orange', ls='--', alpha=0.3)
    ax.axhline(-2, color='orange', ls='--', alpha=0.3)
    ax.axhline(3, color='red', ls='--', alpha=0.3)
    ax.axhline(-3, color='red', ls='--', alpha=0.3)
    ax.axvspan(10, 50, alpha=0.08, color='blue', label='ISF range')
    ax.set_xscale('log')
    ax.set_xlabel('Comoving separation (Mpc)')
    ax.set_ylabel('z-score')
    ax.set_title('Signal before/after web correction (KEY)')
    ax.legend(fontsize=7)

    # Panel 5: w(r) original vs web-corrected
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(bin_centers, w_orig_A, 'o--', color='gray', ms=4, lw=1,
            alpha=0.7, label='Original')
    ax.plot(bin_centers, w_web_A, 'o-', color='darkred', ms=5, lw=2,
            label='Web-corrected (A)')
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.axvspan(10, 50, alpha=0.08, color='blue')
    ax.set_xscale('log')
    ax.set_xlabel('Comoving separation (Mpc)')
    ax.set_ylabel('$w(r)$')
    ax.set_title('Marked correlation: original vs corrected')
    ax.legend(fontsize=8)

    # Panel 6: Variance reduction by web features
    ax = fig.add_subplot(gs[1, 2])
    bar_labels = ['Step 2\nresidual', 'After\nweb corr.']
    bar_vals = [var_orig_A, var_web_A]
    bars = ax.bar(bar_labels, bar_vals, color=['gray', 'steelblue'],
                   edgecolor='black', width=0.5)
    ax.set_ylabel('Residual variance')
    ax.set_title(f'Variance reduction: {var_reduction*100:.1f}%')
    for bar, val in zip(bars, bar_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.6f}', ha='center', va='bottom', fontsize=9)

    # Panel 7: Residual by web type (after correction — should be flat)
    ax = fig.add_subplot(gs[2, 0])
    bp_data_web = [resid_web_A[web_type == i] for i in range(4)]
    bp = ax.boxplot(bp_data_web, labels=labels_cw, patch_artist=True,
                    showfliers=False, widths=0.6)
    for patch, color in zip(bp['boxes'], colors_cw):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.set_ylabel('Web-corrected residual')
    ax.set_title('Residuals by web type (post-correction)')

    # Panel 8: Signal retention ratio (web / original z-scores)
    ax = fig.add_subplot(gs[2, 1])
    ratio = np.where(np.abs(z_orig_A) > 0.5,
                      z_web_A / z_orig_A, np.nan)
    valid_ratio = ~np.isnan(ratio)
    if valid_ratio.sum() > 0:
        ax.plot(bin_centers[valid_ratio], ratio[valid_ratio],
                'o-', color='darkred', ms=5, lw=1.5)
    ax.axhline(1, color='gray', ls='--', alpha=0.5, label='No change')
    ax.axhline(0, color='gray', ls=':', alpha=0.5, label='Signal killed')
    ax.axvspan(10, 50, alpha=0.08, color='blue')
    ax.set_xscale('log')
    ax.set_xlabel('Comoving separation (Mpc)')
    ax.set_ylabel('z-score ratio (web / orig)')
    ax.set_title('Signal retention after web correction')
    ax.set_ylim(-0.5, 2.0)
    ax.legend(fontsize=8)

    # Panel 9: Summary verdict
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')

    # Determine verdict
    if isf_mask.sum() > 0:
        signal_killed = max_z_web_A < 2 and max_z_web_B < 2
        signal_reduced = max_z_web_A < max_z_orig * 0.5
        signal_survives = max_z_web_A > 3

    verdict_lines = [
        "Step 5 Summary",
        "─" * 30,
        f"N galaxies: {len(df):,}",
        f"Smoothing:  σ = {SMOOTH_SCALE} Mpc",
        f"Threshold:  λ_th = {LAMBDA_TH}",
        "",
        f"Web census:",
        f"  Void:     {counts_cw[0]:>6,} ({counts_cw[0]/len(df)*100:.0f}%)",
        f"  Wall:     {counts_cw[1]:>6,} ({counts_cw[1]/len(df)*100:.0f}%)",
        f"  Filament: {counts_cw[2]:>6,} ({counts_cw[2]/len(df)*100:.0f}%)",
        f"  Node:     {counts_cw[3]:>6,} ({counts_cw[3]/len(df)*100:.0f}%)",
        "",
        f"Var. reduction: {var_reduction*100:.1f}%",
        "",
    ]

    if isf_mask.sum() > 0:
        verdict_lines.append(f"ISF range max |z|:")
        verdict_lines.append(f"  Original:  {max_z_orig:.1f}σ")
        verdict_lines.append(f"  Web-corr.: {max_z_web_A:.1f}σ (A)")
        verdict_lines.append(f"             {max_z_web_B:.1f}σ (B)")
        verdict_lines.append("")

        if signal_killed:
            verdict_lines.append("VERDICT: Assembly bias")
            verdict_lines.append("Web type explains signal.")
            verdict_lines.append("ISF hypothesis: REJECTED")
        elif signal_survives:
            verdict_lines.append("VERDICT: Signal persists")
            verdict_lines.append("Not assembly bias alone.")
            verdict_lines.append("ISF hypothesis: VIABLE")
        else:
            verdict_lines.append("VERDICT: Partial reduction")
            verdict_lines.append("Assembly bias contributes")
            verdict_lines.append("but doesn't fully explain.")

    ax.text(0.05, 0.95, "\n".join(verdict_lines),
            transform=ax.transAxes, fontsize=9, va='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle('Phase B Step 5: Cosmic Web Classification Control',
                 fontsize=14, fontweight='bold', y=0.99)

    out_fig = os.path.join(DATA_DIR, "phaseB_step5_diagnostics.png")
    plt.savefig(out_fig, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_fig}")

    # ── Stage F: Summary ─────────────────────────────────────────────
    lines = []
    lines.append("Phase B Step 5: Cosmic Web Classification Control Summary")
    lines.append("=" * 60)
    lines.append(f"Sample: {len(df):,} galaxies")
    lines.append(f"Density field: cell={GRID_CELL} Mpc, σ_smooth={SMOOTH_SCALE} Mpc")
    lines.append(f"Classification threshold: λ_th = {LAMBDA_TH}")
    lines.append(f"")
    lines.append(f"Cosmic web census:")
    for wt, name in [(0, 'Void'), (1, 'Wall'), (2, 'Filament'), (3, 'Node')]:
        n = counts_cw[wt]
        lines.append(f"  {name:10s}: {n:>7,} ({n/len(df)*100:.1f}%)")

    lines.append(f"")
    if not use_residual_regression:
        lines.append(f"Regression R² (5-fold CV):")
        lines.append(f"  Basic (Step 2 features):  {cv_basic.mean():.4f}")
        lines.append(f"  With web classification:  {cv_web.mean():.4f}")
        lines.append(f"  R² gain from web info:    {r2_gain:+.4f}")
    else:
        lines.append(f"Web features → old residuals R²: {cv_resid.mean():.4f}")

    lines.append(f"")
    lines.append(f"Residual variance:")
    lines.append(f"  Original (Step 2):  {var_orig_A:.6f}")
    lines.append(f"  After web correct.: {var_web_A:.6f}")
    lines.append(f"  Variance reduction: {var_reduction*100:.1f}%")

    lines.append(f"")
    lines.append(f"Spatial autocorrelation (ISF range 10-50 Mpc):")
    if isf_mask.sum() > 0:
        lines.append(f"  Original max |z|:      {max_z_orig:.2f}")
        lines.append(f"  Web-corrected max |z|: {max_z_web_A:.2f} (A), "
                      f"{max_z_web_B:.2f} (B)")

        # Per-bin detail
        lines.append(f"")
        lines.append(f"  Per-bin (ISF range):")
        for i in np.where(isf_mask)[0]:
            lines.append(f"    r={bin_centers[i]:6.1f} Mpc: "
                          f"z_orig={z_orig_A[i]:+5.2f} → "
                          f"z_web_A={z_web_A[i]:+5.2f}, "
                          f"z_web_B={z_web_B[i]:+5.2f}")

    lines.append(f"")
    lines.append(f"Interpretation:")
    if isf_mask.sum() > 0:
        if signal_killed:
            lines.append(f"  The spatial autocorrelation signal VANISHES after")
            lines.append(f"  including cosmic web classification in the regression.")
            lines.append(f"  This confirms the signal was ASSEMBLY BIAS:")
            lines.append(f"  galaxies in different large-scale environments")
            lines.append(f"  (void/wall/filament/node) have systematically")
            lines.append(f"  different f_DM at fixed stellar mass, σ_v, etc.")
            lines.append(f"")
            lines.append(f"  CONCLUSION: Clean null result for ISF hypothesis.")
            lines.append(f"  No evidence for invisible branch structure.")
        elif signal_survives:
            lines.append(f"  The spatial signal PERSISTS after cosmic web")
            lines.append(f"  correction ({max_z_web_A:.1f}σ vs {max_z_orig:.1f}σ "
                          f"original).")
            lines.append(f"  Assembly bias alone cannot explain the signal.")
            lines.append(f"  This is consistent with (but does not prove)")
            lines.append(f"  the ISF hypothesis — some spatial structure in")
            lines.append(f"  f_DM residuals is not captured by observable")
            lines.append(f"  large-scale environment classification.")
        else:
            lines.append(f"  The signal is PARTIALLY REDUCED by cosmic web")
            lines.append(f"  correction. Assembly bias contributes to the")
            lines.append(f"  signal but does not fully explain it.")
            lines.append(f"  The remaining signal could reflect:")
            lines.append(f"  (a) ISF structure not in the observable web")
            lines.append(f"  (b) Higher-order environment effects not captured")
            lines.append(f"      by the 4-class tidal tensor scheme")
            lines.append(f"  (c) Residual systematic effects")

    summary = "\n".join(lines)
    print(f"\n{summary}")

    out_txt = os.path.join(DATA_DIR, "phaseB_step5_summary.txt")
    with open(out_txt, 'w') as f:
        f.write(summary)
    print(f"\n  Saved: {out_txt}")

    print("\n" + "=" * 65)
    print("Step 5 complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()
