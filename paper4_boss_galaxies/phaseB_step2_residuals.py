#!/usr/bin/env python3
"""
phaseB_step2_residuals.py
=========================
Phase B, Step 2: Regress f_DM against galaxy properties and local environment,
extract residuals for spatial autocorrelation analysis.

The ISF hypothesis predicts that after removing dependence on:
  - stellar mass, velocity dispersion, size, color, redshift (galaxy properties)
  - local galaxy density (environment)
the f_DM residuals should show spatial autocorrelation at cosmic web scales
(~10-50 Mpc), reflecting the superposition of independent large-scale structures.

Pipeline:
  Stage A: Galaxy-property regression (OLS + Random Forest)
  Stage B: Local environment density (5th nearest neighbor)
  Stage C: Extended regression including environment
  Stage D: Residual diagnostics and output

Inputs (in DATA_DIR):
  - phaseB_merged.csv  (from Step 1)

Outputs (in DATA_DIR):
  - phaseB_residuals.csv           (catalog with residuals attached)
  - phaseB_step2_diagnostics.png   (9-panel diagnostic figure)
  - phaseB_step2_summary.txt       (text summary)

Usage:
  python3 phaseB_step2_residuals.py
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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ── Configuration ──────────────────────────────────────────────────
DATA_DIR = "/Users/chris/Documents/research_data/cosmic_structure"

# Environment: k-th nearest neighbor for density estimation
K_NEIGHBORS = 5

# Features for regression
FEATURES_BASIC = ['log_Mstar', 'veldisp', 'R50_kpc', 'gmr_rest', 'z']
FEATURES_EXTENDED = FEATURES_BASIC + ['log_rho5']

# Random forest hyperparameters (kept modest — this is detrending, not prediction)
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 8,
    'min_samples_leaf': 50,
    'n_jobs': 1,
    'random_state': 42,
}


# ── Main ──────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("Phase B Step 2: Regression & residuals for spatial analysis")
    print("=" * 65)

    # ── Load ──────────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    infile = os.path.join(DATA_DIR, "phaseB_merged.csv")
    if not os.path.exists(infile):
        print(f"  ERROR: {infile} not found. Run Step 1 first.")
        sys.exit(1)

    df = pd.read_csv(infile)
    print(f"  Loaded {len(df):,} galaxies")

    # Verify required columns
    required = ['fDM', 'fDM_B', 'log_Mstar', 'log_Mstar_A', 'log_Mstar_B',
                'log_Mdyn', 'veldisp', 'R50_kpc', 'gmr_rest', 'z',
                'x', 'y', 'z_cart', 'ra', 'dec', 'd_comoving']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  ERROR: Missing columns: {missing}")
        sys.exit(1)

    # Drop any rows with NaN in key columns
    n_before = len(df)
    df = df.dropna(subset=FEATURES_BASIC + ['fDM', 'x', 'y', 'z_cart']).copy()
    df.reset_index(drop=True, inplace=True)
    if len(df) < n_before:
        print(f"  Dropped {n_before - len(df)} rows with NaN values")
    print(f"  Working sample: {len(df):,} galaxies")

    # ══════════════════════════════════════════════════════════════
    # Stage A: Galaxy-property regression
    # ══════════════════════════════════════════════════════════════
    print("\n[2/5] Stage A: Galaxy-property regression...")

    X_basic = df[FEATURES_BASIC].values
    y = df['fDM'].values

    # Standardize features for OLS interpretability
    scaler_basic = StandardScaler()
    X_basic_scaled = scaler_basic.fit_transform(X_basic)

    # ── OLS regression ────────────────────────────────────────────
    ols_basic = LinearRegression()
    ols_basic.fit(X_basic_scaled, y)
    y_pred_ols = ols_basic.predict(X_basic_scaled)
    resid_ols = y - y_pred_ols

    r2_ols = 1 - np.var(resid_ols) / np.var(y)

    print(f"\n  OLS (galaxy properties only):")
    print(f"    R² = {r2_ols:.4f}")
    print(f"    Residual std = {resid_ols.std():.4f} (original std = {y.std():.4f})")
    print(f"    Coefficients (standardized):")
    for feat, coef in zip(FEATURES_BASIC, ols_basic.coef_):
        print(f"      {feat:>12s}: {coef:+.4f}")

    # ── Random Forest (captures nonlinearities) ───────────────────
    print(f"\n  Random Forest (galaxy properties only):")
    rf_basic = RandomForestRegressor(**RF_PARAMS)
    rf_basic.fit(X_basic, y)  # RF doesn't need scaling
    y_pred_rf = rf_basic.predict(X_basic)
    resid_rf_basic = y - y_pred_rf

    r2_rf = 1 - np.var(resid_rf_basic) / np.var(y)

    # Cross-validated R² to check for overfitting
    cv_scores = cross_val_score(
        RandomForestRegressor(**RF_PARAMS), X_basic, y,
        cv=5, scoring='r2', n_jobs=1
    )

    print(f"    R² (train) = {r2_rf:.4f}")
    print(f"    R² (5-fold CV) = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"    Residual std = {resid_rf_basic.std():.4f}")
    print(f"    Feature importances:")
    for feat, imp in sorted(zip(FEATURES_BASIC, rf_basic.feature_importances_),
                             key=lambda x: -x[1]):
        print(f"      {feat:>12s}: {imp:.4f}")

    # ══════════════════════════════════════════════════════════════
    # Stage B: Local environment density
    # ══════════════════════════════════════════════════════════════
    print(f"\n[3/5] Stage B: Local environment density ({K_NEIGHBORS}th nearest neighbor)...")
    t0 = time.time()

    coords = df[['x', 'y', 'z_cart']].values  # comoving Mpc

    # Build KD-tree
    tree = cKDTree(coords)

    # Query k+1 neighbors (first is self)
    dists, _ = tree.query(coords, k=K_NEIGHBORS + 1, workers=1)

    # k-th neighbor distance (skip self at index 0)
    d_k = dists[:, K_NEIGHBORS]  # distance to k-th neighbor in Mpc

    # Local density: rho_k ~ k / (4/3 * pi * d_k^3)
    rho_k = K_NEIGHBORS / (4/3 * np.pi * d_k**3)
    df['d5'] = d_k
    df['rho5'] = rho_k
    df['log_rho5'] = np.log10(rho_k)

    elapsed = time.time() - t0
    print(f"  KD-tree + query: {elapsed:.1f}s")
    print(f"  5th-neighbor distance: median={np.median(d_k):.1f} Mpc, "
          f"mean={np.mean(d_k):.1f} Mpc")
    print(f"  log(rho5): median={df['log_rho5'].median():.2f}, "
          f"std={df['log_rho5'].std():.2f}")

    # Correlation of environment with f_DM
    rho_fdm_corr = np.corrcoef(df['log_rho5'], df['fDM'])[0, 1]
    print(f"  Pearson corr(log_rho5, f_DM) = {rho_fdm_corr:.4f}")

    # ══════════════════════════════════════════════════════════════
    # Stage C: Extended regression (properties + environment)
    # ══════════════════════════════════════════════════════════════
    print(f"\n[4/5] Stage C: Extended regression (+ environment)...")

    X_ext = df[FEATURES_EXTENDED].values
    scaler_ext = StandardScaler()
    X_ext_scaled = scaler_ext.fit_transform(X_ext)

    # ── OLS extended ──────────────────────────────────────────────
    ols_ext = LinearRegression()
    ols_ext.fit(X_ext_scaled, y)
    y_pred_ols_ext = ols_ext.predict(X_ext_scaled)
    resid_ols_ext = y - y_pred_ols_ext

    r2_ols_ext = 1 - np.var(resid_ols_ext) / np.var(y)

    print(f"\n  OLS (properties + environment):")
    print(f"    R² = {r2_ols_ext:.4f} (was {r2_ols:.4f})")
    print(f"    Residual std = {resid_ols_ext.std():.4f}")
    print(f"    Coefficients (standardized):")
    for feat, coef in zip(FEATURES_EXTENDED, ols_ext.coef_):
        print(f"      {feat:>12s}: {coef:+.4f}")

    # ── Random Forest extended ────────────────────────────────────
    print(f"\n  Random Forest (properties + environment):")
    rf_ext = RandomForestRegressor(**RF_PARAMS)
    rf_ext.fit(X_ext, y)
    y_pred_rf_ext = rf_ext.predict(X_ext)
    resid_rf_ext = y - y_pred_rf_ext

    r2_rf_ext = 1 - np.var(resid_rf_ext) / np.var(y)

    cv_scores_ext = cross_val_score(
        RandomForestRegressor(**RF_PARAMS), X_ext, y,
        cv=5, scoring='r2', n_jobs=1
    )

    print(f"    R² (train) = {r2_rf_ext:.4f} (was {r2_rf:.4f})")
    print(f"    R² (5-fold CV) = {cv_scores_ext.mean():.4f} ± {cv_scores_ext.std():.4f}")
    print(f"    Residual std = {resid_rf_ext.std():.4f}")
    print(f"    Feature importances:")
    for feat, imp in sorted(zip(FEATURES_EXTENDED, rf_ext.feature_importances_),
                             key=lambda x: -x[1]):
        print(f"      {feat:>12s}: {imp:.4f}")

    # ── Store residuals ───────────────────────────────────────────
    # Primary residuals: RF extended (most conservative — removes the most variance)
    # Also store OLS and basic-only for robustness checks
    df['resid_ols_basic'] = resid_ols
    df['resid_rf_basic'] = resid_rf_basic
    df['resid_ols_ext'] = resid_ols_ext
    df['resid_rf_ext'] = resid_rf_ext  # ← primary for Step 3

    # Also compute residuals using Method B stellar mass
    y_B = df['fDM_B'].values
    rf_ext_B = RandomForestRegressor(**RF_PARAMS)
    features_B = ['log_Mstar_B', 'veldisp', 'R50_kpc', 'gmr_rest', 'z', 'log_rho5']
    X_ext_B = df[features_B].values
    rf_ext_B.fit(X_ext_B, y_B)
    df['resid_rf_ext_B'] = y_B - rf_ext_B.predict(X_ext_B)

    print(f"\n  Method B residuals (constant M/L_i):")
    print(f"    Residual std = {df['resid_rf_ext_B'].std():.4f}")
    corr_AB = np.corrcoef(df['resid_rf_ext'], df['resid_rf_ext_B'])[0, 1]
    print(f"    Corr(resid_A, resid_B) = {corr_AB:.4f}")

    # ══════════════════════════════════════════════════════════════
    # Stage D: Diagnostics
    # ══════════════════════════════════════════════════════════════
    print(f"\n[5/5] Diagnostics & output...")

    # Save catalog
    out_csv = os.path.join(DATA_DIR, "phaseB_residuals.csv")
    df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv} ({len(df):,} rows, "
          f"{os.path.getsize(out_csv)/1e6:.1f} MB)")

    # ── Diagnostic figure ─────────────────────────────────────────
    fig = plt.figure(figsize=(18, 15))
    gs = GridSpec(3, 3, figure=fig, hspace=0.38, wspace=0.35)

    # Panel 1: f_DM vs stellar mass with OLS fit
    ax = fig.add_subplot(gs[0, 0])
    ax.hexbin(df['log_Mstar'], df['fDM'], gridsize=80, cmap='inferno',
              mincnt=1, extent=[10.2, 12.2, -0.2, 1.0])
    # Show OLS trend line
    x_grid = np.linspace(10.2, 12.2, 50)
    for xg in x_grid:
        pass  # complex to reconstruct; skip trend line
    ax.set_xlabel('$\\log M_*$')
    ax.set_ylabel('$f_{DM}$')
    ax.set_title(f'$f_{{DM}}$ vs $M_*$ (OLS $R^2$={r2_ols:.3f})')

    # Panel 2: f_DM vs environment density
    ax = fig.add_subplot(gs[0, 1])
    ax.hexbin(df['log_rho5'], df['fDM'], gridsize=80, cmap='inferno', mincnt=1)
    ax.set_xlabel('$\\log_{10}(\\rho_5)$ [Mpc$^{-3}$]')
    ax.set_ylabel('$f_{DM}$')
    ax.set_title(f'$f_{{DM}}$ vs environment ($r$={rho_fdm_corr:.3f})')

    # Panel 3: Environment density distribution
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(df['log_rho5'], bins=80, color='mediumpurple',
            edgecolor='indigo', alpha=0.7)
    ax.set_xlabel('$\\log_{10}(\\rho_5)$ [Mpc$^{-3}$]')
    ax.set_ylabel('Count')
    ax.set_title(f'Environment density (5NN, med $d_5$={np.median(d_k):.0f} Mpc)')
    ax.axvline(df['log_rho5'].median(), color='red', ls='--',
               label=f'med={df["log_rho5"].median():.2f}')
    ax.legend(fontsize=8)

    # Panel 4: OLS residual distribution (basic)
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(resid_ols, bins=100, color='steelblue', edgecolor='navy',
            alpha=0.7, density=True, label='Basic')
    ax.hist(resid_ols_ext, bins=100, color='coral', edgecolor='darkred',
            alpha=0.5, density=True, label='+ Environ')
    ax.set_xlabel('OLS residual')
    ax.set_ylabel('Density')
    ax.set_title('OLS residuals: basic vs extended')
    ax.legend(fontsize=8)

    # Panel 5: RF residual distribution
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(resid_rf_basic, bins=100, color='steelblue', edgecolor='navy',
            alpha=0.7, density=True, label=f'Basic (σ={resid_rf_basic.std():.3f})')
    ax.hist(resid_rf_ext, bins=100, color='coral', edgecolor='darkred',
            alpha=0.5, density=True, label=f'+ Environ (σ={resid_rf_ext.std():.3f})')
    ax.set_xlabel('RF residual')
    ax.set_ylabel('Density')
    ax.set_title('RF residuals: basic vs extended')
    ax.legend(fontsize=8)

    # Panel 6: Residuals vs redshift (systematics check)
    ax = fig.add_subplot(gs[1, 2])
    # Running median of residuals in redshift bins
    z_bins = np.linspace(0.1, 0.48, 30)
    z_mids = 0.5 * (z_bins[:-1] + z_bins[1:])
    resid_meds = []
    resid_mads = []
    for i in range(len(z_bins) - 1):
        mask = (df['z'] >= z_bins[i]) & (df['z'] < z_bins[i+1])
        if mask.sum() > 50:
            r = df.loc[mask, 'resid_rf_ext']
            resid_meds.append(r.median())
            resid_mads.append(1.4826 * np.median(np.abs(r - r.median())))
        else:
            resid_meds.append(np.nan)
            resid_mads.append(np.nan)
    resid_meds = np.array(resid_meds)
    resid_mads = np.array(resid_mads)
    ax.fill_between(z_mids, resid_meds - resid_mads, resid_meds + resid_mads,
                    color='coral', alpha=0.3)
    ax.plot(z_mids, resid_meds, 'o-', color='darkred', ms=3, lw=1.5)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Redshift')
    ax.set_ylabel('RF residual (median ± MAD)')
    ax.set_title('Residuals vs redshift (flat = good)')
    ax.set_ylim(-0.05, 0.05)

    # Panel 7: Residuals vs stellar mass
    ax = fig.add_subplot(gs[2, 0])
    m_bins = np.linspace(10.2, 12.0, 25)
    m_mids = 0.5 * (m_bins[:-1] + m_bins[1:])
    rm = []
    for i in range(len(m_bins) - 1):
        mask = (df['log_Mstar'] >= m_bins[i]) & (df['log_Mstar'] < m_bins[i+1])
        if mask.sum() > 50:
            rm.append(df.loc[mask, 'resid_rf_ext'].median())
        else:
            rm.append(np.nan)
    ax.plot(m_mids, rm, 'o-', color='darkred', ms=3, lw=1.5)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('$\\log M_*$')
    ax.set_ylabel('RF residual (median)')
    ax.set_title('Residuals vs $M_*$ (flat = good)')
    ax.set_ylim(-0.03, 0.03)

    # Panel 8: Residuals vs environment
    ax = fig.add_subplot(gs[2, 1])
    rho_bins = np.linspace(df['log_rho5'].quantile(0.02),
                            df['log_rho5'].quantile(0.98), 25)
    rho_mids = 0.5 * (rho_bins[:-1] + rho_bins[1:])
    rr = []
    for i in range(len(rho_bins) - 1):
        mask = (df['log_rho5'] >= rho_bins[i]) & (df['log_rho5'] < rho_bins[i+1])
        if mask.sum() > 50:
            rr.append(df.loc[mask, 'resid_rf_ext'].median())
        else:
            rr.append(np.nan)
    ax.plot(rho_mids, rr, 'o-', color='darkred', ms=3, lw=1.5)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('$\\log_{10}(\\rho_5)$')
    ax.set_ylabel('RF residual (median)')
    ax.set_title('Residuals vs environment (flat = good)')
    ax.set_ylim(-0.03, 0.03)

    # Panel 9: Method A vs B residual comparison
    ax = fig.add_subplot(gs[2, 2])
    # Subsample for speed
    idx = np.random.RandomState(42).choice(len(df), min(20000, len(df)), replace=False)
    ax.scatter(df['resid_rf_ext'].values[idx], df['resid_rf_ext_B'].values[idx],
               s=1, alpha=0.15, color='navy')
    ax.plot([-0.5, 0.5], [-0.5, 0.5], 'r--', lw=1, alpha=0.5)
    ax.set_xlabel('Residual (Method A: color-mass)')
    ax.set_ylabel('Residual (Method B: const $M/L_i$)')
    ax.set_title(f'A vs B residuals ($r$={corr_AB:.3f})')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_aspect('equal')

    fig.suptitle('Phase B Step 2: Regression Diagnostics',
                 fontsize=14, fontweight='bold', y=0.99)

    out_fig = os.path.join(DATA_DIR, "phaseB_step2_diagnostics.png")
    plt.savefig(out_fig, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_fig}")

    # ── Summary ───────────────────────────────────────────────────
    lines = []
    lines.append("Phase B Step 2: Summary")
    lines.append("=" * 55)
    lines.append(f"Sample: {len(df):,} galaxies")
    lines.append(f"")
    lines.append(f"Stage A: Galaxy-property regression")
    lines.append(f"  OLS R²:                  {r2_ols:.4f}")
    lines.append(f"  RF R² (train):           {r2_rf:.4f}")
    lines.append(f"  RF R² (5-fold CV):       {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    lines.append(f"  Residual std (OLS):      {resid_ols.std():.4f}")
    lines.append(f"  Residual std (RF):       {resid_rf_basic.std():.4f}")
    lines.append(f"")
    lines.append(f"Stage B: Environment density")
    lines.append(f"  5th-neighbor dist median: {np.median(d_k):.1f} Mpc")
    lines.append(f"  Corr(log_rho5, f_DM):    {rho_fdm_corr:.4f}")
    lines.append(f"")
    lines.append(f"Stage C: Extended regression (+ environment)")
    lines.append(f"  OLS R²:                  {r2_ols_ext:.4f} (Δ={r2_ols_ext-r2_ols:+.4f})")
    lines.append(f"  RF R² (train):           {r2_rf_ext:.4f} (Δ={r2_rf_ext-r2_rf:+.4f})")
    lines.append(f"  RF R² (5-fold CV):       {cv_scores_ext.mean():.4f} ± {cv_scores_ext.std():.4f}")
    lines.append(f"  Residual std (OLS):      {resid_ols_ext.std():.4f}")
    lines.append(f"  Residual std (RF):       {resid_rf_ext.std():.4f}")
    lines.append(f"")
    lines.append(f"Method B cross-check:")
    lines.append(f"  Resid std (Method B):    {df['resid_rf_ext_B'].std():.4f}")
    lines.append(f"  Corr(resid_A, resid_B):  {corr_AB:.4f}")
    lines.append(f"")
    lines.append(f"Variance decomposition:")
    var_total = np.var(y)
    var_props = var_total - np.var(resid_rf_basic)
    var_env = np.var(resid_rf_basic) - np.var(resid_rf_ext)
    var_resid = np.var(resid_rf_ext)
    lines.append(f"  Total f_DM variance:       {var_total:.6f}")
    lines.append(f"  Explained by properties:   {var_props:.6f} ({var_props/var_total*100:.1f}%)")
    lines.append(f"  Explained by environment:  {var_env:.6f} ({var_env/var_total*100:.1f}%)")
    lines.append(f"  Unexplained (residual):    {var_resid:.6f} ({var_resid/var_total*100:.1f}%)")
    lines.append(f"")
    lines.append(f"Residual columns in output:")
    lines.append(f"  resid_ols_basic  — OLS, galaxy properties only")
    lines.append(f"  resid_rf_basic   — RF, galaxy properties only")
    lines.append(f"  resid_ols_ext    — OLS, properties + environment")
    lines.append(f"  resid_rf_ext     — RF, properties + environment (PRIMARY)")
    lines.append(f"  resid_rf_ext_B   — RF extended, Method B stellar mass")

    summary = "\n".join(lines)
    print(f"\n{summary}")

    out_txt = os.path.join(DATA_DIR, "phaseB_step2_summary.txt")
    with open(out_txt, 'w') as f:
        f.write(summary)
    print(f"\n  Saved: {out_txt}")

    print("\n" + "=" * 65)
    print("Step 2 complete. Next: Step 3 (spatial autocorrelation)")
    print("=" * 65)


if __name__ == "__main__":
    main()
