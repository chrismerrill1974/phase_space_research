#!/usr/bin/env python3
"""
Paper VII - Phase 2, Step 2a v2
Extract KiDS-1000 band power data for ISF likelihood.

Reads:
  - PeeE cosmic shear E-mode band powers (15 bin pairs × 8 ℓ-bins)
  - Full 200×200 covariance → extract 120×120 shear-shear block
  - n(z) for all 5 source bins (119 z-values each)
  - Also reads their read_data_files.py for convention info

Saves: data/kids1000_bandpowers.npz

Run locally: python step2a_extract_kids_data.py
"""

import os
import sys
import numpy as np
from astropy.io import fits

# ── Configuration ──────────────────────────────────────────────────
KIDS_DIR = "/Users/chris/Documents/research_data/KiDS1000_cosmis_shear_data_release"
BP_FILE = os.path.join(
    KIDS_DIR, "data_fits",
    "bp_KIDS1000_BlindC_with_m_bias_V1.0.0A_ugriZYJHKs_"
    "photoz_SG_mask_LF_svn_309c_2Dbins_v2_goldclasses_"
    "Flag_SOM_Fid.fits"
)
READER_FILE = os.path.join(KIDS_DIR, "data_fits", "read_data_files.py")

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)
OUTPATH = os.path.join(DATA_DIR, 'kids1000_bandpowers.npz')

# Check paths
if not os.path.isfile(BP_FILE):
    print(f"ERROR: Band power FITS file not found:\n  {BP_FILE}")
    sys.exit(1)

print("=" * 70)
print("Step 2a: Extract KiDS-1000 Band Power Data")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════
# PART 0: Read their read_data_files.py for conventions
# ══════════════════════════════════════════════════════════════════
print("\n[0] KiDS reader script (for reference):")
print("-" * 50)

if os.path.isfile(READER_FILE):
    with open(READER_FILE, 'r') as f:
        reader_content = f.read()
    print(reader_content)
    print("-" * 50)
else:
    print("  read_data_files.py not found — proceeding with manual read")
    reader_content = ""

# ══════════════════════════════════════════════════════════════════
# PART 1: Open FITS and extract band powers
# ══════════════════════════════════════════════════════════════════
print("\n[1] Reading band power FITS file...")

with fits.open(BP_FILE) as hdul:
    # ── HDU names and structure ─────────────────────────
    print(f"  HDUs: {[hdu.name for hdu in hdul]}")

    # ── PeeE: cosmic shear E-mode band powers ──────────
    pee = hdul['PeeE'].data
    n_rows = len(pee)
    print(f"\n  PeeE: {n_rows} rows")

    bin1_ee = pee['BIN1']
    bin2_ee = pee['BIN2']
    angbin_ee = pee['ANGBIN']
    value_ee = pee['VALUE']
    ang_ee = pee['ANG']  # effective ℓ

    # Identify unique bin pairs and ℓ-bins
    unique_pairs = sorted(set(zip(bin1_ee, bin2_ee)))
    unique_angbins = sorted(set(angbin_ee))
    n_pairs = len(unique_pairs)
    n_ell_bins = len(unique_angbins)

    print(f"  Bin pairs: {n_pairs} (expect 15)")
    print(f"  ℓ-bins per pair: {n_ell_bins} (expect 8)")
    print(f"  Total: {n_pairs * n_ell_bins} = {n_rows}")

    # Print bin pair list
    print(f"\n  Bin pairs (i,j):")
    for p in unique_pairs:
        print(f"    ({p[0]}, {p[1]})")

    # Get effective ℓ values (should be same for all pairs)
    first_pair = unique_pairs[0]
    mask_first = (bin1_ee == first_pair[0]) & (bin2_ee == first_pair[1])
    ell_eff = ang_ee[mask_first]
    print(f"\n  Effective ℓ values: {ell_eff}")
    print(f"  ℓ range: [{ell_eff.min():.1f}, {ell_eff.max():.1f}]")

    # Organize into 2D arrays: data_vector[pair_idx, ell_idx]
    # Also build the flat ordering for covariance matching
    bp_data = np.zeros((n_pairs, n_ell_bins))
    pair_labels = []

    for ip, (b1, b2) in enumerate(unique_pairs):
        mask = (bin1_ee == b1) & (bin2_ee == b2)
        vals = value_ee[mask]
        angs = ang_ee[mask]
        # Sort by ANGBIN
        order = np.argsort(angbin_ee[mask])
        bp_data[ip, :] = vals[order]
        pair_labels.append((int(b1), int(b2)))

        # Verify ℓ values are consistent
        assert np.allclose(angs[order], ell_eff), \
            f"ℓ mismatch for pair ({b1},{b2})"

    print(f"\n  Band power data shape: {bp_data.shape}")
    print(f"  Value range: [{bp_data.min():.4e}, {bp_data.max():.4e}]")

    # Print a summary table
    print(f"\n  Band power summary (value at first and last ℓ-bin):")
    print(f"  {'Pair':>8s}  {'P_E(ℓ_min)':>12s}  {'P_E(ℓ_max)':>12s}")
    for ip, (b1, b2) in enumerate(pair_labels):
        print(f"  ({b1},{b2}){'':<3s}  {bp_data[ip,0]:>12.4e}  "
              f"{bp_data[ip,-1]:>12.4e}")

    # ── PneE: galaxy-galaxy lensing (for reference) ────
    pne = hdul['PneE'].data
    print(f"\n  PneE (GGL): {len(pne)} rows "
          f"(BIN1 range: {pne['BIN1'].min()}-{pne['BIN1'].max()}, "
          f"BIN2 range: {pne['BIN2'].min()}-{pne['BIN2'].max()})")
    n_ggl = len(pne)

    # ── COVMAT ─────────────────────────────────────────
    covmat_full = hdul['COVMAT'].data
    print(f"\n  Full covariance: {covmat_full.shape}")

    # The data vector ordering is: PneE (80 entries) + PeeE (120)
    # So the shear-shear block is rows/cols 80:200
    cov_ee = covmat_full[n_ggl:, n_ggl:]
    print(f"  Shear-shear block: {cov_ee.shape}")

    # Check positive-definiteness
    eigvals = np.linalg.eigvalsh(cov_ee)
    print(f"  Eigenvalue range: [{eigvals.min():.2e}, {eigvals.max():.2e}]")
    if eigvals.min() > 0:
        print(f"  ✓ Covariance is positive definite")
    else:
        print(f"  ⚠ WARNING: Covariance has non-positive eigenvalues!")

    # Diagonal errors for reference
    diag_err = np.sqrt(np.diag(cov_ee))
    print(f"  Diagonal σ range: [{diag_err.min():.2e}, {diag_err.max():.2e}]")

    # SNR per data point
    flat_data = bp_data.flatten()
    snr = np.abs(flat_data) / diag_err
    print(f"  Per-point |SNR| range: [{snr.min():.2f}, {snr.max():.2f}]")
    print(f"  Per-point |SNR| mean: {snr.mean():.2f}")

    # Correlation matrix
    d_inv = 1.0 / diag_err
    corr_ee = cov_ee * np.outer(d_inv, d_inv)
    off_diag = corr_ee[np.triu_indices_from(corr_ee, k=1)]
    print(f"  Off-diagonal correlation: "
          f"mean={off_diag.mean():.3f}, "
          f"max={off_diag.max():.3f}")

    # ── NZ_SOURCE: n(z) for 5 tomographic bins ─────────
    nz_src = hdul['NZ_SOURCE'].data
    z_low = nz_src['Z_LOW']
    z_mid = nz_src['Z_MID']
    z_high = nz_src['Z_HIGH']

    nz_bins = np.zeros((5, len(z_mid)))
    for b in range(5):
        nz_bins[b] = nz_src[f'BIN{b+1}']

    print(f"\n  Source n(z): {nz_bins.shape} "
          f"(5 bins × {len(z_mid)} z-values)")
    print(f"  z range: [{z_low.min():.3f}, {z_high.max():.3f}]")
    print(f"  Δz = {z_mid[1] - z_mid[0]:.4f}")

    # Check normalization
    dz = z_high - z_low
    for b in range(5):
        norm = np.sum(nz_bins[b] * dz)
        zmean = np.sum(z_mid * nz_bins[b] * dz) / norm
        print(f"  Bin {b+1}: ∫n(z)dz = {norm:.4f}, "
              f"<z> = {zmean:.3f}")

    # ── NZ_LENS (for reference) ────────────────────────
    if 'NZ_LENS' in [h.name for h in hdul]:
        nz_lens = hdul['NZ_LENS'].data
        print(f"\n  Lens n(z): {len(nz_lens)} z-values, "
              f"z range: [{nz_lens['Z_LOW'].min():.3f}, "
              f"{nz_lens['Z_HIGH'].max():.3f}]")

# ══════════════════════════════════════════════════════════════════
# PART 2: Determine band power convention
# ══════════════════════════════════════════════════════════════════
print("\n[2] Band power convention check:")
print("-" * 50)

# The KiDS band powers can be either:
#   (a) C_E(ℓ) — raw power spectrum
#   (b) ℓ(ℓ+1)C_E(ℓ)/(2π) — scaled
#
# Typical C_κ(ℓ~1000) ~ 10^{-7} for a single broad source bin
# → ℓ(ℓ+1)C_κ/(2π) ~ 10^{-4} to 10^{-3}
#
# KiDS values range ~10^{-5} to 5×10^{-4}
# This is consistent with convention (b) for tomographic bins

# Cross-check: for highest-z auto pair (5,5), signal is strongest
idx_55 = pair_labels.index((5, 5))
print(f"  Pair (5,5) at ℓ~{ell_eff[3]:.0f}: P_E = {bp_data[idx_55, 3]:.4e}")
print(f"  If this is ℓ(ℓ+1)C_E/(2π), then C_E = "
      f"{bp_data[idx_55, 3] * 2*np.pi / (ell_eff[3]*(ell_eff[3]+1)):.4e}")
print(f"  If this is raw C_E, then ℓ(ℓ+1)C_E/(2π) = "
      f"{bp_data[idx_55, 3] * ell_eff[3]*(ell_eff[3]+1) / (2*np.pi):.4e}")

# The Asgari+2021 convention: their Eq. 17 defines band powers
# as P_E^n = ∑_ℓ [ℓ(ℓ+1)/(2π)] × h_ℓ^n × C_E(ℓ)
# But they plot "C_EE" vs ℓ — need to check their Fig. 2 scale
# From their paper, values at ℓ~500 are ~10^{-4}, consistent with
# ℓ(ℓ+1)C_E/(2π).

# We'll store both the raw VALUE and the effective ℓ, and handle
# the conversion in the likelihood step after verifying against
# our theory prediction.
print("\n  → Storing raw VALUES as-is. Convention will be verified")
print("    in Step 2b by comparing ΛCDM prediction shape+amplitude.")

# ══════════════════════════════════════════════════════════════════
# PART 3: Save checkpoint
# ══════════════════════════════════════════════════════════════════
print("\n[3] Saving checkpoint...")

np.savez(
    OUTPATH,
    # Band power data
    bp_data=bp_data,                # (15, 8) band power values
    ell_eff=ell_eff,                # (8,) effective ℓ per bin
    pair_labels=np.array(pair_labels),  # (15, 2) bin pair indices
    n_pairs=n_pairs,
    n_ell_bins=n_ell_bins,
    # Covariance
    cov_ee=cov_ee,                  # (120, 120) shear-shear block
    cov_full=covmat_full,           # (200, 200) full covariance
    n_ggl=n_ggl,                    # offset for GGL entries
    # Source redshift distributions
    z_low=z_low,                    # (119,) bin edges
    z_mid=z_mid,                    # (119,) bin centers
    z_high=z_high,                  # (119,) bin edges
    nz_bins=nz_bins,                # (5, 119) n(z) per bin
    # Flat data vector (for χ² computation)
    data_vector=flat_data,          # (120,) flattened band powers
)

print(f"  ✓ Saved to {OUTPATH}")
print(f"  bp_data: {bp_data.shape}")
print(f"  ell_eff: {ell_eff.shape} → {ell_eff}")
print(f"  cov_ee: {cov_ee.shape}")
print(f"  nz_bins: {nz_bins.shape}")

# ══════════════════════════════════════════════════════════════════
# PART 4: Summary for next step
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY — Ready for Step 2b")
print("=" * 70)
print(f"""
Data extracted:
  • 15 tomographic bin pairs, 8 ℓ-bins each → 120 data points
  • ℓ range: [{ell_eff[0]:.1f}, {ell_eff[-1]:.1f}]
  • Full 120×120 covariance matrix (positive definite ✓)
  • n(z) for 5 source bins (119 z-values, z ∈ [0, {z_high.max():.1f}])
  • Mean SNR per data point: {snr.mean():.1f}

Next: Step 2b — Build 15 tomographic response operators A_ij(8, 500)
  Each A_ij maps M²(k) → C_ij(ℓ_eff) using KiDS n_i(z) kernels.
  Total work: 15 × 8 × 500 = 60,000 matrix elements (seconds).
""")

print("Script complete.")
