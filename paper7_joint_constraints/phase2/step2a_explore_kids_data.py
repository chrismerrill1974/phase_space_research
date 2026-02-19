#!/usr/bin/env python3
"""
Step 2a — Explore KiDS-1000 cosmic shear data release.
Run locally: python step2a_explore_kids_data.py

Reports:
  1. Directory listing of the data release
  2. FITS file structure (HDUs, columns, shapes)
  3. Band power ℓ-bins and data vector layout
  4. n(z) distributions for all tomographic bins
  5. Covariance matrix structure

Output: printed summary + saves step2a_kids_exploration.npz
"""

import os
import sys
import glob
import numpy as np

# ── Configuration ──────────────────────────────────────────────────
KIDS_DIR = "/Users/chris/Documents/research_data/KiDS1000_cosmis_shear_data_release"

# Check path exists
if not os.path.isdir(KIDS_DIR):
    print(f"ERROR: KiDS data directory not found at:\n  {KIDS_DIR}")
    print("Please update KIDS_DIR in this script.")
    sys.exit(1)

print("=" * 70)
print("Step 2a: KiDS-1000 Data Exploration")
print("=" * 70)

# ── 1. Directory listing ──────────────────────────────────────────
print("\n[1] Directory contents:")
print("-" * 50)

all_files = []
for root, dirs, files in os.walk(KIDS_DIR):
    for f in sorted(files):
        rel = os.path.relpath(os.path.join(root, f), KIDS_DIR)
        size = os.path.getsize(os.path.join(root, f))
        all_files.append((rel, size))
        print(f"  {rel:60s}  ({size:>10,} bytes)")

print(f"\nTotal: {len(all_files)} files")

# ── 2. Try to load FITS files ─────────────────────────────────────
print("\n[2] FITS file inspection:")
print("-" * 50)

try:
    from astropy.io import fits
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False
    print("  astropy not installed — trying manual inspection")

fits_files = [f for f, s in all_files if f.endswith('.fits') or f.endswith('.FITS')]
txt_files = [f for f, s in all_files if f.endswith('.txt') or f.endswith('.dat') or f.endswith('.asc')]

if HAS_ASTROPY and fits_files:
    for ff in fits_files:
        fpath = os.path.join(KIDS_DIR, ff)
        print(f"\n  File: {ff}")
        with fits.open(fpath) as hdul:
            print(f"    HDUs: {len(hdul)}")
            for i, hdu in enumerate(hdul):
                hdu_name = hdu.name if hdu.name else f"HDU{i}"
                if hasattr(hdu, 'columns') and hdu.columns:
                    print(f"    [{i}] {hdu_name}: BinTable, "
                          f"{len(hdu.columns)} columns, "
                          f"{hdu.data.shape[0] if hdu.data is not None else '?'} rows")
                    for col in hdu.columns:
                        print(f"         col '{col.name}': format={col.format}")
                elif hasattr(hdu, 'data') and hdu.data is not None:
                    print(f"    [{i}] {hdu_name}: shape={hdu.data.shape}, "
                          f"dtype={hdu.data.dtype}")
                else:
                    print(f"    [{i}] {hdu_name}: (header only / empty)")

# ── 3. Look for text/ASCII data files ─────────────────────────────
print("\n[3] Text/ASCII data files:")
print("-" * 50)

for tf in txt_files[:20]:  # limit output
    fpath = os.path.join(KIDS_DIR, tf)
    print(f"\n  File: {tf}")
    with open(fpath, 'r') as f:
        lines = f.readlines()
    print(f"    Lines: {len(lines)}")
    # Show first 5 lines
    for line in lines[:5]:
        print(f"    | {line.rstrip()}")
    if len(lines) > 5:
        print(f"    | ... ({len(lines) - 5} more lines)")

# ── 4. Look for n(z) files ────────────────────────────────────────
print("\n[4] Searching for n(z) / redshift distribution files:")
print("-" * 50)

nz_candidates = [f for f, s in all_files 
                 if 'nz' in f.lower() or 'n_z' in f.lower() 
                 or 'nofz' in f.lower() or 'redshift' in f.lower()
                 or 'source' in f.lower()]
if nz_candidates:
    for nc in nz_candidates:
        print(f"  Found: {nc}")
else:
    print("  No obvious n(z) files found by name — check FITS contents above")

# ── 5. Look for band power / Cl files ─────────────────────────────
print("\n[5] Searching for band power / C_ell files:")
print("-" * 50)

bp_candidates = [f for f, s in all_files
                 if 'band' in f.lower() or 'cl' in f.lower()
                 or 'power' in f.lower() or 'spectrum' in f.lower()
                 or 'bp' in f.lower()]
if bp_candidates:
    for bc in bp_candidates:
        print(f"  Found: {bc}")
else:
    print("  No obvious band power files found by name — check FITS contents above")

# ── 6. Look for covariance files ──────────────────────────────────
print("\n[6] Searching for covariance files:")
print("-" * 50)

cov_candidates = [f for f, s in all_files
                  if 'cov' in f.lower() or 'covariance' in f.lower()
                  or 'noise' in f.lower()]
if cov_candidates:
    for cc in cov_candidates:
        print(f"  Found: {cc}")
else:
    print("  No obvious covariance files found by name — check FITS contents above")

# ── 7. Try to read band power data if found ───────────────────────
print("\n[7] Attempting to read band power data vectors:")
print("-" * 50)

if HAS_ASTROPY:
    # Try common filenames from Asgari+2021
    possible_bp = glob.glob(os.path.join(KIDS_DIR, '**/*band*'), recursive=True)
    possible_bp += glob.glob(os.path.join(KIDS_DIR, '**/*BP*'), recursive=True)
    possible_bp += glob.glob(os.path.join(KIDS_DIR, '**/*Cl*'), recursive=True)
    possible_bp += glob.glob(os.path.join(KIDS_DIR, '**/*cl*'), recursive=True)
    
    for bp in set(possible_bp):
        rel = os.path.relpath(bp, KIDS_DIR)
        print(f"\n  Trying: {rel}")
        try:
            if bp.endswith('.fits'):
                with fits.open(bp) as hdul:
                    for i, hdu in enumerate(hdul):
                        if hasattr(hdu, 'data') and hdu.data is not None:
                            if hasattr(hdu.data, 'dtype') and hdu.data.dtype.names:
                                print(f"    HDU[{i}] columns: {hdu.data.dtype.names}")
                                for name in hdu.data.dtype.names[:5]:
                                    col = hdu.data[name]
                                    print(f"      {name}: shape={col.shape}, "
                                          f"min={col.min():.4e}, max={col.max():.4e}")
                            else:
                                print(f"    HDU[{i}]: shape={hdu.data.shape}")
            elif bp.endswith('.txt') or bp.endswith('.dat') or bp.endswith('.asc'):
                data = np.loadtxt(bp)
                print(f"    shape={data.shape}")
                if data.ndim == 2:
                    print(f"    columns: {data.shape[1]}")
                    print(f"    col[0] range: [{data[:,0].min():.2f}, {data[:,0].max():.2f}]")
        except Exception as e:
            print(f"    Error: {e}")

# ── 8. Summary ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY — Key questions for Phase 2 design:")
print("=" * 70)
print("""
Please paste/send the output above so we can determine:

  Q1: How many tomographic bins, and what format are the band powers?
  Q2: Are individual n(z) per bin available?
  Q3: Is the full covariance matrix included?
  Q4: What ℓ-range and binning scheme is used?
  Q5: Are the band powers C_E (E-mode) or something else?

Based on this, we'll decide:
  Option A: Full tomographic treatment (15 bin pairs, most rigorous)
  Option B: Effective single-bin (simpler, matches Phase 1 pipeline)
  Option C: Published S₈ comparison (simplest, least constraining)
""")

print("Script complete.")
