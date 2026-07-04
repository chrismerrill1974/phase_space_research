#!/usr/bin/env python3
"""
Paper VIII — Step 1: Euclid Q1 lens demographics

Parses the Q1 Strong Lensing Discovery Engine catalogue (Walmsley et al.
2025; Zenodo 15025832) and the automated mass-model catalogue to extract:

  - grade A / B / C / X counts and the grade-A/B rate per deg^2
  - the Einstein radius (theta_E) distribution of modelled systems
  - wide-survey and DR1 extrapolations of the grade-A/B lens count

These replace Paper VII's assumed survey inputs (75,000 grade A/B lenses;
no theta_E information).

Inputs:  ../data/raw/q1_discovery_engine_lens_catalog.csv
         ../data/raw/modeling_lens_mass.csv
Output:  ../data/q1_demographics.npz
"""

import csv
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
RAW = os.path.join(ROOT, 'data', 'raw')
OUTPATH = os.path.join(ROOT, 'data', 'q1_demographics.npz')

Q1_AREA_DEG2 = 63.1       # Q1 footprint (EDF-N + EDF-S + EDF-F single visits)
WIDE_AREA_DEG2 = 14000.0  # Euclid Wide Survey
DR1_AREA_DEG2 = 1900.0    # approximate DR1 wide-survey coverage

# ── Catalogue: grades ─────────────────────────────────────────────
# Restrict to the 'discovery_engine' subset (the published 497-candidate
# sample with a single well-defined selection function). The catalogue
# also contains gz_euclid and serendipitous candidates whose selection
# differs; excluded from rate calculations.
grades = {}
with open(os.path.join(RAW, 'q1_discovery_engine_lens_catalog.csv')) as f:
    for row in csv.DictReader(f):
        if row['subset'].strip() != 'discovery_engine':
            continue
        grades[row['id_str']] = row['grade'].strip()

grade_vals = list(grades.values())
counts = {g: grade_vals.count(g) for g in sorted(set(grade_vals))}
print("Grade counts:", counts)

n_A = sum(v for g, v in counts.items() if g.startswith('A'))
n_B = counts.get('B', 0)
n_AB = n_A + n_B
rate_AB = n_AB / Q1_AREA_DEG2
print(f"\nGrade A: {n_A}   Grade B: {n_B}   A+B: {n_AB}")
print(f"Rate: {rate_AB:.2f} grade-A/B per deg^2 over {Q1_AREA_DEG2} deg^2")

# Extrapolations (Poisson error on Q1 count propagated)
frac_err = 1.0 / np.sqrt(n_AB)
N_wide = rate_AB * WIDE_AREA_DEG2
N_dr1 = rate_AB * DR1_AREA_DEG2
print(f"\nWide-survey extrapolation: {N_wide:.0f} +/- {N_wide*frac_err:.0f} grade A/B")
print(f"DR1 extrapolation:         {N_dr1:.0f} +/- {N_dr1*frac_err:.0f} grade A/B")
print(f"(Paper VII assumed 75,000 — ratio {N_wide/75000:.2f}x)")

# ── Mass models: Einstein radii ───────────────────────────────────
# einstein_radius_median_pdf is the SIE Einstein radius in arcsec.
theta_all, theta_grade, ids_model = [], [], []
with open(os.path.join(RAW, 'modeling_lens_mass.csv')) as f:
    for row in csv.DictReader(f):
        try:
            te = float(row['einstein_radius_median_pdf'])
            lo = float(row['einstein_radius_lower_1_sigma'])
            hi = float(row['einstein_radius_upper_1_sigma'])
        except (ValueError, KeyError):
            continue
        # Quality cuts: physical range and <50% relative 1-sigma width
        if not (0.1 < te < 5.0):
            continue
        if (hi - lo) / (2 * te) > 0.5:
            continue
        theta_all.append(te)
        theta_grade.append(grades.get(row['id_str'], '?'))
        ids_model.append(row['id_str'])

theta_all = np.array(theta_all)
theta_grade = np.array(theta_grade)
is_AB = np.array([g.startswith('A') or g == 'B' for g in theta_grade])
theta_AB = theta_all[is_AB]

print(f"\nModelled systems passing quality cuts: {len(theta_all)}")
print(f"  of which grade A/B: {len(theta_AB)}")
for label, arr in [('all modelled', theta_all), ('grade A/B', theta_AB)]:
    q25, q50, q75 = np.percentile(arr, [25, 50, 75])
    print(f"  theta_E ({label}): median {q50:.2f}\" "
          f"[{q25:.2f}, {q75:.2f}]  frac<1\": {np.mean(arr < 1.0):.2f}  "
          f"frac<0.5\": {np.mean(arr < 0.5):.2f}")

np.savez(
    OUTPATH,
    q1_area_deg2=Q1_AREA_DEG2,
    wide_area_deg2=WIDE_AREA_DEG2,
    dr1_area_deg2=DR1_AREA_DEG2,
    n_A=n_A, n_B=n_B, n_AB=n_AB,
    rate_AB_per_deg2=rate_AB,
    N_wide=N_wide, N_dr1=N_dr1, frac_err=frac_err,
    theta_E_modelled=theta_all,
    theta_E_grade=theta_grade,
    theta_E_AB=theta_AB,
    grade_counts_keys=np.array(list(counts.keys())),
    grade_counts_vals=np.array(list(counts.values())),
)
print(f"\nSaved {OUTPATH}")
