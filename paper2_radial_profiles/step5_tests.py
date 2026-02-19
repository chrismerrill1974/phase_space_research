#!/usr/bin/env python3
"""
Step 5: Statistical Tests for Radial Steps
============================================
Five independent tests for staircase vs smooth f_DM(R) profiles:

1. Piecewise-constant vs smooth model comparison (BIC)
2. Derivative analysis: bimodal (spikes + zeros) vs smooth
3. Stacked residual structure at predicted transition densities
4. Plateau duration: dwell time at predicted (N-1)/N values
5. Transition sharpness: radial interval for step changes

Author: C. K. Merrill
Date: 2026-02-11
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize
from scipy.stats import ks_2samp, norm, kurtosis
from scipy.interpolate import UnivariateSpline
import csv
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────
PROFILES_CSV = "sparc_radial_profiles.csv"
CATALOG_CSV  = "sparc_dm_fractions.csv"
NULLS_CSV    = "sparc_radial_nulls.csv"
UPSILON      = 0.5

OUTPUT_MAIN  = "step5_tests.png"
OUTPUT_CSV   = "step5_results.csv"

MIN_PTS      = 10
MIN_RMAX_RD  = 3.0

# Predicted levels
PREDICTED_FDM = {N: (N-1)/N for N in range(2, 11)}
PREDICTED_STEPS = {N: 1.0/(N*(N+1)) for N in range(2, 10)}


# ── Load data ─────────────────────────────────────────────────────
def load_profiles(filepath):
    data = defaultdict(list)
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gname = row['Galaxy']
            data[gname].append({
                'R_kpc': float(row['R_kpc']),
                'R_Rd': float(row['R_Rd']),
                'fDM': float(row['fDM']),
                'e_fDM': float(row['e_fDM']),
                'Vobs': float(row['Vobs']),
                'eVobs': float(row['eVobs']),
                'Q': int(row['Q']),
                'Vflat': float(row['Vflat']),
                'T': int(row['T']),
                'Rdisk': float(row['Rdisk']),
                'L36': float(row['L36']),
                'Vgas': float(row['Vgas']),
                'Vdisk_raw': float(row['Vdisk_raw']),
                'Vbul_raw': float(row['Vbul_raw']),
            })
    return data


def load_catalog(filepath):
    meta = {}
    with open(filepath, 'r') as f:
        for row in csv.DictReader(f):
            meta[row['Galaxy']] = {
                'Vflat': float(row['Vflat_kms']),
                'Q': int(row['Q']),
                'Rdisk': float(row['Rdisk_kpc']),
                'Rmax': float(row['Rmax_kpc']),
                'T': int(row['T']),
                'L36': float(row['L36_1e9Lsun']),
                'Npts': int(row['Npts']),
            }
    return meta


def load_nulls(filepath):
    """Load Step 4 null model fits."""
    nulls = defaultdict(list)
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gname = row['Galaxy']
            nulls[gname].append({
                'R_Rd': float(row['R_Rd']),
                'fDM_obs': float(row['fDM_obs']),
                'e_fDM': float(row['e_fDM']),
                'fDM_spline': float(row['fDM_spline']) if row['fDM_spline'] else np.nan,
                'res_spline': float(row['res_spline']) if row['res_spline'] else np.nan,
                'fDM_nfw': float(row['fDM_nfw']) if row['fDM_nfw'] else np.nan,
                'fDM_iso': float(row['fDM_iso']) if row['fDM_iso'] else np.nan,
            })
    return nulls


def select_galaxies(data, meta):
    selected = []
    for gname, pts in data.items():
        if gname not in meta:
            continue
        m = meta[gname]
        if m['Q'] > 1 or m['Vflat'] <= 0:
            continue
        if len(pts) < MIN_PTS:
            continue
        Rmax = max(p['R_kpc'] for p in pts)
        Rd = m['Rdisk']
        if Rd <= 0 or Rmax / Rd < MIN_RMAX_RD:
            continue
        selected.append(gname)
    return sorted(selected)


def classify(meta, gname):
    vf = meta[gname]['Vflat']
    if vf < 80: return 'dwarf'
    elif vf < 150: return 'intermediate'
    else: return 'massive'


# ══════════════════════════════════════════════════════════════════
# TEST 1: Piecewise-Constant vs Smooth (BIC comparison)
# ══════════════════════════════════════════════════════════════════
def fit_piecewise_constant(R, f, e, n_steps=1):
    """Fit a piecewise-constant function with n_steps transitions.
    Levels are CONSTRAINED to (N-1)/N for integer N.
    Returns BIC and fitted values.
    """
    n = len(R)
    if n < 2 * (n_steps + 1):
        return np.inf, None

    # Candidate levels: (N-1)/N for N = 2..10
    cand_levels = [(N-1)/N for N in range(2, 11)]

    # For n_steps=1: two plateaus with one breakpoint
    # For n_steps=2: three plateaus with two breakpoints
    # Search over all combinations of levels and breakpoint positions

    best_bic = np.inf
    best_fit = None

    if n_steps == 0:
        # Just a constant: find best single level
        for level in cand_levels:
            resid = f - level
            sigma = np.maximum(e, 0.01)
            chi2 = np.sum((resid / sigma) ** 2)
            k = 1  # one parameter
            bic = chi2 + k * np.log(n)
            if bic < best_bic:
                best_bic = bic
                best_fit = np.full(n, level)
        return best_bic, best_fit

    if n_steps == 1:
        for i_break in range(3, n - 3):
            for lev1 in cand_levels:
                for lev2 in cand_levels:
                    if lev2 <= lev1:
                        continue  # f_DM must increase
                    fitted = np.concatenate([
                        np.full(i_break, lev1),
                        np.full(n - i_break, lev2)
                    ])
                    sigma = np.maximum(e, 0.01)
                    chi2 = np.sum(((f - fitted) / sigma) ** 2)
                    k = 3  # two levels + one breakpoint
                    bic = chi2 + k * np.log(n)
                    if bic < best_bic:
                        best_bic = bic
                        best_fit = fitted.copy()
        return best_bic, best_fit

    if n_steps == 2:
        # Three plateaus: coarser search for speed
        step_size = max(1, n // 20)
        for i1 in range(3, n - 6, step_size):
            for i2 in range(i1 + 3, n - 3, step_size):
                for lev1 in cand_levels[::2]:  # coarser level search
                    for lev2 in cand_levels[::2]:
                        if lev2 <= lev1:
                            continue
                        for lev3 in cand_levels[::2]:
                            if lev3 <= lev2:
                                continue
                            fitted = np.concatenate([
                                np.full(i1, lev1),
                                np.full(i2 - i1, lev2),
                                np.full(n - i2, lev3)
                            ])
                            sigma = np.maximum(e, 0.01)
                            chi2 = np.sum(((f - fitted) / sigma) ** 2)
                            k = 5
                            bic = chi2 + k * np.log(n)
                            if bic < best_bic:
                                best_bic = bic
                                best_fit = fitted.copy()
        return best_bic, best_fit

    return np.inf, None


def fit_smooth_model(R, f, e):
    """Fit a smooth spline model and return BIC."""
    n = len(R)
    w = 1.0 / np.maximum(e, 0.01)
    try:
        s = n * 1.0
        spline = UnivariateSpline(R, f, w=w, s=s, k=3)
        fitted = spline(R)
        sigma = np.maximum(e, 0.01)
        chi2 = np.sum(((f - fitted) / sigma) ** 2)
        # Effective parameters ~ n - residual_dof
        # For cubic spline with smoothing, approximate k ~ n/4
        k_eff = max(4, n // 4)
        bic = chi2 + k_eff * np.log(n)
        return bic, fitted
    except:
        return np.inf, None


def run_test1(data, selected, meta):
    """Test 1: BIC comparison piecewise-constant vs smooth for each galaxy."""
    print("\n  TEST 1: Piecewise-constant vs smooth (BIC)")
    print("  " + "-" * 50)

    results = []
    pw_wins = 0
    smooth_wins = 0

    for gname in selected:
        pts = sorted(data[gname], key=lambda p: p['R_Rd'])
        # Use only R/Rd > 0.5 to avoid extreme inner baryon-dominated region
        pts = [p for p in pts if p['R_Rd'] > 0.5]
        if len(pts) < 8:
            continue

        R = np.array([p['R_Rd'] for p in pts])
        f = np.array([p['fDM'] for p in pts])
        e = np.array([p['e_fDM'] for p in pts])

        bic_smooth, _ = fit_smooth_model(R, f, e)
        bic_pw0, _ = fit_piecewise_constant(R, f, e, n_steps=0)
        bic_pw1, fit_pw1 = fit_piecewise_constant(R, f, e, n_steps=1)
        bic_pw2, fit_pw2 = fit_piecewise_constant(R, f, e, n_steps=2)

        best_pw_bic = min(bic_pw0, bic_pw1, bic_pw2)
        best_pw_steps = [0, 1, 2][np.argmin([bic_pw0, bic_pw1, bic_pw2])]

        delta_bic = best_pw_bic - bic_smooth  # negative = PW preferred
        if delta_bic < 0:
            pw_wins += 1
        else:
            smooth_wins += 1

        results.append({
            'Galaxy': gname,
            'class': classify(meta, gname),
            'Vflat': meta[gname]['Vflat'],
            'Npts': len(pts),
            'BIC_smooth': bic_smooth,
            'BIC_pw_best': best_pw_bic,
            'pw_steps': best_pw_steps,
            'delta_BIC': delta_bic,
        })

    frac_pw = pw_wins / (pw_wins + smooth_wins) if (pw_wins + smooth_wins) > 0 else 0
    print(f"    Piecewise preferred: {pw_wins}/{pw_wins + smooth_wins} "
          f"({frac_pw:.1%})")
    print(f"    Smooth preferred:    {smooth_wins}/{pw_wins + smooth_wins}")
    print(f"    Prediction A threshold: >25% piecewise preferred")
    print(f"    Result: {'SUPPORTED' if frac_pw > 0.25 else 'NOT SUPPORTED'}")

    return results, frac_pw


# ══════════════════════════════════════════════════════════════════
# TEST 2: Derivative Distribution Analysis
# ══════════════════════════════════════════════════════════════════
def compute_derivatives(pts):
    sorted_pts = sorted(pts, key=lambda p: p['R_Rd'])
    R = np.array([p['R_Rd'] for p in sorted_pts])
    f = np.array([p['fDM'] for p in sorted_pts])
    if len(R) < 3:
        return np.array([]), np.array([])
    dfdR = np.zeros(len(R))
    for i in range(1, len(R)-1):
        dR = R[i+1] - R[i-1]
        if dR > 0:
            dfdR[i] = (f[i+1] - f[i-1]) / dR
    if R[1] - R[0] > 0:
        dfdR[0] = (f[1] - f[0]) / (R[1] - R[0])
    if R[-1] - R[-2] > 0:
        dfdR[-1] = (f[-1] - f[-2]) / (R[-1] - R[-2])
    return R, dfdR


def run_test2(data, selected):
    """Test 2: Is the derivative distribution bimodal (plateaus + transitions)?"""
    print("\n  TEST 2: Derivative distribution bimodality")
    print("  " + "-" * 50)

    all_derivs = []
    for gname in selected:
        pts = sorted(data[gname], key=lambda p: p['R_Rd'])
        pts = [p for p in pts if p['R_Rd'] > 1.0]
        if len(pts) < 5:
            continue
        R, dfdR = compute_derivatives(pts)
        # Only interior points (skip edge derivatives)
        if len(dfdR) > 4:
            all_derivs.extend(dfdR[1:-1])

    all_derivs = np.array(all_derivs)
    all_derivs = all_derivs[np.isfinite(all_derivs)]
    clip = np.percentile(np.abs(all_derivs), 99)
    all_derivs = all_derivs[np.abs(all_derivs) < clip]

    # Kurtosis: staircase (bimodal with peak at 0) should give
    # positive excess kurtosis from heavy tails (transition spikes)
    kurt = kurtosis(all_derivs, fisher=True)

    # Fraction near zero: staircase predicts high fraction at |df/dR| ~ 0
    frac_zero = np.mean(np.abs(all_derivs) < 0.01)
    frac_zero_05 = np.mean(np.abs(all_derivs) < 0.005)

    # Compare to null: generate smooth-model derivatives
    # For a smooth monotonic function, derivatives should be
    # approximately normally distributed (or chi-like)
    # Use bootstrap of actual derivatives to test
    n_boot = 1000
    boot_frac_zero = []
    for _ in range(n_boot):
        # Shuffle: if structure exists, destroying radial order
        # should reduce near-zero fraction
        perm = np.random.permutation(all_derivs)
        # Recompute "derivatives" from shuffled sequence
        # This doesn't make physical sense, so instead generate
        # from a Gaussian with same mean/std
        sim = np.random.normal(np.mean(all_derivs), np.std(all_derivs),
                               len(all_derivs))
        boot_frac_zero.append(np.mean(np.abs(sim) < 0.01))

    p_excess_zero = np.mean(np.array(boot_frac_zero) >= frac_zero)

    print(f"    N derivatives: {len(all_derivs)}")
    print(f"    Excess kurtosis: {kurt:.3f}")
    print(f"    |df/dR| < 0.01:  {frac_zero:.3%}")
    print(f"    |df/dR| < 0.005: {frac_zero_05:.3%}")
    print(f"    p(Gaussian has ≥ observed near-zero fraction): {p_excess_zero:.3f}")
    print(f"    Prediction: bimodal (spike at 0 + tails)")
    bimodal = kurt > 1.0 and p_excess_zero < 0.05
    print(f"    Result: {'SUPPORTED (excess near-zero + heavy tails)' if bimodal else 'NOT SUPPORTED'}")

    return all_derivs, kurt, frac_zero, p_excess_zero


# ══════════════════════════════════════════════════════════════════
# TEST 3: Stacked Residual Structure at Predicted Levels
# ══════════════════════════════════════════════════════════════════
def run_test3(nulls, selected):
    """Test 3: Do spline residuals show coherent structure at predicted levels?"""
    print("\n  TEST 3: Residual structure at predicted (N-1)/N levels")
    print("  " + "-" * 50)

    # Collect all (f_DM_obs, residual) pairs at R/Rd > 1
    fdm_all = []
    res_all = []
    for gname in selected:
        if gname not in nulls:
            continue
        for pt in nulls[gname]:
            if pt['R_Rd'] > 1.0 and np.isfinite(pt['res_spline']):
                fdm_all.append(pt['fDM_obs'])
                res_all.append(pt['res_spline'])
    fdm_all = np.array(fdm_all)
    res_all = np.array(res_all)

    # For each predicted level, compute mean absolute residual in a
    # narrow window vs broader background
    window = 0.015  # half-width of window around predicted level
    results = {}
    for N in range(3, 9):
        level = (N-1) / N
        near = np.abs(fdm_all - level) < window
        far = np.abs(fdm_all - level) >= window
        if near.sum() < 3:
            results[N] = {'n_near': near.sum(), 'mean_abs_near': np.nan,
                          'mean_abs_far': np.nan, 'ratio': np.nan}
            continue
        mean_abs_near = np.mean(np.abs(res_all[near]))
        mean_abs_far = np.mean(np.abs(res_all[far]))
        ratio = mean_abs_near / mean_abs_far if mean_abs_far > 0 else np.nan

        results[N] = {
            'n_near': int(near.sum()),
            'mean_abs_near': mean_abs_near,
            'mean_abs_far': mean_abs_far,
            'ratio': ratio,
        }
        print(f"    N={N} (f={level:.4f}): {near.sum()} pts, "
              f"|res|_near={mean_abs_near:.4f}, |res|_far={mean_abs_far:.4f}, "
              f"ratio={ratio:.3f}")

    # If galaxies dwell at predicted levels (plateaus), residuals NEAR
    # those levels should be SMALLER (ratio < 1). If smooth, ratio ~ 1.
    ratios = [r['ratio'] for r in results.values() if np.isfinite(r['ratio'])]
    mean_ratio = np.mean(ratios) if ratios else np.nan
    print(f"    Mean ratio (near/far): {mean_ratio:.3f}")
    print(f"    Prediction: ratio < 1 (smaller residuals near predicted levels)")
    supported = mean_ratio < 0.9 if np.isfinite(mean_ratio) else False
    print(f"    Result: {'SUPPORTED' if supported else 'NOT SUPPORTED'}")

    return results, fdm_all, res_all


# ══════════════════════════════════════════════════════════════════
# TEST 4: Plateau Duration (Dwell Time)
# ══════════════════════════════════════════════════════════════════
def run_test4(data, selected):
    """Test 4: Do galaxies spend more radial extent near predicted levels
    than expected from a smooth rise?"""
    print("\n  TEST 4: Plateau duration (dwell time at predicted levels)")
    print("  " + "-" * 50)

    window = 0.02  # half-width: f_DM within ± window of predicted level

    # For each galaxy, measure fraction of radial extent spent near
    # each predicted level
    dwell_fractions = {N: [] for N in range(3, 9)}

    for gname in selected:
        pts = sorted(data[gname], key=lambda p: p['R_Rd'])
        pts = [p for p in pts if p['R_Rd'] > 0.5]
        if len(pts) < 8:
            continue

        R = np.array([p['R_Rd'] for p in pts])
        f = np.array([p['fDM'] for p in pts])
        total_dR = R[-1] - R[0]
        if total_dR <= 0:
            continue

        for N in range(3, 9):
            level = (N-1)/N
            near = np.abs(f - level) < window
            # Sum radial intervals spent near this level
            dR_near = 0.0
            for i in range(len(R) - 1):
                if near[i] and near[i+1]:
                    dR_near += R[i+1] - R[i]
                elif near[i] or near[i+1]:
                    dR_near += 0.5 * (R[i+1] - R[i])
            dwell_fractions[N].append(dR_near / total_dR)

    # Compare observed dwell to null expectation
    # Null: for a smooth monotonic rise from f_low to f_high,
    # the fraction of time near any particular value is ~ 2*window / (f_high - f_low)
    # Typical f range: 0.2 to 0.85, so null fraction ~ 0.04/0.65 ~ 0.06

    print(f"    Window: ±{window}")
    excess_count = 0
    total_tests = 0
    for N in range(3, 9):
        level = (N-1)/N
        fracs = np.array(dwell_fractions[N])
        fracs = fracs[fracs > 0]  # Only galaxies that pass through this level
        if len(fracs) < 3:
            print(f"    N={N} (f={level:.3f}): <3 galaxies pass through")
            continue

        med_dwell = np.median(fracs)
        # Null expectation for smooth rise
        null_dwell = 2 * window / 0.6  # approximate
        excess = med_dwell / null_dwell if null_dwell > 0 else np.nan
        total_tests += 1
        if excess > 1.5:
            excess_count += 1

        print(f"    N={N} (f={level:.3f}): {len(fracs)} gals, "
              f"median dwell={med_dwell:.3f}, "
              f"null≈{null_dwell:.3f}, "
              f"excess={excess:.2f}x")

    print(f"    Levels with >1.5x excess dwell: {excess_count}/{total_tests}")
    supported = excess_count >= 2
    print(f"    Result: {'SUPPORTED' if supported else 'NOT SUPPORTED'}")

    return dwell_fractions


# ══════════════════════════════════════════════════════════════════
# TEST 5: Transition Sharpness
# ══════════════════════════════════════════════════════════════════
def run_test5(data, selected, meta):
    """Test 5: When f_DM increases by ~Δf = predicted step, how sharp
    is the transition (narrow radial interval = step; wide = smooth)?"""
    print("\n  TEST 5: Transition sharpness")
    print("  " + "-" * 50)

    # For each galaxy, find radial intervals where f_DM increases by
    # approximately a predicted step size
    transition_widths = []  # (galaxy, N_transition, delta_R_Rd, delta_fDM)

    for gname in selected:
        pts = sorted(data[gname], key=lambda p: p['R_Rd'])
        pts = [p for p in pts if p['R_Rd'] > 0.5 and np.isfinite(p['fDM'])]
        if len(pts) < 8:
            continue

        R = np.array([p['R_Rd'] for p in pts])
        f = np.array([p['fDM'] for p in pts])

        # Scan for transitions: look at all pairs of points within
        # a limited radial range (1-3 R/Rd apart)
        for i in range(len(R)):
            for j in range(i+1, min(i+6, len(R))):
                dR = R[j] - R[i]
                df = f[j] - f[i]
                if dR <= 0 or df <= 0.02:
                    continue
                # Is this transition consistent with a predicted step?
                for N in range(3, 8):
                    expected_step = PREDICTED_STEPS[N]
                    if abs(df - expected_step) < 0.015:
                        # Sharpness = df / dR (steeper = sharper)
                        sharpness = df / dR
                        transition_widths.append({
                            'Galaxy': gname,
                            'N': N,
                            'R_start': R[i],
                            'R_end': R[j],
                            'dR': dR,
                            'df': df,
                            'sharpness': sharpness,
                        })

    print(f"    Candidate transitions found: {len(transition_widths)}")

    if len(transition_widths) == 0:
        print("    Result: NO TRANSITIONS FOUND — NOT SUPPORTED")
        return transition_widths

    # Compare sharpness to null expectation
    # For a smooth NFW/ISO profile, the gradient df/d(R/Rd) is typically
    # 0.02-0.05 per R/Rd unit. Steps should be sharper.
    sharpness_vals = [t['sharpness'] for t in transition_widths]
    dR_vals = [t['dR'] for t in transition_widths]
    med_sharpness = np.median(sharpness_vals)
    med_dR = np.median(dR_vals)

    # Generate null: from smooth spline derivatives, what is the
    # typical df/dR?
    null_gradients = []
    for gname in selected:
        pts = sorted(data[gname], key=lambda p: p['R_Rd'])
        pts = [p for p in pts if p['R_Rd'] > 1.0]
        if len(pts) < 5:
            continue
        R = np.array([p['R_Rd'] for p in pts])
        f = np.array([p['fDM'] for p in pts])
        for i in range(len(R)-1):
            dR = R[i+1] - R[i]
            if dR > 0:
                null_gradients.append((f[i+1] - f[i]) / dR)
    null_gradients = np.array(null_gradients)
    null_gradients = null_gradients[null_gradients > 0]  # only positive gradients
    med_null_gradient = np.median(null_gradients) if len(null_gradients) > 0 else 0

    print(f"    Median transition sharpness: {med_sharpness:.4f} per R/Rd")
    print(f"    Median transition width: {med_dR:.2f} R/Rd")
    print(f"    Median overall positive gradient: {med_null_gradient:.4f} per R/Rd")
    sharp_ratio = med_sharpness / med_null_gradient if med_null_gradient > 0 else np.nan
    print(f"    Sharpness ratio (transition/overall): {sharp_ratio:.2f}")
    print(f"    Prediction: ratio >> 1 (transitions much sharper than smooth)")

    # Count by N
    n_by_N = defaultdict(int)
    for t in transition_widths:
        n_by_N[t['N']] += 1
    for N in sorted(n_by_N.keys()):
        print(f"      N={N}→{N+1}: {n_by_N[N]} candidates")

    supported = sharp_ratio > 2.0
    print(f"    Result: {'SUPPORTED' if supported else 'NOT SUPPORTED'}")

    return transition_widths


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    print("Step 5: Statistical Tests for Radial Steps")
    print("=" * 60)

    data = load_profiles(PROFILES_CSV)
    meta = load_catalog(CATALOG_CSV)
    nulls = load_nulls(NULLS_CSV)
    selected = select_galaxies(data, meta)

    dwarfs = [g for g in selected if classify(meta, g) == 'dwarf']
    intermediate = [g for g in selected if classify(meta, g) == 'intermediate']
    massive = [g for g in selected if classify(meta, g) == 'massive']

    print(f"Selected: {len(selected)} galaxies")

    # Run all 5 tests
    t1_results, t1_frac = run_test1(data, selected, meta)
    t2_derivs, t2_kurt, t2_frac_zero, t2_pval = run_test2(data, selected)
    t3_results, t3_fdm, t3_res = run_test3(nulls, selected)
    t4_dwell = run_test4(data, selected)
    t5_trans = run_test5(data, selected, meta)

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: OVERALL RESULTS")
    print("=" * 60)

    # Prediction A sub-tests:
    t1_pass = t1_frac > 0.25
    t2_pass = t2_kurt > 1.0 and t2_pval < 0.05
    t3_ratios = [r['ratio'] for r in t3_results.values() if np.isfinite(r['ratio'])]
    t3_pass = np.mean(t3_ratios) < 0.9 if t3_ratios else False
    t4_excesses = []
    for N in range(3, 9):
        fracs = np.array(t4_dwell[N])
        fracs = fracs[fracs > 0]
        if len(fracs) >= 3:
            null_dwell = 2 * 0.02 / 0.6
            excess = np.median(fracs) / null_dwell
            t4_excesses.append(excess)
    t4_pass = sum(1 for e in t4_excesses if e > 1.5) >= 2
    t5_pass = False
    if t5_trans:
        sharpness_vals = [t['sharpness'] for t in t5_trans]
        null_g = []
        for gname in selected:
            pts = sorted(data[gname], key=lambda p: p['R_Rd'])
            pts = [p for p in pts if p['R_Rd'] > 1.0]
            if len(pts) < 5: continue
            R = np.array([p['R_Rd'] for p in pts])
            f = np.array([p['fDM'] for p in pts])
            for i in range(len(R)-1):
                dR = R[i+1] - R[i]
                if dR > 0:
                    null_g.append((f[i+1] - f[i]) / dR)
        null_g = [g for g in null_g if g > 0]
        if null_g:
            t5_pass = np.median(sharpness_vals) / np.median(null_g) > 2.0

    tests_passed = sum([t1_pass, t2_pass, t3_pass, t4_pass, t5_pass])

    print(f"\n  Test 1 (BIC: piecewise > smooth for >25%): "
          f"{'PASS' if t1_pass else 'FAIL'} ({t1_frac:.1%})")
    print(f"  Test 2 (Derivative bimodality):             "
          f"{'PASS' if t2_pass else 'FAIL'} (kurt={t2_kurt:.2f}, p={t2_pval:.3f})")
    print(f"  Test 3 (Residuals smaller at (N-1)/N):      "
          f"{'PASS' if t3_pass else 'FAIL'} (ratio={np.mean(t3_ratios):.3f})")
    print(f"  Test 4 (Excess dwell at ≥2 levels):         "
          f"{'PASS' if t4_pass else 'FAIL'}")
    print(f"  Test 5 (Transitions sharper than smooth):    "
          f"{'PASS' if t5_pass else 'FAIL'}")
    print(f"\n  Tests passed: {tests_passed}/5")
    print(f"  Prediction A (staircase morphology): "
          f"{'SUPPORTED' if tests_passed >= 3 else 'NOT SUPPORTED'}")

    # ── Figure ────────────────────────────────────────────────────
    print(f"\nGenerating {OUTPUT_MAIN}...")

    fig = plt.figure(figsize=(18, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle('Step 5: Statistical Tests for Radial Steps',
                 fontsize=15, fontweight='bold', y=0.98)

    # Panel (0,0): Test 1 — ΔBIC distribution
    ax = fig.add_subplot(gs[0, 0])
    delta_bics = [r['delta_BIC'] for r in t1_results]
    colors_bic = ['green' if d < 0 else 'red' for d in delta_bics]
    ax.hist([d for d in delta_bics if d < 0], bins=30, color='green',
            alpha=0.7, label=f'PW preferred ({sum(1 for d in delta_bics if d<0)})')
    ax.hist([d for d in delta_bics if d >= 0], bins=30, color='red',
            alpha=0.7, label=f'Smooth preferred ({sum(1 for d in delta_bics if d>=0)})')
    ax.axvline(0, color='black', ls='-', lw=1)
    ax.set_xlabel('ΔBIC (piecewise − smooth)')
    ax.set_ylabel('Count')
    ax.set_title(f'Test 1: BIC comparison ({t1_frac:.0%} PW preferred)')
    ax.legend(fontsize=7)

    # Panel (0,1): Test 1 by mass class
    ax = fig.add_subplot(gs[0, 1])
    for cls, color in [('dwarf', 'blue'), ('intermediate', 'green'),
                        ('massive', 'red')]:
        dbics = [r['delta_BIC'] for r in t1_results if r['class'] == cls]
        frac = sum(1 for d in dbics if d < 0) / len(dbics) if dbics else 0
        ax.hist(dbics, bins=20, alpha=0.4, color=color,
                label=f'{cls} ({frac:.0%} PW)')
    ax.axvline(0, color='black', ls='-', lw=1)
    ax.set_xlabel('ΔBIC')
    ax.set_ylabel('Count')
    ax.set_title('Test 1 by mass class')
    ax.legend(fontsize=7)

    # Panel (0,2): Test 2 — Derivative distribution + Gaussian overlay
    ax = fig.add_subplot(gs[0, 2])
    clip = np.percentile(np.abs(t2_derivs), 98)
    d_clip = t2_derivs[np.abs(t2_derivs) < clip]
    ax.hist(d_clip, bins=80, density=True, color='steelblue',
            edgecolor='white', alpha=0.8, label='Observed')
    # Overlay Gaussian
    x_g = np.linspace(-clip, clip, 200)
    ax.plot(x_g, norm.pdf(x_g, np.mean(d_clip), np.std(d_clip)),
            'r-', lw=1.5, label='Gaussian fit')
    ax.axvline(0, color='black', ls='-', lw=0.5)
    ax.set_xlabel('df$_{DM}$/d(R/R$_d$)')
    ax.set_ylabel('Density')
    ax.set_title(f'Test 2: Derivatives (kurt={t2_kurt:.2f})')
    ax.legend(fontsize=7)

    # Panel (1,0): Test 3 — Residual structure at predicted levels
    ax = fig.add_subplot(gs[1, 0])
    fDM_bins = np.arange(0.3, 1.01, 0.02)
    fDM_centers = 0.5 * (fDM_bins[:-1] + fDM_bins[1:])
    med_res = []
    for j in range(len(fDM_centers)):
        mask = (t3_fdm >= fDM_bins[j]) & (t3_fdm < fDM_bins[j+1])
        if mask.sum() >= 3:
            med_res.append(np.mean(np.abs(t3_res[mask])))
        else:
            med_res.append(np.nan)
    med_res = np.array(med_res)
    m = ~np.isnan(med_res)
    ax.bar(fDM_centers[m], med_res[m], width=0.018, color='steelblue', alpha=0.7)
    for N in range(3, 9):
        ax.axvline((N-1)/N, color='red', ls='--', alpha=0.5, lw=1)
        ax.text((N-1)/N, ax.get_ylim()[1]*0.95 if ax.get_ylim()[1] > 0 else 0.02,
                f'{N}', fontsize=8, color='red', ha='center', va='top')
    ax.set_xlabel('f$_{DM,obs}$')
    ax.set_ylabel('Mean |spline residual|')
    ax.set_title('Test 3: Residual magnitude at predicted levels')

    # Panel (1,1): Test 4 — Dwell fractions
    ax = fig.add_subplot(gs[1, 1])
    Ns = list(range(3, 9))
    med_dwells = []
    for N in Ns:
        fracs = np.array(t4_dwell[N])
        fracs = fracs[fracs > 0]
        med_dwells.append(np.median(fracs) if len(fracs) >= 3 else 0)
    null_dwell = 2 * 0.02 / 0.6
    ax.bar([f'{N}' for N in Ns], med_dwells, color='steelblue', alpha=0.8)
    ax.axhline(null_dwell, color='red', ls='--', lw=1.5,
               label=f'Smooth null ≈ {null_dwell:.3f}')
    ax.set_xlabel('N (predicted plateau level)')
    ax.set_ylabel('Median dwell fraction')
    ax.set_title('Test 4: Dwell time at predicted levels')
    ax.legend(fontsize=8)

    # Panel (1,2): Test 5 — Transition sharpness histogram
    ax = fig.add_subplot(gs[1, 2])
    if t5_trans:
        sharp = [t['sharpness'] for t in t5_trans]
        ax.hist(sharp, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
        ax.axvline(np.median(sharp), color='red', ls='--',
                   label=f'Median = {np.median(sharp):.3f}')
        ax.set_xlabel('Transition sharpness (df/d(R/R$_d$))')
        ax.set_ylabel('Count')
        ax.set_title(f'Test 5: Transition sharpness (N={len(t5_trans)})')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No transitions found', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title('Test 5: Transition sharpness')

    # Panel (2,0): Example galaxy with best PW fit overlaid
    ax = fig.add_subplot(gs[2, 0])
    # Find galaxy with strongest PW preference
    best_pw = sorted(t1_results, key=lambda r: r['delta_BIC'])
    if best_pw and best_pw[0]['delta_BIC'] < 0:
        gname = best_pw[0]['Galaxy']
        pts = sorted(data[gname], key=lambda p: p['R_Rd'])
        pts = [p for p in pts if p['R_Rd'] > 0.5]
        R = np.array([p['R_Rd'] for p in pts])
        f = np.array([p['fDM'] for p in pts])
        e = np.array([p['e_fDM'] for p in pts])
        ax.errorbar(R, f, yerr=e, fmt='o', ms=3, color='black', capsize=1.5)
        # Show PW fit
        best_steps = best_pw[0]['pw_steps']
        _, pw_fit = fit_piecewise_constant(R, f, e, n_steps=best_steps)
        if pw_fit is not None:
            ax.plot(R, pw_fit, 'r-', lw=2, label=f'PW ({best_steps} steps)')
        # Show spline
        _, sp_fit = fit_smooth_model(R, f, e)
        if sp_fit is not None:
            ax.plot(R, sp_fit, 'g-', lw=1.5, label='Spline', alpha=0.7)
        for N in range(3, 9):
            ax.axhline((N-1)/N, color='orange', ls='--', alpha=0.2, lw=0.5)
        vf = meta[gname]['Vflat']
        ax.set_title(f'Best PW galaxy: {gname} (Vf={vf:.0f}, '
                     f'ΔBIC={best_pw[0]["delta_BIC"]:.1f})')
        ax.set_xlabel('R / R$_d$')
        ax.set_ylabel('f$_{DM}$')
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, 'No galaxy prefers PW', ha='center', va='center',
                transform=ax.transAxes)

    # Panel (2,1): Example galaxy with worst PW fit (smooth clearly better)
    ax = fig.add_subplot(gs[2, 1])
    worst_pw = sorted(t1_results, key=lambda r: -r['delta_BIC'])
    if worst_pw:
        gname = worst_pw[0]['Galaxy']
        pts = sorted(data[gname], key=lambda p: p['R_Rd'])
        pts = [p for p in pts if p['R_Rd'] > 0.5]
        R = np.array([p['R_Rd'] for p in pts])
        f = np.array([p['fDM'] for p in pts])
        e = np.array([p['e_fDM'] for p in pts])
        ax.errorbar(R, f, yerr=e, fmt='o', ms=3, color='black', capsize=1.5)
        _, sp_fit = fit_smooth_model(R, f, e)
        if sp_fit is not None:
            ax.plot(R, sp_fit, 'g-', lw=1.5, label='Spline')
        for N in range(3, 9):
            ax.axhline((N-1)/N, color='orange', ls='--', alpha=0.2, lw=0.5)
        vf = meta[gname]['Vflat']
        ax.set_title(f'Worst PW galaxy: {gname} (Vf={vf:.0f}, '
                     f'ΔBIC={worst_pw[0]["delta_BIC"]:.1f})')
        ax.set_xlabel('R / R$_d$')
        ax.set_ylabel('f$_{DM}$')
        ax.legend(fontsize=7)

    # Panel (2,2): Summary
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')

    summary = (
        f"STEP 5: TEST RESULTS SUMMARY\n"
        f"{'='*44}\n\n"
        f"Test 1 (BIC: PW > smooth >25%):\n"
        f"  {'PASS' if t1_pass else 'FAIL'} — "
        f"{t1_frac:.1%} prefer piecewise\n\n"
        f"Test 2 (Derivative bimodality):\n"
        f"  {'PASS' if t2_pass else 'FAIL'} — "
        f"kurtosis={t2_kurt:.2f}, p={t2_pval:.3f}\n\n"
        f"Test 3 (Residuals at (N-1)/N):\n"
        f"  {'PASS' if t3_pass else 'FAIL'} — "
        f"near/far ratio={np.mean(t3_ratios):.3f}\n\n"
        f"Test 4 (Dwell time excess):\n"
        f"  {'PASS' if t4_pass else 'FAIL'}\n\n"
        f"Test 5 (Transition sharpness):\n"
        f"  {'PASS' if t5_pass else 'FAIL'}\n\n"
        f"{'='*44}\n"
        f"Tests passed: {tests_passed}/5\n\n"
        f"PREDICTION A (staircase):\n"
        f"  {'SUPPORTED' if tests_passed >= 3 else 'NOT SUPPORTED'}\n"
        f"  (requires ≥3/5 tests passing)"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=9.5, fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.savefig(OUTPUT_MAIN, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {OUTPUT_MAIN}")

    # ── Save results CSV ──────────────────────────────────────────
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Test', 'Metric', 'Value', 'Threshold', 'Pass'])
        writer.writerow(['T1_BIC', 'frac_PW_preferred', f'{t1_frac:.4f}', '>0.25', t1_pass])
        writer.writerow(['T2_Deriv', 'kurtosis', f'{t2_kurt:.4f}', '>1.0', t2_kurt > 1.0])
        writer.writerow(['T2_Deriv', 'p_excess_zero', f'{t2_pval:.4f}', '<0.05', t2_pval < 0.05])
        writer.writerow(['T3_Resid', 'mean_near_far_ratio', f'{np.mean(t3_ratios):.4f}', '<0.9', t3_pass])
        writer.writerow(['T4_Dwell', 'levels_with_excess', f'{sum(1 for e in t4_excesses if e > 1.5)}',
                         '>=2', t4_pass])
        writer.writerow(['T5_Sharp', 'sharpness_ratio',
                         f'{np.median([t["sharpness"] for t in t5_trans]) / np.median([g for g in null_g if g > 0]):.4f}' if t5_trans and null_g else 'N/A',
                         '>2.0', t5_pass])
        writer.writerow(['OVERALL', 'tests_passed', f'{tests_passed}', '>=3',
                         tests_passed >= 3])

    print(f"  Saved {OUTPUT_CSV}")
    print("\nStep 5 complete.")


if __name__ == '__main__':
    main()
