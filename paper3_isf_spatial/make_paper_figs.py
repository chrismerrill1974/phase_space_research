#!/usr/bin/env python3
"""
Generate paper figures for the radial f_DM staircase test.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
from scipy.stats import kurtosis, norm
import csv
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

PROFILES_CSV = "sparc_radial_profiles.csv"
CATALOG_CSV  = "sparc_dm_fractions.csv"
NULLS_CSV    = "sparc_radial_nulls.csv"


def load_profiles(filepath):
    data = defaultdict(list)
    with open(filepath, 'r') as f:
        for row in csv.DictReader(f):
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
                'Rdisk': float(row['Rdisk']),
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
                'Npts': int(row['Npts']),
            }
    return meta


def load_nulls(filepath):
    nulls = defaultdict(list)
    with open(filepath, 'r') as f:
        for row in csv.DictReader(f):
            gname = row['Galaxy']
            nulls[gname].append({
                'R_Rd': float(row['R_Rd']),
                'fDM_obs': float(row['fDM_obs']),
                'e_fDM': float(row['e_fDM']),
                'fDM_spline': float(row['fDM_spline']) if row['fDM_spline'] else np.nan,
                'fDM_nfw': float(row['fDM_nfw']) if row['fDM_nfw'] else np.nan,
                'fDM_iso': float(row['fDM_iso']) if row['fDM_iso'] else np.nan,
            })
    return nulls


def select(data, meta):
    s = []
    for g, pts in data.items():
        if g not in meta: continue
        m = meta[g]
        if m['Q'] > 1 or m['Vflat'] <= 0: continue
        if len(pts) < 10: continue
        Rmax = max(p['R_kpc'] for p in pts)
        Rd = m['Rdisk']
        if Rd <= 0 or Rmax / Rd < 3.0: continue
        s.append(g)
    return sorted(s)


def stack_profiles(data, gnames, R_bins):
    Rc = 0.5 * (R_bins[:-1] + R_bins[1:])
    med = np.full(len(Rc), np.nan)
    q25 = np.full(len(Rc), np.nan)
    q75 = np.full(len(Rc), np.nan)
    for j in range(len(Rc)):
        vals = [p['fDM'] for g in gnames for p in data[g]
                if R_bins[j] <= p['R_Rd'] < R_bins[j+1]]
        if len(vals) >= 3:
            med[j] = np.median(vals)
            q25[j] = np.percentile(vals, 25)
            q75[j] = np.percentile(vals, 75)
    return Rc, med, q25, q75


data = load_profiles(PROFILES_CSV)
meta = load_catalog(CATALOG_CSV)
nulls = load_nulls(NULLS_CSV)
selected = select(data, meta)

dwarfs = [g for g in selected if meta[g]['Vflat'] < 80]
interm = [g for g in selected if 80 <= meta[g]['Vflat'] < 150]
massive = [g for g in selected if meta[g]['Vflat'] >= 150]

R_bins = np.arange(0, 16.0, 0.5)

# ══════════════════════════════════════════════════════════════════
# FIGURE 1: Data overview (2 panels)
# ══════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# Left: All profiles overlaid colored by Vflat
for gname in selected:
    pts = [(p['R_Rd'], p['fDM']) for p in data[gname] if p['R_Rd'] > 0]
    if not pts: continue
    rr, ff = zip(*pts)
    vf = meta[gname]['Vflat']
    color = plt.cm.viridis(np.clip((vf - 30) / 270, 0, 1))
    ax1.plot(rr, ff, '-', lw=0.4, alpha=0.35, color=color)
for N in range(3, 9):
    ax1.axhline((N-1)/N, color='red', ls='--', alpha=0.35, lw=0.7)
    ax1.text(14.3, (N-1)/N, f'$N$={N}', fontsize=7, color='red',
             va='center', alpha=0.7)
ax1.set_xlabel('$R / R_d$', fontsize=10)
ax1.set_ylabel('$f_\\mathrm{DM}(R)$', fontsize=10)
ax1.set_title(f'All profiles ($N_\\mathrm{{gal}}$ = {len(selected)}, color = $V_\\mathrm{{flat}}$)',
              fontsize=10)
ax1.set_xlim(0, 15)
ax1.set_ylim(-0.5, 1.08)
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(30, 300))
sm.set_array([])
plt.colorbar(sm, ax=ax1, label='$V_\\mathrm{flat}$ (km/s)', shrink=0.85)

# Right: Stacked median profiles by mass class
for label, gnames, color in [
        (f'Dwarfs ($N$={len(dwarfs)})', dwarfs, 'blue'),
        (f'Interm. ($N$={len(interm)})', interm, 'green'),
        (f'Massive ($N$={len(massive)})', massive, 'red')]:
    Rc, med, q2, q7 = stack_profiles(data, gnames, R_bins)
    m = ~np.isnan(med)
    ax2.plot(Rc[m], med[m], 'o-', color=color, ms=2.5, lw=1.5, label=label)
    ax2.fill_between(Rc[m], q2[m], q7[m], alpha=0.12, color=color)
for N in range(3, 9):
    ax2.axhline((N-1)/N, color='red', ls='--', alpha=0.3, lw=0.6)
    ax2.text(0.3, (N-1)/N + 0.005, f'$N$={N}', fontsize=7, color='red', alpha=0.6)
ax2.set_xlabel('$R / R_d$', fontsize=10)
ax2.set_ylabel('Median $f_\\mathrm{DM}(R)$', fontsize=10)
ax2.set_title('Stacked profiles by mass class', fontsize=10)
ax2.set_xlim(0, 12)
ax2.set_ylim(-0.3, 1.05)
ax2.legend(fontsize=7, loc='lower right')

plt.tight_layout()
plt.savefig('fig1_data_overview.pdf', bbox_inches='tight')
plt.savefig('fig1_data_overview.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig1_data_overview")

# ══════════════════════════════════════════════════════════════════
# FIGURE 2: Test results (4 panels)
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# (0,0): Test 1 — BIC comparison summary by mass class
ax = axes[0, 0]
classes = ['Dwarfs\n(15)', 'Interm.\n(27)', 'Massive\n(34)', 'All\n(76)']
pw_fracs = [20.0, 11.1, 0.0, 7.9]
ax.bar(classes, pw_fracs, color=['blue', 'green', 'red', 'purple'], alpha=0.7)
ax.axhline(25, color='red', ls='--', lw=1.5, label='25% threshold')
ax.set_ylabel('% preferring piecewise', fontsize=10)
ax.set_title('Test 1: BIC model comparison', fontsize=10)
ax.legend(fontsize=8)
ax.set_ylim(0, 35)

# (0,1): Test 2 — Derivative distribution
ax = axes[0, 1]
all_derivs = []
for gname in selected:
    pts = sorted(data[gname], key=lambda p: p['R_Rd'])
    pts = [p for p in pts if p['R_Rd'] > 1.0]
    if len(pts) < 5: continue
    R = np.array([p['R_Rd'] for p in pts])
    f = np.array([p['fDM'] for p in pts])
    dfdR = np.zeros(len(R))
    for i in range(1, len(R)-1):
        dR = R[i+1] - R[i-1]
        if dR > 0:
            dfdR[i] = (f[i+1] - f[i-1]) / dR
    all_derivs.extend(dfdR[1:-1])
all_derivs = np.array(all_derivs)
all_derivs = all_derivs[np.isfinite(all_derivs)]
clip = np.percentile(np.abs(all_derivs), 98)
d_clip = all_derivs[np.abs(all_derivs) < clip]
ax.hist(d_clip, bins=80, density=True, color='steelblue', edgecolor='white',
        alpha=0.8, label='Observed')
x_g = np.linspace(-clip, clip, 200)
ax.plot(x_g, norm.pdf(x_g, np.mean(d_clip), np.std(d_clip)),
        'r-', lw=1.5, label='Gaussian fit')
ax.axvline(0, color='black', ls='-', lw=0.5)
kurt = kurtosis(d_clip, fisher=True)
ax.text(0.97, 0.95, f'$\\kappa$ = {kurt:.2f}', transform=ax.transAxes,
        fontsize=10, ha='right', va='top',
        bbox=dict(facecolor='white', alpha=0.8))
ax.set_xlabel('$df_\\mathrm{DM}/d(R/R_d)$', fontsize=10)
ax.set_ylabel('Density', fontsize=10)
ax.set_title('Test 2: Derivative distribution', fontsize=10)
ax.legend(fontsize=8)

# (1,0): Corrected Test 4 — Dwell time vs smooth null
ax = axes[1, 0]
# Run corrected dwell
np.random.seed(42)
window = 0.02
obs_dwells = {N: [] for N in range(3, 9)}
null_dwells = {N: [] for N in range(3, 9)}
for gname in selected:
    pts = sorted(data[gname], key=lambda p: p['R_Rd'])
    pts = [p for p in pts if p['R_Rd'] > 0.5]
    if len(pts) < 8: continue
    R = np.array([p['R_Rd'] for p in pts])
    f = np.array([p['fDM'] for p in pts])
    e = np.array([p['e_fDM'] for p in pts])
    total_dR = R[-1] - R[0]
    if total_dR <= 0: continue
    for N in range(3, 9):
        level = (N-1)/N
        near = np.abs(f - level) < window
        dR_near = sum(R[i+1]-R[i] for i in range(len(R)-1)
                     if near[i] and near[i+1])
        obs_dwells[N].append(dR_near / total_dR)
    w = 1.0 / np.maximum(e, 0.01)
    try:
        spline = UnivariateSpline(R, f, w=w, s=len(R)*1.0, k=3)
        f_smooth = spline(R)
    except: continue
    for _ in range(3):
        f_sim = f_smooth + np.random.normal(0, e)
        for N in range(3, 9):
            level = (N-1)/N
            near = np.abs(f_sim - level) < window
            dR_near = sum(R[i+1]-R[i] for i in range(len(R)-1)
                         if near[i] and near[i+1])
            null_dwells[N].append(dR_near / total_dR)

Ns = list(range(3, 9))
obs_v = [np.median(obs_dwells[N]) for N in Ns]
null_v = [np.median(null_dwells[N]) for N in Ns]
x = np.arange(len(Ns))
w_bar = 0.35
ax.bar(x - w_bar/2, obs_v, w_bar, label='Observed', color='steelblue', alpha=0.8)
ax.bar(x + w_bar/2, null_v, w_bar, label='Smooth null', color='salmon', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f'$N$={N}' for N in Ns])
ax.set_ylabel('Median dwell fraction', fontsize=10)
ax.set_title('Test 4 (corrected): Dwell time', fontsize=10)
ax.legend(fontsize=8)

# (1,1): Robustness — T1 across variations
ax = axes[1, 1]
var_names = ['Fiducial', '$\\Upsilon_\\star$=0.3', '$\\Upsilon_\\star$=0.7',
             '$Q \\leq 2$', '$R/R_d \\geq 5$', '$R/R_d \\geq 7$',
             '$N_\\mathrm{pts} \\geq 15$', '$N_\\mathrm{pts} \\geq 20$',
             'Dwarfs', 'Interm.', 'Massive', 'No bulge', 'Bulge']
t1_vals = [7.9, 5.3, 9.2, 7.8, 3.5, 2.4, 3.6, 2.7, 20.0, 11.1, 0.0, 10.5, 0.0]
y_pos = range(len(var_names))
ax.barh(y_pos, t1_vals, color='steelblue', alpha=0.8)
ax.axvline(25, color='red', ls='--', lw=1.5, label='Threshold')
ax.set_yticks(y_pos)
ax.set_yticklabels(var_names, fontsize=7)
ax.set_xlabel('% PW preferred', fontsize=10)
ax.set_title('Robustness: Test 1 across variations', fontsize=10)
ax.legend(fontsize=7)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('fig2_test_results.pdf', bbox_inches='tight')
plt.savefig('fig2_test_results.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig2_test_results")

# ══════════════════════════════════════════════════════════════════
# FIGURE 3: Example galaxy fits (6 panels)
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(12, 7))

examples = []
# Pick 2 massive, 2 intermediate, 2 dwarfs with most points
for glist in [massive, interm, dwarfs]:
    sorted_g = sorted(glist, key=lambda g: len(data[g]), reverse=True)
    examples.extend(sorted_g[:2])

for idx, gname in enumerate(examples):
    row, col = divmod(idx, 3)
    ax = axes[row][col]

    # Get null model fits
    if gname in nulls:
        npts = nulls[gname]
        R_n = np.array([p['R_Rd'] for p in npts])
        f_obs = np.array([p['fDM_obs'] for p in npts])
        e_obs = np.array([p['e_fDM'] for p in npts])
        f_nfw = np.array([p['fDM_nfw'] for p in npts])
        f_iso = np.array([p['fDM_iso'] for p in npts])
        f_spl = np.array([p['fDM_spline'] for p in npts])
    else:
        pts = sorted(data[gname], key=lambda p: p['R_Rd'])
        R_n = np.array([p['R_Rd'] for p in pts])
        f_obs = np.array([p['fDM'] for p in pts])
        e_obs = np.array([p['e_fDM'] for p in pts])
        f_nfw = f_iso = f_spl = np.full(len(R_n), np.nan)

    ax.errorbar(R_n, f_obs, yerr=e_obs, fmt='o', ms=2.5, color='black',
                capsize=1, elinewidth=0.4, zorder=5, label='Observed')
    m_nfw = np.isfinite(f_nfw)
    m_iso = np.isfinite(f_iso)
    m_spl = np.isfinite(f_spl)
    if np.any(m_nfw):
        ax.plot(R_n[m_nfw], f_nfw[m_nfw], '-', color='red', lw=1.2, label='NFW', alpha=0.7)
    if np.any(m_iso):
        ax.plot(R_n[m_iso], f_iso[m_iso], '-', color='blue', lw=1.2, label='ISO', alpha=0.7)
    if np.any(m_spl):
        ax.plot(R_n[m_spl], f_spl[m_spl], '-', color='green', lw=1.2, label='Spline', alpha=0.7)

    for N in range(3, 9):
        ax.axhline((N-1)/N, color='orange', ls='--', alpha=0.2, lw=0.5)

    vf = meta[gname]['Vflat']
    cls = 'D' if vf < 80 else ('I' if vf < 150 else 'M')
    ax.set_title(f'{gname} ($V_f$={vf:.0f}, {cls})', fontsize=9)
    ax.set_ylim(-0.5, 1.08)
    ax.set_xlabel('$R / R_d$', fontsize=9)
    if col == 0:
        ax.set_ylabel('$f_\\mathrm{DM}$', fontsize=9)
    ax.tick_params(labelsize=7)
    if idx == 0:
        ax.legend(fontsize=6, loc='lower right')

plt.tight_layout()
plt.savefig('fig3_example_fits.pdf', bbox_inches='tight')
plt.savefig('fig3_example_fits.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig3_example_fits")

print("All paper figures complete.")
