#!/usr/bin/env python3
"""
Paper X — Step 5: the outer slopes of Milky-Way-mass spirals in SPARC

Peer review (2026-10-10, finding 3) asked that the present-day tension
between the predicted declining outer curves and external spirals be
quantified rather than deferred. This script measures, for the SPARC
galaxies that resemble the Milky Way in rotation speed, the same
quantity the paper predicts: beta = d ln v_c / d ln R over 15-30 kpc,
by least squares on the observed rotation curve points in that range.

Selection (stated before looking): quality flag Q <= 2, inclination
>= 30 deg, asymptotically flat speed V_flat in [180, 260] km/s (the
Milky Way's ~230), and at least three observed points in 15-30 kpc
spanning at least 8 kpc. A second, scaled selection uses the same
galaxies with points in 6-13 disc scale lengths (the Milky Way's 15-30
kpc at R_d = 2.35 kpc), to guard against the physical-radius choice.

Reported: per-galaxy beta with its formal error, the sample median and
16-84 percentile range, and the fraction of galaxies with beta below
each predicted value. The Paper X predictions (-0.42, -0.30, -0.22,
-0.17 at N = 4, 6, 8, 10) and the NFW reference are printed beside the
data for comparison; nothing is fitted.

Inputs:  ../data/sparc/SPARC_Lelli2016c.mrt (Table 1: galaxy properties)
         ../data/sparc/MassModels_Lelli2016c.mrt (rotation curves)
         (Lelli, McGaugh & Schombert 2016; copied from the series' data)
Output:  ../data/sparc_mw_mass_slopes.npz
Runtime: under a second.
"""

import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA = os.path.join(ROOT, 'data')
OUTPATH = os.path.join(DATA, 'sparc_mw_mass_slopes.npz')

def tokens(path, n_min):
    """Whitespace-split rows of a CDS table with at least n_min fields whose
    second field is numeric (the data lines; names are right-justified in
    Table 1, so byte positions cannot be trusted)."""
    out = []
    for l in open(path, encoding='utf-8', errors='replace'):
        t = l.split()
        if len(t) >= n_min:
            try:
                float(t[1]); float(t[2])
            except ValueError:
                continue
            out.append(t)
    return out

# ── Table 1: galaxy properties (columns in header order) ───────────
gal = {}
for t in tokens(os.path.join(DATA, 'sparc', 'SPARC_Lelli2016c.mrt'), 18):
    gal[t[0]] = dict(T=int(t[1]), D=float(t[2]), e_D=float(t[3]), inc=float(t[5]), Rdisk=float(t[11]),
                     Vflat=float(t[15]), e_Vflat=float(t[16]), Q=int(t[17]))

# ── Mass models: rotation curves ────────────────────────────────────
rc = {}
for t in tokens(os.path.join(DATA, 'sparc', 'MassModels_Lelli2016c.mrt'), 10):
    rc.setdefault(t[0], []).append((float(t[2]), float(t[3]), float(t[4])))
for k in rc:
    rc[k] = np.array(rc[k])

print(f"SPARC: {len(gal)} galaxies in Table 1, {len(rc)} with rotation curves")

def slope(R, V, eV):
    """Weighted least-squares slope of ln V on ln R with its formal error."""
    x, y, w = np.log(R), np.log(V), (V / eV)**2
    W = w.sum(); xm = (w * x).sum() / W; ym = (w * y).sum() / W
    sxx = (w * (x - xm)**2).sum()
    b = (w * (x - xm) * (y - ym)).sum() / sxx
    return b, np.sqrt(1.0 / sxx)

VFLAT_LO, VFLAT_HI, Q_MAX, INC_MIN = 180.0, 260.0, 2, 30.0
R_LO, R_HI, N_MIN, SPAN_MIN = 15.0, 30.0, 3, 8.0
RD_LO, RD_HI = 6.0, 13.0
PRED = {4: -0.42, 6: -0.30, 8: -0.22, 10: -0.17}
NFW_REF = -0.074

base = [n for n, g in gal.items() if g['Q'] <= Q_MAX and g['inc'] >= INC_MIN
        and VFLAT_LO <= g['Vflat'] <= VFLAT_HI and g['Vflat'] > 0 and n in rc]
print(f"Milky-Way-mass candidates (Q<={Q_MAX}, inc>={INC_MIN:.0f}, V_flat in [{VFLAT_LO:.0f},{VFLAT_HI:.0f}]): {len(base)}")

rows_phys, rows_scaled = [], []
for n in base:
    R, V, eV = rc[n].T
    sel = (R >= R_LO) & (R <= R_HI)
    if sel.sum() >= N_MIN and (R[sel].max() - R[sel].min()) >= SPAN_MIN:
        b, eb = slope(R[sel], V[sel], eV[sel])
        rows_phys.append((n, gal[n]['Vflat'], R.max(), int(sel.sum()), b, eb))
    rd = gal[n]['Rdisk']
    sel2 = (R >= RD_LO * rd) & (R <= RD_HI * rd)
    if rd > 0 and sel2.sum() >= N_MIN and (R[sel2].max() - R[sel2].min()) >= 0.5 * (RD_HI - RD_LO) * rd:
        b2, eb2 = slope(R[sel2], V[sel2], eV[sel2])
        rows_scaled.append((n, gal[n]['Vflat'], rd, int(sel2.sum()), b2, eb2))

def report(rows, title, col):
    print(f"\n{title}: {len(rows)} galaxies")
    print(f"  {'galaxy':11s} {'V_flat':>6s} {col:>7s} {'n_pts':>5s} {'beta':>7s} {'err':>6s}")
    for n, vf, c, k, b, eb in sorted(rows, key=lambda r: r[4]):
        print(f"  {n:11s} {vf:6.1f} {c:7.1f} {k:5d} {b:+7.3f} {eb:6.3f}")
    b = np.array([r[4] for r in rows]); eb = np.array([r[5] for r in rows])
    if len(b) == 0:
        return b, eb
    p16, p50, p84 = np.percentile(b, [16, 50, 84])
    w = 1.0 / eb**2; mean_w = (w * b).sum() / w.sum(); err_w = np.sqrt(1.0 / w.sum())
    print(f"  median {p50:+.3f}, 16-84 percent range [{p16:+.3f}, {p84:+.3f}], "
          f"inverse-variance mean {mean_w:+.3f} +/- {err_w:.3f}, scatter (std) {b.std(ddof=1) if len(b) > 1 else float('nan'):.3f}")
    print("  fraction of galaxies with beta below the Paper X prediction: "
          + ", ".join(f"N={N}: {(b < v).mean():.2f}" for N, v in PRED.items())
          + f";  below NFW reference {NFW_REF:+.3f}: {(b < NFW_REF).mean():.2f}")
    return b, eb

b_phys, eb_phys = report(rows_phys, f"Physical radii {R_LO:.0f}-{R_HI:.0f} kpc", 'R_max')
b_scaled, eb_scaled = report(rows_scaled, f"Scaled radii {RD_LO:.0f}-{RD_HI:.0f} R_disk", 'R_disk')
print("\nPaper X predictions for Milky-Way-mass spirals, beta(15-30 kpc): "
      + ", ".join(f"N={N}: {v:+.2f}" for N, v in PRED.items()) + f"; NFW c=12 at 1e12: {NFW_REF:+.3f}")


# ═══════════════════════════════════════════════════════════════════
# Systematics (added after the second peer review, 2026-10-10, findings
# 4 and 5): the median carries the claim; its uncertainty and the
# sensitivity to weighting, selection, window and distance are reported.
# ═══════════════════════════════════════════════════════════════════
rng = np.random.default_rng(20261010)
names_p = [r[0] for r in rows_phys]
def slopes_for(sel_names, lo=R_LO, hi=R_HI, dscale=None, perturb=False):
    out = []
    for n in sel_names:
        R, V, eV = rc[n].T
        if dscale is not None:
            R = R * dscale[n]
        if perturb:
            V = V + rng.normal(0.0, eV)
        sel = (R >= lo) & (R <= hi)
        if sel.sum() >= N_MIN and (R[sel].max() - R[sel].min()) >= SPAN_MIN * (hi - lo) / (R_HI - R_LO):
            out.append(slope(R[sel], V[sel], eV[sel])[0])
    return np.array(out)

print("\nSYSTEMATICS on the physical-radius sample (the median carries the claim):")
b_unw = []
for n in names_p:
    R, V, eV = rc[n].T; sel = (R >= R_LO) & (R <= R_HI)
    b_unw.append(np.polyfit(np.log(R[sel]), np.log(V[sel]), 1)[0])
b_unw = np.array(b_unw)
print(f"  weighted median {np.median(b_phys):+.3f}; unweighted median {np.median(b_unw):+.3f}")
NMC = 2000
mc_med = np.empty(NMC); mc_min = np.empty(NMC)
for k in range(NMC):
    bb = slopes_for(names_p, perturb=True)
    mc_med[k] = np.median(bb); mc_min[k] = bb.min()
print(f"  Monte Carlo over the quoted velocity errors ({NMC} draws): median = {mc_med.mean():+.3f} +/- {mc_med.std():.3f};"
      f" P(at least one galaxy <= -0.17) = {(mc_min <= -0.17).mean():.2f}; P(median <= -0.17) = {(mc_med <= -0.17).mean():.3f}")
# leave the outermost in-window point out
b_loo = []
for n in names_p:
    R, V, eV = rc[n].T; sel = (R >= R_LO) & (R <= R_HI)
    idx = np.where(sel)[0][:-1]
    if len(idx) >= N_MIN and (R[idx].max() - R[idx].min()) >= SPAN_MIN:
        b_loo.append(slope(R[idx], V[idx], eV[idx])[0])
b_loo = np.array(b_loo)
print(f"  outermost window point dropped: {len(b_loo)} galaxies, median {np.median(b_loo):+.3f}, min {b_loo.min():+.3f}")
# peak-speed selection (V_flat is low or zero for declining curves)
peak = [n for n, g in gal.items() if g['Q'] <= Q_MAX and g['inc'] >= INC_MIN and n in rc
        and VFLAT_LO <= rc[n][:, 1].max() <= VFLAT_HI]
b_peak = slopes_for(peak)
added = [n for n in peak if n not in base]
b_added = {n: slopes_for([n]) for n in added}
print(f"  selection on peak V_obs in [{VFLAT_LO:.0f},{VFLAT_HI:.0f}] instead of V_flat: {len(b_peak)} galaxies, median {np.median(b_peak):+.3f};"
      f" galaxies added: " + ", ".join(f"{n} ({b_added[n][0]:+.2f})" for n in added if len(b_added[n])))
b_1525 = slopes_for(names_p, lo=15.0, hi=25.0)
print(f"  window 15-25 kpc: {len(b_1525)} galaxies, median {np.median(b_1525):+.3f}")
for sign, lab in ((-1, 'down'), (+1, 'up')):
    dscale = {n: (gal[n]['D'] + sign * gal[n]['e_D']) / gal[n]['D'] for n in names_p}
    bd = slopes_for(names_p, dscale=dscale)
    print(f"  every distance shifted {lab} by its quoted error (coherent worst case): {len(bd)} galaxies, median {np.median(bd):+.3f}")
n_beyond25 = sum(((rc[n][:, 0] > 25.0) & (rc[n][:, 0] <= 30.0)).sum() <= 2 for n in names_p)
print(f"  galaxies with at most two points in 25-30 kpc: {n_beyond25} of {len(names_p)}")
print("  pressure support is neglected in V_obs; at these speeds it is a percent-level effect on the slope.")
sys_out = dict(median_unweighted=np.median(b_unw), mc_median_mean=mc_med.mean(), mc_median_std=mc_med.std(),
               mc_p_any_below=(mc_min <= -0.17).mean(), loo_median=np.median(b_loo), loo_min=b_loo.min(), n_loo=len(b_loo),
               peak_n=len(b_peak), peak_median=np.median(b_peak), w1525_n=len(b_1525), w1525_median=np.median(b_1525),
               n_beyond25=n_beyond25)

np.savez(OUTPATH,
         names_phys=np.array([r[0] for r in rows_phys]), vflat_phys=np.array([r[1] for r in rows_phys]),
         rmax_phys=np.array([r[2] for r in rows_phys]), npts_phys=np.array([r[3] for r in rows_phys]),
         beta_phys=b_phys, err_phys=eb_phys,
         names_scaled=np.array([r[0] for r in rows_scaled]), vflat_scaled=np.array([r[1] for r in rows_scaled]),
         rdisk_scaled=np.array([r[2] for r in rows_scaled]), npts_scaled=np.array([r[3] for r in rows_scaled]),
         beta_scaled=b_scaled, err_scaled=eb_scaled,
         selection=np.array([VFLAT_LO, VFLAT_HI, Q_MAX, INC_MIN, R_LO, R_HI, N_MIN, SPAN_MIN, RD_LO, RD_HI]),
         pred_N=np.array(list(PRED.keys())), pred_beta=np.array(list(PRED.values())), nfw_ref=NFW_REF,
         **{f'sys_{k}': v for k, v in sys_out.items()})
print(f"\nSaved {OUTPATH}")
