#!/usr/bin/env python3
"""
Paper X — Step 5b: the captured-component chain run per SPARC galaxy

The second peer review (2026-10-10, finding 1) showed that step 5's
comparison was not like-for-like: it set one predicted slope, computed
for the Milky Way's compact B2 baryons, against galaxies whose discs are
larger, while the predicted declines are largely the baryons' own. Here
the Paper X chain is run for each galaxy from its own SPARC baryonic
curve, so that observed and predicted slopes are compared galaxy by
galaxy. Adapted from the reviewer's probe (review/probes_2/
probe_sparc_per_galaxy_chain.py) with its two fixed choices bracketed.

Per galaxy (same selection as step 5: Q <= 2, inc >= 30, V_flat in
[180, 260], >= 3 points spanning >= 8 kpc in 15-30 kpc):
  v_bar^2(R) = Vgas|Vgas| + ML_disc Vdisk|Vdisk| + 0.7 Vbul^2
               (SPARC's standard [3.6] mass-to-light 0.5 for the disc,
               bracketed at 0.3 and 0.7; bulge 0.7)
  M_bar      = ML_disc L[3.6] + 1.33 M_HI
  datum      = V_obs interpolated at the anchor radius: the Milky Way's
               8.178 kpc (as the paper), bracketed by 3 R_disk
  M_cap      = (N-1) eps_inf(N) m M_bar in a Hernquist spheroid whose
               extent is solved so the model reproduces the datum
  beta_pred  = weighted least-squares slope of the model curve at the
               SAME observed radii and weights as the observed slope
GATE: the Milky Way B2 model through the same code must reproduce
step 2's chain at N = 6, m = 1 (beta = -0.331, r_half = 20.2 kpc).

Population statistics per (N, m): median of observed minus predicted,
16-84 percent range, fraction of galaxies at or below their prediction,
a one-sided sign test (binomial probability of that many or fewer
below prediction if the model were right), and the inverse-variance
mean offset with its error. Nothing is fitted.

Inputs:  ../data/sparc/*.mrt, ../data/dissipated_component.npz (gate)
Output:  ../data/sparc_per_galaxy_chain.npz
Runtime: seconds.
"""

import os
import numpy as np
from scipy.optimize import brentq
from scipy.special import i0, i1, k0, k1
from scipy.stats import binom

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA = os.path.join(ROOT, 'data')
OUTPATH = os.path.join(DATA, 'sparc_per_galaxy_chain.npz')
G = 4.30091e-6
EPS = {4: 0.40, 6: 0.58, 8: 0.68, 10: 0.74}
CASES = [(4, 1), (6, 1), (8, 1), (10, 1), (8, 2), (10, 2), (8, 3), (10, 3)]
ML_DISC = [0.5, 0.3, 0.7]            # first is fiducial
ANCHORS = ['8.178 kpc', '3 R_disk']   # first is fiducial
VFLAT_LO, VFLAT_HI, Q_MAX, INC_MIN = 180.0, 260.0, 2, 30.0
R_LO, R_HI, N_MIN, SPAN_MIN = 15.0, 30.0, 3, 8.0

def tokens(path, n_min):
    out = []
    for l in open(path, encoding='utf-8', errors='replace'):
        t = l.split()
        if len(t) >= n_min:
            try: float(t[1]); float(t[2])
            except ValueError: continue
            out.append(t)
    return out
gal = {}
for t in tokens(os.path.join(DATA, 'sparc', 'SPARC_Lelli2016c.mrt'), 18):
    gal[t[0]] = dict(D=float(t[2]), inc=float(t[5]), L36=float(t[7]), Rdisk=float(t[11]), MHI=float(t[13]),
                     Vflat=float(t[15]), Q=int(t[17]))
rc = {}
for t in tokens(os.path.join(DATA, 'sparc', 'MassModels_Lelli2016c.mrt'), 10):
    rc.setdefault(t[0], []).append([float(x) for x in t[2:8]])   # R, Vobs, eV, Vgas, Vdisk, Vbul
for k in rc: rc[k] = np.array(rc[k])

def wslope(R, V, eV):
    x, y, w = np.log(R), np.log(V), (V / eV)**2
    W = w.sum(); xm = (w * x).sum() / W; ym = (w * y).sum() / W
    sxx = (w * (x - xm)**2).sum()
    return (w * (x - xm) * (y - ym)).sum() / sxx, np.sqrt(1.0 / sxx)

def chain(R_win, eV_win, vbar2_fn, m_bar, R_a, v_a, N, m):
    """Solve the Hernquist extent so the model reproduces v_a at R_a, then
    return the predicted slope over the window radii, r_half, or None."""
    f = (N - 1) * EPS[N] * m
    M_cap = f * m_bar
    need = v_a**2 - vbar2_fn(R_a)
    if need <= 0 or M_cap <= 0:
        return None
    g = lambda a: G * M_cap * (R_a / (R_a + a))**2 / R_a - need
    if g(1e-3) < 0:                       # even a point mass cannot reach the datum
        return None
    a = brentq(g, 1e-3, 1e5)
    v = np.sqrt(vbar2_fn(R_win) + G * M_cap * (R_win / (R_win + a))**2 / R_win)
    return wslope(R_win, v, eV_win)[0], a * (1 + np.sqrt(2))

# ── GATE: Milky Way B2 through the same code ─────────────────────────
BULGE_M, BULGE_A = 1.55e10, 0.7
DISCS = [(3.65e10, 2.35), (8.2e9, 18.24), (1.3e9, 2.57), (7.0e7, 5.00), (2.2e5, 3.30)]
M_BAR_MW = BULGE_M + sum(m for m, _ in DISCS)
def vbar2_mw(r):
    r = np.asarray(r, float); m = BULGE_M * r**2 / (r + BULGE_A)**2
    v2 = G * m / r
    for md, rd in DISCS:
        y = r / (2 * rd); s0 = md / (2 * np.pi * rd**2)
        v2 = v2 + 4 * np.pi * G * s0 * rd * y**2 * (i0(y) * k0(y) - i1(y) * k1(y))
    return v2
R_mw = np.linspace(15, 30, 60); eV_mw = np.ones_like(R_mw)
h2 = np.load(os.path.join(DATA, 'dissipated_component.npz'))
i6 = int(np.where(h2['N_GRID'] == 6)[0][0])
b_mw, rh_mw = chain(R_mw, eV_mw, vbar2_mw, M_BAR_MW, 8.178, 230.0, 6, 1)
gate = abs(b_mw - h2['beta_pred'][i6]) < 2e-3 and abs(rh_mw - h2['rh_pred'][i6]) < 0.1
print(f"GATE Milky Way B2 at (6,1): beta {b_mw:+.3f} (step 2 {h2['beta_pred'][i6]:+.3f}), r_half {rh_mw:.1f} (step 2 {h2['rh_pred'][i6]:.1f}): {'PASS' if gate else 'FAIL'}")
if not gate:
    raise SystemExit("gate failed")

# ── Sample ───────────────────────────────────────────────────────────
base = [n for n, g in gal.items() if g['Q'] <= Q_MAX and g['inc'] >= INC_MIN
        and VFLAT_LO <= g['Vflat'] <= VFLAT_HI and n in rc]
sample = []
for n in base:
    R = rc[n][:, 0]; sel = (R >= R_LO) & (R <= R_HI)
    if sel.sum() >= N_MIN and (R[sel].max() - R[sel].min()) >= SPAN_MIN:
        sample.append(n)

results = {}   # (ml, anchor) -> dict of arrays
for ml in ML_DISC:
    for anc in ANCHORS:
        obs, pred, names, rh = [], {c: [] for c in CASES}, [], {c: [] for c in CASES}
        dropped = {c: [] for c in CASES}
        for n in sample:
            R, V, eV, Vg, Vd, Vb = rc[n].T
            R_a = 8.178 if anc == '8.178 kpc' else 3.0 * gal[n]['Rdisk']
            if R_a < R.min() or R_a > R.max():
                continue
            vbar2_tab = Vg * np.abs(Vg) + ml * Vd * np.abs(Vd) + 0.7 * Vb**2
            vbar2_fn = lambda r, R=R, t=vbar2_tab: np.interp(r, R, t)
            m_bar = (ml * gal[n]['L36'] + 1.33 * gal[n]['MHI']) * 1e9
            v_a = np.interp(R_a, R, V)
            sel = (R >= R_LO) & (R <= R_HI)
            b_obs = wslope(R[sel], V[sel], eV[sel])[0]
            names.append(n); obs.append(b_obs)
            for c in CASES:
                out = chain(R[sel], eV[sel], vbar2_fn, m_bar, R_a, v_a, *c)
                if out is None:
                    pred[c].append(np.nan); rh[c].append(np.nan); dropped[c].append(n)
                else:
                    pred[c].append(out[0]); rh[c].append(out[1])
        results[(ml, anc)] = dict(names=np.array(names), obs=np.array(obs),
                                  pred={c: np.array(v) for c, v in pred.items()},
                                  rh={c: np.array(v) for c, v in rh.items()}, dropped=dropped)

def stats(obs, pred):
    ok = np.isfinite(pred); d = obs[ok] - pred[ok]; n = ok.sum()
    k_below = int((d <= 0).sum())
    p_sign = binom.cdf(k_below, n, 0.5)          # one-sided: this few (or fewer) below if the model were right
    return n, np.median(d), np.percentile(d, 16), np.percentile(d, 84), k_below, p_sign

# ── Report ───────────────────────────────────────────────────────────
fid = results[(0.5, '8.178 kpc')]
print(f"\nFiducial (M/L_disc 0.5, anchor 8.178 kpc): {len(fid['names'])} galaxies with data at the anchor")
print(f"  {'galaxy':11s} {'obs':>7s} | " + " ".join(f"({N:2d},{m})" for N, m in CASES))
for i, n in enumerate(fid['names']):
    print(f"  {n:11s} {fid['obs'][i]:+7.3f} | " + " ".join(f"{fid['pred'][c][i]:+7.3f}" if np.isfinite(fid['pred'][c][i]) else "    ---" for c in CASES))
print(f"\n  population, observed median {np.median(fid['obs']):+.3f}:")
print(f"  {'(N,m)':>7s} {'n':>3s} {'median obs-pred':>15s} {'16-84':>17s} {'k<=pred':>7s} {'P(sign)':>8s} {'pred median':>11s}")
summary = {}
for c in CASES:
    n, md, p16, p84, k, p = stats(fid['obs'], fid['pred'][c])
    summary[c] = (n, md, p16, p84, k, p)
    print(f"  ({c[0]:2d},{c[1]}) {n:3d} {md:+15.3f} [{p16:+.3f}, {p84:+.3f}] {k:7d} {p:8.3f} {np.nanmedian(fid['pred'][c]):+11.3f}")
print("  P(sign): one-sided binomial probability of seeing that few galaxies at or below their prediction if the model held;")
print("  galaxies dropped from a case when their baryons alone exceed the datum: "
      + "; ".join(f"({c[0]},{c[1]}): {len(fid['dropped'][c])}" for c in CASES if fid['dropped'][c]))

print("\nBrackets (median obs - pred; n):")
print(f"  {'M/L, anchor':>20s} | " + " ".join(f"({N:2d},{m})" for N, m in CASES))
for key, r in results.items():
    row = []
    for c in CASES:
        n, md, *_ = stats(r['obs'], r['pred'][c]); row.append(f"{md:+.3f}/{n:2d}")
    print(f"  {key[0]:.1f}, {key[1]:>9s} | " + " ".join(f"{x:>9s}" for x in row))

np.savez(OUTPATH, cases=np.array(CASES), ml_disc=np.array(ML_DISC), anchors=np.array(ANCHORS),
         fid_names=fid['names'], fid_obs=fid['obs'],
         fid_pred=np.array([fid['pred'][c] for c in CASES]), fid_rh=np.array([fid['rh'][c] for c in CASES]),
         summary=np.array([summary[c] for c in CASES]),
         bracket_keys=np.array([f"{k[0]},{k[1]}" for k in results]),
         bracket_median=np.array([[stats(r['obs'], r['pred'][c])[1] for c in CASES] for r in results.values()]),
         bracket_n=np.array([[stats(r['obs'], r['pred'][c])[0] for c in CASES] for r in results.values()]),
         gate_beta=b_mw, gate_rhalf=rh_mw)
print(f"\nSaved {OUTPATH}")
