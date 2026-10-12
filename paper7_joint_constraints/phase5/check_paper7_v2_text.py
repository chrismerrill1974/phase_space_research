#!/usr/bin/env python3
"""
Paper VII version 2 — check that paper_vii.tex carries the version-2
numbers and none of the version-1 numbers the Jacobian correction changed.

Every quantity below is read from the version-1 and version-2 forecast
products (../data/euclid_forecast.npz, ../data/euclid_forecast_v2.npz)
and ../data/joint_constraint.npz, formatted the way the paper prints it,
and looked for in ../12_phase5/paper_vii.tex: the version-1 string must
be ABSENT from the body (the version-2 note's deliberate mentions of the
old numbers are excluded) and the version-2 string PRESENT. Exit status
1 if any check fails. Read-only.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, '..'))
TEX = os.path.join(WS, '12_phase5', 'paper_vii.tex')
v1 = np.load(os.path.join(WS, 'data', 'euclid_forecast.npz'), allow_pickle=True)
v2 = np.load(os.path.join(WS, 'data', 'euclid_forecast_v2.npz'), allow_pickle=True)
jc = np.load(os.path.join(WS, 'data', 'joint_constraint.npz'), allow_pickle=True)
tex = open(TEX, encoding='utf-8').read()
_i, _j = tex.find('\\textbf{Version 2'), tex.find('\\medskip')
tex_body = tex[:_i] + tex[_j:] if 0 <= _i < _j else tex

N = v1['N_grid']; a = float(v1['N_lenses']) * float(v1['eta'])
i6 = int(np.where(N == 6)[0][0]); i4 = int(np.where(N == 4)[0][0])

def num(x):           # 4,050-style with the paper's {,}
    return f"{x:,.0f}".replace(',', '{,}')

def rows(v):
    nd = v['N_det_per_lens']; sig = v['sigma_Ndet']; thr = v['N_det_threshold_95']
    n6 = nd[i6] * a
    out = {
        'C_sub': f"C_\\mathrm{{sub}} = {float(v['C_sub']):.3f}",
        'N_det at 6 (prose)': f"N_\\mathrm{{det}} = {nd[i6]:.2f}$ at $N = 6$",
        'table N_det column': " ".join(f"{x:.2f}" for x in nd),
        'range N=2..10': f"${num(nd[0] * a)}$ ($N = 2$) to ${num(nd[-1] * a)}$ ($N = 10$)",
        'n_exp fiducial': f"n_\\mathrm{{exp}} = {num(n6)}$ detections",
        'Poisson sigma': f"\\sigma = {np.sqrt(n6):.1f}$",
        'fractional precision': f"${sig / nd[i6] * 100:.1f}\\%$",
        'P(N=6) ISF-true': f"P(N = 6) = {v['P_N_euclid_isf'][i6]:.2f}$",
        'N=2 from 6': f"$N = 2$ excluded at ${abs(nd[0] - nd[i6]) / sig:.1f}\\sigma$",
        'N=3 from 6': f"$N = 3$ excluded at ${abs(nd[1] - nd[i6]) / sig:.1f}\\sigma$",
        'N=4 vs 6': f"$N = 4$ vs $N = 6$: ${abs(nd[i4] - nd[i6]) / sig:.1f}\\sigma$",
        'abstract N=2 vs 6': f"${abs(nd[0] - nd[i6]) / sig:.0f}\\sigma$;",
        'null N=2': f"$N = 2$ at ${np.sqrt(2 * nd[0] * a):.0f}\\sigma$",
        'null N=6': f"$N = 6$ at ${np.sqrt(2 * nd[i6] * a):.0f}\\sigma$",
        'null N=10 ': f"${np.sqrt(2 * nd[-1] * a):.0f}\\sigma$.  ISF is decisively",
        'abstract null': f"$> {np.sqrt(2 * nd[0] * a):.0f}\\sigma$",
        'threshold all ISF': f"fewer than ${num(thr[0] * a)}$",
        'threshold all ISF per lens': f"< {thr[0]:.2f}$)",
        'threshold N>=4': f"fewer than ${num(thr[i4] * a)}$",
        'threshold N>=4 per lens': f"< {thr[i4]:.2f}$)",
    }
    return out

r1, r2 = rows(v1), rows(v2)
# the table column is checked row by row
ok = True
print(f"{'quantity':28s} {'v1 (absent?)':>44s} {'v2 (present?)':>44s}")
for key in r1:
    if key == 'table N_det column':
        for n_, x1, x2 in zip(N, v1['N_det_per_lens'], v2['N_det_per_lens']):
            s1 = f"& {x1:.2f} &"; s2 = f"& {x2:.2f} &"
            # version-1 and version-2 strings can coincide across rows (0.77 at N=3 v1, N=6 v2): test the full row
            e = float(jc['eps_inf_grid'][list(N).index(n_)])
            s1 = f"{int(n_):<2d} & {e:.2f} & {(n_ - 1) * (1 - e):.2f} & {x1:.2f} &"
            s2 = f"{int(n_):<2d} & {e:.2f} & {(n_ - 1) * (1 - e):.2f} & {x2:.2f} &"
            absent, present = s1 not in tex_body, s2 in tex
            ok &= absent and present
            print(f"{'table row N=' + str(int(n_)):28s} {s1:>40s} {'yes' if absent else 'NO':>3s} {s2:>40s} {'yes' if present else 'NO':>3s}")
        continue
    s1, s2 = r1[key], r2[key]
    absent, present = s1 not in tex_body, s2 in tex
    ok &= absent and present
    print(f"{key:28s} {s1:>40s} {'yes' if absent else 'NO':>3s} {s2:>40s} {'yes' if present else 'NO':>3s}")
print("ALL CHECKS PASS" if ok else "CHECKS FAILED")
sys.exit(0 if ok else 1)
