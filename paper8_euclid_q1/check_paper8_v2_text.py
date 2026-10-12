#!/usr/bin/env python3
"""
Paper VIII version 2 — print every number the paper quotes from the
forecast, version 1 beside version 2, and check paper_viii.tex carries
the version-2 form and none of the version-1 forms.

Reads ../data/recalibrated_forecast.npz (v1) and
../data/recalibrated_forecast_v2.npz (v2), formats each quantity the way
the paper prints it, and looks for it in ../04_paper/paper_viii.tex: the
version-1 string must be ABSENT from the body (the version-2 note's
deliberate mentions excluded) and the version-2 string PRESENT. Also
prints a few quantities the paper states in words (count precisions,
the DR1 N = 10 vs 6 separation, the wide P(N = 6) per model) so the
prose can be checked by eye against them. Exit 1 on any failure.
Read-only.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, '..'))
TEX = os.path.join(WS, '04_paper', 'paper_viii.tex')
v1 = np.load(os.path.join(WS, 'data', 'recalibrated_forecast.npz'), allow_pickle=True)
v2 = np.load(os.path.join(WS, 'data', 'recalibrated_forecast_v2.npz'), allow_pickle=True)
tex = open(TEX, encoding='utf-8').read()
_i, _j = tex.find('\\textbf{Version 2'), tex.find('\\medskip')
tex_body = tex[:_i] + tex[_j:] if 0 <= _i < _j else tex

N = v1['N_grid']; M = ('flat', 'linear', 'area')
i2, i4, i6, i10 = [int(np.where(N == n)[0][0]) for n in (2, 4, 6, 10)]

def num(x):
    return f"{x:,.0f}".replace(',', '{,}')
def rng(lo, hi, f):
    return f"{f(lo)}--{f(hi)}"
def br(v, key, idx):
    vals = [v[f'{key.replace("MODEL", m)}'][idx] for m in M]
    return min(vals), max(vals)

def rows(v):
    out = {}
    for stage in ('Q1', 'DR1', 'Wide'):
        lo, hi = br(v, f'{stage}_MODEL_n_exp', i6)
        out[f'{stage} n_exp(6)'] = rng(lo, hi, num)
        lo, hi = br(v, f'{stage}_MODEL_sig_vs_fid', i2); out[f'{stage} sig(2v6)'] = rng(lo, hi, lambda x: f"{x:.1f}")
        lo, hi = br(v, f'{stage}_MODEL_sig_vs_fid', i4); out[f'{stage} sig(4v6)'] = rng(lo, hi, lambda x: f"{x:.1f}")
        lo, hi = br(v, f'{stage}_MODEL_sig_null', i2); out[f'{stage} null(2)'] = rng(lo, hi, lambda x: f"{x:.1f}")
    lo, hi = br(v, 'Wide_MODEL_P_N', i6); out['Wide P(6) table'] = rng(lo, hi, lambda x: f"{x:.3f}")
    lo, hi = br(v, 'Wide_MODEL_thr95_total', i2); out['Wide thr all-ISF'] = rng(lo, hi, num)
    lo, hi = br(v, 'Wide_MODEL_thr95_total', i4); out['Wide thr N>=4'] = rng(lo, hi, num)
    lo, hi = br(v, 'DR1_MODEL_thr95_total', i2); out['DR1 thr all-ISF'] = rng(lo, hi, num)
    p7 = v['P7_flat_n_exp'][i6]
    out['P7 row'] = f"{num(p7)}      & {v['P7_flat_sig_vs_fid'][i2]:.1f}       & {v['P7_flat_sig_vs_fid'][i4]:.1f}       & {v['P7_flat_sig_null'][i2]:.1f}       & {v['P7_flat_P_N'][i6]:.3f}"
    lo, hi = br(v, 'q1ip_MODEL_n_exp', i6); out['Q1ip n_exp(6)'] = rng(lo, hi, lambda x: f"{x:.1f}")
    lo, hi = br(v, 'q1ip_MODEL_sig_null', i6); out['Q1ip null(6)'] = rng(lo, hi, lambda x: f"{x:.1f}") + "$\\sigma$"
    lo, hi = br(v, 'q1ip_MODEL_sig_null', i10); out['Q1ip null(10)'] = rng(lo, hi, lambda x: f"{x:.1f}") + "$\\sigma$"
    lo, hi = br(v, 'q1ip_MODEL_sig_null', i2); out['Q1ip null(2)'] = rng(lo, hi, lambda x: f"{x:.1f}") + "$\\sigma$"
    return out

r1, r2 = rows(v1), rows(v2)
# absence tests use the paper's own forms, which for some quantities differ from the generic range form
ok = True
print(f"{'quantity':20s} {'v1 (absent?)':>52s} {'v2 (present?)':>52s}")
for key in r1:
    s1, s2 = r1[key], r2[key]
    absent, present = s1 not in tex_body, s2 in tex
    ok &= absent and present
    print(f"{key:20s} {s1:>48s} {'yes' if absent else 'NO':>3s} {s2:>48s} {'yes' if present else 'NO':>3s}")
print("ALL CHECKS PASS" if ok else "CHECKS FAILED")

jc = np.load(os.path.join(WS, 'data', 'from_paper_vii', 'joint_constraint.npz'), allow_pickle=True)
i8 = int(np.where(N == 8)[0][0])
chi2j = jc['chi2_joint_grid']
print(f"\nCurrent joint (WL + sub) chi2 minimum over k_t: N=6 {chi2j[i6].min():.1f}, N=8 {chi2j[i8].min():.1f}, "
      f"N=10 {chi2j[i10].min():.1f}; N=8 favoured over N=6 by dchi2 {chi2j[i6].min() - chi2j[i8].min():.1f}, "
      f"N=10 over N=6 by {chi2j[i6].min() - chi2j[i10].min():.1f}")
for v, lab in ((v1, 'v1'), (v2, 'v2')):
    s8 = [v[f'Wide_{m}_sig_vs_fid'][i8] for m in M]
    print(f"  Wide {lab}: N=8 vs 6 {min(s8):.1f}-{max(s8):.1f} sigma (dchi2 {min(s8)**2:.0f}-{max(s8)**2:.0f})")
print("\nFor the prose (v1 -> v2):")
for stage in ('DR1', 'Wide'):
    for v, lab in ((v1, 'v1'), (v2, 'v2')):
        n6 = [v[f'{stage}_{m}_n_exp'][i6] for m in M]
        prec = [100 / np.sqrt(x) for x in n6]
        s10 = [v[f'{stage}_{m}_sig_vs_fid'][i10] for m in M]
        print(f"  {stage} {lab}: count precision {min(prec):.1f}-{max(prec):.1f}%; N=10 vs 6 {min(s10):.1f}-{max(s10):.1f} sigma "
              f"(dchi2 {min(s10)**2:.0f}-{max(s10)**2:.0f}); P(N=6) per model " + ", ".join(f"{m} {v[f'{stage}_{m}_P_N'][i6]:.3f}" for m in M))
sys.exit(0 if ok else 1)
