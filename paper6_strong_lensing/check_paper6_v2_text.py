#!/usr/bin/env python3
"""
Paper VI version 2 — check that paper6.tex carries the version-2 numbers
and none of the version-1 numbers the Jacobian correction changed.

Every quantity below is read from the version-1 and version-2 results
files (the phase-2 JSON and the figure-4 metrics), formatted the way the
paper prints it, and looked for in 005_writeup/paper6.tex: the
version-1 string must be ABSENT and the version-2 string PRESENT. The
LaTeX strings are the paper's own forms. Exit status 1 if any check
fails. Read-only.

Inputs:  ../004_phase3/paper6_handoff/phase2_outputs/paper6_phase2_results.json (v1)
         ../004_phase3/phase3B/files/paper6_phase3b_metrics.json (v1)
         paper6_phase2_results.json, paper6_phase3b_metrics.json (v2)
         ../005_writeup/paper6.tex
Runtime: instant.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, '..'))
TEX = os.path.join(WS, '005_writeup', 'paper6.tex')
v1 = json.load(open(os.path.join(WS, '004_phase3', 'paper6_handoff', 'phase2_outputs', 'paper6_phase2_results.json')))
m1 = json.load(open(os.path.join(WS, '004_phase3', 'phase3B', 'files', 'paper6_phase3b_metrics.json')))
v2 = json.load(open(os.path.join(HERE, 'paper6_phase2_results.json')))
m2 = json.load(open(os.path.join(HERE, 'paper6_phase3b_metrics.json')))
tex = open(TEX, encoding='utf-8').read()
# the version-2 note deliberately names the version-1 numbers; exclude it from the absence test
_i, _j = tex.find('\\textbf{Version 2'), tex.find('\\medskip')
tex_body = tex[:_i] + tex[_j:] if 0 <= _i < _j else tex

def pla(v, key, field):
    return v['per_lens_abundance'][key][field]

N_AB, EFF = 75000, 0.05
checks = []   # (description, v1 string, v2 string)
for key, label in (('isf_conservative', 'per lens 0.58'), ('isf_optimistic', 'per lens 0.30'), ('isf_maximum', 'per lens 0.00')):
    a, b = pla(v1, key, 'N_detect_total'), pla(v2, key, 'N_detect_total')
    checks.append((f'{label}, table', f'& {a:.2f} & \\textbf{{{a:.2f}}}', f'& {b:.2f} & \\textbf{{{b:.2f}}}'))
    sa, sb = pla(v1, key, 'N_sub_detect'), pla(v2, key, 'N_sub_detect')
    checks.append((f'{label} N_sub', f'${sa*1e4:.1f}{{\\times}}10^{{-4}}$', f'${sb*1e4:.1f}{{\\times}}10^{{-4}}$'))
    checks.append((f'{label} N_sub, prose', f'{sa*1e4:.1f} \\times 10^{{-4}}', f'{sb*1e4:.1f} \\times 10^{{-4}}' if key == 'isf_conservative' else ''))
a, b = pla(v1, 'isf_conservative', 'N_detect_total'), pla(v2, 'isf_conservative', 'N_detect_total')
cdm = pla(v1, 'lcdm_baseline', 'N_sub_detectable')
checks.append(('per lens 0.58 in prose', f'{{\\sim}}\\,{a:.2f}$', f'{{\\sim}}\\,{b:.2f}$'))
checks.append(('ratio to CDM baseline', f'${a/cdm:.1f}\\times$ the established', f'${b/cdm:.1f}\\times$ the established'))
checks.append(('Euclid table row 0.58', f'& {a:.3f} & {a*N_AB:,.0f} & {a*N_AB*EFF:,.0f}', f'& {b:.3f} & {b*N_AB:,.0f} & {b*N_AB*EFF:,.0f}'))
a30, b30 = pla(v1, 'isf_optimistic', 'N_detect_total'), pla(v2, 'isf_optimistic', 'N_detect_total')
checks.append(('Euclid table row 0.30', f'& {a30:.3f} & {a30*N_AB:,.0f} & {a30*N_AB*EFF:,.0f}', f'& {b30:.3f} & {b30*N_AB:,.0f} & {b30*N_AB*EFF:,.0f}'))
checks.append(('~N Euclid at 5 percent', '${\\sim}\\,4{,}000$', '${\\sim}\\,2{,}900$'))
c1, c2 = v1['los_subhalo_fraction']['combined'], v2['los_subhalo_fraction']['combined']
checks.append(('combined f_LOS', f'f_\\mathrm{{LOS}} = {c1["f_LOS"]:.3f}', f'f_\\mathrm{{LOS}} = {c2["f_LOS"]:.3f}'))
checks.append(('LOS shift', f'+{c1["shift_from_lcdm"]*100:.1f} percentage point', f'+{c2["shift_from_lcdm"]*100:.1f} percentage point'))
checks.append(('LOS table f_LOS', f'& {c1["f_LOS"]*100:.1f}\\% \\\\', f'& {c2["f_LOS"]*100:.1f}\\% \\\\'))
checks.append(('LOS table f_sub', f'& {c1["f_sub"]*100:.1f}\\% \\\\', f'& {c2["f_sub"]*100:.1f}\\% \\\\'))
checks.append(('LOS table N_total', f'& {a:.2f}  & {c1["N_total"]:.2f} \\\\', f'& {b:.2f}  & {c2["N_total"]:.2f} \\\\'))
checks.append(('fraction at 1e7', f'{m1["isf_fraction_at_logM7"]*100:.2f}\\%', f'{m2["isf_fraction_at_logM7"]*100:.2f}\\%'))
checks.append(('fraction at 1e8', f'{m1["isf_fraction_at_logM8"]*100:.2f}\\%', f'{m2["isf_fraction_at_logM8"]*100:.2f}\\%'))
checks.append(('fraction at 1e9', f'{m1["isf_fraction_at_logM9"]*100:.1f}\\% of', f'{m2["isf_fraction_at_logM9"]*100:.1f}\\% of'))
checks.append(('10 percent mass', f'\\approx {m1["logM_isf_10pct"]:.1f}$', f'\\approx {m2["logM_isf_10pct"]:.1f}$'))
checks.append(('crossover', f'\\approx {m1["logM_crossover"]:.2f}', f'\\approx {m2["logM_crossover"]:.2f}'))
checks.append(('integrated ratio', f'$= {m1["integrated_ratio_10_12"]:.2f}\\times$', f'$= {m2["integrated_ratio_10_12"]:.2f}\\times$'))
checks.append(('Schechter phi*', '\\Phi^* = 0.012', '\\Phi^* = 3.96\\times10^{-3}'))
checks.append(('Schechter alpha', '\\alpha_S = -1.05$.', '\\alpha_S = -1.16$'))

ok = True
print(f"{'quantity':24s} {'v1 (absent?)':>36s} {'v2 (present?)':>36s}")
for desc, s1, s2 in checks:
    absent, present = s1 not in tex_body, (s2 in tex if s2 else True)
    ok &= absent and present
    print(f"{desc:24s} {s1:>30s} {'yes' if absent else 'NO':>5s} {s2:>30s} {'yes' if present else 'NO':>5s}")
print("ALL CHECKS PASS" if ok else "CHECKS FAILED")
sys.exit(0 if ok else 1)
