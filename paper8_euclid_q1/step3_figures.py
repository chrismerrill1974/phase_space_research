#!/usr/bin/env python3
"""
Paper VIII — Step 3: Figures — VERSION 2 (2026-10-11): reads
recalibrated_forecast_v2.npz (Papers VI and VII version 2, the Jacobian
corrected) and writes to this folder's figures/. Otherwise version 1.

Fig 1: Q1 Einstein-radius distribution (grade A/B, modelled) with the
       three detection-efficiency models overlaid.
Fig 2: Expected ISF detections vs survey stage (Q1/DR1/Wide) for each N,
       with eta-model bracket bands and 95% falsification thresholds.
Fig 3: Marginalised P(N) under the ISF-true (N=6) scenario: current
       joint constraint vs +DR1 vs +Wide.

Inputs:  ../data/q1_demographics.npz, ../data/recalibrated_forecast.npz,
         ../data/from_paper_vii/joint_constraint.npz
Output:  ../figures/fig{1,2,3}_*.pdf and .png
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA = os.path.join(ROOT, 'data')
FIGDIR = os.path.join(SCRIPT_DIR, 'figures')   # version 2: this folder's figures/

demo = np.load(os.path.join(DATA, 'q1_demographics.npz'), allow_pickle=True)
rc = np.load(os.path.join(DATA, 'recalibrated_forecast_v2.npz'), allow_pickle=True)   # version 2 product
jc = np.load(os.path.join(DATA, 'from_paper_vii', 'joint_constraint.npz'),
             allow_pickle=True)

N_grid = rc['N_grid']
theta_AB = demo['theta_E_AB']
ETA0, THETA_REF, ETA_CAP = float(rc['eta0']), float(rc['theta_ref']), float(rc['eta_cap'])

def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIGDIR, f'{name}.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved figures/{name}.pdf/.png")

# ── Figure 1: theta_E distribution + eta models ───────────────────
fig, ax = plt.subplots(figsize=(6.0, 4.2))
bins = np.linspace(0, 3.0, 31)
ax.hist(theta_AB, bins=bins, color='steelblue', alpha=0.65,
        edgecolor='k', linewidth=0.4,
        label=f'Q1 grade A/B, modelled (n={len(theta_AB)})')
ax.axvline(np.median(theta_AB), color='steelblue', ls='--', lw=1.2,
           label=f'Q1 median = {np.median(theta_AB):.2f}"')
ax.axvline(THETA_REF, color='crimson', ls=':', lw=1.4,
           label=f'legacy reference = {THETA_REF:.1f}"')
ax.set_xlabel(r'Einstein radius $\theta_{\rm E}$ [arcsec]')
ax.set_ylabel('systems per bin')
ax.set_xlim(0, 3)

ax2 = ax.twinx()
tt = np.linspace(0.01, 3, 300)
ax2.plot(tt, np.full_like(tt, ETA0), 'k-', lw=1.3, label=r'$\eta$ flat')
ax2.plot(tt, np.minimum(ETA0 * tt / THETA_REF, ETA_CAP), 'k--', lw=1.3,
         label=r'$\eta \propto \theta_{\rm E}$')
ax2.plot(tt, np.minimum(ETA0 * (tt / THETA_REF) ** 2, ETA_CAP), 'k:', lw=1.5,
         label=r'$\eta \propto \theta_{\rm E}^2$')
ax2.set_ylabel(r'detection efficiency $\eta(\theta_{\rm E})$')
ax2.set_ylim(0, 0.17)

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper right')
ax.set_title('Euclid Q1 lens demographics and detection-efficiency models')
save(fig, 'fig1_thetaE_eta')

# ── Figure 2: expected detections vs survey stage ─────────────────
stages = ['Q1', 'DR1', 'Wide']
stage_nl = {s: n for s, n in zip([str(x) for x in rc['stages']],
                                 rc['stage_nlenses'])}
fig, ax = plt.subplots(figsize=(6.0, 4.2))
colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(N_grid)))
x = np.arange(len(stages))
for iN, N in enumerate(N_grid):
    lo = [min(rc[f'{s}_{m}_n_exp'][iN] for m in ('flat', 'linear', 'area'))
          for s in stages]
    hi = [max(rc[f'{s}_{m}_n_exp'][iN] for m in ('flat', 'linear', 'area'))
          for s in stages]
    mid = np.sqrt(np.array(lo) * np.array(hi))
    ax.fill_between(x, lo, hi, color=colors[iN], alpha=0.30)
    ax.plot(x, mid, 'o-', color=colors[iN], lw=1.4, ms=4, label=f'N={N}')

# 95% all-ISF falsification threshold band (Wide only)
thr_lo = min(rc[f'Wide_{m}_thr95_total'][0] for m in ('flat', 'linear', 'area'))
thr_hi = max(rc[f'Wide_{m}_thr95_total'][0] for m in ('flat', 'linear', 'area'))
ax.fill_between([1.7, 2.3], [thr_lo] * 2, [thr_hi] * 2, color='crimson',
                alpha=0.35, label='95% all-ISF exclusion (Wide)')
ax.axhline(1307, color='grey', ls=':', lw=1.2,   # version 2: Paper VII version 2's threshold
           label='Paper VII threshold (1307)')

ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels([f'{s}\n({stage_nl[s]:.0f} lenses)' for s in stages])
ax.set_ylabel('expected ISF perturber detections')
ax.set_title('ISF detection counts by survey stage (bands: $\\eta$ models)')
ax.legend(fontsize=7.5, ncol=2, loc='upper left')
save(fig, 'fig2_counts_by_stage')

# ── Figure 3: P(N) collapse, ISF-true ─────────────────────────────
fig, ax = plt.subplots(figsize=(6.0, 4.2))
P_current = jc['P_N_joint']
# eta bracket per stage: min/max over models
width = 0.27
xN = np.arange(len(N_grid))
ax.bar(xN - width, P_current, width, color='dimgrey',
       label='Current joint (Paper VII)')
for off, stage, col in [(0.0, 'DR1', 'steelblue'), (width, 'Wide', 'seagreen')]:
    P_models = np.array([rc[f'{stage}_{m}_P_N']
                         for m in ('flat', 'linear', 'area')])
    P_mid = P_models.mean(axis=0)
    yerr = np.vstack([P_mid - P_models.min(axis=0),
                      P_models.max(axis=0) - P_mid])
    ax.bar(xN + off, P_mid, width, color=col, yerr=yerr,
           error_kw=dict(lw=0.9, capsize=2),
           label=f'+ Euclid {stage} (ISF-true, $N_{{\\rm fid}}=6$)')
ax.set_yscale('log')
ax.set_ylim(1e-12, 3)
ax.set_xticks(xN)
ax.set_xticklabels([str(N) for N in N_grid])
ax.set_xlabel('N (branch count)')
ax.set_ylabel('marginalised P(N)')
ax.set_title('Posterior by survey stage (ISF-true scenario)')   # version 2: no longer a collapse
ax.legend(fontsize=8)
save(fig, 'fig3_PN_stages')

print("Done.")
