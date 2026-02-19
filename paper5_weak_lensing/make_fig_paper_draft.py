"""
make_fig_paper_draft.py
=======================
Clean 2-panel figure for Paper V:
  Left:  ε(k) constrained by S₈ with σ₈ window
  Right: Modification factor showing ISF vs ΛCDM bias shape
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

def epsilon_sigmoid(k, k_t, alpha, eps_floor=0.0):
    return eps_floor + (1.0 - eps_floor) / (1.0 + (k / k_t)**alpha)

def isf_mod_factor(k, N, k_t, alpha, eps_floor=0.0):
    eps = epsilon_sigmoid(k, k_t, alpha, eps_floor)
    return (1.0 + (N - 1) * eps) / N

# Load S₈-constrained models for N=6
models_n6 = []
with open('/home/claude/paper5_wl/isf_s8_constrained_grid.csv') as f:
    for row in csv.DictReader(f):
        if int(row['N']) == 6 and float(row['alpha']) == 2:
            models_n6.append({k_: float(v) if k_ != 'N' else int(v)
                             for k_, v in row.items()})

k_plot = np.logspace(-3, 1, 1000)

# Select 3 representative models
sel_kts = [0.03, 0.07, 0.15]
sel_models = [m for m in models_n6 if m['k_t'] in sel_kts]
colors = ['#2166ac', '#d6604d', '#b2182b']
labels = [f"$k_t = {m['k_t']}$" + r" $h$/Mpc, " +
          f"$\\varepsilon_\\infty = {m['eps_floor']:.2f}$"
          for m in sel_models]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# ── Left panel: ε(k) ──────────────────────────────────────────
for m, col, lab in zip(sel_models, colors, labels):
    eps = epsilon_sigmoid(k_plot, m['k_t'], m['alpha'], m['eps_floor'])
    ax1.semilogx(k_plot, eps, color=col, lw=2.5, label=lab)

# σ₈ window
ax1.axvspan(0.05, 0.8, alpha=0.10, color='#4daf4a', zorder=0)
ax1.text(0.18, 0.555, r'$\sigma_8$ window', fontsize=10,
         color='#4daf4a', alpha=0.7, style='italic')

# S₈ constraint line
ax1.axhline(0.81, color='#4daf4a', ls=':', alpha=0.6, lw=1.5)
ax1.text(1.5e-3, 0.825, r'$\varepsilon_{\rm eff} = 0.81$',
         fontsize=9, color='#4daf4a')

ax1.set_xlabel(r'$k$ [$h$/Mpc]', fontsize=12)
ax1.set_ylabel(r'$\varepsilon(k)$', fontsize=12)
ax1.set_xlim(1e-3, 10)
ax1.set_ylim(0.55, 1.03)
ax1.legend(fontsize=9, loc='lower left',
           framealpha=0.9, edgecolor='gray')
ax1.set_title(r'Cross-branch correlation ($N=6$, $\alpha=2$, S$_8$ constrained)',
              fontsize=11)

# ── Right panel: Modification factor with ΛCDM comparison ─────
for m, col, lab in zip(sel_models, colors, labels):
    mod = isf_mod_factor(k_plot, 6, m['k_t'], m['alpha'], m['eps_floor'])
    ax2.semilogx(k_plot, mod, color=col, lw=2.5,
                 label=f"ISF $k_t = {m['k_t']}$")

# ΛCDM scale-dependent bias (perturbative, for comparison)
k_NL = 0.15
c2_vals = [0.3, 0.6]
for c2, ls in zip(c2_vals, ['--', ':']):
    b_sq_norm = 1.0 / (1 + c2 * (k_plot / k_NL)**2)
    ax2.semilogx(k_plot, b_sq_norm, color='gray', ls=ls, lw=2, alpha=0.6,
                 label=f'$\\Lambda$CDM bias ($c_2={c2}$)' if ls == '--' else
                 f'$\\Lambda$CDM bias ($c_2={c2}$)')

ax2.axhline(1.0, color='black', ls='-', lw=0.5, alpha=0.3)

# Shade detectable region
ax2.fill_between(k_plot, 0.9, 1.0, alpha=0.05, color='orange')
ax2.text(1.5e-3, 0.905, '< 10% deviation', fontsize=8,
         color='orange', alpha=0.5)

ax2.set_xlabel(r'$k$ [$h$/Mpc]', fontsize=12)
ax2.set_ylabel(r'$P_{\rm total}(k) / P_{\rm total}^{\Lambda\rm CDM}(k)$',
               fontsize=12)
ax2.set_xlim(1e-3, 10)
ax2.set_ylim(0.6, 1.05)
ax2.legend(fontsize=8, loc='lower left',
           framealpha=0.9, edgecolor='gray', ncol=1)
ax2.set_title(r'ISF signal shape vs $\Lambda$CDM bias',
              fontsize=11)

plt.tight_layout()
outpath = '/home/claude/paper5_wl/fig_paper_draft.png'
plt.savefig(outpath, dpi=200, bbox_inches='tight')
plt.savefig(outpath.replace('.png', '.pdf'), bbox_inches='tight')
print(f"Saved {outpath}")
