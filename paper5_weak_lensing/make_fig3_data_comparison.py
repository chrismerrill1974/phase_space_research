"""
make_fig3_data_comparison.py
============================
The key figure: ISF predictions vs published lensing amplitude data.

Panel (a): A(R) from Amon+2023 vs ISF S₈-constrained models
Panel (b): S₈ measurements vs ISF prediction band
Panel (c): Residuals (data - ISF best fit)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

# ── Load data ──────────────────────────────────────────────────
with open('/home/claude/paper5_wl/published_data_compiled.json') as f:
    pub = json.load(f)

# ── ISF functions ──────────────────────────────────────────────
def epsilon_sigmoid(k, k_t, alpha, eps_floor=0.0):
    return eps_floor + (1.0 - eps_floor) / (1.0 + (k / k_t)**alpha)

def isf_mod_factor(k, N, k_t, alpha, eps_floor=0.0):
    eps = epsilon_sigmoid(k, k_t, alpha, eps_floor)
    return (1.0 + (N - 1) * eps) / N

# ── Published A(R) data ────────────────────────────────────────
R_data = np.array(pub['amon_A_vs_R']['R_Mpch'])
A_data = np.array(pub['amon_A_vs_R']['A'])
A_err = np.array(pub['amon_A_vs_R']['A_err'])

# Convert R (Mpc/h) to effective k (h/Mpc) 
# For projected separation R, the 3D power spectrum sensitivity peaks at
# k ~ 1/R (not π/R). The exact mapping depends on the projection kernel
# but 1/R is a reasonable first approximation.
k_data = 1.0 / R_data

# ── Load S₈-constrained models for N=6 ────────────────────────
import csv
models_n6 = []
with open('/home/claude/paper5_wl/isf_s8_constrained_grid.csv') as f:
    for row in csv.DictReader(f):
        if int(row['N']) == 6 and float(row['alpha']) == 2:
            models_n6.append({k_: float(v) if k_ != 'N' else int(v)
                             for k_, v in row.items()})

k_plot = np.logspace(-2, 1, 500)

# ── Figure ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 6))

# ── Panel (a): A(R) vs ISF predictions ────────────────────────
ax1 = fig.add_subplot(121)

# Data points
ax1.errorbar(R_data, A_data, yerr=A_err, fmt='ko', ms=8, capsize=4,
             label='Amon+2023 (BOSS×DES/HSC/KiDS)', zorder=5)

# ISF models (converted to R space)
R_model = np.logspace(-0.5, 2.0, 200)
k_model = 1.0 / R_model

sel_models = [m for m in models_n6 if m['k_t'] in [0.03, 0.07, 0.1, 0.2]]
colors = ['#2166ac', '#67a9cf', '#d6604d', '#b2182b']

for m, col in zip(sel_models, colors):
    mod = isf_mod_factor(k_model, 6, m['k_t'], m['alpha'], m['eps_floor'])
    label = (f"ISF $k_t={m['k_t']}$, "
             f"$\\varepsilon_\\infty={m['eps_floor']:.2f}$")
    ax1.plot(R_model, mod, color=col, lw=2, label=label)

ax1.axhline(1.0, color='gray', ls=':', lw=1, alpha=0.5)
ax1.set_xscale('log')
ax1.set_xlabel(r'$R_\perp$ [Mpc/$h$]', fontsize=12)
ax1.set_ylabel(r'$A(R) = \Delta\Sigma_{\rm obs}/\Delta\Sigma_{\rm pred}$',
               fontsize=12)
ax1.set_title(r'(a) Lensing amplitude vs ISF predictions ($N=6$, $\alpha=2$)',
              fontsize=11)
ax1.set_xlim(0.3, 60)
ax1.set_ylim(0.4, 1.15)
ax1.legend(fontsize=8, loc='lower right', framealpha=0.9)

# Mark the ΛCDM expectation
ax1.fill_between([0.3, 60], 0.95, 1.05, alpha=0.1, color='green')
ax1.text(35, 1.06, r'$\Lambda$CDM', fontsize=9, color='green', ha='right')

# ── Panel (b): χ² goodness of fit for each k_t ───────────────
ax2 = fig.add_subplot(122)

kt_vals = []
chi2_vals = []
eps_floor_vals = []

for m in models_n6:
    kt = m['k_t']
    ef = m['eps_floor']
    
    # Compute ISF prediction at data k values
    mod_at_data = isf_mod_factor(k_data, 6, kt, m['alpha'], ef)
    
    # χ²
    chi2 = np.sum(((A_data - mod_at_data) / A_err)**2)
    ndof = len(A_data) - 1  # 1 free parameter (k_t, since ε_∞ is fixed by S₈)
    
    kt_vals.append(kt)
    chi2_vals.append(chi2 / ndof)
    eps_floor_vals.append(ef)

kt_arr = np.array(kt_vals)
chi2_arr = np.array(chi2_vals)
ef_arr = np.array(eps_floor_vals)

# Color by ε_∞
sc = ax2.scatter(kt_arr, chi2_arr, c=ef_arr, cmap='RdBu', 
                  s=80, edgecolors='black', lw=0.5, zorder=5)
plt.colorbar(sc, ax=ax2, label=r'$\varepsilon_\infty$')

# Find best fit
best_idx = np.argmin(chi2_arr)
ax2.plot(kt_arr[best_idx], chi2_arr[best_idx], '*', color='gold', 
         ms=20, zorder=10, markeredgecolor='black', markeredgewidth=1)
ax2.annotate(f'Best: $k_t$={kt_arr[best_idx]:.2f}\n'
             f'$\\varepsilon_\\infty$={ef_arr[best_idx]:.2f}\n'
             f'$\\chi^2/\\nu$={chi2_arr[best_idx]:.1f}',
             xy=(kt_arr[best_idx], chi2_arr[best_idx]),
             xytext=(kt_arr[best_idx]+0.05, chi2_arr[best_idx]+1),
             fontsize=9, arrowprops=dict(arrowstyle='->', color='black'))

# ΛCDM comparison (constant A = 1, no scale dependence)
chi2_lcdm = np.sum(((A_data - 1.0) / A_err)**2) / (len(A_data) - 1)
ax2.axhline(chi2_lcdm, color='green', ls='--', lw=2, alpha=0.7,
            label=f'$\\Lambda$CDM ($A=1$): $\\chi^2/\\nu$={chi2_lcdm:.1f}')

# Constant A = mean(data) — no scale dependence
A_const = np.average(A_data, weights=1/A_err**2)
chi2_const = np.sum(((A_data - A_const) / A_err)**2) / (len(A_data) - 1)
ax2.axhline(chi2_const, color='orange', ls=':', lw=2, alpha=0.7,
            label=f'Constant $A={A_const:.2f}$: $\\chi^2/\\nu$={chi2_const:.1f}')

ax2.axhline(1.0, color='gray', ls=':', lw=0.5, alpha=0.5)

ax2.set_xlabel(r'$k_t$ [$h$/Mpc]', fontsize=12)
ax2.set_ylabel(r'$\chi^2/\nu$', fontsize=12)
ax2.set_title(r'(b) Goodness of fit: ISF models ($N=6$, $\alpha=2$)',
              fontsize=11)
ax2.set_xlim(0, 0.55)
ax2.set_ylim(0, max(chi2_arr.max(), chi2_lcdm) * 1.1)
ax2.legend(fontsize=9, loc='upper right')

plt.tight_layout()
outpath = '/home/claude/paper5_wl/fig3_data_comparison.png'
plt.savefig(outpath, dpi=200, bbox_inches='tight')
plt.savefig(outpath.replace('.png', '.pdf'), bbox_inches='tight')
print(f"Saved {outpath}")

# ── Print best-fit results ────────────────────────────────────
print(f"\n=== BEST FIT RESULTS ===")
print(f"Best ISF model: k_t = {kt_arr[best_idx]:.3f} h/Mpc,"
      f" ε_∞ = {ef_arr[best_idx]:.3f}")
print(f"  χ²/ν = {chi2_arr[best_idx]:.2f}")
print(f"\nΛCDM (A=1): χ²/ν = {chi2_lcdm:.2f}")
print(f"Constant A={A_const:.3f}: χ²/ν = {chi2_const:.2f}")
print(f"\nΔχ²(ISF vs ΛCDM) = {(chi2_lcdm - chi2_arr[best_idx]) * (len(A_data)-1):.1f}"
      f" for 1 extra parameter")
