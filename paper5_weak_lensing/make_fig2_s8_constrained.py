"""
make_fig2_s8_constrained.py
===========================
Figure 2: S₈-constrained ISF predictions — the money figure.

Panel (a): ε(k) curves that solve the S₈ tension for N=6
Panel (b): Modification factor R_ISF/R_ΛCDM for these models
Panel (c): Comparison of ISF sigmoid vs ΛCDM b²(k) shape
Panel (d): Maximum observable deviation as function of k_t
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Load data ──────────────────────────────────────────────────
data = np.load('/home/claude/paper5_wl/camb_pk_planck2018.npz')
k = data['k']
Pk_lin_z03 = data['Pk_lin_z03']

n6_data = np.load('/home/claude/paper5_wl/isf_s8_constrained_N6.npz')

# Read CSV for parameter values
import csv
models_n6 = []
with open('/home/claude/paper5_wl/isf_s8_constrained_grid.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row['N']) == 6:
            models_n6.append({k_: float(v) if k_ != 'N' else int(v)
                             for k_, v in row.items()})

def epsilon_sigmoid(k, k_t, alpha, eps_floor=0.0):
    return eps_floor + (1.0 - eps_floor) / (1.0 + (k / k_t)**alpha)

def isf_mod_factor(k, N, k_t, alpha, eps_floor=0.0):
    eps = epsilon_sigmoid(k, k_t, alpha, eps_floor)
    return (1.0 + (N - 1) * eps) / N

k_plot = np.logspace(-3, 1, 1000)

# Select representative models (α=2 only for clarity)
rep_models = [m for m in models_n6 if m['alpha'] == 2
              and m['k_t'] in [0.02, 0.05, 0.1, 0.2]]

colors = ['#2166ac', '#67a9cf', '#d6604d', '#b2182b']

# ── Figure ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(r'S$_8$-Constrained ISF Predictions ($N=6$, $\alpha=2$)',
             fontsize=14, fontweight='bold', y=0.98)

# ── Panel (a): ε(k) constrained by S₈ ─────────────────────────
ax = axes[0, 0]
for m, col in zip(rep_models, colors):
    eps = epsilon_sigmoid(k_plot, m['k_t'], m['alpha'], m['eps_floor'])
    ax.semilogx(k_plot, eps, color=col, lw=2,
                label=f"$k_t={m['k_t']}$, $\\varepsilon_\\infty={m['eps_floor']:.2f}$")

ax.axhline(0.81, color='green', ls=':', alpha=0.5, lw=1)
ax.text(3, 0.83, r'$\varepsilon_{\rm eff} \approx 0.81$ (from S$_8$)',
        fontsize=8, color='green')
ax.set_xlabel(r'$k$ [$h$/Mpc]')
ax.set_ylabel(r'$\varepsilon(k)$')
ax.set_title(r'(a) Cross-branch correlation constrained by S$_8$')
ax.set_xlim(1e-3, 10)
ax.set_ylim(0.5, 1.05)
ax.legend(fontsize=8, loc='lower left')

# Shade the σ₈ window
ax.axvspan(0.05, 0.8, alpha=0.08, color='green')
ax.text(0.15, 0.52, r'$\sigma_8$ window', fontsize=8, color='green', alpha=0.6)

# ── Panel (b): Modification factor ────────────────────────────
ax = axes[0, 1]
for m, col in zip(rep_models, colors):
    mod = isf_mod_factor(k_plot, 6, m['k_t'], m['alpha'], m['eps_floor'])
    ax.semilogx(k_plot, mod, color=col, lw=2,
                label=f"$k_t={m['k_t']}$")

ax.axhline(1.0, color='black', ls='-', lw=0.5, alpha=0.5)

# Show ΛCDM scale-dependent bias shape for comparison
# b²(k) ≈ b₁² (1 + c₂ k² / k_NL²) — perturbative bias
b1_sq = 1.0  # normalised
k_NL = 0.2  # h/Mpc, typical nonlinear scale
c2 = 0.5  # typical second-order bias coefficient
b_sq = b1_sq * (1 + c2 * (k_plot / k_NL)**2)
# The RATIO P_total/P_galaxy = 1/b²(k) in ΛCDM
# So ΛCDM "modification" relative to constant bias = 1/b²(k) × b₁²
lcdm_mod = 1.0 / (1 + c2 * (k_plot / k_NL)**2)
ax.semilogx(k_plot, lcdm_mod, 'k--', lw=2, alpha=0.6,
            label=r'$\Lambda$CDM bias ($1/b^2(k)$ shape)')

ax.set_xlabel(r'$k$ [$h$/Mpc]')
ax.set_ylabel(r'$R_{\rm ISF}(k) / R_{\Lambda\rm CDM}(k)$')
ax.set_title(r'(b) ISF modification to $P_{\rm total}/P_{\rm gal}$ ratio')
ax.set_xlim(1e-3, 10)
ax.set_ylim(0.55, 1.05)
ax.legend(fontsize=8, loc='lower left')

# ── Panel (c): ISF P_total vs ΛCDM ───────────────────────────
ax = axes[1, 0]
ax.loglog(k, Pk_lin_z03, 'k-', lw=2.5, label=r'$\Lambda$CDM $P_m(k)$', alpha=0.9)

for m, col in zip(rep_models, colors):
    mod = isf_mod_factor(k, 6, m['k_t'], m['alpha'], m['eps_floor'])
    ax.loglog(k, Pk_lin_z03 * mod, color=col, ls='--', lw=1.5,
              label=f"ISF $k_t={m['k_t']}$", alpha=0.85)

ax.set_xlabel(r'$k$ [$h$/Mpc]')
ax.set_ylabel(r'$P(k)$ [(Mpc/$h$)$^3$]')
ax.set_title(r'(c) Total matter $P(k)$ at $z=0.3$')
ax.set_xlim(1e-3, 2)
ax.set_ylim(10, 2e5)
ax.legend(fontsize=7, loc='lower left')

# ── Panel (d): Maximum deviation vs k_t for different N ───────
ax = axes[1, 1]

# For each N and k_t (α=2), find the max deviation
for N, col, ls in [(3, '#fdb863', '--'), (6, '#2166ac', '-'),
                    (10, '#b2182b', ':')]:
    kts_plot = []
    deviations = []
    for m in [row for row in csv.DictReader(
              open('/home/claude/paper5_wl/isf_s8_constrained_grid.csv'))
              if int(row['N']) == N and float(row['alpha']) == 2]:
        kt = float(m['k_t'])
        min_mod = float(m['min_mod'])
        kts_plot.append(kt)
        deviations.append((1.0 - min_mod) * 100)  # percent deviation

    if kts_plot:
        ax.plot(kts_plot, deviations, f'{ls}o', color=col, lw=2,
                ms=6, label=f'$N={N}$')

ax.set_xlabel(r'$k_t$ [$h$/Mpc]')
ax.set_ylabel('Max deviation from $\\Lambda$CDM [%]')
ax.set_title(r'(d) Observable signal strength ($\alpha=2$, S$_8$ constrained)')
ax.set_xlim(0, 0.55)
ax.set_ylim(0, 60)
ax.legend(fontsize=9)
ax.axhline(5, color='gray', ls=':', alpha=0.5)
ax.text(0.52, 5.5, '5% level', fontsize=8, color='gray', ha='right')
ax.axhline(10, color='gray', ls=':', alpha=0.3)
ax.text(0.52, 10.5, '10% level', fontsize=8, color='gray', ha='right')

plt.tight_layout(rect=[0, 0, 1, 0.96])
outpath = '/home/claude/paper5_wl/fig2_s8_constrained.png'
plt.savefig(outpath, dpi=200, bbox_inches='tight')
plt.savefig(outpath.replace('.png', '.pdf'), bbox_inches='tight')
print(f"Saved {outpath}")
