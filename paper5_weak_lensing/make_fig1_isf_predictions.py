"""
make_fig1_isf_predictions.py
============================
Figure 1: ISF modification factor and P_total(k)/P_galaxy(k)
predictions for Paper V.

Produces a 2×2 panel figure:
  (a) ε(k) sigmoid for various k_t and α
  (b) ISF modification factor for N=6 at various (k_t, α)
  (c) P_total(k) for ISF vs ΛCDM
  (d) S₈ constraint: ε_eff vs N with observed tension band
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

# ── Load CAMB baseline ─────────────────────────────────────────
data = np.load('/home/claude/paper5_wl/camb_pk_planck2018.npz')
k = data['k']
Pk_lin = data['Pk_lin_z03']  # Use z=0.3 as representative
sigma8 = float(data['sigma8'][0])
Omega_b = float(data['Omega_b'][0])
Omega_m = float(data['Omega_m'][0])
f_b = Omega_b / Omega_m

# ── ISF functions ──────────────────────────────────────────────
def epsilon_sigmoid(k, k_t, alpha):
    return 1.0 / (1.0 + (k / k_t)**alpha)

def isf_mod_factor(k, N, k_t, alpha):
    eps = epsilon_sigmoid(k, k_t, alpha)
    return (1.0 + (N - 1) * eps) / N

def sigma8_ratio(N, eps_eff):
    return np.sqrt((1.0 + (N - 1) * eps_eff) / N)

def eps_from_s8(N, s8_ratio):
    return (N * s8_ratio**2 - 1.0) / (N - 1.0)

# ── Figure setup ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('ISF Power Spectrum Predictions (Paper V)',
             fontsize=14, fontweight='bold', y=0.98)

colors_kt = {0.01: '#2166ac', 0.03: '#67a9cf',
             0.05: '#d6604d', 0.1: '#b2182b'}
ls_alpha = {1: '-', 2: '--', 4: ':'}
N_ref = 6
k_plot = np.logspace(-3, 1, 1000)

# ── Panel (a): ε(k) for various parameters ────────────────────
ax = axes[0, 0]
for k_t in [0.01, 0.03, 0.05, 0.1]:
    for alpha in [1, 2, 4]:
        eps = epsilon_sigmoid(k_plot, k_t, alpha)
        label = None
        if alpha == 2:
            label = f'$k_t={k_t}$'
        ax.semilogx(k_plot, eps, color=colors_kt[k_t],
                     ls=ls_alpha[alpha], alpha=0.8, label=label)

# Add legend for line styles
from matplotlib.lines import Line2D
style_legend = [Line2D([0],[0], color='gray', ls='-', label=r'$\alpha=1$'),
                Line2D([0],[0], color='gray', ls='--', label=r'$\alpha=2$'),
                Line2D([0],[0], color='gray', ls=':', label=r'$\alpha=4$')]
ax.legend(loc='lower left', fontsize=8, ncol=2)
leg2 = ax.legend(handles=style_legend, loc='upper right', fontsize=8)
ax.add_artist(ax.get_legend())

ax.set_xlabel(r'$k$ [$h$/Mpc]')
ax.set_ylabel(r'$\varepsilon(k)$')
ax.set_title(r'(a) Cross-branch correlation $\varepsilon(k)$')
ax.set_xlim(1e-3, 10)
ax.set_ylim(-0.05, 1.05)
ax.axhline(0.81, color='green', ls=':', alpha=0.5, lw=1)
ax.text(2e-3, 0.83, r'$\varepsilon \approx 0.81$ (S$_8$ tension)',
        fontsize=7, color='green')

# ── Panel (b): Modification factor for N=6 ────────────────────
ax = axes[0, 1]
for k_t in [0.01, 0.03, 0.05, 0.1]:
    for alpha in [1, 2, 4]:
        mod = isf_mod_factor(k_plot, N_ref, k_t, alpha)
        label = None
        if alpha == 2:
            label = f'$k_t={k_t}$'
        ax.semilogx(k_plot, mod, color=colors_kt[k_t],
                     ls=ls_alpha[alpha], alpha=0.8, label=label)

ax.axhline(1.0, color='black', ls='-', lw=0.5, alpha=0.5)
ax.axhline(1.0/N_ref, color='gray', ls=':', lw=1, alpha=0.5)
ax.text(2, 1.0/N_ref + 0.02, f'$1/N = 1/{N_ref}$', fontsize=8,
        color='gray', ha='right')

ax.set_xlabel(r'$k$ [$h$/Mpc]')
ax.set_ylabel(r'$[1+(N-1)\varepsilon(k)]/N$')
ax.set_title(f'(b) ISF modification factor ($N={N_ref}$)')
ax.set_xlim(1e-3, 10)
ax.set_ylim(0, 1.1)
ax.legend(loc='lower left', fontsize=8)

# ── Panel (c): P_total(k) comparison ──────────────────────────
ax = axes[1, 0]

# ΛCDM = full P(k)
ax.loglog(k, Pk_lin, 'k-', lw=2, label=r'$\Lambda$CDM $P_m(k)$', alpha=0.9)

# ISF P_total for a few representative models
isf_models = [
    (6, 0.03, 2, '#2166ac', r'$N\!=\!6, k_t\!=\!0.03, \alpha\!=\!2$'),
    (6, 0.05, 2, '#d6604d', r'$N\!=\!6, k_t\!=\!0.05, \alpha\!=\!2$'),
    (6, 0.03, 4, '#67a9cf', r'$N\!=\!6, k_t\!=\!0.03, \alpha\!=\!4$'),
    (3, 0.03, 2, '#fdb863', r'$N\!=\!3, k_t\!=\!0.03, \alpha\!=\!2$'),
]
for N, kt, alp, col, lab in isf_models:
    mod = isf_mod_factor(k, N, kt, alp)
    ax.loglog(k, Pk_lin * mod, color=col, ls='--', lw=1.5,
              label=lab, alpha=0.85)

ax.set_xlabel(r'$k$ [$h$/Mpc]')
ax.set_ylabel(r'$P(k)$ [(Mpc/$h$)$^3$]')
ax.set_title(r'(c) Total matter $P(k)$: ISF vs $\Lambda$CDM ($z=0.3$)')
ax.set_xlim(1e-3, 2)
ax.set_ylim(1, 2e5)
ax.legend(fontsize=7, loc='lower left')

# ── Panel (d): S₈ constraint ─────────────────────────────────
ax = axes[1, 1]

N_arr = np.arange(2, 15, 0.1)
# S₈ tension band
s8_ratio_central = 0.76 / 0.83
s8_ratio_lo = (0.76 - 0.02) / (0.83 + 0.01)  # conservative
s8_ratio_hi = (0.76 + 0.02) / (0.83 - 0.01)

eps_central = np.array([eps_from_s8(N, s8_ratio_central) for N in N_arr])
eps_lo = np.array([eps_from_s8(N, s8_ratio_lo) for N in N_arr])
eps_hi = np.array([eps_from_s8(N, s8_ratio_hi) for N in N_arr])

ax.fill_between(N_arr, eps_lo, eps_hi, alpha=0.3, color='steelblue',
                label=r'S$_8$ tension ($1\sigma$)')
ax.plot(N_arr, eps_central, '-', color='steelblue', lw=2)

# Mark N=6
eps_n6 = eps_from_s8(6, s8_ratio_central)
ax.plot(6, eps_n6, 'ro', ms=10, zorder=5,
        label=f'$N=6$: $\\varepsilon_{{eff}}={eps_n6:.2f}$')
ax.axvline(6, color='red', ls=':', alpha=0.3)

# Physical limits
ax.axhline(0, color='gray', ls=':', alpha=0.3)
ax.axhline(1, color='gray', ls=':', alpha=0.3)
ax.text(14, 1.02, r'$\varepsilon=1$ (locked)', fontsize=7,
        color='gray', ha='right')
ax.text(14, 0.02, r'$\varepsilon=0$ (independent)', fontsize=7,
        color='gray', ha='right')

ax.set_xlabel('$N$ (number of branches)')
ax.set_ylabel(r'$\varepsilon_{\rm eff}$ at $\sigma_8$ scale')
ax.set_title(r'(d) S$_8$ tension $\rightarrow$ constraint on $\varepsilon_{\rm eff}(N)$')
ax.set_xlim(2, 14)
ax.set_ylim(-0.1, 1.15)
ax.legend(fontsize=9, loc='center right')

plt.tight_layout(rect=[0, 0, 1, 0.96])
outpath = '/home/claude/paper5_wl/fig1_isf_predictions.png'
plt.savefig(outpath, dpi=200, bbox_inches='tight')
print(f"Saved {outpath}")

# Also save PDF for paper
plt.savefig(outpath.replace('.png', '.pdf'), bbox_inches='tight')
print(f"Saved PDF version")
