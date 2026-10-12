#!/usr/bin/env python3
"""
Paper VII — Phase 5b: Forecast Figure — VERSION 2 (2026-10-11): reads
euclid_forecast_v2.npz (Jacobian-corrected, see step5a) and writes to
this folder's figures/; the ISF-true annotation says the posterior narrows
to N = 6-8 rather than collapsing onto N = 6. Otherwise version 1. (The
list below says four figures; the script also writes the WL dchi2
profiles figure, five in all, as version 1 did.)

Produces two key figures:

  Figure 1: Joint constraint contour plot (N, k_t)
            WL-only (blue), sub-only (red), joint (black filled)
            + Paper V and Paper VII fiducial markers

  Figure 2: Euclid forecast overlay
            Current joint (black), ISF-true forecast (green),
            + annotation for null exclusion

Also produces:
  Figure 3: Marginalised P(N) comparison (current vs Euclid)
  Figure 4: 1D marginalised P(k_t) with confidence intervals

Depends on:
  ../data/joint_constraint.npz  (Phase 4)
  ../data/euclid_forecast.npz   (Phase 5a)

Output: ../figures/fig_joint_constraint.pdf
        ../figures/fig_euclid_forecast.pdf
        ../figures/fig_marginalised_PN.pdf
        ../figures/fig_marginalised_kt.pdf
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

# ── Paths ─────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA_DIR = os.path.join(REPO_ROOT, 'data')
FIG_DIR = os.path.join(SCRIPT_DIR, 'figures')   # version 2 (2026-10-11): this folder's figures/
os.makedirs(FIG_DIR, exist_ok=True)

jc = np.load(os.path.join(DATA_DIR, 'joint_constraint.npz'), allow_pickle=True)
ec = np.load(os.path.join(DATA_DIR, 'euclid_forecast_v2.npz'), allow_pickle=True)   # version 2 product

N_grid = jc['N_grid']
eps_inf_grid = jc['eps_inf_grid']
k_t_grid = jc['k_t_grid']

# ── Plotting style ────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})


# ══════════════════════════════════════════════════════════════════
# Figure 1: Joint constraint heatmap (N, k_t)
# ══════════════════════════════════════════════════════════════════
print("Figure 1: Joint constraint heatmap...")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

# Compute Delta_chi^2 surfaces relative to LCDM
chi2_lcdm = float(jc['chi2_lcdm'])
dchi2_wl = jc['chi2_wl_grid'] - chi2_lcdm
dchi2_sub = jc['chi2_sub_grid']  # already relative
dchi2_joint = jc['dchi2_joint_grid']

# Clip for display
vmin, vmax = -55, 5

panels = [
    (dchi2_wl, 'WL-only', 'Blues_r'),
    (dchi2_sub, 'Substructure-only', 'Reds_r'),
    (dchi2_joint, 'Joint', 'viridis_r'),
]

for ax, (dchi2, title, cmap) in zip(axes, panels):
    # Use imshow for discrete N axis
    # Interpolate k_t to regular log grid for display
    im = ax.imshow(np.clip(dchi2, vmin, vmax), aspect='auto',
                   origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[np.log10(k_t_grid[0]), np.log10(k_t_grid[-1]),
                           -0.5, len(N_grid) - 0.5])

    ax.set_yticks(range(len(N_grid)))
    ax.set_yticklabels(N_grid)
    # Custom x-ticks in log space
    xticks_vals = [0.1, 0.2, 0.5, 1.0, 2.0]
    ax.set_xticks([np.log10(v) for v in xticks_vals])
    ax.set_xticklabels([str(v) for v in xticks_vals])
    ax.set_xlabel(r'$k_t^\mathrm{app}$ [$h\,\mathrm{Mpc}^{-1}$]')
    ax.set_title(title)

    # Mark Paper V fiducial
    ax.axvline(np.log10(0.20), color='grey', ls='--', lw=0.8, alpha=0.5)
    # Mark Paper VII physical k_t
    ax.axvline(np.log10(0.52), color='orange', ls='--', lw=0.8, alpha=0.5)
    # Mark best-fit
    idx = np.unravel_index(np.argmin(dchi2), dchi2.shape)
    ax.plot(np.log10(k_t_grid[idx[1]]), idx[0], 'w*', ms=12, mec='k', mew=0.5)

    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label(r'$\Delta\chi^2$ vs $\Lambda$CDM')

axes[0].set_ylabel(r'$N$ (branch count)')

# Add annotations
axes[0].text(np.log10(0.22), 5.2, 'Paper V', fontsize=7, color='grey', rotation=90)
axes[0].text(np.log10(0.55), 5.2, 'Paper VII', fontsize=7, color='orange', rotation=90)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_joint_constraint.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(FIG_DIR, 'fig_joint_constraint.png'), bbox_inches='tight')
print("  Saved fig_joint_constraint.pdf/png")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# Figure 2: Euclid forecast overlay
# ══════════════════════════════════════════════════════════════════
print("Figure 2: Euclid forecast overlay...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Panel (a): Current joint vs Joint+Euclid (ISF-true)
dchi2_current = jc['dchi2_joint_grid']
dchi2_euclid = ec['chi2_joint_euclid_isf'] - ec['chi2_joint_euclid_isf'].min()

for ax, (dchi2, title, cmap) in zip(axes, [
    (dchi2_current, r'Current joint ($\mathcal{L}_\mathrm{WL} \times \mathcal{L}_\mathrm{sub}$)',
     'viridis_r'),
    (dchi2_euclid, r'+ Euclid forecast (ISF-true, $N_\mathrm{fid}=6$)',
     'Greens_r'),
]):
    im = ax.imshow(np.clip(dchi2, 0, 60), aspect='auto',
                   origin='lower', cmap=cmap, vmin=0, vmax=60,
                   extent=[np.log10(k_t_grid[0]), np.log10(k_t_grid[-1]),
                           -0.5, len(N_grid) - 0.5])

    ax.set_yticks(range(len(N_grid)))
    ax.set_yticklabels(N_grid)
    xticks_vals = [0.1, 0.2, 0.5, 1.0, 2.0]
    ax.set_xticks([np.log10(v) for v in xticks_vals])
    ax.set_xticklabels([str(v) for v in xticks_vals])
    ax.set_xlabel(r'$k_t^\mathrm{app}$ [$h\,\mathrm{Mpc}^{-1}$]')
    ax.set_title(title, fontsize=10)

    # Mark best-fit
    idx = np.unravel_index(np.argmin(dchi2), dchi2.shape)
    ax.plot(np.log10(k_t_grid[idx[1]]), idx[0], 'w*', ms=12, mec='k', mew=0.5)

    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label(r'$\Delta\chi^2$ from best-fit')

axes[0].set_ylabel(r'$N$ (branch count)')

# Annotations on Euclid panel
axes[1].annotate('Euclid narrows\nN posterior to N=6-8',
                 xy=(np.log10(k_t_grid[15]), 3), fontsize=8, ha='center',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_euclid_forecast.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(FIG_DIR, 'fig_euclid_forecast.png'), bbox_inches='tight')
print("  Saved fig_euclid_forecast.pdf/png")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# Figure 3: Marginalised P(N) comparison
# ══════════════════════════════════════════════════════════════════
print("Figure 3: Marginalised P(N)...")

fig, ax = plt.subplots(figsize=(6, 4.5))

x = np.arange(len(N_grid))
width = 0.25

# Current joint
bars1 = ax.bar(x - width, jc['P_N_joint'], width, label='Current joint',
               color='#2c3e50', alpha=0.8, edgecolor='k', linewidth=0.5)
# Current WL-only
bars2 = ax.bar(x, jc['P_N_wl'], width, label='WL-only',
               color='#3498db', alpha=0.7, edgecolor='k', linewidth=0.5)
# + Euclid (ISF-true)
bars3 = ax.bar(x + width, ec['P_N_euclid_isf'], width,
               label=r'+ Euclid (ISF-true, $N_\mathrm{fid}=6$)',
               color='#27ae60', alpha=0.8, edgecolor='k', linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(N_grid)
ax.set_xlabel(r'$N$ (branch count)')
ax.set_ylabel(r'$P(N)$ (marginalised)')
ax.set_yscale('log')
ax.set_ylim(1e-12, 2)
ax.legend(loc='upper left', framealpha=0.9)
ax.set_title(r'Marginalised $P(N)$: current vs Euclid forecast')

# Annotate key features
ax.annotate(f'P(N=6) = {ec["P_N_euclid_isf"][3]:.2f}',
            xy=(3 + width, ec['P_N_euclid_isf'][3]),
            xytext=(1.6, 2e-3), fontsize=8,   # version 2: placed below the bars, clear of the legend and the N = 10 bars
            arrowprops=dict(arrowstyle='->', color='#27ae60'),
            color='#27ae60')

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_marginalised_PN.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(FIG_DIR, 'fig_marginalised_PN.png'), bbox_inches='tight')
print("  Saved fig_marginalised_PN.pdf/png")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# Figure 4: Marginalised P(k_t)
# ══════════════════════════════════════════════════════════════════
print("Figure 4: Marginalised P(k_t)...")

fig, ax = plt.subplots(figsize=(6, 4))

# Normalise for plotting
P_kt_wl = jc['P_kt_wl']
P_kt_joint = jc['P_kt_joint']

# Scale to peak = 1
P_kt_wl_norm = P_kt_wl / P_kt_wl.max()
P_kt_joint_norm = P_kt_joint / P_kt_joint.max()

ax.plot(k_t_grid, P_kt_wl_norm, 'b-', lw=2, label='WL-only', alpha=0.7)
ax.plot(k_t_grid, P_kt_joint_norm, 'k-', lw=2, label='Current joint')

# Shade confidence intervals
kt_68 = jc['kt_joint_68']
kt_95 = jc['kt_joint_95']
ax.axvspan(kt_68[0], kt_68[1], alpha=0.2, color='grey', label='68% CI')
ax.axvspan(kt_95[0], kt_95[1], alpha=0.08, color='grey', label='95% CI')

# Fiducial markers
ax.axvline(0.20, color='grey', ls='--', lw=0.8, label='Paper V ($k_t=0.20$)')
ax.axvline(0.52, color='orange', ls='--', lw=0.8,
           label=r'Paper VII ($k_t^\mathrm{phys}=0.52$)')

ax.set_xscale('log')
ax.set_xlabel(r'$k_t^\mathrm{app}$ [$h\,\mathrm{Mpc}^{-1}$]')
ax.set_ylabel(r'$P(k_t)$ (normalised)')
ax.set_title(r'Marginalised $P(k_t)$')
ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
ax.set_xlim(k_t_grid[0], k_t_grid[-1])
ax.set_ylim(0, 1.15)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_marginalised_kt.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(FIG_DIR, 'fig_marginalised_kt.png'), bbox_inches='tight')
print("  Saved fig_marginalised_kt.pdf/png")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# Figure 5: Per-N WL Delta_chi^2 profiles
# ══════════════════════════════════════════════════════════════════
print("Figure 5: Per-N WL profiles...")

fig, ax = plt.subplots(figsize=(7, 4.5))

colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db', '#9b59b6']

for iN in range(len(N_grid)):
    dchi2 = jc['chi2_wl_grid'][iN] - chi2_lcdm
    ax.plot(k_t_grid, dchi2, '-', color=colors[iN], lw=1.5,
            label=f'N={N_grid[iN]}', alpha=0.8)
    # Mark minimum
    idx = np.argmin(dchi2)
    ax.plot(k_t_grid[idx], dchi2[idx], 'o', color=colors[iN], ms=5)

ax.axhline(0, color='k', ls=':', lw=0.8, alpha=0.5)
ax.axvline(0.52, color='orange', ls='--', lw=0.8, alpha=0.5)

ax.set_xscale('log')
ax.set_xlabel(r'$k_t^\mathrm{app}$ [$h\,\mathrm{Mpc}^{-1}$]')
ax.set_ylabel(r'$\Delta\chi^2_\mathrm{WL}$ vs $\Lambda$CDM')
ax.set_title(r'WL $\Delta\chi^2$ profiles by $N$')
ax.legend(loc='upper right', ncol=2, fontsize=9)
ax.set_xlim(k_t_grid[0], k_t_grid[-1])

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_wl_dchi2_profiles.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(FIG_DIR, 'fig_wl_dchi2_profiles.png'), bbox_inches='tight')
print("  Saved fig_wl_dchi2_profiles.pdf/png")
plt.close(fig)


print("\nAll figures saved to:", FIG_DIR)
print("Phase 5b complete.")
