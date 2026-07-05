#!/usr/bin/env python3
"""
Paper IX — Step 3: Figures

Fig 1: mechanism schematic — two branches' 1D density fields at high z
       (independent) and at z=0 (co-located peaks in shared wells).
Fig 2: eps(k,z) prediction family with eps_inf(z) and k_t(z) evolution,
       against the Paper V/VII fitted sigmoid and the feedback-mimic
       (SFRD-shaped) evolution for contrast.
Fig 3: Soltan gamma(N) exclusion band and early-universe overmassive
       headroom vs the JWST z>6 anomaly range.

Inputs:  ../data/eps_evolution.npz, ../data/bh_accounting.npz
Output:  ../figures/fig{1,2,3}_*.pdf/.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA = os.path.join(ROOT, 'data')
FIGDIR = os.path.join(ROOT, 'figures')

ev = np.load(os.path.join(DATA, 'eps_evolution.npz'), allow_pickle=True)
bh = np.load(os.path.join(DATA, 'bh_accounting.npz'), allow_pickle=True)

def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIGDIR, f'{name}.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved figures/{name}.pdf/.png")

# ── Figure 1: mechanism schematic (1D toy fields) ─────────────────
rng = np.random.default_rng(42)
x = np.linspace(0, 100, 2000)

def smooth_field(seed, scale=8.0):
    r = np.random.default_rng(seed)
    f = r.normal(size=len(x))
    kern = np.exp(-0.5 * (np.arange(-100, 101) / (scale * 20 / 5))**2)
    kern /= kern.sum()
    return np.convolve(f, kern, mode='same')

fA_early = smooth_field(1)
fA_early /= fA_early.std()
fB_early = smooth_field(2)
fB_early /= fB_early.std()

# late times: same independent large-scale fields plus shared sharp
# wells at common positions — local maxima of the SUM field with a
# minimum separation, so the wells are distinct
total = fA_early + fB_early
peaks = [i for i in range(5, len(x) - 5)
         if total[i] == total[i-5:i+6].max()]
peaks = sorted(peaks, key=lambda i: -total[i])
well_pos, min_sep = [], 12.0
for i in peaks:
    if all(abs(x[i] - p) > min_sep for p in well_pos):
        well_pos.append(x[i])
    if len(well_pos) == 4:
        break

def wells(amp):
    w = np.zeros_like(x)
    for p in well_pos:
        w += amp * np.exp(-0.5 * ((x - p) / 1.2)**2)
    return w

fA_late = 0.7 * fA_early + wells(2.5)
fB_late = 0.7 * fB_early + wells(2.5)

fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.6), sharex=True)
axes[0].plot(x, fA_early, color='steelblue', lw=1.0, label='branch A')
axes[0].plot(x, fB_early, color='crimson', lw=1.0, label='branch B')
axes[0].set_ylabel(r'$\delta$ (early)')
axes[0].legend(fontsize=8, loc='upper right')
axes[0].set_title('Early times: independent fields '
                  r'($\varepsilon \to 0$)')
axes[1].plot(x, fA_late, color='steelblue', lw=1.0)
axes[1].plot(x, fB_late, color='crimson', lw=1.0)
for p in well_pos:
    axes[1].axvline(p, color='k', ls=':', lw=0.6, alpha=0.5)
axes[1].set_ylabel(r'$\delta$ (late)')
axes[1].set_xlabel('comoving position (arbitrary)')
axes[1].set_title('Late times: capture into shared wells '
                  r'(dotted) $\Rightarrow$ small-scale co-location '
                  r'($\varepsilon \to \varepsilon_\infty$)')
fig.tight_layout()
save(fig, 'fig1_mechanism')

# ── Figure 2: eps(k,z) prediction family ──────────────────────────
z_grid = ev['z_grid']
eps_inf_z = ev['eps_inf_z']       # (3 M_min, nz)
kNL_z = ev['kNL_z']
kt_fit = ev['kt_fit']             # (0.52, 0.46, 0.55)
sfrd = ev['sfrd_shape']
M_min_list = ev['M_min_list']

# calibrate k_t(z) = c * k_NL(z) with c fixed at z=0 to the fitted k_t
c_cal = kt_fit[0] / kNL_z[0]

kk = np.logspace(-3, 1.5, 400)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.9))

iM = 1   # M_min = 1e11 for the family plot
zs_show = [0.0, 0.5, 1.0, 2.0]
cols = plt.cm.plasma(np.linspace(0.05, 0.75, len(zs_show)))
for zshow, col in zip(zs_show, cols):
    iz = np.argmin(np.abs(z_grid - zshow))
    e_inf = eps_inf_z[iM, iz]
    kt = c_cal * kNL_z[iz]
    eps_k = e_inf / (1 + (kt / kk)**2)
    ax1.plot(kk, eps_k, color=col, lw=1.6, label=f'z={zshow:.1f}')
eps_fit = 0.58 / (1 + (kt_fit[0] / kk)**2)
ax1.plot(kk, eps_fit, 'k--', lw=1.4, label='Paper V/VII fit (z=0)')
ax1.set_xscale('log')
ax1.set_xlabel(r'$k$ [$h\,{\rm Mpc}^{-1}$]')
ax1.set_ylabel(r'$\varepsilon(k, z)$')
ax1.set_title(r'Capture mechanism: $\varepsilon(k,z)$ '
              r'($M_{\rm min}=10^{11}\,M_\odot/h$)')
ax1.legend(fontsize=8)

labels = [r'$M_{\rm min}=10^{10}$', r'$M_{\rm min}=10^{11}$',
          r'$M_{\rm min}=10^{12}$']
for iMm, (lab, ls) in enumerate(zip(labels, ['-', '--', ':'])):
    ax2.plot(z_grid, eps_inf_z[iMm] / eps_inf_z[iMm, 0], 'k', ls=ls,
             lw=1.5, label=lab + r'$\,M_\odot/h$')
ax2.plot(z_grid, sfrd, color='crimson', lw=1.8, alpha=0.8,
         label='feedback mimic (SFRD shape)')
ax2.axhline(1.0, color='grey', lw=0.8, ls=':',
            label='Papers V-VIII assumption (z-indep.)')
ax2.set_xlabel('redshift z')
ax2.set_ylabel(r'$\varepsilon_\infty(z)\,/\,\varepsilon_\infty(0)$')
ax2.set_xlim(0, 4)
ax2.set_ylim(0, 3.2)
ax2.set_title('The tomographic discriminant')
ax2.legend(fontsize=7.5)
fig.tight_layout()
save(fig, 'fig2_eps_evolution')

# ── Figure 3: Soltan gamma(N) + JWST headroom ─────────────────────
N_grid = bh['N_grid']
gamma_95 = bh['gamma_95']
gamma_best = bh['gamma_best']
overmassive = bh['overmassive_max']
jwst = bh['jwst_overmassive_range']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.9))
ax1.fill_between(N_grid, 0, gamma_95, color='seagreen', alpha=0.25,
                 label='allowed (95%, Soltan)')
ax1.plot(N_grid, gamma_95, 'o-', color='seagreen', lw=1.5,
         label=r'$\gamma_{95}(N)$')
ax1.plot(N_grid, gamma_best, 's--', color='dimgrey', lw=1.2,
         label=r'$\gamma_{\rm best}(N)$')
for N_mark, lab, off in [(6, 'N=6 (fiducial)', (8, 20)),
                         (10, 'N=10 (joint best-fit)', (-100, -34))]:
    i = int(np.where(N_grid == N_mark)[0][0])
    ax1.annotate(f'{lab}\n' + rf'$\gamma_{{95}}={gamma_95[i]:.2f}$',
                 (N_mark, gamma_95[i]), textcoords='offset points',
                 xytext=off, fontsize=8)
ax1.set_xlabel('N (branch count)')
ax1.set_ylabel(r'cross-branch capture efficiency $\gamma$ (today)')
ax1.set_title('Late-time Soltan constraint')
ax1.set_ylim(0, 1.05)
ax1.legend(fontsize=8, loc='upper right')

ax2.fill_between([1.5, 10.5], [jwst[0]]*2, [jwst[1]]*2, color='crimson',
                 alpha=0.18, label=r'JWST $z>6$ overmassive range')
ax2.plot(N_grid, overmassive, 'o-', color='navy', lw=1.6,
         label=r'max ISF factor $1+(N-1)$ ($\gamma_{\rm early}=1$)')
ax2.set_yscale('log')
ax2.set_xlabel('N (branch count)')
ax2.set_ylabel(r'BH mass excess factor')
ax2.set_title('Early-universe headroom vs JWST anomaly')
ax2.set_xlim(1.5, 10.5)
ax2.legend(fontsize=8, loc='upper left')
fig.tight_layout()
save(fig, 'fig3_gamma_N')

print("Done.")
