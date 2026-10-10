#!/usr/bin/env python3
"""
Paper X — Step 4: Figures

Fig 1: the two components in a shared well (schematic).
Fig 2: (a) the step 1 failure — isothermal other-branch gas at the
       virial temperature against the solar-radius datum and NFW;
       (b) the step 2 chain — captured mass required to reach the datum
       versus extent, with the series' available f(N) marking the
       N -> r_half mapping.
Fig 3: (a) combined rotation curves to 300 kpc for N = 4, 6, 8, 10
       (step 2b, pass 2, M_200 = 2e12, T = 2 T_vir) with the datum,
       NFW, and the stream masses as data to be tested;
       (b) the predicted outer slope beta(15-30 kpc) per N against the
       Gaia-era Milky Way slopes and the Jiao et al. 2026 value.
Fig 4: (a) perturber convergence at 1e10 Msun: ISF spheroid (scaling ii,
       N = 6) against Paper VI's intact and truncated NFW;
       (b) Paper XI in-principle expected detections at N = 6 for the
       312 Q1 systems: Paper VIII, headline, and fallback.

Palette for N = 4, 6, 8, 10 is fixed in that order and was validated
with the dataviz skill's six-check script (all checks pass):
#2255aa, #cc3311, #009988, #882255. Direct labels carry identity too.

Inputs:  ../data/{hydrostatic_halo,dissipated_component,combined_halo,perturber_profile}.npz
Output:  ../figures/fig{1,2,3,4}_*.pdf/.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA = os.path.join(ROOT, 'data')
FIGDIR = os.path.join(ROOT, 'figures')
os.makedirs(FIGDIR, exist_ok=True)

h1 = np.load(os.path.join(DATA, 'hydrostatic_halo.npz'), allow_pickle=True)
h2 = np.load(os.path.join(DATA, 'dissipated_component.npz'), allow_pickle=True)
hb = np.load(os.path.join(DATA, 'combined_halo.npz'), allow_pickle=True)
h3 = np.load(os.path.join(DATA, 'perturber_profile.npz'), allow_pickle=True)

G = 4.30091e-6
COL = {4: '#2255aa', 6: '#cc3311', 8: '#009988', 10: '#882255'}
GREY = '#6b6b6b'
plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9.5, 'legend.fontsize': 8,
                     'axes.spines.top': False, 'axes.spines.right': False})

def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIGDIR, f'{name}.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved figures/{name}.pdf/.png")

# ── Figure 1: schematic ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.4, 3.4))
x = np.linspace(-120, 120, 800)
well = -1.0 / np.sqrt(1 + (x / 25.0)**2)
ax.plot(x, well, color='k', lw=1.4)
ax.text(62, -0.62, 'shared potential well\n(one metric, all branches)', ha='left', va='top', fontsize=8.5)
# hot other-branch gas: wide, low
gas = 0.22 * np.exp(-0.5 * (x / 70.0)**2) - 0.08
ax.fill_between(x, -0.08, gas, color='#009988', alpha=0.25, lw=0)
ax.text(78, 0.08, 'other-branch diffuse gas\n(hot, extended; step 1)', color='#007766', fontsize=8.5, ha='left')
# captured spheroid: compact
cap = 0.55 * np.exp(-0.5 * (x / 14.0)**2) - 0.08
ax.fill_between(x, -0.08, cap, color='#cc3311', alpha=0.35, lw=0)
ax.text(18, 0.44, 'captured other-branch galaxies\n(stars, remnants; step 2)', color='#aa2200', fontsize=8.5)
# our disc
ax.plot([-10, 10], [-0.02, -0.02], color='#2255aa', lw=5, solid_capstyle='round')
ax.text(-12, 0.02, 'our disc', color='#2255aa', fontsize=8.5, ha='right')
ax.set_xlim(-120, 120); ax.set_ylim(-1.3, 0.75)
ax.set_xlabel('radius (schematic, ~kpc)')
ax.set_yticks([]); ax.spines['left'].set_visible(False)
save(fig, 'fig1_components')

# ── Figure 2: step 1 failure and the step 2 chain ──────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8))
R = h1['R_PROFILE']; T = h1['T_GRID']; M200 = h1['M200_GRID']
i2 = int(np.argmin(np.abs(M200 - 2e12)))
jT = int(np.argmin(np.abs(T - float(h1['t_min_feasible'][i2]))))   # coldest feasible T at this M_200
v_gas_tot = h1['vc_profiles'][i2, jT]
T_SHOWN = float(T[jT])
v_bar = h2['v_bar']; Rb = h2['R_PROFILE']
from_nfw = lambda r, m200, c=12.0: None
ax1.plot(Rb, v_bar, color=GREY, lw=1.6, ls='--', label='baryons alone (B2)')
ax1.plot(R, v_gas_tot, color='#009988', lw=2.0, label=r'+ isothermal gas at $%.1f\,T_{\rm vir}$ (coldest feasible), $M_{200}=2\times10^{12}$' % T_SHOWN)
# NFW c=12 reference at 1e12 with same baryons (recompute, as in step 1)
RHO_CRIT = float(h1['RHO_CRIT']); M_BAR = float(h1['M_BAR'])
BULGE_M, BULGE_A = 1.55e10, 0.7          # B2 baryons, identical to steps 1-3
DISCS = [(3.65e10, 2.35), (8.2e9, 18.24), (1.3e9, 2.57), (7.0e7, 5.00), (2.2e5, 3.30)]
def r200_of(m): return (3.0 * m / (800.0 * np.pi * RHO_CRIT))**(1.0 / 3.0)
def m_bar(r):
    """Spherical-equivalent baryonic mass inside r [kpc]: Hernquist bulge
    mass plus, for each exponential disc, r v_c^2 / G with the razor-thin
    Freeman (1970) circular speed v_c^2 = 4 pi G Sigma_0 R_d y^2
    [I0(y)K0(y) - I1(y)K1(y)], y = r / 2R_d (concession 4 fixed
    2026-10-10; previously the discs were spherically averaged)."""
    from scipy.special import i0, i1, k0, k1
    r = np.asarray(r, dtype=float)
    m = BULGE_M * r**2 / (r + BULGE_A)**2
    for md, rd in DISCS:
        sigma0 = md / (2.0 * np.pi * rd**2)
        y = np.maximum(r, 1e-6) / (2.0 * rd)
        vc2 = 4.0 * np.pi * G * sigma0 * rd * y**2 * (i0(y) * k0(y) - i1(y) * k1(y))
        m = m + r * vc2 / G
    return m
def v_nfw(r, m200, c=12.0):
    rs = r200_of(m200) / c; fc = np.log(1 + c) - c / (1 + c); xx = r / rs
    return np.sqrt(G * ((m200 - M_BAR) * (np.log(1 + xx) - xx / (1 + xx)) / fc + m_bar(r)) / r)
ax1.plot(R, v_nfw(R, 1e12), color='k', lw=1.2, ls=':', label=r'NFW $c{=}12$, $M_{200}=10^{12}$ (reference)')
ax1.errorbar([8.178], [230], yerr=[[1], [7]], fmt='o', color='k', ms=5, capsize=2, zorder=5)
ax1.text(9.5, 236, 'datum: $v_c(R_\\odot)$', fontsize=8)
ax1.set_xscale('log'); ax1.set_xlim(1, 300); ax1.set_ylim(0, 260)
ax1.set_xlabel('$R$ [kpc]'); ax1.set_ylabel('$v_c$ [km s$^{-1}$]')
ax1.set_title('(a) step 1: the hot atmosphere cannot be the halo', fontsize=9.5)
ax1.legend(loc='lower left')

rh = h2['R_HALF']; f_req = h2['f_req_B']; f_A = float(h2['f_req_A'])
rhf = np.logspace(0, np.log10(150), 200)
need = (230.0**2 - np.interp(8.178, Rb, v_bar)**2) * 8.178 / (G * M_BAR)
a_of = rhf / (1 + np.sqrt(2)); f_curve = need / (8.178 / (8.178 + a_of))**2
ax2.plot(rhf, f_curve, color='k', lw=1.8, label=r'$f$ required for $v_c(R_\odot)=230$')
for N in (4, 6, 8, 10):
    i = int(np.where(h2['N_GRID'] == N)[0][0]); f = h2['f_avail_m1'][i]; r_pred = h2['rh_pred'][i]
    ax2.axhline(f, color=COL[N], lw=1.2, ls='--')
    ax2.plot([r_pred], [f], 'o', color=COL[N], ms=6)
    ax2.text(r_pred * 1.15, f * 1.06, f'$N={N}$: $r_{{1/2}}={r_pred:.0f}$ kpc', color=COL[N], fontsize=8)
ax2.set_xscale('log'); ax2.set_yscale('log'); ax2.set_xlim(1, 150); ax2.set_ylim(0.8, 60)
ax2.set_xlabel('captured-component half-mass radius $r_{1/2}$ [kpc]')
ax2.set_ylabel(r'$f = M_{\rm cap}/M_{\rm bar}$')
ax2.set_title(r'(b) step 2: $N \to f \to r_{1/2}$', fontsize=9.5)
ax2.legend(loc='upper left')
save(fig, 'fig2_step1_step2')

# ── Figure 3: combined curves and the slope prediction ─────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8))
Nb = hb['N_GRID']; M2b = hb['M200_GRID']; Tb = hb['T_GRID']; Rp = hb['R_PROF']
j2 = int(np.argmin(np.abs(M2b - 2e12))); k2 = int(np.argmin(np.abs(Tb - 2.0)))
for i, N in enumerate(Nb):
    prof = hb['prof_p2'][i, j2, k2]
    if np.isfinite(prof).all():
        ax1.plot(Rp, prof, color=COL[int(N)], lw=1.8, label=f'$N={N}$')
ax1.plot(Rb, v_bar, color=GREY, lw=1.4, ls='--', label='baryons alone')
ax1.plot(R, v_nfw(R, 1e12), color='k', lw=1.1, ls=':', label='NFW $c{=}12$, $10^{12}$')
ax1.errorbar([8.178], [230], yerr=[[1], [7]], fmt='o', color='k', ms=5, capsize=2, zorder=5)
for Rs, Ms, lab in ((50.0, 3.77e11, 'Orphan–Chenab'), (100.0, 5.6e11, 'Sagittarius')):
    vs = np.sqrt(G * Ms / Rs)
    ax1.plot([Rs], [vs], 's', color='k', ms=5, mfc='white', zorder=5)
    ax1.text(Rs * 1.1, vs + 6, lab, fontsize=7.5)
ax1.set_xscale('log'); ax1.set_xlim(1, 330); ax1.set_ylim(0, 310)
ax1.set_xlabel('$R$ [kpc]'); ax1.set_ylabel('$v_c$ [km s$^{-1}$]')
ax1.set_title(r'(a) captured + hot remainder, $M_{200}=2\times10^{12}$, $T=2T_{\rm vir}$', fontsize=9.5)
ax1.legend(loc='lower left', ncol=2)

beta_lo = np.nanmin(hb['beta_p2'], axis=(1, 2)); beta_hi = np.nanmax(hb['beta_p2'], axis=(1, 2))
beta_2 = h2['beta_pred'][[int(np.where(h2['N_GRID'] == N)[0][0]) for N in Nb]]
ax2.axhspan(-0.20, -0.16, color=GREY, alpha=0.18, lw=0)
ax2.text(4.6, -0.18, 'Gaia-era MW linear slopes', fontsize=7.5, va='center', color='#444444')
ax2.axhspan(-0.60, -0.40, color='#f0c8bf', alpha=0.45, lw=0)
ax2.text(4.05, -0.45, 'Jiao+2026 power law', fontsize=7.5, va='center', color='#884433')
ax2.axhline(0.0, color='k', lw=0.8, ls=':'); ax2.text(3.2, 0.012, 'flat', fontsize=7.5)
beta_nfw_1e12 = float(h1['beta_nfw'][int(np.argmin(np.abs(h1['M200_GRID'] - 1e12)))])   # from step 1, not typed in
ax2.axhline(beta_nfw_1e12, color='k', lw=0.8, ls='-.'); ax2.text(3.2, beta_nfw_1e12 - 0.03, r'NFW $c{=}12$, $10^{12}$', fontsize=7.5)
# SPARC Milky-Way-mass spirals (step 5): median and 16-84 percent band, present-day data
h5 = np.load(os.path.join(DATA, 'sparc_mw_mass_slopes.npz'), allow_pickle=True)
p16, p50, p84 = np.percentile(h5['beta_phys'], [16, 50, 84])
ax2.axhspan(p16, p84, color='#cfe3d8', alpha=0.6, lw=0, zorder=0)
ax2.axhline(p50, color='#2a6f4e', lw=1.0, ls='-')
ax2.text(4.1, p84 + 0.012, f'SPARC, $V_f$=180–260 km/s, 15–30 kpc: {len(h5["beta_phys"])} galaxies, 16–84%% band, median {p50:+.2f}' % (), fontsize=7.2, color='#2a6f4e')

for i, N in enumerate(Nb):
    ax2.errorbar([N], [0.5 * (beta_lo[i] + beta_hi[i])], yerr=[[0.5 * (beta_hi[i] - beta_lo[i])]] * 2,
                 fmt='o', color=COL[int(N)], ms=7, capsize=3, lw=1.6)
    ax2.plot([N], [beta_2[i]], 'x', color=COL[int(N)], ms=7, mew=1.4)
bm = h2['beta_pred_m23']   # rows: m = 2, 3; columns follow h2['N_GRID']
for i, N in enumerate(Nb):
    j = int(np.where(h2['N_GRID'] == N)[0][0])
    ax2.plot([N + 0.25], [bm[0, j]], marker='^', mfc='white', mec=COL[int(N)], ms=7, ls='none', mew=1.4)
    ax2.plot([N + 0.45], [bm[1, j]], marker='s', mfc='white', mec=COL[int(N)], ms=6, ls='none', mew=1.4)
ax2.plot([], [], marker='^', mfc='white', mec='k', ms=7, ls='none', label='captured + baryons, $m=2$')
ax2.plot([], [], marker='s', mfc='white', mec='k', ms=6, ls='none', label='captured + baryons, $m=3$')
ax2.plot([], [], 'o', color='k', label='prediction, with hot remainder, $m=1$ (range over $M_{200}$, $T$)')
ax2.plot([], [], 'x', color='k', label='captured + baryons only, $m=1$')
ax2.set_xlim(3, 12.2); ax2.set_ylim(-0.62, 0.10); ax2.set_xticks([4, 6, 8, 10])
ax2.set_xlabel('number of branches $N$'); ax2.set_ylabel(r'$\beta = d\ln v_c/d\ln R$ at 15–30 kpc')
ax2.set_title('(b) the outer slope follows from $N$', fontsize=9.5)
ax2.legend(loc='lower right', fontsize=6.8)
save(fig, 'fig3_curves_and_slope')

# ── Figure 4: perturber convergence and the Paper XI numbers ───────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8))
rg = h3['r_grid_p6']; k_isf = h3['k_isf_p6']; k_lcdm = h3['k_lcdm_p6']
iN6 = int(np.where(h3['N_GRID'] == 6)[0][0]); rh6 = float(h3['RH_MW'][iN6])
a6 = rh6 / (1 + np.sqrt(2)); M = 1e10
rho_h = M * a6 / (2 * np.pi * rg * (rg + a6)**3)
rho_i = interp1d(rg, rho_h, kind='linear', fill_value=0, bounds_error=False)
k_h = np.zeros_like(rg)
for i, Rr in enumerate(rg):
    if Rr < rg[-1] * 0.99:
        f = lambda r3, _R=Rr: 0.0 if r3 <= _R else rho_i(r3) * r3 / np.sqrt(r3**2 - _R**2)
        k_h[i] = 2.0 * quad(f, Rr * 1.001, rg[-1], limit=100)[0] / 3.5e9
sel = (rg >= 0.3) & (rg <= 40)
ax1.plot(rg[sel], k_isf[sel], color='k', lw=1.4, ls=':', label='Paper VI: intact NFW $c{=}12$')
ax1.plot(rg[sel], k_lcdm[sel], color=GREY, lw=1.4, ls='--', label=r'$\Lambda$CDM: truncated, $r_t=13$ kpc')
ax1.plot(rg[sel], k_h[sel], color=COL[6], lw=2.0, label='Paper X: captured spheroid, $r_{1/2}=%.0f$ kpc ($N{=}6$)' % rh6)
ax1.axvline(5.0, color='k', lw=0.7, alpha=0.5); ax1.text(5.3, 2e-4, 'aperture 5 kpc', fontsize=7.5, rotation=90, va='bottom')
ax1.set_xscale('log'); ax1.set_yscale('log'); ax1.set_xlim(0.3, 40)
ax1.set_xlabel('projected radius [kpc]'); ax1.set_ylabel(r'convergence $\kappa$ ($10^{10}\,M_\odot$, $z_l=0.3$)')
ax1.set_title('(a) the perturber profile', fontsize=9.5); ax1.legend(loc='lower left')

eta_models = [str(m) for m in h3['ETA_MODELS']]; a_q1 = h3['a_q1']
pub6 = float(h3['N_det_pub'][iN6]) * a_q1
head6 = h3['n_exp_head'][iN6]
fall6 = h3['n_exp_fall'][iN6]
xpos = np.arange(3); w = 0.26
b1 = ax2.bar(xpos - w, pub6, w, color='#bbbbbb', label='Paper VIII (intact NFW)')
b2 = ax2.bar(xpos, head6, w, color=COL[6], label='Paper X, extent held (ii) + mapping')
b3 = ax2.bar(xpos + w, fall6, w, color='#e8a79a', label='Paper X, extent scaled (i) + mapping')
for bars in (b1, b2, b3):
    for b in bars:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f'{b.get_height():.1f}', ha='center', fontsize=7.5)
ax2.set_xticks(xpos); ax2.set_xticklabels([f'$\\eta$ {m}' for m in eta_models])
ax2.set_ylabel('expected detections, $N=6$, 312 Q1 systems'); ax2.set_ylim(0, 30)
ax2.set_title('(b) the Paper XI in-principle test', fontsize=9.5); ax2.legend(loc='upper right', fontsize=7.5)
save(fig, 'fig4_perturber_and_paperxi')
print("done")
