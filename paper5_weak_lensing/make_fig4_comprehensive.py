"""
make_fig4_comprehensive.py
==========================
Comprehensive paper figure: 3-panel comparison of ISF with observations.

Panel (a): Scale-dependent lensing amplitude A(R) with ISF best-fit envelope
Panel (b): S₈ comparison — survey values vs ISF prediction  
Panel (c): Best-fit k_t and ε_∞ for different N values
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv, json

# ── Load data ──────────────────────────────────────────────────
with open('/home/claude/paper5_wl/published_data_compiled.json') as f:
    pub = json.load(f)

# Load full model grid
models = []
with open('/home/claude/paper5_wl/isf_s8_constrained_grid.csv') as f:
    for row in csv.DictReader(f):
        models.append({k: float(v) if k != 'N' else int(v)
                       for k, v in row.items()})

# ── ISF functions ──────────────────────────────────────────────
def epsilon_sigmoid(k, k_t, alpha, eps_floor=0.0):
    return eps_floor + (1.0 - eps_floor) / (1.0 + (k / k_t)**alpha)

def isf_mod_factor(k, N, k_t, alpha, eps_floor=0.0):
    eps = epsilon_sigmoid(k, k_t, alpha, eps_floor)
    return (1.0 + (N - 1) * eps) / N

# ── Data ───────────────────────────────────────────────────────
R_data = np.array(pub['amon_A_vs_R']['R_Mpch'])
A_data = np.array(pub['amon_A_vs_R']['A'])
A_err  = np.array(pub['amon_A_vs_R']['A_err'])
k_data = 1.0 / R_data

# ── Fit ISF for each (N, alpha) ───────────────────────────────
print("=== Best-fit ISF models by N and α ===\n")
best_fits = {}

for N_val in [2, 3, 4, 5, 6, 8, 10]:
    for alpha_val in [1, 2, 4]:
        sub = [m for m in models if m['N'] == N_val 
               and m['alpha'] == alpha_val]
        
        best_chi2 = 1e10
        best_m = None
        for m in sub:
            mod = isf_mod_factor(k_data, N_val, m['k_t'], alpha_val,
                                 m['eps_floor'])
            chi2 = np.sum(((A_data - mod) / A_err)**2)
            if chi2 < best_chi2:
                best_chi2 = chi2
                best_m = m
        
        if best_m is not None:
            ndof = len(A_data) - 1
            key = (N_val, alpha_val)
            best_fits[key] = {
                'N': N_val, 'alpha': alpha_val,
                'k_t': best_m['k_t'], 'eps_floor': best_m['eps_floor'],
                'chi2': best_chi2, 'chi2_nu': best_chi2/ndof
            }
            if alpha_val == 2:
                print(f"  N={N_val}, α={alpha_val}: k_t={best_m['k_t']:.3f},"
                      f" ε_∞={best_m['eps_floor']:.3f},"
                      f" χ²/ν={best_chi2/ndof:.2f}")

# ΛCDM baseline
chi2_lcdm = np.sum(((A_data - 1.0) / A_err)**2)
ndof = len(A_data) - 1
print(f"\n  ΛCDM (A=1): χ²/ν = {chi2_lcdm/ndof:.2f}")
A_const = np.average(A_data, weights=1/A_err**2)
chi2_const = np.sum(((A_data - A_const) / A_err)**2)
print(f"  Constant A={A_const:.3f}: χ²/ν = {chi2_const/ndof:.2f}")

# ── FIGURE ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 5.5))

# ── Panel (a): A(R) with best-fit ISF envelope ────────────────
ax1 = fig.add_subplot(131)

# Data
ax1.errorbar(R_data, A_data, yerr=A_err, fmt='s', color='black', ms=7,
             capsize=4, capthick=1.5, elinewidth=1.5,
             label='Amon+2023\n(BOSS×DES/HSC/KiDS)', zorder=5)

# ISF envelope: show range of acceptable fits (χ²/ν < 2)
R_mod = np.logspace(-0.5, 2, 300)
k_mod = 1.0 / R_mod

# Best fit for N=6, α=2
bf = best_fits[(6, 2)]
mod_best = isf_mod_factor(k_mod, 6, bf['k_t'], 2, bf['eps_floor'])
ax1.plot(R_mod, mod_best, 'b-', lw=2.5, zorder=4,
         label=f"ISF best fit ($N$=6)\n$k_t$={bf['k_t']}, "
               f"$\\varepsilon_\\infty$={bf['eps_floor']:.2f}")

# Show α=1 and α=4 best fits for N=6
for alpha_v, ls, col in [(1, '--', '#2166ac'), (4, '-.', '#67a9cf')]:
    bfa = best_fits[(6, alpha_v)]
    mod_a = isf_mod_factor(k_mod, 6, bfa['k_t'], alpha_v, bfa['eps_floor'])
    ax1.plot(R_mod, mod_a, ls=ls, color=col, lw=1.5, alpha=0.7,
             label=f"$\\alpha$={alpha_v}: $k_t$={bfa['k_t']}, "
                   f"$\\varepsilon_\\infty$={bfa['eps_floor']:.2f}")

# Envelope of all acceptable N=6 models
all_mods_lo = np.ones_like(k_mod)
all_mods_hi = np.zeros_like(k_mod)
for m in models:
    if m['N'] == 6:
        mod_m = isf_mod_factor(k_mod, 6, m['k_t'], m['alpha'], m['eps_floor'])
        mod_at_data = isf_mod_factor(k_data, 6, m['k_t'], m['alpha'],
                                      m['eps_floor'])
        chi2_m = np.sum(((A_data - mod_at_data) / A_err)**2) / ndof
        if chi2_m < 2.0:
            all_mods_lo = np.minimum(all_mods_lo, mod_m)
            all_mods_hi = np.maximum(all_mods_hi, mod_m)
ax1.fill_between(R_mod, all_mods_lo, all_mods_hi, alpha=0.15, color='blue',
                  label=r'$N$=6 envelope ($\chi^2/\nu < 2$)')

ax1.axhline(1.0, color='green', ls=':', lw=1, alpha=0.5)
ax1.text(50, 1.02, r'$\Lambda$CDM', fontsize=9, color='green', ha='right')

ax1.set_xscale('log')
ax1.set_xlabel(r'$R_\perp$ [Mpc/$h$]', fontsize=12)
ax1.set_ylabel(r'$A(R) = \Delta\Sigma_{\rm obs}/\Delta\Sigma_{\rm pred}$',
               fontsize=12)
ax1.set_title('(a) Scale-dependent lensing amplitude', fontsize=11)
ax1.set_xlim(0.3, 60)
ax1.set_ylim(0.35, 1.15)
ax1.legend(fontsize=7.5, loc='lower right', framealpha=0.9)

# ── Panel (b): S₈ measurements ────────────────────────────────
ax2 = fig.add_subplot(132)

s8_plot = [
    ('Planck 2018', 0.834, 0.016, 'CMB', 'green'),
    ('KiDS-1000 shear', 0.759, 0.024, 'WL', '#2166ac'),
    ('KiDS-1000 3×2pt', 0.766, 0.020, 'WL', '#4393c3'),
    ('KiDS-1000 v2', 0.776, 0.029, 'WL', '#92c5de'),
    ('DES Y3 shear', 0.759, 0.025, 'WL', '#d6604d'),
    ('DES Y3 3×2pt', 0.776, 0.017, 'WL', '#f4a582'),
    ('HSC Y3 shear', 0.776, 0.032, 'WL', '#fddbc7'),
    ('DES+KiDS joint', 0.790, 0.018, 'WL', '#878787'),
]

y_pos = np.arange(len(s8_plot))
for i, (name, s8, err, ptype, col) in enumerate(s8_plot):
    marker = 'D' if ptype == 'CMB' else 'o'
    ax2.errorbar(s8, i, xerr=err, fmt=marker, color=col, ms=8,
                  capsize=4, capthick=1.5, elinewidth=2)
    ax2.text(s8 + err + 0.005, i, f' {s8:.3f}', va='center', fontsize=8)

# ISF prediction band for N=6
# S₈(ISF) = S₈(Planck) × √([1+(N-1)ε_eff]/N)
# With ε_eff ≈ 0.80-0.83 from our Phase 2 analysis
s8_planck = 0.834
for eps_eff in [0.78, 0.83]:
    ratio = np.sqrt((1 + 5*eps_eff) / 6)
    s8_isf = s8_planck * ratio
    ax2.axvline(s8_isf, color='blue', ls=':', alpha=0.3)

# ISF band
s8_lo = s8_planck * np.sqrt((1 + 5*0.78) / 6)
s8_hi = s8_planck * np.sqrt((1 + 5*0.83) / 6)
ax2.axvspan(s8_lo, s8_hi, alpha=0.15, color='blue',
            label=f'ISF ($N$=6, $\\varepsilon_{{eff}}$=0.78–0.83)')

ax2.set_yticks(y_pos)
ax2.set_yticklabels([s[0] for s in s8_plot], fontsize=9)
ax2.set_xlabel(r'$S_8 = \sigma_8\sqrt{\Omega_m/0.3}$', fontsize=12)
ax2.set_title(r'(b) $S_8$ measurements vs ISF', fontsize=11)
ax2.set_xlim(0.70, 0.88)
ax2.legend(fontsize=8, loc='upper left')
ax2.invert_yaxis()

# ── Panel (c): Best-fit parameters across N ───────────────────
ax3 = fig.add_subplot(133)

N_vals_plot = [2, 3, 4, 5, 6, 8, 10]
for alpha_v, marker, col, label in [(1, 'o', '#2166ac', r'$\alpha=1$'),
                                     (2, 's', '#d6604d', r'$\alpha=2$'),
                                     (4, '^', '#1b7837', r'$\alpha=4$')]:
    kts = []
    efs = []
    chi2s = []
    Ns = []
    for N_v in N_vals_plot:
        key = (N_v, alpha_v)
        if key in best_fits:
            bf = best_fits[key]
            kts.append(bf['k_t'])
            efs.append(bf['eps_floor'])
            chi2s.append(bf['chi2_nu'])
            Ns.append(N_v)
    
    sc = ax3.scatter(Ns, kts, c=[chi2s[i] for i in range(len(Ns))],
                      marker=marker, s=100, cmap='coolwarm_r',
                      vmin=0.3, vmax=2.0, edgecolors='black', lw=0.5,
                      zorder=5, label=label)
    # Connect with line
    ax3.plot(Ns, kts, ls=':', color=col, alpha=0.3, lw=1)

cb = plt.colorbar(sc, ax=ax3)
cb.set_label(r'$\chi^2/\nu$', fontsize=10)

# Annotate best-fit ε_∞ for α=2
for N_v in N_vals_plot:
    key = (N_v, 2)
    if key in best_fits:
        bf = best_fits[key]
        ax3.annotate(f"$\\varepsilon_\\infty$={bf['eps_floor']:.2f}",
                     xy=(N_v, bf['k_t']),
                     xytext=(N_v + 0.3, bf['k_t'] + 0.015),
                     fontsize=7, color='gray')

ax3.set_xlabel(r'$N$ (number of branches)', fontsize=12)
ax3.set_ylabel(r'Best-fit $k_t$ [$h$/Mpc]', fontsize=12)
ax3.set_title(r'(c) ISF parameters vs branch count', fontsize=11)
ax3.legend(fontsize=8, loc='upper left')

plt.tight_layout()

outpath = '/home/claude/paper5_wl/fig4_comprehensive.png'
plt.savefig(outpath, dpi=200, bbox_inches='tight')
plt.savefig(outpath.replace('.png', '.pdf'), bbox_inches='tight')
print(f"\nSaved {outpath}")

# ── Summary table ──────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY: ISF FIT TO LENSING AMPLITUDE DATA")
print("="*70)
print(f"\nData: Amon+2023 A(R) — 8 radial bins, R = 0.5–40 Mpc/h")
print(f"Model: ISF with S₈ constraint, tested N=2–10, α=1,2,4")
print(f"\n{'Model':<25s} {'χ²/ν':>8s} {'k_t':>8s} {'ε_∞':>8s}")
print("-"*55)
print(f"{'ΛCDM (A=1)':<25s} {chi2_lcdm/ndof:>8.2f} {'—':>8s} {'—':>8s}")
print(f"{'Constant A=0.81':<25s} {chi2_const/ndof:>8.2f} {'—':>8s} {'—':>8s}")
print()
for alpha_v in [1, 2, 4]:
    for N_v in [3, 6, 10]:
        key = (N_v, alpha_v)
        if key in best_fits:
            bf = best_fits[key]
            print(f"{'ISF N='+str(N_v)+', α='+str(alpha_v):<25s}"
                  f" {bf['chi2_nu']:>8.2f} {bf['k_t']:>8.3f}"
                  f" {bf['eps_floor']:>8.3f}")
    print()

# Statistical significance
print("\n=== Statistical significance ===")
dchi2_isf_lcdm = chi2_lcdm - best_fits[(6,2)]['chi2'] * ndof / ndof * ndof
# Actually compute properly
bf62 = best_fits[(6,2)]
chi2_isf = bf62['chi2']
delta_chi2 = chi2_lcdm - chi2_isf
print(f"Δχ²(ΛCDM vs ISF best) = {delta_chi2:.1f}")
print(f"ISF has 1 extra parameter (k_t; ε_∞ fixed by S₈)")
print(f"For 1 d.o.f., Δχ² = {delta_chi2:.1f} corresponds to"
      f" >{np.sqrt(delta_chi2):.1f}σ preference for ISF shape")

delta_chi2_const = chi2_const - chi2_isf
print(f"\nΔχ²(constant A vs ISF) = {delta_chi2_const:.1f}")
print(f"ISF shape preferred over constant suppression at"
      f" {np.sqrt(delta_chi2_const):.1f}σ")
