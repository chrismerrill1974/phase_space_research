#!/usr/bin/env python3
"""
Paper VI Phase 3D: LOS Fraction & Profile Tests (Secondary Discriminants).

1. LOS fraction: ISF prediction vs He+2022 constraints
2. Density profile: ISF intact NFW vs LCDM tidally truncated
3. Mass-concentration relation: ISF field c(M) vs CDM subhalo c(M)
4. Figure 6: Two-panel — (a) LOS fraction vs mass, (b) convergence profile

Author: C. Merrill, 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))   # version 2 (2026-10-11): this folder
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
})

# Load Phase 2 results
results_path = os.path.join(os.path.dirname(__file__),
    'paper6_phase2_results.json')
with open(results_path) as f:
    phase2 = json.load(f)

los = phase2['los_subhalo_fraction']
prof = phase2['profile_comparison']
mf = phase2['mass_function']

print("Phase 2 data loaded.")

# ==================================================================
# SECTION 1: LOS Fraction Analysis
# ==================================================================
print("\n" + "="*65)
print("SECTION 1: LOS Fraction — ISF vs He+2022")
print("="*65)

# He+2022 constraints (from Phase 3A):
# CDM: N_LOS/N_sub = 3 (z_l=0.2, z_s=1.0) to 10 (z_l=0.5, z_s=2.0)
# → f_LOS = 0.75 to 0.91 for CDM
# ISF: adds population that is ~100% LOS at 10^10-10^12

# Build mass-dependent LOS fraction model
logM = np.array(mf['log_M'])
lcdm_mf = np.array(mf['lcdm_dndlogM'])
isf_058 = np.array(mf['isf_eps058_dndlogM'])

# CDM LOS fraction varies with mass:
# Low mass (10^6-10^9): mix of sub + LOS, He+2022 says f_LOS ~ 0.67
# High mass (10^10-10^12): ISF adds ~100% LOS population
# We compute the COMBINED f_LOS at each mass bin

# CDM: f_LOS depends on lens/source config. Use He+2022 fiducial
# For subhalo contribution: concentrated near host
# For LOS: distributed along sightline
# At low masses, CDM gives f_LOS ~ 0.67 (He+2022 baseline)
f_los_cdm = 0.67  # He+2022 fiducial for 10^8-10^10

# ISF halos are ALL LOS (independent formation → not bound to host)
# Combined f_LOS at each mass:
# f_LOS(M) = [f_LOS_CDM * N_CDM(M) + 1.0 * N_ISF(M)] / [N_CDM(M) + N_ISF(M)]
combined_flos = np.zeros_like(logM)
for i in range(len(logM)):
    n_cdm = lcdm_mf[i]
    n_isf = isf_058[i]
    total = n_cdm + n_isf
    if total > 0:
        combined_flos[i] = (f_los_cdm * n_cdm + 1.0 * n_isf) / total
    else:
        combined_flos[i] = f_los_cdm

print(f"\nLOS fraction model:")
print(f"  CDM baseline (He+2022): f_LOS = {f_los_cdm}")
print(f"  ISF addition: f_LOS = 1.00 (all LOS)")
print(f"\nCombined f_LOS at key masses:")
for target in [7, 8, 9, 10, 10.7, 11, 12]:
    idx = np.argmin(np.abs(logM - target))
    print(f"  logM={logM[idx]:5.1f}: f_LOS = {combined_flos[idx]:.4f} "
          f"(CDM contrib: {lcdm_mf[idx]:.4f}, ISF: {isf_058[idx]:.4f})")

# Overall combined values from Phase 2
print(f"\nPhase 2 combined per-lens:")
print(f"  N_total = {los['combined']['N_total']:.3f}")
print(f"  f_LOS = {los['combined']['f_LOS']:.4f}")
print(f"  Shift from CDM: +{los['discriminant_assessment']['shift_percent']:.1f}%")

# ==================================================================
# SECTION 2: Density Profile Comparison
# ==================================================================
print("\n" + "="*65)
print("SECTION 2: Density Profile — ISF Intact vs LCDM Truncated")
print("="*65)

r_kpc = np.array(prof['r_kpc'])
kappa_isf = np.array(prof['kappa_isf'])
kappa_lcdm = np.array(prof['kappa_lcdm'])
rho_isf = np.array(prof['rho_isf'])
rho_lcdm = np.array(prof['rho_lcdm'])

params = prof['parameters']
conv_ratios = prof['convergence_ratios']
mass_comp = prof['mass_comparison']

print(f"\nHalo parameters:")
print(f"  M_halo = {params['M_halo']:.1e} M☉")
print(f"  c_ISF = {params['c_isf']} (intact NFW)")
print(f"  c_LCDM = {params['c_lcdm']} (before stripping)")
print(f"  r_trunc = {params['r_trunc_kpc']:.1f} kpc (tidal)")
print(f"  R_vir = {params['R_vir_kpc']:.1f} kpc")

print(f"\nConvergence ratios:")
print(f"  κ_ISF/κ_LCDM at 1 kpc:  {conv_ratios['kappa_isf_over_lcdm_at_1kpc']:.3f}")
print(f"  κ_ISF/κ_LCDM at 5 kpc:  {conv_ratios['kappa_isf_over_lcdm_at_5kpc']:.3f}")
print(f"  κ_ISF/κ_LCDM at 10 kpc: {conv_ratios['kappa_isf_over_lcdm_at_10kpc']:.3f}")

print(f"\nMass comparison:")
print(f"  M_bound(ISF) = {mass_comp['M_bound_isf']:.2e} M☉")
print(f"  M_bound(LCDM) = {mass_comp['M_bound_lcdm']:.2e} M☉")
print(f"  LCDM mass loss: {mass_comp['mass_loss_fraction_lcdm']*100:.1f}%")


# ==================================================================
# FIGURE 6: Two-panel — (a) LOS fraction, (b) Convergence profile
# ==================================================================
def make_figure6():
    """Two-panel figure for secondary discriminants."""

    fig = plt.figure(figsize=(10, 4.5))
    gs = GridSpec(1, 2, figure=fig, wspace=0.35)

    # ---- Panel (a): LOS fraction vs mass ----
    ax1 = fig.add_subplot(gs[0])

    # CDM-only baseline
    ax1.axhline(y=f_los_cdm, color='black', linewidth=1.5,
                linestyle='--', label=r'$\Lambda$CDM (He+2022)', zorder=3)

    # CDM range from He+2022: f_LOS = 0.75 to 0.91 depending on z_l, z_s
    ax1.axhspan(0.60, 0.91, alpha=0.10, color='grey', zorder=1,
                label=r'CDM range ($z_l$, $z_s$ dependent)')

    # Combined ISF + CDM
    ax1.plot(logM, combined_flos, '-', color='#B2182B', linewidth=2.0,
             label=r'$\Lambda$CDM + ISF ($\varepsilon=0.58$)', zorder=5)

    # Highlight the transition
    ax1.fill_between(logM, f_los_cdm, combined_flos,
                     where=combined_flos > f_los_cdm + 0.01,
                     alpha=0.15, color='#B2182B', zorder=2)

    # Annotations
    ax1.annotate('ISF signal:\ntransition\nto ~100% LOS',
                 xy=(11.0, 0.96), fontsize=7.5, ha='center',
                 color='#B2182B', fontstyle='italic',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                           edgecolor='#B2182B', alpha=0.8))

    # Mark key mass ranges
    ax1.axvspan(6, 9, alpha=0.04, color='green', zorder=0)
    ax1.axvspan(10, 13, alpha=0.04, color='salmon', zorder=0)

    ax1.set_xlim(6, 13)
    ax1.set_ylim(0.55, 1.02)
    ax1.set_xlabel(r'$\log_{10}(M_{\rm halo}/M_\odot)$')
    ax1.set_ylabel(r'$f_{\rm LOS}$ (LOS fraction)')
    ax1.set_title('(a) LOS Fraction vs Halo Mass')
    ax1.legend(loc='lower right', fontsize=7, framealpha=0.9)
    ax1.grid(True, alpha=0.15)

    # ---- Panel (b): Convergence profile comparison ----
    ax2 = fig.add_subplot(gs[1])

    # Only plot where both are nonzero
    mask = (np.array(kappa_isf) > 0) & (np.array(kappa_lcdm) > 0)
    r_plot = np.array(r_kpc)[mask]
    k_isf_plot = np.array(kappa_isf)[mask]
    k_lcdm_plot = np.array(kappa_lcdm)[mask]

    ax2.plot(r_plot, k_isf_plot, '-', color='#2166AC', linewidth=2.0,
             label='ISF (intact NFW)', zorder=4)
    ax2.plot(r_plot, k_lcdm_plot, '--', color='black', linewidth=1.8,
             label=r'$\Lambda$CDM (truncated at $r_t$)', zorder=3)

    # Highlight the ratio enhancement
    ratio_plot = k_isf_plot / k_lcdm_plot
    # Add inset or second axis for ratio
    ax2_r = ax2.twinx()
    ax2_r.plot(r_plot, ratio_plot, ':', color='#D6604D', linewidth=1.2,
               alpha=0.7, zorder=2)
    ax2_r.set_ylabel(r'$\kappa_{\rm ISF}/\kappa_{\Lambda\rm CDM}$',
                     color='#D6604D', fontsize=9)
    ax2_r.tick_params(axis='y', labelcolor='#D6604D')
    ax2_r.set_ylim(0.8, 50)
    ax2_r.set_yscale('log')

    # Mark truncation radius
    r_trunc = params['r_trunc_kpc']
    ax2.axvline(x=r_trunc, color='grey', linewidth=0.8, linestyle=':',
                alpha=0.6, zorder=1)
    ax2.text(r_trunc * 1.1, 0.03, r'$r_t$',
             fontsize=9, color='grey', va='bottom')

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim(0.5, 100)
    ax2.set_ylim(1e-5, 0.1)
    ax2.set_xlabel(r'$r$ [kpc]')
    ax2.set_ylabel(r'$\kappa(r)$ (convergence)')
    ax2.set_title(r'(b) Convergence Profile ($10^{10}\,M_\odot$ halo)')
    ax2.legend(loc='upper right', fontsize=7, framealpha=0.9)
    ax2.grid(True, alpha=0.15, which='major')

    outpath = os.path.join(FIG_DIR,
                           'paper6_fig6_los_profile.png')
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"\nFigure 6 saved to {outpath}")
    plt.close(fig)
    return outpath


# ==================================================================
# SECTION 3: Mass-Concentration Relation
# ==================================================================
def print_concentration_test():
    """Mass-concentration diagnostic for ISF vs CDM subhalos."""
    print("\n" + "="*65)
    print("SECTION 3: Mass-Concentration Relation Diagnostic")
    print("="*65)

    print("""
ISF prediction for detected perturbers:
  ISF halos are field halos (never tidally processed)
  → Follow the Dutton & Macciò (2014) FIELD c(M) relation
  → At 10^10 M☉, z=0: c ≈ 9-12
  → At 10^10 M☉, z=0.5 (typical lens): c ≈ 7-10

CDM prediction for subhalos:
  CDM subhalos are tidally stripped
  → Bound mass decreases, central density preserved
  → APPARENT concentration increases: c_app = c_infall × (M_infall/M_bound)^0.4
  → For M_infall = 2 × M_bound: c_app ≈ 1.3 × c_infall
  → For M_infall = 5 × M_bound: c_app ≈ 2.0 × c_infall
  → This gives c_app = 15-25 for typical subhalos

Diagnostic:
  CDM subhalos: c_apparent >> c_field(M)     (tidal boost)
  ISF halos:    c ≈ c_field(M)               (no tidal processing)
  LOS halos:    c ≈ c_field(M)               (also unprocessed)

  KEY: The existing Vegetti detection (c ~ 60) is problematic
  for BOTH models at face value:
    - CDM needs extreme stripping (M_infall/M_bound > 10)
    - ISF needs high-z formation or unusual formation history

  The discriminating test is STATISTICAL: across a sample of
  ≥20 detections at 10^10-10^12, ISF predicts:
    - Tight scatter in c around c_field(M) ~ 8-12
    - No correlation between c and projected distance to host
    - c independent of host halo mass

  CDM predicts:
    - Wide scatter in c (reflecting varied tidal histories)
    - c increases for subhalos closer to host center
    - c correlates with host mass (stronger tides → more stripping)

  Current data: INSUFFICIENT (3 detections with weak c constraints)
  Required: ≥20 detections with joint M, c, projected distance
  Timeline: Euclid era (2028+)
""")


# ==================================================================
# Main execution
# ==================================================================
if __name__ == '__main__':
    print("\n" + "="*65)
    print("Paper VI Phase 3D: LOS Fraction & Profile Tests")
    print("="*65)

    fig6_path = make_figure6()
    print_concentration_test()

    # Compile metrics
    metrics = {
        'los_fraction': {
            'cdm_baseline_flos': f_los_cdm,
            'cdm_range_flos': '0.60-0.91',
            'combined_flos_at_logM7': float(combined_flos[np.argmin(np.abs(logM-7))]),
            'combined_flos_at_logM9': float(combined_flos[np.argmin(np.abs(logM-9))]),
            'combined_flos_at_logM10': float(combined_flos[np.argmin(np.abs(logM-10))]),
            'combined_flos_at_logM11': float(combined_flos[np.argmin(np.abs(logM-11))]),
            'combined_flos_at_logM12': float(combined_flos[np.argmin(np.abs(logM-12))]),
            'overall_combined_flos': los['combined']['f_LOS'],
            'shift_from_cdm_pct': los['discriminant_assessment']['shift_percent'],
            'discriminant_strength': los['discriminant_assessment']['strength'],
        },
        'profile_comparison': {
            'M_halo': params['M_halo'],
            'c_isf': params['c_isf'],
            'c_lcdm': params['c_lcdm'],
            'r_trunc_kpc': params['r_trunc_kpc'],
            'kappa_ratio_1kpc': conv_ratios['kappa_isf_over_lcdm_at_1kpc'],
            'kappa_ratio_5kpc': conv_ratios['kappa_isf_over_lcdm_at_5kpc'],
            'kappa_ratio_10kpc': conv_ratios['kappa_isf_over_lcdm_at_10kpc'],
            'mass_loss_lcdm_pct': mass_comp['mass_loss_fraction_lcdm'] * 100,
        },
        'concentration_diagnostic': {
            'isf_predicted_c': '8-12 (field c(M) at z~0.5)',
            'cdm_subhalo_c': '15-25 (tidally boosted)',
            'vegetti_observed_c': 60,
            'required_sample': '>=20 detections with M, c, r_proj',
            'discriminant_strength': 'Strong (future)',
            'timeline': 'Euclid era (2028+)',
        },
        'key_predictions': {
            'prediction_1': 'f_LOS transitions from 0.67 to ~1.0 at logM > 10',
            'prediction_2': 'ISF convergence exceeds CDM by 1.4x at 5 kpc',
            'prediction_3': 'ISF perturber c clusters tightly at c_field(M)',
            'prediction_4': 'No c vs host-distance correlation for ISF halos',
        }
    }

    metrics_path = os.path.join(OUTPUT_DIR,
                                'paper6_phase3d_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    print("\n" + "="*65)
    print("Phase 3D complete.")
    print("="*65)
