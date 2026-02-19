#!/usr/bin/env python3
"""
Paper VI Phase 3C: Abundance & Meneghetti Comparison.

1. Per-lens abundance: ISF prediction vs published detection rates
2. Meneghetti GGSL: quantitative bracket comparison (Figure 5)
3. Multi-scale epsilon test: what ε at GGSL scales?

Author: C. Merrill, 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

OUTPUT_DIR = '/mnt/user-data/outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
})

# Load Phase 2 results
results_path = os.path.join(os.path.dirname(__file__),
    '../paper6_handoff/phase2_outputs/paper6_phase2_results.json')
with open(results_path) as f:
    phase2 = json.load(f)

pla = phase2['per_lens_abundance']
men = phase2['meneghetti']
evm = phase2['epsilon_vs_mass']

print("Phase 2 data loaded.")

# Load Phase 3A observational data
data3a_path = os.path.join(OUTPUT_DIR, 'paper6_phase3a_data.json')
with open(data3a_path) as f:
    obs = json.load(f)

print("Phase 3A observational data loaded.")

# ==================================================================
# SECTION 1: Per-Lens Abundance Comparison Table
# ==================================================================
print("\n" + "="*65)
print("SECTION 1: Per-Lens Abundance — ISF vs Published Rates")
print("="*65)

# Published detection rates from Phase 3A
# Vegetti+2014: 1 detection in 11 SLACS lenses (10^8-10^10 range)
vegetti_rate = 1.0 / 11.0  # = 0.091 per lens
vegetti_range = "10^8 - 10^10"

# Despali+2018: detection probability 10-20% for 1-in-11
despali_det_prob = 0.15  # midpoint

# Gilman+2020: f_sub = 0.035 → consistent with CDM at 10^6-10^9
gilman_fsub = 0.035

# ISF predictions at 10^10-10^12 (the NEW population)
isf_cons = pla['isf_conservative']
isf_opt = pla['isf_optimistic']
isf_max = pla['isf_maximum']
lcdm_base = pla['lcdm_baseline']

print(f"\nΛCDM baseline (Despali+2018, 10^8-10^10):")
print(f"  N_sub total per host: {lcdm_base['N_sub_total']:.3f}")
print(f"  N_sub detectable:     {lcdm_base['N_sub_detectable']:.3f}")
print(f"  Detection efficiency: {lcdm_base['detection_efficiency']}")

print(f"\nISF ADDITIONAL population (10^10-10^12):")
print(f"{'Scenario':<25s} {'N_exist':>8s} {'N_detect':>8s} "
      f"{'N_LOS':>8s} {'N_Euclid':>10s}")
print("-"*65)
for label, data in [("Conservative (ε=0.58)", isf_cons),
                     ("Optimistic (ε=0.30)", isf_opt),
                     ("Maximum (ε=0.00)", isf_max)]:
    print(f"{label:<25s} {data['N_exist_total']:8.3f} "
          f"{data['N_detect_total']:8.3f} "
          f"{data['N_LOS_detect']:8.3f} "
          f"{data['N_Euclid']:10.0f}")

print(f"\nKey ratio: ISF detectable / ΛCDM detectable (10^8-10^10)")
for label, data in [("Conservative", isf_cons),
                     ("Optimistic", isf_opt)]:
    ratio = data['N_detect_total'] / lcdm_base['N_sub_detectable']
    print(f"  {label}: {data['N_detect_total']:.3f} / "
          f"{lcdm_base['N_sub_detectable']:.3f} = {ratio:.2f}×")

# Euclid expectations (updated: 170,000 lenses from Phase 3A)
N_euclid_lenses = 170000
print(f"\nEuclid forecast ({N_euclid_lenses:,} lenses):")
for label, data in [("Conservative (ε=0.58)", isf_cons),
                     ("Optimistic (ε=0.30)", isf_opt)]:
    n_total = data['N_detect_total'] * N_euclid_lenses
    print(f"  {label}: {n_total:,.0f} ISF perturbers detectable")

# ==================================================================
# SECTION 2: Meneghetti GGSL Quantitative Bracket
# ==================================================================
print("\n" + "="*65)
print("SECTION 2: Meneghetti GGSL Anomaly — Quantitative Bracket")
print("="*65)

obs_original = men['observational']['gap_original']  # 10×
obs_residual = men['observational']['gap_residual']   # 4×
baryonic_boost = men['observational']['baryonic_boost']  # 3×

print(f"\nObservational gaps:")
print(f"  Original (Meneghetti+2020 vs GADGET-X): {obs_original}×")
print(f"  Baryonic boost (AGN feedback, GIZMO-SIMBA): {baryonic_boost}×")
print(f"  Residual gap (after baryonic): {obs_residual}×")

print(f"\nISF predictions:")
for key, label in [('scenario_A', 'Conservative (ε=0.58)'),
                    ('scenario_B', 'Optimistic (ε=0.30)'),
                    ('scenario_C', 'Maximum (ε=0.00)')]:
    s = men[key]
    print(f"  {label}:")
    print(f"    ISF-only excess: {s['isf_excess']:.2f}×")
    print(f"    ISF × baryonic:  {s['isf_x_baryonic']:.2f}×")
    print(f"    vs original gap: {s['vs_original_gap']*100:.0f}%")
    print(f"    vs residual gap: {s['vs_residual_gap']*100:.0f}%")


# ==================================================================
# FIGURE 5: Meneghetti GGSL — ISF Bracket
# ==================================================================
def make_figure5():
    """Bar/waterfall chart: GGSL anomaly decomposition."""

    fig, ax = plt.subplots(1, 1, figsize=(8.0, 5.0))

    # Categories on x-axis
    categories = [
        'ΛCDM\n(GADGET-X)',
        'Baryonic\nboost\n(×3)',
        'ISF\n(ε=0.58)',
        'ISF\n(ε=0.30)',
        'ISF\n(ε=0.00)',
        'Observed\n(Meneghetti+20)'
    ]
    x = np.arange(len(categories))

    # Cumulative GGSL cross-section (normalised to GADGET-X = 1)
    # GADGET-X baseline = 1
    # Baryonic (GIZMO-SIMBA) = 3× baseline
    # ISF additions are MULTIPLICATIVE on top of baryonic
    baseline = 1.0
    baryonic = 3.0
    isf_058_total = men['scenario_A']['isf_x_baryonic']   # 6.29
    isf_030_total = men['scenario_B']['isf_x_baryonic']   # 10.5
    isf_000_total = men['scenario_C']['isf_x_baryonic']   # 15.0
    observed = obs_original  # 10×

    values = [baseline, baryonic, isf_058_total, isf_030_total,
              isf_000_total, observed]
    colors = ['#4575B4', '#74ADD1', '#F46D43', '#D73027',
              '#A50026', '#2CA02C']
    edge_colors = ['#2166AC', '#4393C3', '#D6604D', '#B2182B',
                   '#67001F', '#1B7837']

    bars = ax.bar(x, values, width=0.6, color=colors,
                  edgecolor=edge_colors, linewidth=1.2, zorder=3)

    # Observed line with uncertainty band
    ax.axhline(y=observed, color='#2CA02C', linewidth=2.0,
               linestyle='--', zorder=2, alpha=0.7)
    ax.axhspan(observed * 0.5, observed * 1.5, alpha=0.08,
               color='green', zorder=1)
    ax.text(5.45, observed * 1.05, f'Observed: {observed:.0f}×',
            ha='right', va='bottom', fontsize=8.5,
            color='#1B7837', fontweight='bold')

    # Residual gap line (after baryonic = 4×)
    ax.axhline(y=baryonic + obs_residual, color='orange',
               linewidth=1.2, linestyle=':', zorder=2, alpha=0.6)
    ax.text(0.02, (baryonic + obs_residual)*1.05,
            f'Residual gap = {obs_residual:.0f}× above baryonic',
            ha='left', va='bottom', fontsize=7.5,
            color='darkorange', transform=ax.get_yaxis_transform())

    # Value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        va = 'bottom'
        y_offset = val + 0.3
        label = f'{val:.1f}×' if val != int(val) else f'{val:.0f}×'
        ax.text(bar.get_x() + bar.get_width()/2, y_offset, label,
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color=edge_colors[i])

    # Bracket annotations
    # Conservative ISF explains 63% of original gap
    pct_058 = men['scenario_A']['vs_original_gap'] * 100
    pct_030 = men['scenario_B']['vs_original_gap'] * 100
    ax.annotate(f'{pct_058:.0f}% of gap',
                xy=(2, isf_058_total), xytext=(2, isf_058_total + 3.5),
                ha='center', fontsize=7.5, color='#D6604D',
                arrowprops=dict(arrowstyle='->', color='#D6604D', lw=1.0))
    ax.annotate(f'{pct_030:.0f}% of gap',
                xy=(3, isf_030_total), xytext=(3, isf_030_total + 3.0),
                ha='center', fontsize=7.5, color='#B2182B',
                arrowprops=dict(arrowstyle='->', color='#B2182B', lw=1.0))

    # Formatting
    ax.set_ylabel('GGSL cross-section (× ΛCDM baseline)', fontsize=11)
    ax.set_title('Figure 5: Meneghetti GGSL Anomaly — ISF Contribution',
                 fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8.5)
    ax.set_ylim(0, 18)
    ax.grid(True, axis='y', alpha=0.2, zorder=0)

    # Explanatory text box
    textstr = (
        'ISF adds dark galaxy-mass perturbers\n'
        'that boost the GGSL cross-section.\n'
        r'Conservative: $\varepsilon=0.58$ (Paper V)'
        '\n'
        r'Optimistic: $\varepsilon=0.30$ (lower bound)'
    )
    props = dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                 alpha=0.9, edgecolor='grey')
    ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=7.5,
            verticalalignment='top', bbox=props)

    outpath = os.path.join(OUTPUT_DIR,
                           'paper6_fig5_meneghetti_bracket.png')
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"\nFigure 5 saved to {outpath}")
    plt.close(fig)
    return outpath


# ==================================================================
# SECTION 3: Concentration Anomaly — ISF Interpretation
# ==================================================================
def print_concentration_analysis():
    """The c~60 anomaly as ISF diagnostic."""
    print("\n" + "="*65)
    print("SECTION 3: Concentration Anomaly — ISF Interpretation")
    print("="*65)

    # From Phase 3A: Vegetti detections show c ~ 60
    # CDM c(M) relation at 10^9 M☉ predicts c ~ 15-20
    # Factor 3-4× higher than expected

    print("""
Observed:
  Vegetti+2010 (J0946+1006): M = 3.51 × 10⁹ M☉, c ~ 60
  CDM c(M) relation at 10⁹ M☉: c ~ 15-20 (Dutton & Macciò 2014)
  Discrepancy: factor ~3-4×

Standard CDM explanation:
  Tidal stripping removes outer mass, leaving compact remnant
  → Apparent concentration increases as M_bound/M_infall decreases
  → But requires fine-tuning: enough stripping to boost c,
    not so much that subhalo is destroyed

ISF interpretation:
  ISF halos form independently (no tidal environment)
  → Intact NFW profiles with concentration set at FORMATION
  → ISF halos at 10⁹-10¹⁰ M☉ collapsed at higher redshift
    in their own branch → naturally higher c
  → No tidal stripping needed to explain high concentration
  → Prediction: ISF-dominated perturbers (10¹⁰-10¹²) should
    show SYSTEMATIC c-excess relative to CDM c(M) relation

Discriminating test:
  If concentration excess correlates with mass in the ISF bump
  range (10¹⁰-10¹²) but NOT in the CDM-dominated range
  (10⁷-10⁹), this would be a clean ISF signature.

  Required: ≥10 subhalo detections with mass AND concentration
  constraints. Current sample (3 detections) is insufficient.
""")


# ==================================================================
# SECTION 4: Multi-Scale Epsilon Test
# ==================================================================
def print_multiscale_test():
    """What ε is required at GGSL cluster scales?"""
    print("\n" + "="*65)
    print("SECTION 4: Multi-Scale Epsilon Consistency")
    print("="*65)

    # The ε(k) sigmoid from Paper V gives ε at each scale
    logM = np.array(evm['log_M'])
    eps = np.array(evm['epsilon'])
    k = np.array(evm['k_hMpc'])

    # Cluster scales: M ~ 10^14-10^15 (where GGSL operates)
    # But GGSL excess is from GALAXY-MASS substructure within clusters
    # → relevant scale is 10^10-10^12 (the perturbers, not the host)
    print(f"\nε at key mass scales (from Paper V sigmoid):")
    for target_logM in [8, 9, 10, 10.7, 11, 12, 13]:
        idx = np.argmin(np.abs(logM - target_logM))
        print(f"  logM = {logM[idx]:5.1f}  →  ε = {eps[idx]:.4f}  "
              f"(k = {k[idx]:.1f} h/Mpc)")

    print(f"""
Key insight:
  The ε(k) sigmoid is nearly flat at ε ≈ 0.58 across 10⁸-10¹²
  because all these masses correspond to k > k_t (= 0.20 h/Mpc)
  where ISF branches are fully correlated.

  This means the ISF predictions for:
    - Flux ratio anomalies (10⁶-10⁹): ε ≈ 0.580 ✓
    - Gravitational imaging (10⁸-10¹⁰): ε ≈ 0.580 ✓
    - GGSL cross-section (10¹⁰-10¹²): ε ≈ 0.580 ✓
  are ALL derived from the SAME ε value.

  There are NO additional free parameters between regimes.
  This is the strongest self-consistency test of the ISF framework.

  If Meneghetti excess requires ε < 0.58 (more independent branches)
  to explain the full gap, this would indicate either:
    (a) additional baryonic physics not yet modelled, OR
    (b) scale-dependent deviation from the Paper V sigmoid
  The conservative ε=0.58 explains 63% of the original gap;
  the full gap requires ε ≈ 0.30 at these scales.
""")


# ==================================================================
# Main execution
# ==================================================================
if __name__ == '__main__':
    print("\n" + "="*65)
    print("Paper VI Phase 3C: Abundance & Meneghetti Comparison")
    print("="*65)

    fig5_path = make_figure5()
    print_concentration_analysis()
    print_multiscale_test()

    # Compile Phase 3C metrics
    metrics = {
        'per_lens_abundance': {
            'lcdm_detectable_per_lens_8_10': lcdm_base['N_sub_detectable'],
            'isf_detectable_per_lens_10_12_eps058': isf_cons['N_detect_total'],
            'isf_detectable_per_lens_10_12_eps030': isf_opt['N_detect_total'],
            'isf_los_fraction_eps058': (isf_cons['N_LOS_detect'] /
                                         isf_cons['N_detect_total']),
            'euclid_isf_total_eps058': isf_cons['N_detect_total'] * N_euclid_lenses,
            'euclid_isf_total_eps030': isf_opt['N_detect_total'] * N_euclid_lenses,
            'N_euclid_lenses': N_euclid_lenses,
        },
        'meneghetti_bracket': {
            'gap_original': obs_original,
            'gap_residual': obs_residual,
            'isf_eps058_x_baryonic': men['scenario_A']['isf_x_baryonic'],
            'isf_eps030_x_baryonic': men['scenario_B']['isf_x_baryonic'],
            'isf_eps058_pct_original': men['scenario_A']['vs_original_gap'] * 100,
            'isf_eps030_pct_original': men['scenario_B']['vs_original_gap'] * 100,
            'isf_eps058_pct_residual': men['scenario_A']['vs_residual_gap'] * 100,
            'isf_eps030_pct_residual': men['scenario_B']['vs_residual_gap'] * 100,
        },
        'concentration_anomaly': {
            'observed_c': 60,
            'cdm_expected_c': '15-20',
            'excess_factor': '3-4×',
            'isf_explanation': 'Intact NFW, high-z formation',
            'required_detections': '>=10 with mass+concentration',
        },
        'multiscale_epsilon': {
            'eps_at_logM8': 0.580,
            'eps_at_logM10': 0.580,
            'eps_at_logM12': 0.580,
            'note': 'All scales use same ε from Paper V sigmoid',
        }
    }

    metrics_path = os.path.join(OUTPUT_DIR,
                                'paper6_phase3c_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    print("\n" + "="*65)
    print("Phase 3C complete.")
    print("="*65)
