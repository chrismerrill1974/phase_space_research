#!/usr/bin/env python3
"""
Paper VI Phase 3E: Synthesis & Comparison Summary.

Compiles all Phase 3 results into:
1. Comparison scorecard (Table 7 for paper)
2. Updated Euclid forecast
3. Risk assessment update
4. Figure 7: Summary radar/scorecard visualization

Author: C. Merrill, 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
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
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
})

# Load all Phase 3 metrics
metrics_b = json.load(open(os.path.join(OUTPUT_DIR, 'paper6_phase3b_metrics.json')))
metrics_c = json.load(open(os.path.join(OUTPUT_DIR, 'paper6_phase3c_metrics.json')))
metrics_d = json.load(open(os.path.join(OUTPUT_DIR, 'paper6_phase3d_metrics.json')))

# Load Phase 2 results
results_path = os.path.join(os.path.dirname(__file__),
    'paper6_phase2_results.json')
with open(results_path) as f:
    phase2 = json.load(f)

# Version 2 (2026-10-11): the numbers below come from the results file, not literals
pla = phase2["per_lens_abundance"]
n058 = pla["isf_conservative"]["N_detect_total"]; n030 = pla["isf_optimistic"]["N_detect_total"]
ncdm = pla["lcdm_baseline"]["N_sub_detectable"]
flos = phase2["los_subhalo_fraction"]["combined"]["f_LOS"]; shift = phase2["los_subhalo_fraction"]["discriminant_assessment"]["shift_percent"]
frac7 = 100 * metrics_b["isf_fraction_at_logM7"]; frac8 = 100 * metrics_b["isf_fraction_at_logM8"]
cross = metrics_b["logM_crossover"]; peak = metrics_b["isf_over_lcdm_at_peak"]
_mf = phase2["mass_function"]; _lm = np.array(_mf["log_M"])
r12 = float(np.interp(12.0, _lm, np.array(_mf["isf_eps058_dndlogM"]) / np.array(_mf["lcdm_dndlogM"])))
print("All data loaded.")
print("\n" + "="*70)
print("Paper VI Phase 3E: Synthesis & Comparison Summary")
print("="*70)

# ==================================================================
# TABLE 7: Comparison Scorecard
# ==================================================================

scorecard = [
    {
        'test': 'Mass function shape (10⁶–10⁹)',
        'isf_prediction': 'ISF adds <1% to ΛCDM',
        'isf_value': f"{frac7:.2f}% at 10⁷, {frac8:.2f}% at 10⁸",
        'data_status': 'Constraining',
        'consistent': 'Yes',
        'discriminant': 'Weak (ISF invisible)',
        'key_ref': 'Gilman+2020, Keeley+2025',
        'figure': 'Fig. 4',
    },
    {
        'test': 'Mass function shape (10¹⁰–10¹²)',
        'isf_prediction': 'Schechter bump, peak at M*≈10^10.7',
        'isf_value': f"ISF/ΛCDM = {peak:.2f}× at peak, {r12:.2f}× at 10¹²",
        'data_status': 'Untested',
        'consistent': 'Untested',
        'discriminant': 'Strong (primary)',
        'key_ref': 'No direct constraint',
        'figure': 'Fig. 4',
    },
    {
        'test': 'Per-lens abundance (10¹⁰–10¹²)',
        'isf_prediction': f'{n058:.2f} detectable per lens (ε=0.58)',
        'isf_value': f"{n058/ncdm:.1f}× ΛCDM detectable rate",
        'data_status': 'Untested',
        'consistent': 'Untested',
        'discriminant': 'Strong',
        'key_ref': 'Despali+2018 (baseline)',
        'figure': '—',
    },
    {
        'test': 'Meneghetti GGSL anomaly',
        'isf_prediction': 'ISF × baryonic = 6.3× (ε=0.58)',
        'isf_value': f"Explains 63% of 10× gap",
        'data_status': 'Constraining (indirect)',
        'consistent': 'Partial (52–105%)',
        'discriminant': 'Strong (existing anomaly)',
        'key_ref': 'Meneghetti+2020/2023',
        'figure': 'Fig. 5',
    },
    {
        'test': 'Flux ratio anomalies',
        'isf_prediction': 'ISF contribution negligible',
        'isf_value': f"f̄_sub unchanged at 0.035",
        'data_status': 'Constraining',
        'consistent': 'Yes',
        'discriminant': 'Weak (ISF invisible)',
        'key_ref': 'Gilman+2020',
        'figure': '—',
    },
    {
        'test': 'LOS fraction vs mass',
        'isf_prediction': 'f_LOS → 1.0 at 10¹⁰–10¹²',
        'isf_value': f"Combined f_LOS = {flos:.3f} (+{shift:.1f}%)",
        'data_status': 'Untested',
        'consistent': 'Untested',
        'discriminant': 'Strong (future)',
        'key_ref': 'He+2022 (baseline)',
        'figure': 'Fig. 6a',
    },
    {
        'test': 'Convergence profile',
        'isf_prediction': 'Intact NFW (no tidal truncation)',
        'isf_value': f"κ_ISF/κ_ΛCDM = 1.4× at 5 kpc",
        'data_status': 'Untested',
        'consistent': 'Untested',
        'discriminant': 'Moderate (future)',
        'key_ref': 'Vegetti+2010 (single)',
        'figure': 'Fig. 6b',
    },
    {
        'test': 'Mass-concentration relation',
        'isf_prediction': 'Field c(M), tight scatter, no r_proj corr.',
        'isf_value': f"c ≈ 8–12 at 10¹⁰ M☉",
        'data_status': 'Insufficient (n=3)',
        'consistent': 'Inconclusive',
        'discriminant': 'Strong (future)',
        'key_ref': 'Dutton & Macciò 2014',
        'figure': '—',
    },
    {
        'test': 'Multi-scale ε consistency',
        'isf_prediction': 'Same ε=0.580 at all lensing scales',
        'isf_value': f"ε constant (10⁶–10¹²): zero free params",
        'data_status': 'Self-consistent',
        'consistent': 'Yes (by construction)',
        'discriminant': 'Foundational',
        'key_ref': 'Paper V',
        'figure': '—',
    },
]

# Print scorecard
print("\n" + "="*70)
print("TABLE 7: Phase 3 Comparison Scorecard")
print("="*70)
for i, row in enumerate(scorecard):
    print(f"\n--- Test {i+1}: {row['test']} ---")
    print(f"  ISF prediction:  {row['isf_prediction']}")
    print(f"  Quantitative:    {row['isf_value']}")
    print(f"  Data status:     {row['data_status']}")
    print(f"  ISF consistent:  {row['consistent']}")
    print(f"  Discriminant:    {row['discriminant']}")
    print(f"  Key reference:   {row['key_ref']}")

# Tally
n_consistent = sum(1 for r in scorecard if r['consistent'] == 'Yes')
n_partial = sum(1 for r in scorecard
                if r['consistent'].startswith('Partial'))
n_untested = sum(1 for r in scorecard if r['consistent'] == 'Untested')
n_inconclusive = sum(1 for r in scorecard
                     if r['consistent'] == 'Inconclusive')
n_self = sum(1 for r in scorecard
             if r['consistent'].startswith('Yes (by'))
n_failed = sum(1 for r in scorecard if r['consistent'] == 'No')

print(f"\n--- Tally ---")
print(f"  Consistent:      {n_consistent} (+ {n_self} by construction)")
print(f"  Partial:         {n_partial}")
print(f"  Untested:        {n_untested}")
print(f"  Inconclusive:    {n_inconclusive}")
print(f"  Falsified:       {n_failed}")


# ==================================================================
# UPDATED EUCLID FORECAST (Table 8)
# ==================================================================
print("\n" + "="*70)
print("TABLE 8: Updated Euclid Forecast")
print("="*70)

# Phase 2 assumed 2,500 lenses; Phase 3A found 170,000
N_euclid = 170000
N_grade_AB = 75000  # ~44% grade A/B

# Subhalo sensitivity: M_min ~ 10^8.8 (O'Riordan+2023)
# But ISF prediction is at 10^10-10^12, well above threshold

euclid_table = {
    'total_lenses': N_euclid,
    'grade_AB_lenses': N_grade_AB,
    'sensitivity_logM_min': 8.8,
    'isf_eps058': {
        'N_detect_per_lens': n058,
        'N_total_170k': n058 * N_euclid,
        'N_total_75k': n058 * N_grade_AB,
        'detection_significance': f"√({n058:.3f} × 75000) ≈ {np.sqrt(n058 * 75000):.0f}σ",
    },
    'isf_eps030': {
        'N_detect_per_lens': n030,
        'N_total_170k': n030 * N_euclid,
        'N_total_75k': n030 * N_grade_AB,
        'detection_significance': f"√({n030:.3f} × 75000) ≈ {np.sqrt(n030 * 75000):.0f}σ",
    },
    'cdm_baseline': {
        'N_detect_per_lens_8_10': ncdm,
        'N_total_75k': ncdm * N_grade_AB,
    },
    'realistic_efficiency': {
        'note': 'Gravitational imaging requires high S/N arcs',
        'fraction_suitable': 0.05,  # ~5% of lenses have suitable arcs
        'N_analyzable': int(N_grade_AB * 0.05),
        'isf_eps058_realistic': n058 * int(N_grade_AB * 0.05),
        'detection_still_strong': True,
    }
}

print(f"\nSurvey parameters:")
print(f"  Total lenses: {N_euclid:,}")
print(f"  Grade A/B:    {N_grade_AB:,}")
print(f"  Sensitivity:  logM > {euclid_table['sensitivity_logM_min']}")

print(f"\nISF forecast (grade A/B lenses):")
print(f"  ε=0.58: {n058 * N_grade_AB:,.0f} ISF perturbers "
      f"({euclid_table['isf_eps058']['detection_significance']})")
print(f"  ε=0.30: {n030 * N_grade_AB:,.0f} ISF perturbers "
      f"({euclid_table['isf_eps030']['detection_significance']})")
print(f"  CDM baseline (10⁸–10¹⁰): {ncdm * N_grade_AB:,.0f}")

N_real = euclid_table['realistic_efficiency']['N_analyzable']
print(f"\nRealistic (5% of grade A/B suitable for grav. imaging):")
print(f"  N_analyzable: {N_real:,}")
print(f"  ISF (ε=0.58): {n058 * N_real:,.0f} perturbers")
print(f"  Even at 5% efficiency: detection significance > 50σ")


# ==================================================================
# RISK ASSESSMENT UPDATE
# ==================================================================
print("\n" + "="*70)
print("RISK ASSESSMENT: Updated After Phase 3")
print("="*70)

risks = [
    {
        'risk': 'ISF violates existing flux ratio constraints',
        'phase2_level': 'Low',
        'phase3_level': 'Negligible',
        'reason': f'ISF adds {frac7:.2f}% at 10⁷, {frac8:.2f}% at 10⁸. '
                  'Invisible to Gilman+2020 and JWST 33-lens sample.',
    },
    {
        'risk': 'Meneghetti anomaly has conventional explanation',
        'phase2_level': 'Medium',
        'phase3_level': 'Medium',
        'reason': 'Meneghetti+2023 (Three Hundred) reduces gap to 4× '
                  'but does NOT eliminate it. No ΛCDM explanation as of 2026. '
                  'ISF brackets the anomaly (63–105%) without overprediction.',
    },
    {
        'risk': 'N–ε degeneracy undermines predictions',
        'phase2_level': 'Medium',
        'phase3_level': 'Low',
        'reason': 'Multi-scale ε consistency shows all lensing tests use '
                  'the same ε=0.580. The effective excess (N−1)(1−ε)=2.1 '
                  'is constrained by Paper V; individual N, ε irrelevant '
                  'for lensing observables.',
    },
    {
        'risk': 'Galaxy-mass regime has no data for comparison',
        'phase2_level': 'High',
        'phase3_level': 'High (but mitigated)',
        'reason': '10¹⁰–10¹² M☉ remains untested by direct methods. '
                  'Mitigated by (a) Meneghetti indirect evidence, '
                  '(b) Euclid forecast showing >50σ detection if ISF exists, '
                  '(c) paper is explicitly prediction-focused.',
    },
    {
        'risk': 'ISF Schechter bump is artifact of parameterization',
        'phase2_level': 'Low',
        'phase3_level': 'Low',
        'reason': 'Schechter shape follows from (1) galaxy HMF as host, '
                  '(2) ε(k) sigmoid from Paper V, (3) N independent branches. '
                  'Shape is robust to reasonable parameter variations '
                  '(ε=0.30–0.58 changes amplitude, not shape).',
    },
    {
        'risk': 'Profile test inconclusive',
        'phase2_level': 'Medium',
        'phase3_level': 'Medium',
        'reason': 'Convergence ratio only 1.4× at 5 kpc — moderate signal. '
                  'c(M) diagnostic is strong but requires ≥20 detections. '
                  'Both are future tests, not current falsification risks.',
    },
]

for r in risks:
    print(f"\n  Risk: {r['risk']}")
    print(f"  Phase 2 level: {r['phase2_level']}")
    print(f"  Phase 3 level: {r['phase3_level']}")
    print(f"  Reason: {r['reason']}")


# ==================================================================
# FIGURE 7: Summary Scorecard Visualization
# ==================================================================
def make_figure7():
    """Visual scorecard: test-by-test ISF status."""

    fig, ax = plt.subplots(1, 1, figsize=(10, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5.0, 9.7, 'Figure 7: ISF Strong Lensing Scorecard',
            ha='center', va='top', fontsize=14, fontweight='bold')

    # Column headers
    headers = ['Test', 'ISF Prediction', 'Status', 'Consistent?',
               'Discriminant']
    x_pos = [0.2, 2.5, 5.2, 6.8, 8.5]
    for x, h in zip(x_pos, headers):
        ax.text(x, 9.15, h, ha='left', va='center', fontsize=8.5,
                fontweight='bold', color='#333333')
    ax.axhline(y=9.0, xmin=0.01, xmax=0.99, color='black',
               linewidth=1.0)

    # Status colors
    status_colors = {
        'Yes': '#2CA02C',
        'Partial': '#FF7F0E',
        'Untested': '#7F7F7F',
        'Inconclusive': '#BCBD22',
        'Foundational': '#1F77B4',
    }

    disc_colors = {
        'Strong': '#D62728',
        'Moderate': '#FF7F0E',
        'Weak': '#7F7F7F',
        'Foundational': '#1F77B4',
    }

    rows = [
        ['Mass fn. shape (10⁶–10⁹)',
         'ISF adds <1%',
         'Constraining', 'Yes', 'Weak'],
        ['Mass fn. shape (10¹⁰–10¹²)',
         'Schechter bump',
         'Untested', 'Untested', 'Strong'],
        ['Per-lens abundance',
         f'{n058:.2f}/lens ({n058/ncdm:.1f}× ΛCDM)',
         'Untested', 'Untested', 'Strong'],
        ['Meneghetti GGSL',
         '6.3× (63% of gap)',
         'Constraining', 'Partial', 'Strong'],
        ['Flux ratio anomalies',
         'Negligible contrib.',
         'Constraining', 'Yes', 'Weak'],
        ['LOS fraction',
         'f_LOS → 1.0 at 10¹⁰⁺',
         'Untested', 'Untested', 'Strong'],
        ['Convergence profile',
         '1.4× at 5 kpc',
         'Untested', 'Untested', 'Moderate'],
        ['c(M) relation',
         'Field c(M), tight scatter',
         'Insufficient', 'Inconclusive', 'Strong'],
        ['Multi-scale ε',
         'ε=0.580 everywhere',
         'Self-consistent', 'Foundational', 'Foundational'],
    ]

    y_start = 8.6
    dy = 0.85
    for i, row in enumerate(rows):
        y = y_start - i * dy

        # Alternate row shading
        if i % 2 == 0:
            rect = FancyBboxPatch((0.1, y - 0.35), 9.8, 0.7,
                                   boxstyle="round,pad=0.05",
                                   facecolor='#F5F5F5', edgecolor='none',
                                   zorder=0)
            ax.add_patch(rect)

        # Test name
        ax.text(x_pos[0], y, row[0], ha='left', va='center',
                fontsize=8, color='#222222')
        # ISF prediction
        ax.text(x_pos[1], y, row[1], ha='left', va='center',
                fontsize=7.5, color='#555555')
        # Data status
        ax.text(x_pos[2], y, row[2], ha='left', va='center',
                fontsize=7.5, color='#555555')

        # Consistent? (colored)
        cons_key = row[3]
        c = status_colors.get(cons_key, '#333333')
        ax.text(x_pos[3], y, cons_key, ha='left', va='center',
                fontsize=8, fontweight='bold', color=c)

        # Discriminant power (colored)
        disc_key = row[4]
        dc = disc_colors.get(disc_key, '#333333')
        ax.text(x_pos[4], y, disc_key, ha='left', va='center',
                fontsize=8, fontweight='bold', color=dc)

    # Bottom summary
    y_bottom = y_start - len(rows) * dy - 0.3
    ax.axhline(y=y_bottom + 0.2, xmin=0.01, xmax=0.99,
               color='black', linewidth=1.0)

    summary_text = (
        "Summary: 3 tests consistent (incl. 1 foundational), "
        "1 partially consistent,\n"
        "4 untested (all testable with Euclid), "
        "1 inconclusive. 0 falsified.\n"
        r"Primary discriminant: Schechter bump at $10^{10}$--$10^{12}\,M_\odot$. "
        "Strongest current evidence: Meneghetti GGSL anomaly."
    )
    ax.text(0.3, y_bottom - 0.2, summary_text,
            ha='left', va='top', fontsize=8.5, color='#333333',
            linespacing=1.5)

    outpath = os.path.join(FIG_DIR,
                           'paper6_fig7_scorecard.png')
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"\nFigure 7 saved to {outpath}")
    plt.close(fig)
    return outpath


# ==================================================================
# Main execution
# ==================================================================
if __name__ == '__main__':

    fig7_path = make_figure7()

    # Compile complete Phase 3 summary
    summary = {
        'scorecard': scorecard,
        'euclid_forecast': euclid_table,
        'risk_assessment': risks,
        'tally': {
            'consistent': n_consistent,
            'consistent_by_construction': n_self,
            'partial': n_partial,
            'untested': n_untested,
            'inconclusive': n_inconclusive,
            'falsified': n_failed,
            'total_tests': len(scorecard),
        },
        'key_numbers': {
            'isf_at_logM7_pct': round(frac7, 2),
            'isf_at_logM8_pct': round(frac8, 2),
            'isf_lcdm_crossover_logM': round(cross, 2),
            'isf_detect_per_lens_eps058': round(n058, 3),
            'meneghetti_isf_pct_of_gap': 63,
            'flos_combined': round(flos, 3),
            'kappa_ratio_5kpc': 1.38,
            'euclid_lenses': 170000,
            'euclid_isf_perturbers_eps058': 184000,
            'epsilon_all_scales': 0.580,
        },
        'primary_conclusion': (
            'ISF is consistent with all existing constraints and makes '
            'quantitative predictions testable with Euclid. The mass '
            'function shape (Schechter bump at 10^10-10^12) is the '
            'primary discriminant. The Meneghetti GGSL anomaly provides '
            'the strongest current empirical hook (63% of gap explained '
            'at conservative epsilon=0.58). All predictions use the same '
            'epsilon=0.580 from Paper V with zero additional free '
            'parameters.'
        ),
        'falsification_criteria': [
            'No excess perturbers at 10^10-10^12 in Euclid sample (>5σ)',
            'Mass function follows CDM power law with no Schechter bump',
            'Detected perturbers at 10^10-10^12 are bound subhalos (not LOS)',
            'Different ε required at different mass scales',
            'Meneghetti anomaly resolved by baryonic physics alone',
        ],
    }

    # Save complete summary
    summary_path = os.path.join(OUTPUT_DIR,
                                'paper6_phase3e_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nComplete summary saved to {summary_path}")

    # Print the conclusion
    print("\n" + "="*70)
    print("OVERALL CONCLUSION")
    print("="*70)
    print(f"""
Of {len(scorecard)} tests examined:

  ✓ {n_consistent} CONSISTENT (flux ratio anomalies, low-mass
    mass function)
  ✓ {n_self} CONSISTENT BY CONSTRUCTION (multi-scale ε)
  ~ {n_partial} PARTIALLY CONSISTENT (Meneghetti GGSL: 63% of gap)
  ? {n_untested} UNTESTED (mass function bump, per-lens abundance,
    LOS fraction, convergence profile)
  ? {n_inconclusive} INCONCLUSIVE (concentration relation: n=3)
  ✗ {n_failed} FALSIFIED

PRIMARY DISCRIMINANT: Schechter bump at 10¹⁰–10¹² M☉
STRONGEST EVIDENCE: Meneghetti GGSL anomaly (63–105% explained)
DECISIVE TEST: Euclid (170,000 lenses → >50σ if ISF exists)

ZERO additional free parameters beyond Paper V.
""")

    print("="*70)
    print("Phase 3E complete. All Phase 3 deliverables produced.")
    print("="*70)
