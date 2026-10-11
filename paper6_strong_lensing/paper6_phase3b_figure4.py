#!/usr/bin/env python3
"""
Paper VI Phase 3B: Figure 4 — Mass Function Shape Comparison.

The 'money figure': ISF prediction vs ΛCDM vs published observational
constraints across the full 10^6–10^12 M☉ range.

Three regimes:
  (i)   Gilman-constrained (10^6–10^9): CDM-consistent, ISF-compatible
  (ii)  Gap (10^9–10^10): sparse data, ISF crossover
  (iii) ISF bump (10^10–10^12): untested, ISF predicts Schechter peak

Author: C. Merrill, 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import LogLocator, NullFormatter
import json
import os
import sys

# Import Phase 2 prediction functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
    '.'))

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))   # version 2 (2026-10-11): this folder
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
})

# ------------------------------------------------------------------
# Load Phase 2 mass function data
# ------------------------------------------------------------------
results_path = os.path.join(os.path.dirname(__file__),
    'paper6_phase2_results.json')
with open(results_path) as f:
    phase2 = json.load(f)

mf = phase2['mass_function']
logM = np.array(mf['log_M'])
lcdm = np.array(mf['lcdm_dndlogM'])
isf_058 = np.array(mf['isf_eps058_dndlogM'])
isf_030 = np.array(mf['isf_eps030_dndlogM'])
isf_000 = np.array(mf['isf_eps000_dndlogM'])

print(f"Loaded mass function: {len(logM)} points, "
      f"logM = {logM[0]:.1f} to {logM[-1]:.1f}")
print(f"LCDM range: {lcdm.min():.4e} to {lcdm.max():.4e}")
print(f"ISF(0.58) range: {isf_058.min():.4e} to {isf_058.max():.4e}")

# ------------------------------------------------------------------
# Observational data points
# ------------------------------------------------------------------

# Vegetti detections (individual subhalo masses)
# Units: these are individual detections, not mass function values.
# We plot them as data points with indicative vertical position.
vegetti_detections = {
    'J0946+1006': {'logM': 9.545, 'logM_err': 0.019,
                   'ref': 'Vegetti+2010'},
    'B1938+666':  {'logM': 8.279, 'logM_err': 0.023,
                   'ref': 'Vegetti+2012'},
    'SDP.81':     {'logM': 8.96,  'logM_err': 0.12,
                   'ref': 'Hezaveh+2016'},
}

# Gilman+2020 constraint: CDM-consistent at 10^6-10^9
# f_sub = 0.035 (+0.021, -0.017) for projected mass fraction
# This constrains the TOTAL (sub + LOS) population. ISF adds <1%.
# We represent this as a shaded band showing the LCDM prediction
# is consistent with data in this range.
gilman_constraint = {
    'logM_range': [6, 9],
    'status': 'CDM-consistent',
    'isf_fraction_added': 0.01,  # <1%
}

# Meneghetti implied excess at galaxy-mass scales
# The GGSL excess implies more galaxy-mass perturbers than CDM predicts.
# We show this as an upward arrow / shaded region at 10^10-10^12.
meneghetti_excess = {
    'logM_range': [10, 12],
    'excess_factor_original': 10,    # vs GADGET-X
    'excess_factor_residual': 4,     # vs GIZMO-SIMBA
    'note': 'Persistent GGSL anomaly; no LCDM explanation',
}

# Despali+2018 CDM baseline slope
despali_slope = -1.9  # dn/dM ~ M^(-1.9)

print("Observational data loaded.")

# ------------------------------------------------------------------
# Compute combined mass functions (LCDM + ISF)
# ------------------------------------------------------------------
total_058 = lcdm + isf_058
total_030 = lcdm + isf_030
total_000 = lcdm + isf_000

# Compute ISF/LCDM ratio at key masses for compatibility metrics
def get_val(arr, target_logM):
    idx = np.argmin(np.abs(logM - target_logM))
    return arr[idx]

print("\n=== Quantitative Compatibility Metrics ===")
for lm in [7, 8, 9, 10, 10.7, 11, 12]:
    lv = get_val(lcdm, lm)
    iv = get_val(isf_058, lm)
    ratio = iv / lv if lv > 0 else np.inf
    print(f"  logM={lm:5.1f}: LCDM={lv:.4f}, ISF(0.58)={iv:.4f}, "
          f"ISF/LCDM={ratio:.3f} ({ratio*100:.1f}%)")

# Crossover mass (where ISF exceeds LCDM)
for i in range(len(logM)-1):
    if isf_058[i] < lcdm[i] and isf_058[i+1] >= lcdm[i+1]:
        frac = (lcdm[i] - isf_058[i]) / ((isf_058[i+1] - isf_058[i])
                - (lcdm[i+1] - lcdm[i]))
        crossover = logM[i] + frac * (logM[i+1] - logM[i])
        print(f"\nISF-LCDM crossover at logM = {crossover:.2f}")
        break

print("=" * 50)

# ------------------------------------------------------------------
# FIGURE 4: The Money Figure
# ------------------------------------------------------------------
def make_figure4():
    """Mass function: LCDM vs ISF vs observational constraints."""

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.5))

    # --- Regime shading (background) ---
    # (i) Gilman-constrained
    ax.axvspan(6, 9, alpha=0.06, color='green', zorder=0)
    # (ii) Transition gap
    ax.axvspan(9, 10, alpha=0.06, color='gold', zorder=0)
    # (iii) ISF prediction zone
    ax.axvspan(10, 12.5, alpha=0.06, color='salmon', zorder=0)

    # --- LCDM total (sub + LOS field halos) ---
    ax.plot(logM, lcdm, 'k-', linewidth=2.0, label=r'$\Lambda$CDM total',
            zorder=4)

    # --- ISF additions (dashed) ---
    ax.plot(logM, isf_058, '--', color='#2166AC', linewidth=1.8,
            label=r'ISF only ($\varepsilon_\infty=0.58$)', zorder=3)
    ax.plot(logM, isf_030, ':', color='#4393C3', linewidth=1.4,
            label=r'ISF only ($\varepsilon_\infty=0.30$)', zorder=3)

    # --- Combined (LCDM + ISF) ---
    ax.plot(logM, total_058, '-', color='#B2182B', linewidth=2.0,
            label=r'$\Lambda$CDM + ISF ($\varepsilon=0.58$)', zorder=5)
    ax.fill_between(logM, total_058, total_030, alpha=0.15,
                    color='#B2182B', zorder=2,
                    label=r'ISF range ($\varepsilon=0.58$–$0.30$)')

    # --- Gilman+2020 constraint band ---
    # Show as a green checkmark zone: data says LCDM is correct here.
    # Gilman constrains f_sub = 0.035 +/- 0.02 which is CDM-consistent.
    # ISF adds <1% here, so both LCDM and LCDM+ISF fall in this band.
    gilman_y_upper = lcdm.copy()
    gilman_y_lower = lcdm.copy()
    mask_g = (logM >= 6) & (logM <= 9)
    # Approximate Gilman posterior width as ~50% uncertainty on normalization
    gilman_y_upper[mask_g] *= 1.6
    gilman_y_lower[mask_g] *= 0.5
    ax.fill_between(logM[mask_g], gilman_y_lower[mask_g],
                    gilman_y_upper[mask_g],
                    alpha=0.18, color='green', zorder=1,
                    label='Gilman+2020 (CDM-consistent)')

    # --- Vegetti individual detections ---
    # These are individual masses, not mass function values.
    # Plot them at approximate implied mass function position.
    # Vegetti+2014: 1 detection in 11 lenses → ~0.09/lens at that mass
    # Detection efficiency ~50% → underlying ~0.18/lens/dex
    # These are order-of-magnitude indicators, shown as stars.
    veg_logM = [9.545, 8.279, 8.96]
    # Approximate: 1 detection per 11 lenses / detection efficiency / dex bin
    # Very rough: place at ~0.1-1 per lens per dex to indicate scale
    veg_y = [0.18, 0.5, 0.3]  # approximate implied dN/dlogM per lens
    veg_labels = ['J0946\n(Vegetti+10)', 'B1938\n(Vegetti+12)',
                  'SDP.81\n(Hezaveh+16)']

    ax.scatter(veg_logM, veg_y, marker='*', s=180, c='darkgreen',
               edgecolors='black', linewidths=0.5, zorder=8,
               label='Individual detections')
    for i, (x, y, lab) in enumerate(zip(veg_logM, veg_y, veg_labels)):
        va = 'bottom' if i != 1 else 'top'
        offset = 1.4 if i != 1 else 0.7
        ax.annotate(lab, (x, y*offset), fontsize=6.5, ha='center',
                    va=va, color='darkgreen', style='italic')

    # --- Meneghetti excess indicator ---
    ax.annotate('', xy=(10.3, 0.10), xytext=(10.3, 0.022),
                arrowprops=dict(arrowstyle='->', color='#D6604D',
                                lw=2.5, mutation_scale=15),
                zorder=7)
    ax.text(10.3, 0.13, 'Meneghetti\n   excess\n ' + r'($\sim\!10\times$)',
            ha='center', va='bottom', fontsize=7.5,
            color='#D6604D', fontweight='bold')

    # --- Regime labels (top) ---
    ax.text(7.5, 3e2, '(i) Gilman-constrained\nCDM-consistent\nISF adds <1%',
            ha='center', va='top', fontsize=7.5, color='#2ca02c',
            fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#2ca02c', alpha=0.8))
    ax.text(9.5, 3e2, '(ii)\nGap',
            ha='center', va='top', fontsize=7.5, color='#bcbd22',
            fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#bcbd22', alpha=0.8))
    ax.text(11.2, 3e2, '(iii) ISF prediction\nSchechter bump\nLargely untested',
            ha='center', va='top', fontsize=7.5, color='#d62728',
            fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#d62728', alpha=0.8),
            zorder=10)

    # --- Formatting ---
    ax.set_yscale('log')
    ax.set_xlim(6, 13)
    ax.set_ylim(5e-4, 1e3)
    ax.set_xlabel(r'$\log_{10}(M_{\rm halo}/M_\odot)$', fontsize=12)
    ax.set_ylabel(r'$dN/d\log_{10}M$ per lens sightline', fontsize=12)
    ax.set_title('Figure 4: Mass Function — ISF vs '
                 r'$\Lambda$CDM vs Observations', fontsize=11)

    # Vertical line at crossover
    ax.axvline(x=10.7, color='grey', linestyle=':', linewidth=0.7,
               alpha=0.5, zorder=1)
    ax.text(10.75, 1e-3, r'$M^*_{\rm ISF}$', fontsize=7, color='grey',
            rotation=90, va='bottom')

    ax.legend(loc='upper right', framealpha=0.9, edgecolor='grey',
              ncol=1, fontsize=7.5, bbox_to_anchor=(0.99, 0.72))
    ax.grid(True, alpha=0.15, which='major')
    ax.grid(True, alpha=0.08, which='minor')

    outpath = os.path.join(FIG_DIR, 'paper6_fig4_mass_function_vs_data.png')
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"\nFigure 4 saved to {outpath}")
    plt.close(fig)
    return outpath

# ------------------------------------------------------------------
# Compatibility metrics computation
# ------------------------------------------------------------------
def compute_metrics():
    """Quantitative compatibility metrics for the paper."""
    metrics = {}

    # 1. ISF contribution at Gilman-constrained masses
    for lm in [7, 8, 9]:
        lv = get_val(lcdm, lm)
        iv = get_val(isf_058, lm)
        metrics[f'isf_fraction_at_logM{lm}'] = iv / lv if lv > 0 else 0

    # 2. Mass at which ISF first exceeds 10% of LCDM
    for i in range(len(logM)):
        if lcdm[i] > 0 and isf_058[i] / lcdm[i] >= 0.10:
            metrics['logM_isf_10pct'] = logM[i]
            break

    # 3. Mass at which ISF exceeds LCDM (crossover)
    for i in range(len(logM)-1):
        if isf_058[i] < lcdm[i] and isf_058[i+1] >= lcdm[i+1]:
            frac = (lcdm[i] - isf_058[i]) / ((isf_058[i+1] - isf_058[i])
                    - (lcdm[i+1] - lcdm[i]))
            metrics['logM_crossover'] = logM[i] + frac*(logM[i+1]-logM[i])
            break

    # 4. ISF dominance at Schechter peak
    metrics['isf_over_lcdm_at_peak'] = (get_val(isf_058, 10.7) /
                                         get_val(lcdm, 10.7))

    # 5. Total excess in 10^10-10^12 range (integrated)
    mask = (logM >= 10) & (logM <= 12)
    dlogM = logM[1] - logM[0]
    lcdm_int = np.sum(lcdm[mask]) * dlogM
    isf_int = np.sum(isf_058[mask]) * dlogM
    metrics['integrated_lcdm_10_12'] = lcdm_int
    metrics['integrated_isf_10_12'] = isf_int
    metrics['integrated_ratio_10_12'] = (lcdm_int + isf_int) / lcdm_int

    # 6. At what mass is ISF detectable given Gilman uncertainties?
    # Gilman uncertainty ~50% on normalization → ISF detectable when
    # ISF/LCDM > 0.5 (would shift total outside Gilman band)
    for i in range(len(logM)):
        if lcdm[i] > 0 and isf_058[i] / lcdm[i] >= 0.5:
            metrics['logM_detectable_gilman'] = logM[i]
            break

    return metrics


# ------------------------------------------------------------------
# Main execution
# ------------------------------------------------------------------
if __name__ == '__main__':
    print("\n" + "="*60)
    print("Paper VI Phase 3B: Mass Function Shape Comparison")
    print("="*60)

    fig4_path = make_figure4()

    metrics = compute_metrics()
    print("\n=== Compatibility Metrics (for paper) ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print("\n=== Key Conclusions ===")
    print(f"1. ISF adds <1% at 10^7-10^8 (Gilman-consistent: "
          f"{metrics['isf_fraction_at_logM7']*100:.2f}% and "
          f"{metrics['isf_fraction_at_logM8']*100:.2f}%)")
    print(f"2. ISF first exceeds 10% of LCDM at logM = "
          f"{metrics.get('logM_isf_10pct', 'N/A')}")
    print(f"3. ISF crosses LCDM at logM = "
          f"{metrics.get('logM_crossover', 'N/A'):.2f}")
    print(f"4. At Schechter peak (10^10.7): ISF/LCDM = "
          f"{metrics['isf_over_lcdm_at_peak']:.2f}x")
    print(f"5. Integrated 10^10-10^12: LCDM+ISF/LCDM = "
          f"{metrics['integrated_ratio_10_12']:.2f}x")
    print(f"6. ISF detectable (>50% of LCDM) at logM > "
          f"{metrics.get('logM_detectable_gilman', 'N/A')}")

    # Save metrics
    metrics_path = os.path.join(OUTPUT_DIR,
                                'paper6_phase3b_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    print("\nPhase 3B complete.")
