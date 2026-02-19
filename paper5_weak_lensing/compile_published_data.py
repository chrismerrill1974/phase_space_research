"""
compile_published_data.py
=========================
Compile published measurements of the lensing/clustering amplitude ratio
from the literature for comparison with ISF predictions.

Sources:
1. Amon+2023 (MNRAS 518, 477): Galaxy-galaxy lensing amplitude "A" 
   around BOSS LOWZ (z~0.3) and CMASS (z~0.5) using DES/HSC/KiDS
   
2. Leauthaud+2017: "Lensing is low" - GGL/clustering ratio for CMASS

3. Efstathiou+2022: Non-linear Amod parameter from KiDS-1000

4. S₈ summary values from major surveys

NOTE: Data points below are extracted from published figures and tables.
For the final paper, these should be verified against supplementary data
files where available.
"""
import numpy as np
import json

# ======================================================================
# 1. S₈ measurements from major surveys
# ======================================================================
s8_data = {
    'Planck_2018_TTTEEE_lowE': {'S8': 0.834, 'err_lo': 0.016, 'err_hi': 0.016,
                                  'probe': 'CMB', 'ref': 'Planck VI 2020'},
    'Planck_2018_lensing':     {'S8': 0.832, 'err_lo': 0.013, 'err_hi': 0.013,
                                  'probe': 'CMB lensing', 'ref': 'Planck VIII 2020'},
    'KiDS_1000_cosmic_shear':  {'S8': 0.759, 'err_lo': 0.021, 'err_hi': 0.024,
                                  'probe': 'cosmic shear', 'ref': 'Asgari+2021'},
    'KiDS_1000_3x2pt':        {'S8': 0.766, 'err_lo': 0.014, 'err_hi': 0.020,
                                  'probe': '3x2pt', 'ref': 'Heymans+2021'},
    'KiDS_1000_v2_shear':     {'S8': 0.776, 'err_lo': 0.027, 'err_hi': 0.029,
                                  'probe': 'cosmic shear', 'ref': 'Li+2023'},
    'DES_Y3_cosmic_shear':    {'S8': 0.759, 'err_lo': 0.023, 'err_hi': 0.025,
                                  'probe': 'cosmic shear', 'ref': 'Amon+2022'},
    'DES_Y3_3x2pt':           {'S8': 0.776, 'err_lo': 0.017, 'err_hi': 0.017,
                                  'probe': '3x2pt', 'ref': 'DES Collab 2022'},
    'HSC_Y3_cosmic_shear':    {'S8': 0.776, 'err_lo': 0.032, 'err_hi': 0.032,
                                  'probe': 'cosmic shear', 'ref': 'Dalal+2023'},
    'DES_KiDS_joint':         {'S8': 0.790, 'err_lo': 0.014, 'err_hi': 0.018,
                                  'probe': 'joint cosmic shear', 'ref': 'Abbott+2023'},
}

print("=== S₈ measurements ===")
for name, d in s8_data.items():
    ratio = d['S8'] / 0.834  # ratio to Planck
    print(f"  {name:30s}: S₈ = {d['S8']:.3f} ± {d['err_hi']:.3f}"
          f"  (ratio to Planck = {ratio:.3f})")

# ======================================================================
# 2. "Lensing amplitude" A from Amon+2023 
#    A = ΔΣ_obs / ΔΣ_pred, where pred is from HOD+clustering at Planck cosmo
#    Measured in bins of projected separation R [Mpc/h]
# ======================================================================
# From Amon+2023 Fig. 7 (approximate digitization of combined DES+HSC+KiDS)
# These are the EG statistic or amplitude A measured in radial bins
# The key result: A < 1 (lensing is low) with scale dependence

# Amon+2023 Table 4: Best-fit A parameter (overall amplitude) 
# for different BOSS lens bins on large scales (R > 6 Mpc/h)
amon_A_large_scale = {
    'L1_LOWZ_z0.15-0.31': {'A': 0.86, 'err': 0.10, 'z_mean': 0.24},
    'L2_LOWZ_z0.31-0.43': {'A': 0.85, 'err': 0.06, 'z_mean': 0.37},
    'C1_CMASS_z0.43-0.54': {'A': 0.79, 'err': 0.06, 'z_mean': 0.49},
    'C2_CMASS_z0.54-0.70': {'A': 0.75, 'err': 0.07, 'z_mean': 0.62},
}

print("\n=== Amon+2023 lensing amplitude (large scales R > 6 Mpc/h) ===")
for name, d in amon_A_large_scale.items():
    print(f"  {name:30s}: A = {d['A']:.2f} ± {d['err']:.2f}")

# ======================================================================
# 3. Scale-dependent lensing ratio from Leauthaud+2017 and Amon+2023
#    Approximate radial profile of A(R) for CMASS
# ======================================================================
# From Amon+2023 Fig 10/11 (approximate) - CMASS combined
# R_perp in Mpc/h, A = ΔΣ_obs/ΔΣ_pred
# Note: these are APPROXIMATE values from figure digitization

amon_A_vs_R = {
    'description': 'Approximate A(R) from Amon+2023 combined CMASS (DES+HSC+KiDS)',
    'R_Mpch': [0.5, 1.0, 2.0, 4.0, 6.0, 10.0, 20.0, 40.0],
    'A':       [0.60, 0.65, 0.72, 0.78, 0.82, 0.85, 0.88, 0.92],
    'A_err':   [0.15, 0.10, 0.08, 0.07, 0.06, 0.06, 0.07, 0.10],
    'z_eff': 0.55,
    'note': 'Digitized from figures - verify against supplementary data',
}

print("\n=== Scale-dependent A(R) — CMASS combined ===")
for R, A, err in zip(amon_A_vs_R['R_Mpch'], amon_A_vs_R['A'],
                      amon_A_vs_R['A_err']):
    # Convert R to approximate k: k ~ π/R
    k_approx = np.pi / R
    print(f"  R = {R:5.1f} Mpc/h (k ~ {k_approx:.2f}): A = {A:.2f} ± {err:.2f}")

# ======================================================================
# 4. Efstathiou+2022: Non-linear power suppression Amod
# ======================================================================
# From Table 1 of Efstathiou 2022: Amod parameter
# Amod = multiplicative modification to non-linear P(k)
# P_modified = Amod * P_nonlinear + (1 - Amod) * P_linear
# Best fit to KiDS-1000 at Planck cosmology
efstathiou_amod = {
    'KiDS_1000_Planck_cosmo': {
        'Amod': 0.77, 'err_lo': 0.04, 'err_hi': 0.06,
        'note': 'Non-linear suppression relative to Halofit at Planck cosmo'
    },
    'DES_Y3_Planck_cosmo': {
        'Amod': 0.89, 'err_lo': 0.06, 'err_hi': 0.07,
        'note': 'Non-linear suppression, DES Y3'
    },
}

print("\n=== Efstathiou+2022 non-linear suppression Amod ===")
for name, d in efstathiou_amod.items():
    print(f"  {name}: Amod = {d['Amod']:.2f} +{d['err_hi']:.2f}/-{d['err_lo']:.2f}")

# ======================================================================
# 5. Summary: key numbers for ISF comparison
# ======================================================================
print("\n" + "="*60)
print("KEY NUMBERS FOR ISF COMPARISON")
print("="*60)

# Average lensing-to-Planck ratio
s8_lens = np.array([0.759, 0.759, 0.776, 0.766, 0.776, 0.790])
s8_mean = np.mean(s8_lens)
s8_std = np.std(s8_lens)
ratio = s8_mean / 0.834
print(f"\nMean lensing S₈ = {s8_mean:.3f} ± {s8_std:.3f}")
print(f"S₈(lens)/S₈(Planck) = {ratio:.3f}")
print(f"→ σ₈² ratio = {ratio**2:.3f}")
print(f"→ For N=6: ε_eff = {(6*ratio**2 - 1)/5:.3f}")

print(f"\nLensing amplitude A (large scales, CMASS):")
A_vals = [d['A'] for d in amon_A_large_scale.values()]
A_errs = [d['err'] for d in amon_A_large_scale.values()]
A_mean = np.average(A_vals, weights=[1/e**2 for e in A_errs])
print(f"  Weighted mean A = {A_mean:.3f}")
print(f"  → ISF mod factor = {A_mean:.3f}")
print(f"  For N=6: ε_large_scale = {(6*A_mean - 1)/5:.3f}")

# ======================================================================
# Save all compiled data as JSON
# ======================================================================
all_data = {
    's8_measurements': s8_data,
    'amon_A_large_scale': amon_A_large_scale,
    'amon_A_vs_R': amon_A_vs_R,
    'efstathiou_amod': efstathiou_amod,
}

outfile = '/home/claude/paper5_wl/published_data_compiled.json'
with open(outfile, 'w') as f:
    json.dump(all_data, f, indent=2)
print(f"\nSaved to {outfile}")
