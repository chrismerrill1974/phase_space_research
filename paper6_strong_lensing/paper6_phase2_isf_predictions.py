#!/usr/bin/env python3
"""
Paper VI Phase 2: ISF Substructure Predictions

Computes the ISF-predicted subhalo mass function, abundance per lens,
and LOS-to-subhalo ratio using Paper V's ε(k) as input.

Key inputs from Paper V:
  - ε(k) = ε_∞ + (1-ε_∞)/(1+(k/k_t)^α)
  - Best fit: k_t=0.20 h/Mpc, ε_∞=0.58, α=2, N=6
  - S₈ constraint ties ε_∞ to (N, k_t, α)

Author: C. Merrill, 2026

VERSION 2 (2026-10-11). The version-1 script (003_phase2/) evaluated the
stellar mass function, per dex of STELLAR mass, at the abundance-matched
M_star(M_halo) and summed over d log M_halo without the Jacobian
d log M_star / d log M_halo = 1/1.4 of the matching relation
M_star = 10^10.7 (M_halo/1e12)^(1/1.4). This copy carries the Jacobian
(JACOBIAN below, applied in isf_excess_dndm), so every per-lens count
and ISF excess mass function is 1/1.4 of version 1. Before writing its
results it GATES: with the Jacobian set to 1 it must reproduce the
version-1 results file to 1e-6 in the per-lens counts. Output goes to
this folder (paper6_phase2_results.json here is the version-2 product;
the version-1 file in 003_phase2/ is untouched). Everything else is the
version-1 code, unchanged.
"""

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
import json
import os

# =====================================================================
# SECTION 1: Paper V ε(k) sigmoid
# =====================================================================

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
# The version-1 results the paper and the phase-3 scripts used (the copy in
# 003_phase2/ is an earlier run with older key names and a 3e-4 difference).
V1_RESULTS = os.path.join(OUTPUT_DIR, '..', '004_phase3', 'paper6_handoff', 'phase2_outputs',
                          'paper6_phase2_results.json')

# Version 2: d log M_star / d log M_halo for M_star ∝ M_halo^(1/1.4).
# The main block sets it to 1.0 for the gate, then back to 1/1.4.
JACOBIAN = 1.0 / 1.4


def epsilon_sigmoid(k, k_t=0.20, eps_inf=0.58, alpha=2.0):
    """
    Paper V Eq. (5): Cross-branch correlation coefficient.
    
    ε(k) = ε_∞ + (1 - ε_∞) / (1 + (k/k_t)^α)
    
    ε → 1 at k << k_t (large scales, branches fully correlated)
    ε → ε_∞ at k >> k_t (small scales, partial independence)
    """
    return eps_inf + (1.0 - eps_inf) / (1.0 + (k / k_t)**alpha)


def epsilon_at_mass(M_halo, k_t=0.20, eps_inf=0.58, alpha=2.0):
    """
    Convert halo mass to wavenumber via virial radius, then evaluate ε.
    
    R_vir ≈ 260 kpc × (M/10^12 M_sun)^(1/3) at z~0.3
    k_M ~ 1/R_vir (in h/Mpc)
    """
    # Virial radius in Mpc/h (approximate, z~0.3)
    R_vir = 0.26 * (M_halo / 1e12)**(1.0/3.0)  # Mpc/h
    k_M = 1.0 / R_vir  # h/Mpc
    return epsilon_sigmoid(k_M, k_t, eps_inf, alpha), k_M


# =====================================================================
# SECTION 2: Galaxy Stellar Mass Function (Schechter)
# =====================================================================

def schechter_dndm(M, M_star=10**10.7, alpha_sch=-1.16, phi_star=3.96e-3):
    """
    Schechter galaxy stellar mass function (per dex per Mpc^3).
    
    Default parameters from Baldry+2012 (z~0, total):
      M* = 10^10.7 M_sun
      α = -1.16
      φ* = 3.96 × 10^-3 Mpc^-3 dex^-1
    
    Returns dn/dlog10(M) in units of Mpc^-3 dex^-1.
    """
    x = M / M_star
    # Schechter per dex: φ(M) = ln(10) × φ* × x^(α+1) × exp(-x)
    return np.log(10) * phi_star * x**(alpha_sch + 1) * np.exp(-x)


def schechter_halo_mass(M_star_gal):
    """
    Convert stellar mass to halo mass using abundance matching.
    Approximate relation from Behroozi+2013 (z~0.3):
      M_halo / M_star ~ 50 at M* = 10^10.7 (Milky Way-like)
      Steeper ratio at lower masses
    
    Simple power-law approximation for Paper VI purposes:
      M_halo ≈ 10^12 × (M_star / 10^10.7)^1.4
    
    This gives M_halo ~ 10^12 at M* ~ 10^10.7, and
    M_halo ~ 10^11 at M* ~ 10^9.7
    """
    return 1e12 * (M_star_gal / 10**10.7)**1.4


# =====================================================================
# SECTION 3: ΛCDM subhalo mass function
# =====================================================================

def lcdm_subhalo_dndm(M_sub, M_host=1e13, A=0.012, beta=-1.9):
    """
    ΛCDM subhalo mass function per unit host halo mass.
    
    dN/dM_sub ∝ (M_sub/M_host)^β
    
    Normalised following Springel+2008 / Gao+2012:
      A ~ 0.012 per dex at M_sub/M_host = 0.01
      β ~ -1.9 (slope)
    
    Returns dN/dlog10(M_sub) (number per dex per host halo).
    """
    x = M_sub / M_host
    # Per dex: dN/dlog10(M) = ln(10) × A × (M/M_host)^(β+1) × M_host
    return A * (x / 0.01)**(beta + 1) * np.ones_like(M_sub)


# =====================================================================
# SECTION 4: ISF excess subhalo mass function
# =====================================================================

def isf_excess_dndm(M_halo, N=6, k_t=0.20, eps_inf=0.58, alpha=2.0,
                    V_lens_Mpc3=1.0):
    """
    ISF-predicted excess of dark substructure at galaxy-mass scales.
    
    Each of N-1 non-visible branches contributes a galaxy population.
    At partial correlation ε, the fraction of branch galaxies that
    are NOT coincident with visible structure is approximately (1-ε).
    
    dN_excess/dlog10(M_halo) ≈ (N-1) × [1-ε(k_M)] × Φ_gal(M_star) × V_eff
    
    where M_star = M_star(M_halo) from abundance matching (inverted).
    
    CAVEAT: [1-ε] is a rough estimate. The true ε-to-overlap mapping
    for discrete objects is more complex. We bracket between ε=0 and
    ε=ε_∞ to show the range.
    
    Parameters:
        M_halo: array, halo masses in M_sun
        N: number of branches
        k_t, eps_inf, alpha: Paper V sigmoid parameters
        V_lens_Mpc3: effective lensing volume in Mpc^3
    
    Returns:
        dN/dlog10(M_halo) per lens sightline (number per dex)
    """
    M_halo = np.atleast_1d(M_halo).astype(float)
    result = np.zeros_like(M_halo)
    
    for i, Mh in enumerate(M_halo):
        # Get ε at this mass scale
        eps_at_M, k_M = epsilon_at_mass(Mh, k_t, eps_inf, alpha)
        
        # Invert abundance matching: M_star from M_halo
        M_star = 10**10.7 * (Mh / 1e12)**(1.0/1.4)
        
        # Galaxy number density at this stellar mass (per Mpc^3 per dex)
        phi_gal = schechter_dndm(M_star)
        
        # ISF excess: (N-1) branches × (1-ε) independent fraction
        # × galaxy density × effective volume
        # Version 2: × d log M_star / d log M_halo, since phi_gal is per
        # dex of stellar mass and the sum runs over d log M_halo.
        excess_factor = (N - 1) * (1.0 - eps_at_M)
        result[i] = excess_factor * phi_gal * V_lens_Mpc3 * JACOBIAN

    return result


# =====================================================================
# SECTION 5: Effective lensing volume
# =====================================================================

def lensing_volume(z_L=0.3, z_S=2.0, theta_E_arcsec=1.0,
                   Delta_z_eff=0.1):
    """
    Volumes probed by a strong lens sightline, separated into
    existence volumes (where ISF dark galaxies reside) and detection
    volumes (projected close enough to lensed images to be detected).

    SUBHALO CHANNEL (within host halo):
      V_host = (4/3)pi R_vir^3  (full host — existence volume)
      V_detect_sub = pi R_detect^2 x 2 R_vir  (detection cylinder)
      R_detect ~ 50 kpc for 10^10-10^12 M_sun perturbers.
      Gravitational imaging detects massive perturbers at impact
      parameters well beyond R_E; Vegetti+2012 detected 10^8.9
      at ~6 kpc, scaling by mass^0.3 gives ~40-60 kpc at 10^10.
      eta_sub = V_detect_sub / V_host  (detection efficiency)

    LOS CHANNEL (along sightline):
      V_LOS = pi theta_detect^2 D_A^2 Delta_D
      theta_detect = 3 arcsec (detection region around lensed images).
      This IS already the detection volume — LOS perturbers are only
      detected if projected within ~3" of the images (Gilman+2019).
      No additional detection efficiency needed.
    """
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    # Einstein radius in physical Mpc
    D_A_L = cosmo.angular_diameter_distance(z_L).to('Mpc').value
    R_E_Mpc = theta_E_arcsec * 4.848e-6 * D_A_L

    # --- Subhalo channel ---
    R_vir_host = 0.5  # Mpc, typical group-mass host of strong lens
    V_host = (4.0/3.0) * np.pi * R_vir_host**3  # existence volume

    # Detection cylinder: massive perturbers detectable at impact
    # parameters up to R_detect ~ 50 kpc projected from lensed images.
    # Conservative: from Vegetti+2012 scaling, 10^10 at ~40-60 kpc.
    R_detect_Mpc = 0.050  # 50 kpc
    V_detect_sub = np.pi * R_detect_Mpc**2 * 2 * R_vir_host

    eta_sub = V_detect_sub / V_host  # detection efficiency

    # --- LOS channel ---
    # theta_detect = 3 arcsec IS the detection radius (Gilman+2019).
    # V_LOS computed with this angle is already the detection volume.
    theta_detect_rad = 3.0 * 4.848e-6  # 3 arcsec
    D_comoving_S = cosmo.comoving_distance(z_S).to('Mpc').value
    D_comoving_L = cosmo.comoving_distance(z_L).to('Mpc').value
    Delta_D = D_comoving_S - D_comoving_L
    D_A_avg = 0.5 * (D_comoving_L + D_comoving_S)
    V_LOS_detect = np.pi * (theta_detect_rad * D_A_avg)**2 * Delta_D

    return {
        # Subhalo channel
        'V_host_Mpc3': V_host,
        'V_detect_sub_Mpc3': V_detect_sub,
        'eta_sub': eta_sub,
        'R_detect_kpc': R_detect_Mpc * 1e3,
        'R_vir_host_Mpc': R_vir_host,
        # LOS channel
        'V_LOS_detect_Mpc3': V_LOS_detect,
        'theta_detect_arcsec': 3.0,
        # Geometry
        'R_E_Mpc': R_E_Mpc,
        'z_L': z_L,
        'z_S': z_S,
    }


# =====================================================================
# SECTION 6: N-ε degeneracy from Paper V S₈ constraint
# =====================================================================

# Paper V Table 1: For each N, the S₈ constraint fixes ε_∞ given k_t, α
# These are approximate values for k_t=0.20, α=2:
S8_CONSTRAINED_PARAMS = {
    # N: ε_∞ (from S₈ constraint with k_t=0.20, α=2)
    # The product (N-1)/N² × [1+(N-2)ε_∞] must match S₈
    # At fixed S₈ suppression, larger N requires larger ε_∞
    2: 0.00,   # N=2: only 1 extra branch, ε can be low
    3: 0.25,   # approximate
    4: 0.40,
    6: 0.58,   # Paper V best fit
    8: 0.68,   # approximate
    10: 0.74,  # approximate
}

def isf_effective_excess(N, eps_inf=None):
    """
    Effective ISF substructure excess factor: (N-1)(1-ε_∞).
    
    If eps_inf not provided, uses S₈-constrained value for given N.
    
    This is the key quantity for the N-ε degeneracy test:
    does (N-1)(1-ε_∞(N)) vary significantly with N, or is it
    approximately constant (indicating persistent degeneracy)?
    """
    if eps_inf is None:
        eps_inf = S8_CONSTRAINED_PARAMS.get(N, 0.58)
    return (N - 1) * (1.0 - eps_inf)


# =====================================================================
# SECTION 7: Meneghetti GGSL comparison
# =====================================================================

def meneghetti_prediction(N=6, k_t=0.20, eps_inf=0.58, alpha=2.0):
    """
    ISF prediction for the Meneghetti GGSL anomaly.
    
    GGSL probes galaxy-mass subhalos within clusters.
    Einstein radii ~few kpc → virial radii ~100-300 kpc → k~3-10 h/Mpc
    
    At these scales, the Paper V sigmoid has saturated at ε_∞.
    But Paper V only constrains k up to ~2 h/Mpc.
    
    Returns predictions for three scenarios.
    """
    # Scenario A: ε = ε_∞ = 0.58 (conservative, sigmoid saturated)
    k_GGSL = 5.0  # h/Mpc, representative for galaxy-scale GGSL
    eps_A = epsilon_sigmoid(k_GGSL, k_t, eps_inf, alpha)
    excess_A = (N - 1) * (1.0 - eps_A)
    
    # Scenario B: ε drops further at high k (optimistic)
    # Suppose ε = 0.3 at k~10 h/Mpc (beyond Paper V constraint)
    eps_B = 0.30
    excess_B = (N - 1) * (1.0 - eps_B)
    
    # Scenario C: ε = 0 (maximum, fully independent at galaxy scales)
    eps_C = 0.0
    excess_C = (N - 1) * (1.0 - eps_C)
    
    # Meneghetti gap numbers
    gap_original = 10.0   # vs GADGET-X (Meneghetti+2020)
    gap_residual = 4.0    # vs GIZMO-SIMBA (Meneghetti+2023)
    baryonic_boost = 3.0  # GIZMO-SIMBA vs GADGET-X
    
    return {
        'scenario_A': {
            'label': f'Conservative (ε={eps_A:.2f} at k={k_GGSL})',
            'epsilon': eps_A,
            'isf_excess': excess_A,
            'isf_x_baryonic': excess_A * baryonic_boost,
            'vs_original_gap': excess_A * baryonic_boost / gap_original,
            'vs_residual_gap': excess_A / gap_residual,
        },
        'scenario_B': {
            'label': f'Optimistic (ε={eps_B} at k~10 h/Mpc)',
            'epsilon': eps_B,
            'isf_excess': excess_B,
            'isf_x_baryonic': excess_B * baryonic_boost,
            'vs_original_gap': excess_B * baryonic_boost / gap_original,
            'vs_residual_gap': excess_B / gap_residual,
        },
        'scenario_C': {
            'label': f'Maximum (ε=0, fully independent)',
            'epsilon': eps_C,
            'isf_excess': excess_C,
            'isf_x_baryonic': excess_C * baryonic_boost,
            'vs_original_gap': excess_C * baryonic_boost / gap_original,
            'vs_residual_gap': excess_C / gap_residual,
        },
        'observational': {
            'gap_original': gap_original,
            'gap_residual': gap_residual,
            'baryonic_boost': baryonic_boost,
        }
    }


# =====================================================================
# SECTION 9: Per-lens abundance (Phase 2B, Step 1)
# =====================================================================

def per_lens_abundance(N=6, k_t=0.20, eps_inf=0.58, alpha=2.0,
                       z_L=0.3, z_S=2.0, theta_E_arcsec=1.0):
    """
    Expected ISF dark perturbers per strong lens sightline.

    Reports separately:
      N_exist:  total ISF dark galaxies along sightline (sub + LOS)
      N_detect: those projected close enough to be detectable
      N_Euclid: N_detect x 2500 (O'Riordan+2022 Euclid forecast)

    Subhalo channel: dark galaxies within host halo.
      Existence: V_host = (4/3)pi R_vir^3
      Detection: V_detect_sub = pi R_detect^2 x 2 R_vir  (cylinder)

    LOS channel: dark galaxies along the sightline.
      V_LOS already uses theta_detect = 3", so it IS the detection
      volume. For existence, we report V_LOS as well (the distinction
      is less meaningful for LOS — the detection volume IS the
      relevant volume since perturbers outside it are unobservable).
    """
    vols = lensing_volume(z_L, z_S, theta_E_arcsec)
    V_host = vols['V_host_Mpc3']
    V_detect_sub = vols['V_detect_sub_Mpc3']
    V_LOS = vols['V_LOS_detect_Mpc3']
    eta_sub = vols['eta_sub']

    log_M_isf = np.linspace(10, 12, 201)
    M_isf = 10**log_M_isf
    dlogM = log_M_isf[1] - log_M_isf[0]

    log_M_lcdm = np.linspace(8, 10, 201)
    M_lcdm = 10**log_M_lcdm
    dlogM_lcdm = log_M_lcdm[1] - log_M_lcdm[0]

    def _compute_isf(ei):
        """Compute existence and detection counts for given eps_inf."""
        # Subhalo: existence in full host, detection in cylinder
        sub_exist = isf_excess_dndm(M_isf, N, k_t, ei, alpha, V_host)
        sub_detect = isf_excess_dndm(M_isf, N, k_t, ei, alpha, V_detect_sub)
        N_sub_exist = np.sum(sub_exist) * dlogM
        N_sub_detect = np.sum(sub_detect) * dlogM

        # LOS: V_LOS is already detection volume (theta=3")
        los_detect = isf_excess_dndm(M_isf, N, k_t, ei, alpha, V_LOS)
        N_los_detect = np.sum(los_detect) * dlogM

        N_exist = N_sub_exist + N_los_detect  # sub existence + LOS detection
        N_detect = N_sub_detect + N_los_detect
        N_euclid = N_detect * 2500

        return {
            'eps_inf': ei,
            'N_sub_exist': N_sub_exist,
            'N_sub_detect': N_sub_detect,
            'N_LOS_detect': N_los_detect,
            'N_exist_total': N_exist,
            'N_detect_total': N_detect,
            'N_Euclid': N_euclid,
            'mass_range': '10^10 - 10^12 M_sun',
        }

    # Three epsilon scenarios
    conservative = _compute_isf(eps_inf)
    optimistic = _compute_isf(0.30)
    maximum = _compute_isf(0.0)

    # LCDM baseline (10^8-10^10 for Despali comparison)
    M_host_lens = 1e13
    lcdm_sub = lcdm_subhalo_dndm(M_lcdm, M_host=M_host_lens)
    N_lcdm_sub = np.sum(lcdm_sub) * dlogM_lcdm
    detection_eff = 0.15
    N_lcdm_detectable = N_lcdm_sub * detection_eff

    return {
        'volumes': {
            'V_host_Mpc3': V_host,
            'V_detect_sub_Mpc3': V_detect_sub,
            'eta_sub': eta_sub,
            'R_detect_kpc': vols['R_detect_kpc'],
            'V_LOS_detect_Mpc3': V_LOS,
            'theta_detect_arcsec': 3.0,
            'note': ('V_LOS uses theta=3" and IS the detection volume. '
                     'V_host is the existence volume for subhalos. '
                     'V_detect_sub = pi R_detect^2 x 2R_vir is the '
                     'detection cylinder through the host.'),
        },
        'isf_conservative': conservative,
        'isf_optimistic': optimistic,
        'isf_maximum': maximum,
        'lcdm_baseline': {
            'N_sub_total': N_lcdm_sub,
            'N_sub_detectable': N_lcdm_detectable,
            'detection_efficiency': detection_eff,
            'mass_range': '10^8 - 10^10 M_sun',
            'M_host': M_host_lens,
            'reference': 'Despali+2018: ~0.01 detectable per lens',
        },
    }


# =====================================================================
# SECTION 10: LOS vs Subhalo fraction (Phase 2B, Step 2)
# =====================================================================

def los_subhalo_fraction(N=6, k_t=0.20, eps_inf=0.58, alpha=2.0,
                         z_L=0.3, z_S=2.0):
    """
    ISF-predicted LOS-to-subhalo ratio vs LCDM (He+2022: ~2/3 LOS).

    Uses DETECTION counts (not existence counts) since the observable
    LOS fraction is what matters for comparison to data.

    ISF dark galaxies detected as LOS perturbers vastly outnumber
    those detected as subhalos because V_LOS >> V_detect_sub.
    """
    abundance = per_lens_abundance(N, k_t, eps_inf, alpha, z_L, z_S)

    N_isf_sub = abundance['isf_conservative']['N_sub_detect']
    N_isf_los = abundance['isf_conservative']['N_LOS_detect']

    f_LOS_lcdm = 0.67
    f_sub_lcdm = 0.33
    N_lcdm_total = 0.05

    N_lcdm_los = N_lcdm_total * f_LOS_lcdm
    N_lcdm_sub = N_lcdm_total * f_sub_lcdm

    # Combined LCDM + ISF (at DIFFERENT mass ranges: LCDM at 10^8-10^10,
    # ISF at 10^10-10^12). Total detected perturbers per lens:
    N_total_with_isf = N_lcdm_total + N_isf_sub + N_isf_los

    if N_total_with_isf > 0:
        f_LOS_isf = (N_lcdm_los + N_isf_los) / N_total_with_isf
        f_sub_isf = (N_lcdm_sub + N_isf_sub) / N_total_with_isf
    else:
        f_LOS_isf, f_sub_isf = f_LOS_lcdm, f_sub_lcdm

    # ISF-only LOS fraction (among ISF additions only)
    N_isf_total = N_isf_sub + N_isf_los
    f_LOS_isf_only = (N_isf_los / N_isf_total
                      if N_isf_total > 0 else 0.0)

    shift = f_LOS_isf - f_LOS_lcdm
    # Assessment depends on magnitude of shift
    if abs(shift) > 0.10:
        strength = 'Strong'
    elif abs(shift) > 0.05:
        strength = 'Moderate'
    else:
        strength = 'Weak'

    return {
        'lcdm_baseline': {
            'f_LOS': f_LOS_lcdm, 'f_sub': f_sub_lcdm,
            'N_total_per_lens': N_lcdm_total,
            'reference': 'He+2022',
        },
        'isf_added': {
            'N_sub_detect': N_isf_sub,
            'N_LOS_detect': N_isf_los,
            'f_LOS_isf_only': f_LOS_isf_only,
        },
        'combined': {
            'N_total': N_total_with_isf,
            'f_LOS': f_LOS_isf, 'f_sub': f_sub_isf,
            'shift_from_lcdm': shift,
        },
        'discriminant_assessment': {
            'shift_percent': shift * 100,
            'strength': strength,
            'detectable': abs(shift) > 0.05,
            'note': (f'{strength} discriminant: ISF shifts LOS fraction '
                     f'by {shift*100:+.1f}%. '
                     f'ISF adds {N_isf_total:.3f} detectable perturbers '
                     f'per lens at 10^10-10^12, of which '
                     f'{f_LOS_isf_only:.0%} are LOS.'),
        },
    }


# =====================================================================
# SECTION 11: Density profile comparison (Phase 2B, Step 3)
# =====================================================================

def nfw_profile(r_kpc, M_halo, concentration, z=0.3):
    """NFW density profile rho(r) for a halo of mass M_halo."""
    rho_crit = 1.4e11  # M_sun/Mpc^3 at z~0.3
    Delta_vir = 200
    R_vir_Mpc = (3 * M_halo / (4 * np.pi * Delta_vir * rho_crit))**(1./3.)
    R_vir_kpc = R_vir_Mpc * 1e3

    r_s = R_vir_kpc / concentration
    g_c = np.log(1 + concentration) - concentration / (1 + concentration)
    rho_s = M_halo / (4 * np.pi * r_s**3 * g_c)

    x = np.maximum(np.asarray(r_kpc, dtype=float) / r_s, 1e-6)
    rho = rho_s / (x * (1 + x)**2)
    return rho, r_s, R_vir_kpc


def truncated_nfw_profile(r_kpc, M_halo, concentration, r_trunc_kpc):
    """Truncated NFW (Baltz+2009) for tidally stripped LCDM subhalos."""
    rho_nfw, r_s, R_vir = nfw_profile(r_kpc, M_halo, concentration)
    r = np.asarray(r_kpc, dtype=float)
    truncation = r_trunc_kpc**2 / (r**2 + r_trunc_kpc**2)
    return rho_nfw * truncation, r_s, R_vir


def lensing_convergence(r_kpc, rho_profile, Sigma_cr=3.5e9):
    """
    Lensing convergence kappa(R) = Sigma(R) / Sigma_cr.

    Sigma(R) = 2 * integral_R^inf rho(r) * r / sqrt(r^2 - R^2) dr
    """
    r = np.asarray(r_kpc, dtype=float)
    rho = np.asarray(rho_profile, dtype=float)
    rho_interp = interp1d(r, rho, kind='linear', fill_value=0,
                          bounds_error=False)

    R_proj = r.copy()
    Sigma = np.zeros_like(R_proj)

    for i, R in enumerate(R_proj):
        if R < r[-1] * 0.99:
            def integrand(r_3d, _R=R):
                if r_3d <= _R:
                    return 0.0
                return rho_interp(r_3d) * r_3d / np.sqrt(r_3d**2 - _R**2)
            r_min = R * 1.001
            val, _ = quad(integrand, r_min, r[-1], limit=100)
            Sigma[i] = 2.0 * val

    kappa = Sigma / Sigma_cr
    return kappa, Sigma


def profile_comparison(M_halo=1e10, z_L=0.3, z_S=2.0):
    """
    Compare lensing convergence for ISF (intact NFW, c=12) vs
    LCDM subhalo (truncated NFW, c=12, r_trunc=0.3*R_vir).
    """
    r_kpc = np.logspace(-1, 2.5, 200)

    c_isf = 12.0
    rho_isf, r_s_isf, R_vir_isf = nfw_profile(r_kpc, M_halo, c_isf, z_L)

    c_lcdm = 12.0
    _, _, R_vir_lcdm = nfw_profile(r_kpc, M_halo, c_lcdm, z_L)
    r_trunc = 0.30 * R_vir_lcdm
    rho_lcdm, r_s_lcdm, _ = truncated_nfw_profile(
        r_kpc, M_halo, c_lcdm, r_trunc)

    Sigma_cr = 3.5e9
    kappa_isf, Sigma_isf = lensing_convergence(r_kpc, rho_isf, Sigma_cr)
    kappa_lcdm, Sigma_lcdm = lensing_convergence(r_kpc, rho_lcdm, Sigma_cr)

    def safe_ratio(r_val):
        k_i = np.interp(r_val, r_kpc, kappa_isf)
        k_l = max(np.interp(r_val, r_kpc, kappa_lcdm), 1e-30)
        return k_i / k_l

    ratio_1 = safe_ratio(1.0)
    ratio_5 = safe_ratio(5.0)
    ratio_10 = safe_ratio(10.0)

    # Bound mass comparison
    def nfw_mass_within(r_max, M_h, c):
        _, r_s_loc, _ = nfw_profile(np.array([1.0]), M_h, c)
        g_c = np.log(1 + c) - c / (1 + c)
        x = r_max / r_s_loc
        g_x = np.log(1 + x) - x / (1 + x)
        return M_h * g_x / g_c

    M_bound_lcdm = nfw_mass_within(r_trunc, M_halo, c_lcdm)
    mass_loss = 1.0 - M_bound_lcdm / M_halo

    return {
        'r_kpc': r_kpc.tolist(),
        'rho_isf': rho_isf.tolist(),
        'rho_lcdm': rho_lcdm.tolist(),
        'kappa_isf': kappa_isf.tolist(),
        'kappa_lcdm': kappa_lcdm.tolist(),
        'Sigma_isf': Sigma_isf.tolist(),
        'Sigma_lcdm': Sigma_lcdm.tolist(),
        'parameters': {
            'M_halo': M_halo, 'c_isf': c_isf, 'c_lcdm': c_lcdm,
            'r_trunc_kpc': r_trunc, 'R_vir_kpc': R_vir_isf,
            'Sigma_cr': Sigma_cr,
        },
        'convergence_ratios': {
            'kappa_isf_over_lcdm_at_1kpc': ratio_1,
            'kappa_isf_over_lcdm_at_5kpc': ratio_5,
            'kappa_isf_over_lcdm_at_10kpc': ratio_10,
        },
        'mass_comparison': {
            'M_bound_isf': M_halo,
            'M_bound_lcdm': M_bound_lcdm,
            'mass_loss_fraction_lcdm': mass_loss,
            'note': (f'LCDM subhalo retains {(1-mass_loss)*100:.0f}% '
                     f'of mass after stripping to r_t={r_trunc:.0f} kpc'),
        },
        'discriminant': {
            'note': (f'At 5 kpc: kappa_ISF/kappa_LCDM = {ratio_5:.1f}x. '
                     'ISF perturbers produce stronger convergence peaks.'),
        },
    }


# =====================================================================
# SECTION 8: Main computation
# =====================================================================

def compute_all_predictions():
    """Run all Phase 2 computations and return results dictionary."""
    
    results = {}
    
    # --- 8a. ε(k) evaluation across mass scales ---
    log_M = np.linspace(6, 13, 71)  # 10^6 to 10^13 M_sun
    M_array = 10**log_M
    eps_values = []
    k_values = []
    for M in M_array:
        eps, k = epsilon_at_mass(M)
        eps_values.append(eps)
        k_values.append(k)
    
    results['epsilon_vs_mass'] = {
        'log_M': log_M.tolist(),
        'epsilon': eps_values,
        'k_hMpc': k_values,
    }
    
    # --- 8b. Mass function comparison (ΛCDM vs ISF) ---
    M_host = 1e13  # Typical cluster/group host
    
    # ΛCDM subhalo mass function
    lcdm_dndlogM = lcdm_subhalo_dndm(M_array, M_host=M_host)
    
    # ISF excess at different ε values
    V_eff = 1.0  # Will be normalised in Phase 2B
    isf_eps058 = isf_excess_dndm(M_array, N=6, eps_inf=0.58, V_lens_Mpc3=V_eff)
    isf_eps000 = isf_excess_dndm(M_array, N=6, eps_inf=0.00, V_lens_Mpc3=V_eff)
    isf_eps030 = isf_excess_dndm(M_array, N=6, eps_inf=0.30, V_lens_Mpc3=V_eff)
    
    results['mass_function'] = {
        'log_M': log_M.tolist(),
        'lcdm_dndlogM': lcdm_dndlogM.tolist(),
        'isf_eps058_dndlogM': isf_eps058.tolist(),
        'isf_eps000_dndlogM': isf_eps000.tolist(),
        'isf_eps030_dndlogM': isf_eps030.tolist(),
    }
    
    # --- 8c. N-ε degeneracy ---
    degeneracy = {}
    for N, eps in S8_CONSTRAINED_PARAMS.items():
        eff_excess = isf_effective_excess(N, eps)
        degeneracy[N] = {
            'eps_inf': eps,
            'effective_excess': eff_excess,
            'note': f'(N-1)(1-ε_∞) = {N-1}×{1-eps:.2f} = {eff_excess:.2f}'
        }
    
    results['n_epsilon_degeneracy'] = degeneracy
    
    # --- 8d. Meneghetti comparison ---
    results['meneghetti'] = meneghetti_prediction()
    
    # --- 8e. Key summary numbers ---
    results['summary'] = {
        'paper_v_params': {
            'k_t': 0.20, 'eps_inf': 0.58, 'alpha': 2, 'N': 6
        },
        'isf_excess_factor': {
            'eps_058': isf_effective_excess(6, 0.58),
            'eps_030': isf_effective_excess(6, 0.30),
            'eps_000': isf_effective_excess(6, 0.00),
        },
        'mass_scale_of_bump': {
            'peak_log_M_star': 10.7,
            'peak_log_M_halo': 12.0,
            'description': 'ISF bump peaks at galaxy M* ~ 10^10.7 M_sun'
        },
        'gilman_blind_above': {
            'log_M_ceiling': 9.0,
            'note': 'Flux ratio method sensitive only to 10^6-10^9'
        },
        'euclid_sensitivity': {
            'log_M_threshold': 8.8,
            'predicted_detections': 2500,
            'note': 'ORiordan+2022: 10^(8.8+-0.2) M_sun threshold'
        }
    }

    # --- 8f. Phase 2B: Per-lens abundance (Step 1) ---
    print("  Computing per-lens abundances...")
    results['per_lens_abundance'] = per_lens_abundance()

    # --- 8g. Phase 2B: LOS vs subhalo fraction (Step 2) ---
    print("  Computing LOS/subhalo fractions...")
    results['los_subhalo_fraction'] = los_subhalo_fraction()

    # --- 8h. Phase 2B: Density profile comparison (Step 3) ---
    print("  Computing profile comparison (this takes ~30s)...")
    results['profile_comparison'] = profile_comparison()

    return results


if __name__ == '__main__':
    # ── Version 2 GATE: with the Jacobian off, reproduce version 1 ──
    with open(V1_RESULTS) as f:
        v1 = json.load(f)
    JACOBIAN = 1.0
    gate_results = compute_all_predictions()
    gate_ok = True
    print("VERSION 2 GATE (Jacobian = 1 must reproduce 004_phase3/paper6_handoff/phase2_outputs/paper6_phase2_results.json):")
    for key in ('isf_conservative', 'isf_optimistic', 'isf_maximum'):
        a = v1['per_lens_abundance'][key]['N_detect_total']
        b = gate_results['per_lens_abundance'][key]['N_detect_total']
        ok = abs(a - b) < 1e-6
        gate_ok &= ok
        print(f"  {key:16s}: v1 {a:.6f}  reproduced {b:.6f}  {'PASS' if ok else 'FAIL'}")
    if not gate_ok:
        raise SystemExit("version 2 gate failed")
    JACOBIAN = 1.0 / 1.4
    print(f"  GATE PASS; now with the Jacobian = {JACOBIAN:.4f}\n")

    print("Computing ISF substructure predictions (Phase 2A + 2B)...")
    results = compute_all_predictions()
    results['version'] = {'version': 2, 'date': '2026-10-11', 'jacobian': JACOBIAN,
                          'note': 'd log M_star / d log M_halo applied in isf_excess_dndm; version 1 omitted it'}

    outfile = os.path.join(OUTPUT_DIR, 'paper6_phase2_results.json')

    print("\n" + "="*60)
    print("PHASE 2A RESULTS")
    print("="*60)

    print("\n--- N-epsilon DEGENERACY ---")
    for N, data in results['n_epsilon_degeneracy'].items():
        print(f"  N={N}: {data['note']}")

    print("\n--- MENEGHETTI SCENARIOS ---")
    for key in ['scenario_A', 'scenario_B', 'scenario_C']:
        s = results['meneghetti'][key]
        print(f"  {s['label']}:")
        print(f"    ISF excess: {s['isf_excess']:.1f}x")
        print(f"    ISF x baryonic: {s['isf_x_baryonic']:.1f}x")
        print(f"    vs original 10x gap: {s['vs_original_gap']:.0%}")
        print(f"    vs residual 4x gap: {s['vs_residual_gap']:.0%}")

    print("\n--- ISF EXCESS FACTORS ---")
    for key, val in results['summary']['isf_excess_factor'].items():
        print(f"  {key}: {val:.2f}x")

    print("\n" + "="*60)
    print("PHASE 2B RESULTS")
    print("="*60)

    # Per-lens abundance
    pla = results['per_lens_abundance']
    print("\n--- PER-LENS ABUNDANCE ---")
    v = pla['volumes']
    print(f"  V_host     = {v['V_host_Mpc3']:.3f} Mpc^3  (subhalo existence)")
    print(f"  V_det_sub  = {v['V_detect_sub_Mpc3']:.2e} Mpc^3  "
          f"(R_det={v['R_detect_kpc']:.0f} kpc cylinder)")
    print(f"  eta_sub    = {v['eta_sub']:.4f}  (detection efficiency)")
    print(f"  V_LOS_det  = {v['V_LOS_detect_Mpc3']:.1f} Mpc^3  "
          f"(theta={v['theta_detect_arcsec']:.0f}\" detection cone)")
    for label, key in [('Conservative (eps=0.58)', 'isf_conservative'),
                       ('Optimistic   (eps=0.30)', 'isf_optimistic'),
                       ('Maximum      (eps=0.00)', 'isf_maximum')]:
        sc = pla[key]
        print(f"  {label}:")
        print(f"    Sub exist:  {sc['N_sub_exist']:.4f}  |  "
              f"Sub detect:  {sc['N_sub_detect']:.6f}")
        print(f"    LOS detect: {sc['N_LOS_detect']:.4f}")
        print(f"    Total detect: {sc['N_detect_total']:.4f}  |  "
              f"Euclid total: {sc['N_Euclid']:.0f}")
    lb = pla['lcdm_baseline']
    print(f"  LCDM baseline (10^8-10^10): {lb['N_sub_total']:.3f} total, "
          f"{lb['N_sub_detectable']:.4f} detectable")

    # LOS fraction
    lsf = results['los_subhalo_fraction']
    print("\n--- LOS vs SUBHALO FRACTION ---")
    print(f"  LCDM (He+2022): f_LOS = {lsf['lcdm_baseline']['f_LOS']:.0%}")
    ia = lsf['isf_added']
    print(f"  ISF-only detections: f_LOS = {ia['f_LOS_isf_only']:.0%} "
          f"(N_sub={ia['N_sub_detect']:.6f}, "
          f"N_LOS={ia['N_LOS_detect']:.4f})")
    comb = lsf['combined']
    print(f"  Combined (LCDM+ISF): f_LOS = {comb['f_LOS']:.0%} "
          f"(shift = {comb['shift_from_lcdm']*100:+.1f}%)")
    da = lsf['discriminant_assessment']
    print(f"  Assessment: {da['note']}")

    # Profile comparison
    pc = results['profile_comparison']
    print("\n--- DENSITY PROFILE COMPARISON ---")
    p = pc['parameters']
    print(f"  M_halo = {p['M_halo']:.0e} M_sun")
    print(f"  ISF: intact NFW, c={p['c_isf']}")
    print(f"  LCDM: truncated NFW, c={p['c_lcdm']}, "
          f"r_trunc={p['r_trunc_kpc']:.0f} kpc")
    cr = pc['convergence_ratios']
    print(f"  kappa(ISF)/kappa(LCDM) at  1 kpc: {cr['kappa_isf_over_lcdm_at_1kpc']:.2f}x")
    print(f"  kappa(ISF)/kappa(LCDM) at  5 kpc: {cr['kappa_isf_over_lcdm_at_5kpc']:.2f}x")
    print(f"  kappa(ISF)/kappa(LCDM) at 10 kpc: {cr['kappa_isf_over_lcdm_at_10kpc']:.2f}x")
    mc = pc['mass_comparison']
    print(f"  {mc['note']}")
    print(f"  {pc['discriminant']['note']}")

    # Save JSON
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {outfile}")
