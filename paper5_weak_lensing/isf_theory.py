"""
Paper V: Theoretical Framework — ISF Power Spectrum Predictions
================================================================

This module derives and implements the observable predictions of the
Independent Structure Formation (ISF) hypothesis for weak lensing
and galaxy clustering power spectra.

Key Reference: Paper III (Merrill 2026c), Equation 4.

Mathematical Framework
----------------------

Setup:
  - N ψ-branches, each with identical statistical properties
  - Branch 1 = "ours" (where we observe galaxies)
  - Branches 2..N = "dark matter" (gravitationally coupled, EM isolated)
  - Each branch has density contrast field δ_i(k)
  - Auto-power: <|δ_i|²> = P_single(k) for all i
  - Cross-power: <δ_i δ_j*> = ε(k) P_single(k) for i ≠ j
  - ε(k) = cross-branch correlation coefficient, 0 ≤ ε ≤ 1

Total matter field (all branches):
  δ_total(k) = (1/N) Σ_{i=1}^{N} δ_i(k)

Total matter power spectrum:
  P_total(k) = <|δ_total|²>
             = (1/N²) [N P_single + N(N-1) ε(k) P_single]
             = [1 + (N-1)ε(k)] / N × P_single(k)        ... (*)

Baryon power spectrum (branch 1 only):
  P_baryon(k) = P_single(k)

Galaxy power spectrum (galaxies trace branch 1 with bias b):
  P_galaxy(k) = b²(k) × P_single(k)

Observable ratio:
  R(k) ≡ P_total(k) / P_galaxy(k)
        = [1 + (N-1)ε(k)] / [N b²(k)]

ΛCDM prediction (single fluid, no branch structure):
  R_ΛCDM(k) = 1 / b²(k)

ISF modification factor:
  R_ISF(k) / R_ΛCDM(k) = [1 + (N-1)ε(k)] / N

Limits:
  ε → 1: factor = 1 (indistinguishable from ΛCDM)
  ε → 0: factor = 1/N (maximum suppression)
  N=6, ε=0: factor = 1/6 ≈ 0.167
  N=6, ε=0.5: factor = (1 + 2.5)/6 = 0.583
  N=6, ε=0.8: factor = (1 + 4)/6 = 0.833

σ₈ connection (S₈ tension):
  σ₈²(ISF) / σ₈²(ΛCDM) = [1 + (N-1)ε_eff] / N
  where ε_eff is the effective ε at the σ₈ scale (k ~ 0.1-1 Mpc⁻¹)

  Observed S₈ tension: S₈(lens)/S₈(CMB) ≈ 0.76/0.83 ≈ 0.916
  → σ₈ ratio ≈ 0.916 → σ₈² ratio ≈ 0.839
  For N=6: ε_eff ≈ 0.81

This is the key quantitative result: the S₈ tension is consistent
with ISF at N=6 if branches are ~80% correlated at σ₈ scales.
"""

import numpy as np


def epsilon_sigmoid(k, k_t, alpha):
    """
    Cross-branch correlation coefficient as a function of wavenumber.

    ε(k) → 1 at small k (large scales, gravitational coupling dominates)
    ε(k) → 0 at large k (small scales, independent baryonic physics)

    The transition occurs at k_t with steepness α.

    Parameters
    ----------
    k : array-like
        Wavenumber in h/Mpc
    k_t : float
        Transition wavenumber in h/Mpc
    alpha : float
        Steepness of transition (1 = gradual, 4 = sharp)

    Returns
    -------
    epsilon : array
        Cross-branch correlation, 0 ≤ ε ≤ 1
    """
    k = np.asarray(k, dtype=float)
    return 1.0 / (1.0 + (k / k_t)**alpha)


def isf_modification_factor(k, N, k_t, alpha):
    """
    ISF modification to the total matter power spectrum relative to ΛCDM.

    R_ISF / R_ΛCDM = [1 + (N-1)ε(k)] / N

    Parameters
    ----------
    k : array-like
        Wavenumber in h/Mpc
    N : int
        Number of ψ-branches
    k_t : float
        ε(k) transition wavenumber in h/Mpc
    alpha : float
        ε(k) transition steepness

    Returns
    -------
    factor : array
        Multiplicative modification to P_total/P_galaxy ratio
    """
    eps = epsilon_sigmoid(k, k_t, alpha)
    return (1.0 + (N - 1) * eps) / N


def isf_p_total(k, P_single, N, k_t, alpha):
    """
    Total matter power spectrum under ISF.

    P_total(k) = [1 + (N-1)ε(k)] / N × P_single(k)

    Parameters
    ----------
    k : array-like
        Wavenumber in h/Mpc
    P_single : array-like
        Single-branch power spectrum (= ΛCDM P_m(k) as proxy)
    N : int
        Number of branches
    k_t, alpha : float
        ε(k) parameters

    Returns
    -------
    P_total : array
    """
    factor = isf_modification_factor(k, N, k_t, alpha)
    return factor * np.asarray(P_single)


def isf_sigma8_ratio(N, epsilon_eff):
    """
    Ratio of σ₈(ISF) to σ₈(ΛCDM).

    σ₈² ratio = [1 + (N-1)ε_eff] / N
    σ₈ ratio = sqrt of that

    Parameters
    ----------
    N : int
        Number of branches
    epsilon_eff : float
        Effective ε at σ₈ scales

    Returns
    -------
    ratio : float
        σ₈(ISF) / σ₈(ΛCDM)
    """
    return np.sqrt((1.0 + (N - 1) * epsilon_eff) / N)


def epsilon_from_s8_tension(N, s8_ratio):
    """
    Invert the σ₈ relation to find ε_eff from the observed S₈ tension.

    Given S₈(lens)/S₈(CMB) = s8_ratio, what ε_eff is required?

    σ₈² ratio = s8_ratio² = [1 + (N-1)ε] / N
    → ε = (N × s8_ratio² - 1) / (N - 1)

    Parameters
    ----------
    N : int
        Number of branches
    s8_ratio : float
        S₈(lensing) / S₈(CMB)

    Returns
    -------
    epsilon_eff : float
        Required effective cross-branch correlation
    """
    return (N * s8_ratio**2 - 1.0) / (N - 1.0)


if __name__ == "__main__":
    # Quick verification of key numbers
    print("=== ISF Theoretical Framework Verification ===\n")

    # Table of modification factors
    print("Modification factor [1 + (N-1)ε] / N:")
    print(f"{'N':>3s}  {'ε=0':>8s}  {'ε=0.3':>8s}  {'ε=0.5':>8s}  {'ε=0.8':>8s}  {'ε=1.0':>8s}")
    for N in [2, 3, 4, 5, 6, 8, 10]:
        vals = []
        for eps in [0, 0.3, 0.5, 0.8, 1.0]:
            f = (1 + (N-1)*eps) / N
            vals.append(f"{f:.4f}")
        print(f"{N:3d}  {'  '.join(vals)}")

    print()

    # S₈ tension inversion
    print("S₈ tension: S₈(lens)/S₈(CMB) ≈ 0.76/0.83 = 0.916")
    print("Required ε_eff for each N:")
    s8_ratio = 0.76 / 0.83
    for N in [2, 3, 4, 5, 6, 8, 10]:
        eps = epsilon_from_s8_tension(N, s8_ratio)
        sig_ratio = isf_sigma8_ratio(N, eps)
        print(f"  N={N:2d}: ε_eff = {eps:.4f}  (verify: σ₈ ratio = {sig_ratio:.4f})")

    print()

    # Sigmoid ε(k) at key scales
    k_test = np.array([0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    print("ε(k) sigmoid examples (k_t=0.03 h/Mpc):")
    for alpha in [1, 2, 4]:
        eps_vals = epsilon_sigmoid(k_test, k_t=0.03, alpha=alpha)
        print(f"  α={alpha}: ", end="")
        for ki, ei in zip(k_test, eps_vals):
            print(f"ε({ki})={ei:.3f}  ", end="")
        print()
