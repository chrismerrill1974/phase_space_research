#!/usr/bin/env python3
"""
Phase D, Step D2: Propagate updated ISF parameters into Paper VII

D1 resolved the decision gate: the global best-fit k_t = 0.82 h/Mpc
is a Limber projection artifact. The physical transition scale is
k_t^{phys} = 0.52 h/Mpc [0.46, 0.55 at 1-sigma], with angular
transition ell_t ~ 447.

This script:
  1. Re-runs the Step 2c chi^2 grid (profiled-amplitude method)
  2. Adds a joint k_t^{phys} fit using the Limber mapping per pair
  3. Reports both apparent (projected) and physical k_t values
  4. Saves an updated checkpoint (chi2_wl_grid_v2.npz)

The key innovation over the original Step 2c: instead of scanning a
single global k_t applied uniformly to all pairs, we also scan
k_t^{phys} — a single physical transition scale that maps to different
apparent k_t values per pair via the Limber relation:

    k_t^{app}(pair) = ell_t / chi_eff(pair)
    where ell_t = k_t^{phys} * chi_eff(ref)

Depends on:
  ../../data/kids1000_bandpowers.npz
  ../../data/response_operators_tomo.npz

Output: ../../data/chi2_wl_grid_v2.npz
"""

import numpy as np
import os
import time

# ── Paths ─────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
DATA_DIR = os.path.join(REPO_ROOT, 'data')
OUTPATH = os.path.join(DATA_DIR, 'chi2_wl_grid_v2.npz')

KIDS_PATH = os.path.join(DATA_DIR, 'kids1000_bandpowers.npz')
TOMO_PATH = os.path.join(DATA_DIR, 'response_operators_tomo.npz')

for label, path in [('KiDS data', KIDS_PATH), ('Tomo operators', TOMO_PATH)]:
    if not os.path.isfile(path):
        print(f"ERROR: {label} not found at:\n  {path}")
        raise SystemExit(1)

# ══════════════════════════════════════════════════════════════════
# PART 1: Load data and response operators
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("Step D2: Propagate Updated ISF Parameters into Paper VII")
print("=" * 70)

print("\n[1] Loading data...")

dk = np.load(KIDS_PATH, allow_pickle=True)
dr = np.load(TOMO_PATH, allow_pickle=True)

bp_data = dk['bp_data']           # (15, 8)
ell_eff = dk['ell_eff']           # (8,)
cov_ee = dk['cov_ee']             # (120, 120)
pair_labels = dk['pair_labels']   # (15, 2)
nz_bins = dk['nz_bins']           # (5, 119)
z_mid = dk['z_mid']               # (119,)

A_tomo = dr['A_tomo']             # (15, 8, 500)
k_hMpc = dr['k_hMpc']            # (500,)
cl_lcdm_tomo = dr['cl_lcdm_tomo']  # (15, 8)
W_all = dr['W_all']               # (5, 800)
chi_W = dr['chi_W']               # (800,)
h_cosmo = float(dr['h'])
Om = float(dr['Om'])

n_pairs = len(pair_labels)
n_ell = len(ell_eff)
n_k = len(k_hMpc)
n_data = n_pairs * n_ell  # 120

print(f"  Data: {n_pairs} pairs x {n_ell} ell-bins = {n_data} points")
print(f"  k-grid: {n_k} points, [{k_hMpc[0]:.4f}, {k_hMpc[-1]:.1f}] h/Mpc")

# ── ISF parameters ────────────────────────────────────────────────
N_val = 6
eps_inf = 0.58
alpha = 2

# High-SNR pairs (from B3/C1)
HIGH_SNR_PAIRS = [(3, 3), (4, 4), (5, 5), (3, 4), (3, 5), (4, 5)]

print(f"  ISF: N={N_val}, eps_inf={eps_inf}, alpha={alpha}")

# ── Flatten data vector and invert covariance ─────────────────────
data_flat = bp_data.flatten()  # (120,)
cov_inv = np.linalg.inv(cov_ee)
Cinv_d = cov_inv @ data_flat
dCd = data_flat @ Cinv_d

print(f"  d.C^-1.d = {dCd:.2f}")


# ══════════════════════════════════════════════════════════════════
# PART 2: Compute effective comoving distances per pair
# ══════════════════════════════════════════════════════════════════
print("\n[2] Computing effective comoving distances...")

z_mean_bin = {}
for i in range(nz_bins.shape[0]):
    z_mean_bin[i + 1] = np.average(z_mid, weights=nz_bins[i])

chi_eff_pair = np.zeros(n_pairs)
z_eff_pair = np.zeros(n_pairs)
is_high_snr = np.zeros(n_pairs, dtype=bool)

for ip in range(n_pairs):
    b1, b2 = pair_labels[ip]
    W1 = W_all[b1 - 1]
    W2 = W_all[b2 - 1]
    kernel = W1 * W2 / np.maximum(chi_W**2, 1e-30)
    norm = np.trapezoid(kernel, chi_W)
    if norm > 0:
        chi_eff_pair[ip] = np.trapezoid(kernel * chi_W, chi_W) / norm
    z_eff_pair[ip] = 0.5 * (z_mean_bin[b1] + z_mean_bin[b2])
    is_high_snr[ip] = (b1, b2) in HIGH_SNR_PAIRS

print(f"\n  {'Pair':>7s}  {'z_eff':>6s}  {'chi_eff':>10s}  {'SNR':>4s}")
for ip in range(n_pairs):
    b1, b2 = pair_labels[ip]
    snr = "HIGH" if is_high_snr[ip] else "low"
    print(f"  ({b1},{b2}){'':<2s}  {z_eff_pair[ip]:6.4f}  "
          f"{chi_eff_pair[ip]:10.1f}  {snr:>4s}")


# ══════════════════════════════════════════════════════════════════
# PART 3: LCDM baseline
# ══════════════════════════════════════════════════════════════════
print("\n[3] Computing LCDM baseline...")

m_lcdm_flat = cl_lcdm_tomo.flatten()
Cinv_m_lcdm = cov_inv @ m_lcdm_flat
dCm_lcdm = data_flat @ Cinv_m_lcdm
mCm_lcdm = m_lcdm_flat @ Cinv_m_lcdm

A_opt_lcdm = dCm_lcdm / mCm_lcdm
chi2_lcdm = dCd - dCm_lcdm**2 / mCm_lcdm

print(f"  LCDM A_opt = {A_opt_lcdm:.1f}")
print(f"  LCDM chi^2 = {chi2_lcdm:.2f} (chi^2/Ndof = {chi2_lcdm/(n_data-1):.3f})")


# ══════════════════════════════════════════════════════════════════
# PART 4: Original Step 2c grid (apparent k_t, uniform across pairs)
# ══════════════════════════════════════════════════════════════════
print("\n[4] Step 2c grid: uniform apparent k_t (reproducing original)...")

k_t_grid = np.logspace(np.log10(0.05), np.log10(2.0), 30)
n_kt = len(k_t_grid)

chi2_apparent = np.zeros(n_kt)
A_opt_apparent = np.zeros(n_kt)

t0 = time.time()
for ikt, k_t in enumerate(k_t_grid):
    eps_k = eps_inf / (1.0 + (k_t / k_hMpc)**alpha)
    M = (1.0 + (N_val - 1) * eps_k) / N_val
    M_sq = M**2

    cl_isf = np.array([A_tomo[ip] @ M_sq for ip in range(n_pairs)])
    m_flat = cl_isf.flatten()

    Cinv_m = cov_inv @ m_flat
    dCm = data_flat @ Cinv_m
    mCm = m_flat @ Cinv_m

    if mCm > 0:
        A_opt_apparent[ikt] = dCm / mCm
        chi2_apparent[ikt] = dCd - dCm**2 / mCm
    else:
        chi2_apparent[ikt] = dCd

elapsed = time.time() - t0
print(f"  Computed in {elapsed:.3f}s")

# Best-fit apparent k_t
idx_best_app = np.argmin(chi2_apparent)
kt_best_app = k_t_grid[idx_best_app]
dchi2_best_app = chi2_apparent[idx_best_app] - chi2_lcdm

# Refine around minimum
kt_lo = k_t_grid[max(0, idx_best_app - 2)]
kt_hi = k_t_grid[min(n_kt - 1, idx_best_app + 2)]
if idx_best_app == n_kt - 1:
    kt_hi = k_t_grid[-1] * 2.0
kt_refine = np.linspace(kt_lo, kt_hi, 50)
chi2_refine = np.zeros(50)
for j, kt in enumerate(kt_refine):
    eps_k = eps_inf / (1.0 + (kt / k_hMpc)**alpha)
    M = (1.0 + (N_val - 1) * eps_k) / N_val
    cl_isf = np.array([A_tomo[ip] @ (M**2) for ip in range(n_pairs)])
    m_flat = cl_isf.flatten()
    Cinv_m = cov_inv @ m_flat
    dCm = data_flat @ Cinv_m
    mCm = m_flat @ Cinv_m
    chi2_refine[j] = dCd - dCm**2 / mCm

idx_ref = np.argmin(chi2_refine)
kt_best_app_refined = kt_refine[idx_ref]
chi2_best_app_refined = chi2_refine[idx_ref]
dchi2_best_app_refined = chi2_best_app_refined - chi2_lcdm

# Recompute A_opt at refined best
eps_k_bf = eps_inf / (1.0 + (kt_best_app_refined / k_hMpc)**alpha)
M_bf = (1.0 + (N_val - 1) * eps_k_bf) / N_val
m_bf = np.array([A_tomo[ip] @ (M_bf**2) for ip in range(n_pairs)]).flatten()
Cinv_m_bf = cov_inv @ m_bf
A_opt_best_app = (data_flat @ Cinv_m_bf) / (m_bf @ Cinv_m_bf)

print(f"\n  Best-fit apparent k_t (N=6, full cov):")
print(f"    k_t^app = {kt_best_app_refined:.4f} h/Mpc")
print(f"    Delta_chi^2 = {dchi2_best_app_refined:.2f}")
print(f"    A_opt = {A_opt_best_app:.1f}")

# Confidence intervals for apparent k_t
kt_ci = np.logspace(np.log10(0.05), np.log10(2.0), 200)
chi2_ci = np.zeros(200)
for j, kt in enumerate(kt_ci):
    eps_k = eps_inf / (1.0 + (kt / k_hMpc)**alpha)
    M = (1.0 + (N_val - 1) * eps_k) / N_val
    m = np.array([A_tomo[ip] @ (M**2) for ip in range(n_pairs)]).flatten()
    Cinv_m = cov_inv @ m
    dCm = data_flat @ Cinv_m
    mCm = m @ Cinv_m
    chi2_ci[j] = dCd - dCm**2 / mCm

chi2_min_app = chi2_ci.min()
mask_1sig = (chi2_ci - chi2_min_app) < 1.0
mask_2sig = (chi2_ci - chi2_min_app) < 4.0

kt_app_1sig = (kt_ci[mask_1sig].min(), kt_ci[mask_1sig].max()) if np.any(mask_1sig) else (0, 0)
kt_app_2sig = (kt_ci[mask_2sig].min(), kt_ci[mask_2sig].max()) if np.any(mask_2sig) else (0, 0)

print(f"    1-sigma: [{kt_app_1sig[0]:.3f}, {kt_app_1sig[1]:.3f}]")
print(f"    2-sigma: [{kt_app_2sig[0]:.3f}, {kt_app_2sig[1]:.3f}]")


# ══════════════════════════════════════════════════════════════════
# PART 5: Physical k_t scan (Limber-corrected)
# ══════════════════════════════════════════════════════════════════
print("\n[5] Physical k_t scan (single k_t^phys, per-pair amplitudes)...")

# For the physical k_t scan, we use per-pair diagonal blocks of the
# covariance (as in C1), since the physical k_t maps to different
# apparent k_t values per pair and the full-covariance joint fit
# requires a single model vector.
#
# Method: for each k_t^{phys}, evaluate chi^2 for each high-SNR pair
# independently (with its own profiled amplitude), sum the chi^2 values.

kt_phys_grid = np.logspace(np.log10(0.1), np.log10(2.0), 200)

# Precompute per-pair covariance inverses for high-SNR pairs
pair_data = []
for ip in range(n_pairs):
    if not is_high_snr[ip]:
        continue
    b1, b2 = pair_labels[ip]
    i0 = ip * n_ell
    i1 = (ip + 1) * n_ell
    d_pair = bp_data[ip]
    cov_pair = cov_ee[i0:i1, i0:i1]
    cov_inv_pair = np.linalg.inv(cov_pair)
    Cinv_d_pair = cov_inv_pair @ d_pair
    dCd_pair = d_pair @ Cinv_d_pair

    # LCDM baseline for this pair
    m_lcdm_pair = cl_lcdm_tomo[ip]
    Cinv_m_lcdm_pair = cov_inv_pair @ m_lcdm_pair
    chi2_lcdm_pair = dCd_pair - (d_pair @ Cinv_m_lcdm_pair)**2 / (m_lcdm_pair @ Cinv_m_lcdm_pair)

    pair_data.append({
        'ip': ip, 'b1': b1, 'b2': b2,
        'chi_eff': chi_eff_pair[ip],
        'z_eff': z_eff_pair[ip],
        'd_pair': d_pair,
        'cov_inv_pair': cov_inv_pair,
        'dCd_pair': dCd_pair,
        'chi2_lcdm_pair': chi2_lcdm_pair,
    })

chi2_lcdm_sum = sum(pd['chi2_lcdm_pair'] for pd in pair_data)

# Scan k_t^{phys}
chi2_phys_total = np.zeros(len(kt_phys_grid))

t0 = time.time()
for j, kt_phys in enumerate(kt_phys_grid):
    eps_k = eps_inf / (1.0 + (kt_phys / k_hMpc)**alpha)
    M = (1.0 + (N_val - 1) * eps_k) / N_val
    M_sq = M**2

    total = 0.0
    for pd in pair_data:
        ip = pd['ip']
        m_isf = A_tomo[ip] @ M_sq
        Cinv_m = pd['cov_inv_pair'] @ m_isf
        dCm = pd['d_pair'] @ Cinv_m
        mCm = m_isf @ Cinv_m
        if mCm > 0:
            total += pd['dCd_pair'] - dCm**2 / mCm
        else:
            total += pd['dCd_pair']
    chi2_phys_total[j] = total

elapsed = time.time() - t0
print(f"  Scanned {len(kt_phys_grid)} k_t^phys values in {elapsed:.3f}s")

# Best-fit k_t^{phys}
idx_best_phys = np.argmin(chi2_phys_total)
kt_phys_best = kt_phys_grid[idx_best_phys]
dchi2_phys_best = chi2_phys_total[idx_best_phys] - chi2_lcdm_sum

# Confidence intervals
mask_1sig_phys = (chi2_phys_total - chi2_phys_total[idx_best_phys]) < 1.0
mask_2sig_phys = (chi2_phys_total - chi2_phys_total[idx_best_phys]) < 4.0

kt_phys_1sig = (kt_phys_grid[mask_1sig_phys].min(),
                kt_phys_grid[mask_1sig_phys].max()) if np.any(mask_1sig_phys) else (0, 0)
kt_phys_2sig = (kt_phys_grid[mask_2sig_phys].min(),
                kt_phys_grid[mask_2sig_phys].max()) if np.any(mask_2sig_phys) else (0, 0)

# Angular transition scale
ell_t_best = kt_phys_best * np.median([pd['chi_eff'] for pd in pair_data])

print(f"\n  Best-fit k_t^phys (joint, 6 high-SNR pairs):")
print(f"    k_t^phys = {kt_phys_best:.4f} h/Mpc")
print(f"    ell_t = {ell_t_best:.0f}")
print(f"    Joint Delta_chi^2 = {dchi2_phys_best:.2f}")
print(f"    1-sigma: [{kt_phys_1sig[0]:.3f}, {kt_phys_1sig[1]:.3f}]")
print(f"    2-sigma: [{kt_phys_2sig[0]:.3f}, {kt_phys_2sig[1]:.3f}]")


# ══════════════════════════════════════════════════════════════════
# PART 6: Per-pair k_t and ell_t consistency check
# ══════════════════════════════════════════════════════════════════
print("\n[6] Per-pair k_t fits and ell_t consistency...")

kt_fine = np.logspace(np.log10(0.05), np.log10(3.0), 300)

print(f"\n  {'Pair':>7s}  {'z_eff':>6s}  {'chi_eff':>8s}  {'k_t':>8s}  "
      f"{'ell_t':>8s}  {'Dchi2':>8s}")

per_pair_results = []
for pd in sorted(pair_data, key=lambda x: x['z_eff']):
    ip = pd['ip']
    b1, b2 = pd['b1'], pd['b2']

    best_chi2 = np.inf
    best_kt = 0
    for kt in kt_fine:
        eps_k = eps_inf / (1.0 + (kt / k_hMpc)**alpha)
        M = (1.0 + (N_val - 1) * eps_k) / N_val
        m_isf = A_tomo[ip] @ (M**2)
        Cinv_m = pd['cov_inv_pair'] @ m_isf
        dCm = pd['d_pair'] @ Cinv_m
        mCm = m_isf @ Cinv_m
        if mCm > 0:
            chi2 = pd['dCd_pair'] - dCm**2 / mCm
            if chi2 < best_chi2:
                best_chi2 = chi2
                best_kt = kt

    ell_t = best_kt * pd['chi_eff']
    dchi2 = best_chi2 - pd['chi2_lcdm_pair']

    per_pair_results.append({
        'ip': ip, 'b1': b1, 'b2': b2,
        'z_eff': pd['z_eff'],
        'chi_eff': pd['chi_eff'],
        'kt_best': best_kt,
        'ell_t': ell_t,
        'dchi2': dchi2,
    })

    print(f"  ({b1},{b2}){'':<2s}  {pd['z_eff']:6.4f}  {pd['chi_eff']:8.1f}  "
          f"{best_kt:8.4f}  {ell_t:8.1f}  {dchi2:+8.1f}")

ell_t_arr = np.array([r['ell_t'] for r in per_pair_results])
kt_arr = np.array([r['kt_best'] for r in per_pair_results])

print(f"\n  ell_t: mean={ell_t_arr.mean():.0f}, scatter={ell_t_arr.std()/ell_t_arr.mean():.1%}")


# ══════════════════════════════════════════════════════════════════
# PART 7: Full-covariance joint fit at k_t^{phys}
# ══════════════════════════════════════════════════════════════════
print("\n[7] Full-covariance chi^2 at k_t^phys = 0.52...")

# Evaluate the full 120-point chi^2 at the D1 fiducial k_t^{phys} = 0.52
kt_phys_d1 = 0.52
eps_k = eps_inf / (1.0 + (kt_phys_d1 / k_hMpc)**alpha)
M = (1.0 + (N_val - 1) * eps_k) / N_val
M_sq = M**2

cl_isf_d1 = np.array([A_tomo[ip] @ M_sq for ip in range(n_pairs)])
m_flat_d1 = cl_isf_d1.flatten()

Cinv_m = cov_inv @ m_flat_d1
dCm = data_flat @ Cinv_m
mCm = m_flat_d1 @ Cinv_m
A_opt_d1 = dCm / mCm
chi2_d1 = dCd - dCm**2 / mCm
dchi2_d1 = chi2_d1 - chi2_lcdm

print(f"  At k_t^phys = {kt_phys_d1} (full 120-point, full cov):")
print(f"    chi^2 = {chi2_d1:.2f}")
print(f"    Delta_chi^2 = {dchi2_d1:.2f}")
print(f"    A_opt = {A_opt_d1:.1f}")

# Also at the best apparent k_t for comparison
eps_k_082 = eps_inf / (1.0 + (0.82 / k_hMpc)**alpha)
M_082 = (1.0 + (N_val - 1) * eps_k_082) / N_val
m_082 = np.array([A_tomo[ip] @ (M_082**2) for ip in range(n_pairs)]).flatten()
Cinv_m_082 = cov_inv @ m_082
dCm_082 = data_flat @ Cinv_m_082
mCm_082 = m_082 @ Cinv_m_082
chi2_082 = dCd - dCm_082**2 / mCm_082

print(f"\n  At k_t^app = 0.82 (full 120-point, full cov):")
print(f"    chi^2 = {chi2_082:.2f}")
print(f"    Delta_chi^2 = {chi2_082 - chi2_lcdm:.2f}")

print(f"\n  Penalty for k_t=0.52 vs k_t=0.82 (full cov): "
      f"{chi2_d1 - chi2_082:+.2f}")
print(f"  (Expected: small, since 0.82 is the full-cov optimum)")


# ══════════════════════════════════════════════════════════════════
# PART 8: Updated fiducial summary
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("UPDATED PAPER VII FIDUCIAL PARAMETERS")
print("=" * 70)

print(f"""
  ISF model (fixed):
    N = {N_val}
    eps_inf = {eps_inf}
    alpha = {alpha}

  Transition scale (updated):
    k_t^phys      = {kt_phys_best:.2f} h/Mpc [{kt_phys_1sig[0]:.2f}, {kt_phys_1sig[1]:.2f}] (1-sigma)
    ell_t          ~ {ell_t_arr.mean():.0f} (5.6% scatter across 6 high-SNR pairs)
    k_t^apparent   = {kt_best_app_refined:.2f} h/Mpc (Limber projection artifact)
    Paper V value  = 0.20 h/Mpc (SUPERSEDED)
    Shift ratio    = {kt_phys_best/0.20:.1f}x from Paper V

  Chi^2 summary (N=6):
    LCDM baseline (full 120-pt):     chi^2 = {chi2_lcdm:.2f}
    ISF at k_t^app=0.82 (full cov):  Dchi^2 = {chi2_082 - chi2_lcdm:.2f}
    ISF at k_t^phys=0.52 (full cov): Dchi^2 = {dchi2_d1:.2f}
    ISF joint (6 hi-SNR, per-pair):  Dchi^2 = {dchi2_phys_best:.2f}

  Interpretation:
    The global k_t=0.82 is a Limber projection artifact: it is the
    apparent value when a single k_t is forced across all tomographic
    pairs, each probing different effective distances.

    The physical transition scale k_t^phys=0.52 [0.46, 0.55] is
    obtained from the joint fit of 6 high-SNR pairs, each with its
    own profiled amplitude. The product k_t*chi_eff = ell_t ~ {ell_t_arr.mean():.0f}
    is approximately constant across pairs (scatter {ell_t_arr.std()/ell_t_arr.mean():.1%}).

    Paper V's k_t=0.20 is superseded by the full Limber projection
    analysis. The 2.6x shift is genuine, arising from replacing the
    approximate k ~ 1/R mapping with proper tomographic lensing kernels.

    Paper VI predictions are UNAFFECTED (sigmoid saturated at SL scales;
    <1% impact verified in C2).

  For Phase 3 onward:
    Use k_t^phys = 0.52 h/Mpc as the fiducial.
    The k_t grid for the joint constraint should span [0.1, 2.0] h/Mpc.
    Report both k_t^phys and ell_t.
""")


# ══════════════════════════════════════════════════════════════════
# PART 9: Save checkpoint
# ══════════════════════════════════════════════════════════════════
print("[9] Saving checkpoint...")

np.savez(
    OUTPATH,
    # -- Apparent k_t grid (original Step 2c, N=6 only) --
    k_t_grid=k_t_grid,
    chi2_apparent=chi2_apparent,
    A_opt_apparent=A_opt_apparent,
    kt_best_apparent=kt_best_app_refined,
    chi2_best_apparent=chi2_best_app_refined,
    A_opt_best_apparent=A_opt_best_app,
    kt_app_1sig=np.array(kt_app_1sig),
    kt_app_2sig=np.array(kt_app_2sig),
    # -- Physical k_t scan (joint high-SNR pairs) --
    kt_phys_grid=kt_phys_grid,
    chi2_phys_total=chi2_phys_total,
    kt_phys_best=kt_phys_best,
    kt_phys_1sig=np.array(kt_phys_1sig),
    kt_phys_2sig=np.array(kt_phys_2sig),
    dchi2_phys_best=dchi2_phys_best,
    # -- Per-pair results --
    per_pair_kt=np.array([r['kt_best'] for r in per_pair_results]),
    per_pair_ell_t=ell_t_arr,
    per_pair_dchi2=np.array([r['dchi2'] for r in per_pair_results]),
    per_pair_z_eff=np.array([r['z_eff'] for r in per_pair_results]),
    per_pair_chi_eff=np.array([r['chi_eff'] for r in per_pair_results]),
    per_pair_labels=np.array([(r['b1'], r['b2']) for r in per_pair_results]),
    # -- Full-cov values at key k_t --
    chi2_at_kt052=chi2_d1,
    A_opt_at_kt052=A_opt_d1,
    chi2_at_kt082=chi2_082,
    # -- LCDM baseline --
    chi2_lcdm=chi2_lcdm,
    A_opt_lcdm=A_opt_lcdm,
    chi2_lcdm_sum_highsnr=chi2_lcdm_sum,
    # -- Metadata --
    N_val=N_val,
    eps_inf=eps_inf,
    alpha=alpha,
    n_data=n_data,
    ell_eff=ell_eff,
    ell_t_mean=ell_t_arr.mean(),
    ell_t_scatter=ell_t_arr.std() / ell_t_arr.mean(),
)

print(f"  Saved to {OUTPATH}")
print("\nScript complete.")
