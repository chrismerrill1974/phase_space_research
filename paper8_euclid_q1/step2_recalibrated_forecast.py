#!/usr/bin/env python3
"""
Paper VIII — Step 2: Recalibrated ISF falsification forecast — VERSION 2 (2026-10-11)

Version 1 read Paper VII's per-lens counts, which inherited Paper VI's
missing Jacobian 1/1.4 (Papers VI and VII, version 2). This copy reads
Paper VII's version-2 product (euclid_forecast_v2.npz) and writes
recalibrated_forecast_v2.npz beside the version-1 product. It gates: the
input counts must equal version 1's times 1/1.4, and an existing v2
product must match this run (it is never rewritten). Everything else is
the version-1 code, unchanged.

Reruns the Paper VII Phase 5a Poisson forecast with the assumed survey
inputs replaced by measured Euclid Q1 demographics:

  N_lenses : Paper VII assumed 75,000 grade A/B (wide survey).
             Q1-measured rate (7.88/deg^2) gives ~110,000 wide,
             ~15,000 DR1, and 497 already in hand (Q1).
  eta      : Paper VII assumed a flat 0.05. Here three bracketing models
             normalised to eta_0 = 0.05 at theta_ref = 1.2" (legacy
             SLACS-like median), evaluated over the measured Q1 theta_E
             distribution:
               flat   : eta = 0.05
               linear : eta = eta_0 * (theta_E/theta_ref),   cap 0.15
               area   : eta = eta_0 * (theta_E/theta_ref)^2, cap 0.15
             Rationale for scaling: ISF perturbers (1e10-1e12 Msun) are
             far above the 1e8.8 Msun sensitivity floor (O'Riordan+23),
             so detectability is limited by the sky area within
             ~2 theta_E that the arcs sample.

Also computes the "Q1 in-principle bound": the expected ISF detection
count across the actual modelled grade-A/B Q1 systems, i.e. what a
complete perturber search of the existing sample would already test.

Inputs:  ../data/q1_demographics.npz
         ../data/from_paper_vii/euclid_forecast.npz  (N_det per lens)
         ../data/from_paper_vii/joint_constraint.npz (current joint chi2)
         ../data/from_paper_vii/lnL_sub_grid.npz
Output:  ../data/recalibrated_forecast.npz
"""

import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA = os.path.join(ROOT, 'data')
P7 = os.path.join(DATA, 'from_paper_vii')
OUTPATH = os.path.join(DATA, 'recalibrated_forecast_v2.npz')   # version 2 product
V1PATH = os.path.join(DATA, 'recalibrated_forecast.npz')      # version 1 product

demo = np.load(os.path.join(DATA, 'q1_demographics.npz'), allow_pickle=True)
fc7 = np.load(os.path.join(P7, 'euclid_forecast_v2.npz'), allow_pickle=True)   # Paper VII version 2
fc7_v1 = np.load(os.path.join(P7, 'euclid_forecast.npz'), allow_pickle=True)   # version 1, for the gate
jc = np.load(os.path.join(P7, 'joint_constraint.npz'), allow_pickle=True)
sub = np.load(os.path.join(P7, 'lnL_sub_grid.npz'), allow_pickle=True)

N_grid = fc7['N_grid']
N_det_per_lens = fc7['N_det_per_lens']   # ISF detectable perturbers/lens
k_t_grid = jc['k_t_grid']
chi2_wl_grid = jc['chi2_wl_grid']
chi2_sub_grid = sub['chi2_sub_grid']
chi2_lcdm = float(jc['chi2_lcdm'])
n_N, n_kt = len(N_grid), len(k_t_grid)
iN_fid = int(np.where(N_grid == 6)[0][0])

theta_AB = demo['theta_E_AB']            # measured Q1 grade-A/B theta_E

# ── Detection-efficiency models ───────────────────────────────────
ETA0, THETA_REF, ETA_CAP = 0.05, 1.2, 0.15

def eta_of_theta(theta, model):
    if model == 'flat':
        return np.full_like(theta, ETA0)
    if model == 'linear':
        return np.minimum(ETA0 * (theta / THETA_REF), ETA_CAP)
    if model == 'area':
        return np.minimum(ETA0 * (theta / THETA_REF) ** 2, ETA_CAP)
    raise ValueError(model)

ETA_MODELS = ['flat', 'linear', 'area']
eta_eff = {m: float(np.mean(eta_of_theta(theta_AB, m))) for m in ETA_MODELS}

print("Effective detection efficiency over Q1 theta_E distribution:")
for m in ETA_MODELS:
    print(f"  {m:7s}: <eta> = {eta_eff[m]:.4f}  "
          f"(Paper VII flat 0.05 -> ratio {eta_eff[m]/0.05:.2f})")

# ── Survey stages ─────────────────────────────────────────────────
STAGES = {
    'Q1':   float(demo['n_AB']),     # 497, in hand now
    'DR1':  float(demo['N_dr1']),    # ~15,000, first tranche Nov 2026
    'Wide': float(demo['N_wide']),   # ~110,000, end of mission
}
P7_STAGE = ('P7-assumed', 75000.0, 0.05)  # Paper VII reference

print(f"\nSurvey stages (grade A/B lenses): "
      + ", ".join(f"{k}={v:.0f}" for k, v in STAGES.items()))

# ── Forecast machinery (Paper VII Phase 5a, parametrised) ────────
def forecast(n_lenses, eta):
    """Return dict of forecast quantities for one (N_lenses, eta)."""
    a = n_lenses * eta                        # effective sightlines
    n_exp = N_det_per_lens * a                # expected detections per N

    # ISF-true (fiducial N=6): Gaussian approx to Poisson
    n_true = n_exp[iN_fid]
    sigma_Ndet = np.sqrt(n_true) / a
    sig_vs_fid = np.abs(N_det_per_lens - N_det_per_lens[iN_fid]) / sigma_Ndet

    # chi2 grids
    chi2_isf = ((N_det_per_lens - N_det_per_lens[iN_fid]) / sigma_Ndet) ** 2
    chi2_isf = np.repeat(chi2_isf[:, None], n_kt, axis=1)
    chi2_joint_isf = chi2_wl_grid + chi2_sub_grid + chi2_isf

    # marginalised P(N), log-uniform weight in k_t (as Paper VII)
    dlog = np.diff(np.log(k_t_grid))
    wt = np.zeros(n_kt)
    wt[0], wt[-1] = dlog[0] / 2, dlog[-1] / 2
    wt[1:-1] = (dlog[:-1] + dlog[1:]) / 2
    L = np.exp(-0.5 * (chi2_joint_isf - chi2_joint_isf.min()))
    P_N = np.array([np.sum(L[iN] * wt) for iN in range(n_N)])
    P_N /= P_N.sum()

    # Null scenario: n_obs = 0 -> -2lnL = 2*n_model
    sig_null = np.sqrt(2.0 * n_exp)

    # 95% falsification thresholds (quadratic from Paper VII step 5a)
    thr = np.zeros(n_N)
    for iN in range(n_N):
        mu = N_det_per_lens[iN]
        A, B, C = a, -(2 * a * mu + 3.84), a * mu ** 2
        disc = B * B - 4 * A * C
        thr[iN] = (-B - np.sqrt(disc)) / (2 * A) if disc > 0 else 0.0

    return dict(n_exp=n_exp, sigma_Ndet=sigma_Ndet, sig_vs_fid=sig_vs_fid,
                P_N=P_N, sig_null=sig_null, thr95=thr,
                thr95_total=thr * a, a=a)

results = {}
print("\n" + "=" * 74)
hdr = f"{'stage':>6s} {'eta':>7s} {'a=Nl*eta':>9s} {'n_exp(N=6)':>10s} " \
      f"{'sig(2v6)':>8s} {'sig(4v6)':>8s} {'null(N=2)':>9s} {'P(N=6)':>8s}"
print(hdr)
iN2 = int(np.where(N_grid == 2)[0][0])
iN4 = int(np.where(N_grid == 4)[0][0])
for stage, nl in list(STAGES.items()):
    for m in ETA_MODELS:
        r = forecast(nl, eta_eff[m])
        results[(stage, m)] = r
        print(f"{stage:>6s} {m:>7s} {r['a']:9.0f} {r['n_exp'][iN_fid]:10.0f} "
              f"{r['sig_vs_fid'][iN2]:8.1f} {r['sig_vs_fid'][iN4]:8.1f} "
              f"{r['sig_null'][iN2]:9.1f} {r['P_N'][iN_fid]:8.3f}")

# Paper VII reference row
r7 = forecast(P7_STAGE[1], P7_STAGE[2])
results[('P7', 'flat')] = r7
print(f"{'P7ref':>6s} {'flat':>7s} {r7['a']:9.0f} {r7['n_exp'][iN_fid]:10.0f} "
      f"{r7['sig_vs_fid'][iN2]:8.1f} {r7['sig_vs_fid'][iN4]:8.1f} "
      f"{r7['sig_null'][iN2]:9.1f} {r7['P_N'][iN_fid]:8.3f}")

# ── Headline falsification sentence (Wide, bracketed over eta) ────
print("\nUpdated 95% falsification thresholds (total detections, Wide):")
for m in ETA_MODELS:
    r = results[('Wide', m)]
    print(f"  {m:7s}: all-ISF (N>=2) < {r['thr95_total'][iN2]:.0f};  "
          f"N>=4 < {r['thr95_total'][iN4]:.0f}   "
          f"(Paper VII: 1844 / 3358 at 75k lenses)")

# ── Q1 in-principle bound: actual modelled grade-A/B systems ─────
print("\nQ1 in-principle bound (sum of per-lens eta over 312 modelled A/B):")
q1_inprinciple = {}
for m in ETA_MODELS:
    a_q1 = float(np.sum(eta_of_theta(theta_AB, m)))
    n_exp_q1 = N_det_per_lens * a_q1
    sig_null_q1 = np.sqrt(2.0 * n_exp_q1)
    q1_inprinciple[m] = (a_q1, n_exp_q1, sig_null_q1)
    print(f"  {m:7s}: eff. sightlines {a_q1:6.1f};  "
          f"n_exp(N=6) = {n_exp_q1[iN_fid]:5.1f};  "
          f"null excl: N=2 {sig_null_q1[iN2]:.1f} sig, "
          f"N=6 {sig_null_q1[iN_fid]:.1f} sig, N=10 {sig_null_q1[-1]:.1f} sig")

# ── Save ──────────────────────────────────────────────────────────
save = dict(
    N_grid=N_grid, k_t_grid=k_t_grid, N_det_per_lens=N_det_per_lens,
    eta_models=np.array(ETA_MODELS),
    eta_eff=np.array([eta_eff[m] for m in ETA_MODELS]),
    eta0=ETA0, theta_ref=THETA_REF, eta_cap=ETA_CAP,
    stages=np.array(list(STAGES.keys())),
    stage_nlenses=np.array([STAGES[s] for s in STAGES]),
)
for (stage, m), r in results.items():
    for key, val in r.items():
        save[f'{stage}_{m}_{key}'] = val
for m, (a_q1, n_exp_q1, sig_q1) in q1_inprinciple.items():
    save[f'q1ip_{m}_a'] = a_q1
    save[f'q1ip_{m}_n_exp'] = n_exp_q1
    save[f'q1ip_{m}_sig_null'] = sig_q1
save['jacobian'] = 1.0 / 1.4

# ── Version 2 GATES ───────────────────────────────────────────────
g1 = np.allclose(fc7['N_det_per_lens'], fc7_v1['N_det_per_lens'] / 1.4, rtol=1e-12, atol=0)
print(f"\nVERSION 2 GATE 1: input N_det per lens = version 1 x 1/1.4: {'PASS' if g1 else 'FAIL'}")
if not g1:
    raise SystemExit("gate 1 failed")
if os.path.exists(OUTPATH):
    prev = np.load(OUTPATH, allow_pickle=True)
    common = [k for k in prev.files if k in save]
    g2 = all(np.array_equal(prev[k], save[k]) if np.asarray(prev[k]).dtype.kind in 'US'
             else np.allclose(np.asarray(prev[k], dtype=float), np.asarray(save[k], dtype=float), rtol=1e-12, atol=0)
             for k in common)
    print(f"VERSION 2 GATE 2: existing {os.path.basename(OUTPATH)} matches this run on {len(common)} arrays: "
          f"{'PASS (not rewritten)' if g2 else 'FAIL: archive it and rerun'}")
    if not g2:
        raise SystemExit("gate 2 failed")
else:
    np.savez(OUTPATH, **save)
    print(f"\nSaved {OUTPATH}")
