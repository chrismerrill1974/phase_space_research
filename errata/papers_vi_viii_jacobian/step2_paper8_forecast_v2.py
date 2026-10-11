#!/usr/bin/env python3
"""
Erratum to ISF Papers VI-VIII — Level 2: propagate the Jacobian through
Paper VIII's recalibrated forecast (step 2).

Paper VIII's step2_recalibrated_forecast.py takes Paper VII's per-lens
counts N_det(N) from euclid_forecast.npz and reruns the Phase 5a Poisson
forecast with the measured Euclid Q1 demographics (497 grade-A/B at
7.88/deg^2 -> ~15,000 DR1, ~110,000 wide; three detection-efficiency
models over the measured theta_E distribution; the 312-system
in-principle bound). Every number in it scales with N_det, which Level 1
corrected by 1/1.4 (euclid_forecast_v2.npz). This script reproduces
Paper VIII's step 2 as a function of the input product, then:

  GATE   from the v1 input it must reproduce every array in the published
         recalibrated_forecast.npz;
  V2     from the v2 input it writes recalibrated_forecast_v2.npz BESIDE
         the original (never overwriting it), same keys;
  TABLE  prints every number Paper VIII's text quotes from this step,
         old beside new, and saves the printout.

Inputs:  ../../dark_matter_predictions_8/data/q1_demographics.npz
         ../../dark_matter_predictions_8/data/from_paper_vii/euclid_forecast.npz     (v1, gate)
         ../../dark_matter_predictions_8/data/from_paper_vii/euclid_forecast_v2.npz  (Level 1 product, copied in)
         ../../dark_matter_predictions_8/data/from_paper_vii/joint_constraint.npz, lnL_sub_grid.npz
         ../../dark_matter_predictions_8/data/recalibrated_forecast.npz (gate)
Output:  ../../dark_matter_predictions_8/data/recalibrated_forecast_v2.npz
         ../data/paper8_forecast_v2.txt (this printout)
Runtime: seconds, single-threaded.
"""

import os

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
SERIES = os.path.abspath(os.path.join(ROOT, '..'))
DATA = os.path.join(ROOT, 'data')
P8DATA = os.path.join(SERIES, 'dark_matter_predictions_8', 'data')
P7 = os.path.join(P8DATA, 'from_paper_vii')
PUB_PATH = os.path.join(P8DATA, 'recalibrated_forecast.npz')
OUTPATH = os.path.join(P8DATA, 'recalibrated_forecast_v2.npz')

_lines = []
def say(s=''):
    print(s); _lines.append(s)

demo = np.load(os.path.join(P8DATA, 'q1_demographics.npz'), allow_pickle=True)
jc = np.load(os.path.join(P7, 'joint_constraint.npz'), allow_pickle=True)
sub = np.load(os.path.join(P7, 'lnL_sub_grid.npz'), allow_pickle=True)
k_t_grid = jc['k_t_grid']; chi2_wl_grid = jc['chi2_wl_grid']; chi2_sub_grid = sub['chi2_sub_grid']
theta_AB = demo['theta_E_AB']

# ── Paper VIII step 2, verbatim machinery ────────────────────────────
ETA0, THETA_REF, ETA_CAP = 0.05, 1.2, 0.15
def eta_of_theta(theta, model):
    if model == 'flat':   return np.full_like(theta, ETA0)
    if model == 'linear': return np.minimum(ETA0 * (theta / THETA_REF), ETA_CAP)
    if model == 'area':   return np.minimum(ETA0 * (theta / THETA_REF) ** 2, ETA_CAP)
    raise ValueError(model)
ETA_MODELS = ['flat', 'linear', 'area']
eta_eff = {m: float(np.mean(eta_of_theta(theta_AB, m))) for m in ETA_MODELS}
STAGES = {'Q1': float(demo['n_AB']), 'DR1': float(demo['N_dr1']), 'Wide': float(demo['N_wide'])}
P7_STAGE = (75000.0, 0.05)

def run(fc):
    """Paper VIII step 2 for one Paper VII product; returns the save dict and handles."""
    N_grid = fc['N_grid']; N_det = fc['N_det_per_lens']
    n_N, n_kt = len(N_grid), len(k_t_grid)
    iN_fid = int(np.where(N_grid == 6)[0][0])
    def forecast(n_lenses, eta):
        a = n_lenses * eta
        n_exp = N_det * a
        sigma_Ndet = np.sqrt(n_exp[iN_fid]) / a
        sig_vs_fid = np.abs(N_det - N_det[iN_fid]) / sigma_Ndet
        chi2_isf = np.repeat((((N_det - N_det[iN_fid]) / sigma_Ndet) ** 2)[:, None], n_kt, axis=1)
        chi2_joint_isf = chi2_wl_grid + chi2_sub_grid + chi2_isf
        dlog = np.diff(np.log(k_t_grid)); wt = np.zeros(n_kt)
        wt[0], wt[-1] = dlog[0] / 2, dlog[-1] / 2; wt[1:-1] = (dlog[:-1] + dlog[1:]) / 2
        L = np.exp(-0.5 * (chi2_joint_isf - chi2_joint_isf.min()))
        P_N = np.array([np.sum(L[iN] * wt) for iN in range(n_N)]); P_N /= P_N.sum()
        sig_null = np.sqrt(2.0 * n_exp)
        thr = np.zeros(n_N)
        for iN in range(n_N):
            mu = N_det[iN]; A, B, C = a, -(2 * a * mu + 3.84), a * mu ** 2
            disc = B * B - 4 * A * C
            thr[iN] = (-B - np.sqrt(disc)) / (2 * A) if disc > 0 else 0.0
        return dict(n_exp=n_exp, sigma_Ndet=sigma_Ndet, sig_vs_fid=sig_vs_fid, P_N=P_N,
                    sig_null=sig_null, thr95=thr, thr95_total=thr * a, a=a)
    results = {}
    for stage, nl in STAGES.items():
        for m in ETA_MODELS:
            results[(stage, m)] = forecast(nl, eta_eff[m])
    results[('P7', 'flat')] = forecast(*P7_STAGE)
    q1ip = {}
    for m in ETA_MODELS:
        a_q1 = float(np.sum(eta_of_theta(theta_AB, m)))
        q1ip[m] = (a_q1, N_det * a_q1, np.sqrt(2.0 * N_det * a_q1))
    save = dict(N_grid=N_grid, k_t_grid=k_t_grid, N_det_per_lens=N_det,
                eta_models=np.array(ETA_MODELS), eta_eff=np.array([eta_eff[m] for m in ETA_MODELS]),
                eta0=ETA0, theta_ref=THETA_REF, eta_cap=ETA_CAP,
                stages=np.array(list(STAGES.keys())), stage_nlenses=np.array([STAGES[s] for s in STAGES]))
    for (stage, m), r in results.items():
        for key, val in r.items():
            save[f'{stage}_{m}_{key}'] = val
    for m, (a_q1, n_exp_q1, sig_q1) in q1ip.items():
        save[f'q1ip_{m}_a'] = a_q1; save[f'q1ip_{m}_n_exp'] = n_exp_q1; save[f'q1ip_{m}_sig_null'] = sig_q1
    return save, results, q1ip, N_grid, iN_fid

say("Erratum VI-VIII — Level 2: Paper VIII recalibrated forecast with the Jacobian")
say("=" * 72)

# GATE ────────────────────────────────────────────────────────────────
pub = np.load(PUB_PATH, allow_pickle=True)
old, res_o, q1_o, N_grid, iN_fid = run(np.load(os.path.join(P7, 'euclid_forecast.npz'), allow_pickle=True))
bad = []
for key in pub.files:
    if key not in old:
        bad.append(f"{key}: not recomputed"); continue
    a_, b_ = pub[key], old[key]
    if a_.dtype.kind in 'US':
        ok = np.array_equal(a_, b_)
    else:
        ok = np.allclose(np.asarray(a_, dtype=float), np.asarray(b_, dtype=float), rtol=1e-9, atol=1e-12)
    if not ok:
        bad.append(key)
say(f"\nGATE: published recalibrated_forecast.npz reproduced from the v1 input "
    f"({len(pub.files)} arrays): {'PASS' if not bad else 'FAIL ' + ', '.join(bad)}")
if bad:
    raise SystemExit("gate failed")

# V2 ──────────────────────────────────────────────────────────────────
fc2 = np.load(os.path.join(P7, 'euclid_forecast_v2.npz'), allow_pickle=True)
new, res_n, q1_n, _, _ = run(fc2)
iN2 = int(np.where(N_grid == 2)[0][0]); iN4 = int(np.where(N_grid == 4)[0][0])
say(f"\nInput N_det per lens: " + ", ".join(f"N={int(N)}: {a:.3f} -> {b:.3f}" for N, a, b in zip(N_grid, old['N_det_per_lens'], new['N_det_per_lens'])))
say(f"Effective eta over the Q1 theta_E distribution (unchanged): " + ", ".join(f"{m} {eta_eff[m]:.4f}" for m in ETA_MODELS))
say(f"Stages (grade A/B lenses, unchanged): " + ", ".join(f"{k} {v:.0f}" for k, v in STAGES.items()))

say(f"\nStage table, old -> new:")
say(f"  {'stage':>6s} {'eta':>7s} {'n_exp(N=6)':>18s} {'sig(2v6)':>14s} {'sig(4v6)':>12s} {'null(N=2)':>14s} {'null(N=6)':>14s} {'P(N=6)':>16s}")
for key in list(res_o.keys()):
    ro, rn = res_o[key], res_n[key]
    say(f"  {key[0]:>6s} {key[1]:>7s} {ro['n_exp'][iN_fid]:8.1f} -> {rn['n_exp'][iN_fid]:7.1f} "
        f"{ro['sig_vs_fid'][iN2]:6.1f} -> {rn['sig_vs_fid'][iN2]:5.1f} {ro['sig_vs_fid'][iN4]:5.1f} -> {rn['sig_vs_fid'][iN4]:4.1f} "
        f"{ro['sig_null'][iN2]:6.1f} -> {rn['sig_null'][iN2]:5.1f} {ro['sig_null'][iN_fid]:6.1f} -> {rn['sig_null'][iN_fid]:5.1f} "
        f"{ro['P_N'][iN_fid]:7.3f} -> {rn['P_N'][iN_fid]:6.3f}")

def rng(vals):
    return f"{min(vals):,.0f}-{max(vals):,.0f}"
say(f"\nHeadline brackets over the three eta models:")
say(f"  Wide fiducial expectation n_exp(N=6):   {rng([res_o[('Wide', m)]['n_exp'][iN_fid] for m in ETA_MODELS])} -> "
    f"{rng([res_n[('Wide', m)]['n_exp'][iN_fid] for m in ETA_MODELS])}")
say(f"  Wide all-ISF (N>=2) 95% threshold:     {rng([res_o[('Wide', m)]['thr95_total'][iN2] for m in ETA_MODELS])} -> "
    f"{rng([res_n[('Wide', m)]['thr95_total'][iN2] for m in ETA_MODELS])}")
say(f"  Wide N>=4 95% threshold:               {rng([res_o[('Wide', m)]['thr95_total'][iN4] for m in ETA_MODELS])} -> "
    f"{rng([res_n[('Wide', m)]['thr95_total'][iN4] for m in ETA_MODELS])}")
for stage in ('Q1', 'DR1'):
    say(f"  {stage:>4s} all-ISF (N>=2) 95% threshold:        {rng([res_o[(stage, m)]['thr95_total'][iN2] for m in ETA_MODELS])} -> "
        f"{rng([res_n[(stage, m)]['thr95_total'][iN2] for m in ETA_MODELS])};  N>=4: "
        f"{rng([res_o[(stage, m)]['thr95_total'][iN4] for m in ETA_MODELS])} -> {rng([res_n[(stage, m)]['thr95_total'][iN4] for m in ETA_MODELS])}")
for stage in ('Q1', 'DR1', 'Wide'):
    say(f"  {stage:>4s} null exclusion of N=2 (min over eta): {min(res_o[(stage, m)]['sig_null'][iN2] for m in ETA_MODELS):.1f} -> "
        f"{min(res_n[(stage, m)]['sig_null'][iN2] for m in ETA_MODELS):.1f} sigma;  N=6: "
        f"{min(res_o[(stage, m)]['sig_null'][iN_fid] for m in ETA_MODELS):.1f} -> {min(res_n[(stage, m)]['sig_null'][iN_fid] for m in ETA_MODELS):.1f}")
say(f"  DR1 ISF-true P(N=6) over eta:           " + ", ".join(f"{res_o[('DR1', m)]['P_N'][iN_fid]:.3f} -> {res_n[('DR1', m)]['P_N'][iN_fid]:.3f}" for m in ETA_MODELS))
say(f"  DR1 N=4 vs 6 discrimination over eta:   " + ", ".join(f"{res_o[('DR1', m)]['sig_vs_fid'][iN4]:.1f} -> {res_n[('DR1', m)]['sig_vs_fid'][iN4]:.1f}" for m in ETA_MODELS))

say(f"\nQ1 in-principle bound (312 modelled grade-A/B systems), old -> new:")
for m in ETA_MODELS:
    (a_o, n_o, s_o), (a_n, n_n, s_n) = q1_o[m], q1_n[m]
    say(f"  {m:7s}: eff. sightlines {a_o:6.1f} (unchanged);  n_exp(N=6) {n_o[iN_fid]:5.1f} -> {n_n[iN_fid]:5.1f};  "
        f"null excl N=2 {s_o[iN2]:.1f} -> {s_n[iN2]:.1f}, N=6 {s_o[iN_fid]:.1f} -> {s_n[iN_fid]:.1f}, N=10 {s_o[-1]:.1f} -> {s_n[-1]:.1f} sigma")
say(f"  n_exp(N=6) bracket: {min(q1_o[m][1][iN_fid] for m in ETA_MODELS):.1f}-{max(q1_o[m][1][iN_fid] for m in ETA_MODELS):.1f} -> "
    f"{min(q1_n[m][1][iN_fid] for m in ETA_MODELS):.1f}-{max(q1_n[m][1][iN_fid] for m in ETA_MODELS):.1f};  "
    f"N=6 null sigma: {min(q1_o[m][2][iN_fid] for m in ETA_MODELS):.1f}-{max(q1_o[m][2][iN_fid] for m in ETA_MODELS):.1f} -> "
    f"{min(q1_n[m][2][iN_fid] for m in ETA_MODELS):.1f}-{max(q1_n[m][2][iN_fid] for m in ETA_MODELS):.1f}")
say(f"  Paper X consistency: Paper X's corrected N=6 count 0.756 per lens vs this 0.771.")

# save ────────────────────────────────────────────────────────────────
new['jacobian'] = 1.0 / 1.4
if os.path.exists(OUTPATH):
    # never rewrite a committed product: confirm it is what this run produced
    prev = np.load(OUTPATH, allow_pickle=True)
    same = set(prev.files) == set(new.keys()) and all(
        np.array_equal(prev[k], new[k]) if np.asarray(prev[k]).dtype.kind in 'US'
        else np.allclose(np.asarray(prev[k], dtype=float), np.asarray(new[k], dtype=float), rtol=1e-12, atol=0)
        for k in prev.files)
    say(f"\n{OUTPATH} exists and {'matches this run (not rewritten)' if same else 'DIFFERS from this run: archive it and rerun'}")
    if not same:
        raise SystemExit(1)
else:
    np.savez(OUTPATH, **new)
os.makedirs(DATA, exist_ok=True)
with open(os.path.join(DATA, 'paper8_forecast_v2.txt'), 'w') as f:
    f.write("\n".join(_lines) + "\n")
say(f"\nSaved {OUTPATH} and {os.path.join(DATA, 'paper8_forecast_v2.txt')}")
