"""
generate_camb_baseline.py — ΛCDM P(k) from CAMB, Planck 2018 cosmology.
NOTE: CAMB returns z in ascending order regardless of input.
"""
import numpy as np
import camb
import csv

COSMO = dict(H0=67.36, ombh2=0.02237, omch2=0.12, ns=0.9649,
             As=2.1e-9, tau=0.0544, mnu=0.06)

h = COSMO['H0'] / 100.0
Omega_b = COSMO['ombh2'] / h**2
Omega_c = COSMO['omch2'] / h**2
Omega_m = Omega_b + Omega_c

print(f"Planck 2018: h={h:.4f}, Ω_b={Omega_b:.4f}, Ω_c={Omega_c:.4f}, Ω_m={Omega_m:.4f}")
print(f"  Ω_DM/Ω_m = {Omega_c/Omega_m:.4f},  f_b = {Omega_b/Omega_m:.4f}")

pars = camb.CAMBparams()
pars.set_cosmology(H0=COSMO['H0'], ombh2=COSMO['ombh2'],
                   omch2=COSMO['omch2'], mnu=COSMO['mnu'], tau=COSMO['tau'])
pars.InitPower.set_params(As=COSMO['As'], ns=COSMO['ns'])
pars.set_matter_power(redshifts=[0.0, 0.3, 0.5], kmax=10.0)

# Linear
pars.NonLinear = camb.model.NonLinear_none
res_lin = camb.get_results(pars)
kh, z_arr, pk_lin = res_lin.get_matter_power_spectrum(minkh=1e-4, maxkh=10.0, npoints=500)
sigma8 = res_lin.get_sigma8_0()

# Non-linear
pars.NonLinear = camb.model.NonLinear_both
res_nl = camb.get_results(pars)
_, _, pk_nl = res_nl.get_matter_power_spectrum(minkh=1e-4, maxkh=10.0, npoints=500)

print(f"\nσ₈(z=0) = {sigma8:.4f}")
print(f"CAMB z-order: {z_arr}")
idx01 = np.argmin(np.abs(kh - 0.1))
for iz in range(3):
    print(f"  z={z_arr[iz]:.1f}: P_lin(0.1)={pk_lin[iz,idx01]:.0f}, P_nl(0.1)={pk_nl[iz,idx01]:.0f}")

# Save CSV
outfile = '/home/claude/paper5_wl/camb_pk_planck2018.csv'
header = ['k_hMpc','Pk_lin_z0.0','Pk_nl_z0.0','Pk_lin_z0.3','Pk_nl_z0.3','Pk_lin_z0.5','Pk_nl_z0.5']
with open(outfile, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(header)
    for i in range(len(kh)):
        w.writerow([f"{kh[i]:.6e}",
                     f"{pk_lin[0,i]:.6e}", f"{pk_nl[0,i]:.6e}",
                     f"{pk_lin[1,i]:.6e}", f"{pk_nl[1,i]:.6e}",
                     f"{pk_lin[2,i]:.6e}", f"{pk_nl[2,i]:.6e}"])
print(f"\nSaved {outfile} ({len(kh)} rows)")

# Save npz
np.savez('/home/claude/paper5_wl/camb_pk_planck2018.npz',
         k=kh, Pk_lin_z0=pk_lin[0], Pk_nl_z0=pk_nl[0],
         Pk_lin_z03=pk_lin[1], Pk_nl_z03=pk_nl[1],
         Pk_lin_z05=pk_lin[2], Pk_nl_z05=pk_nl[2],
         sigma8=np.array([sigma8]), h=np.array([h]),
         Omega_b=np.array([Omega_b]), Omega_c=np.array([Omega_c]),
         Omega_m=np.array([Omega_m]))
print("Saved .npz")
