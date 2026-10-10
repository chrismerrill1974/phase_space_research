# Phase Space Dimensionality Research

Analysis scripts and data for a series of papers testing the phase space dimensionality framework for dark matter, specifically the Independent Structure Formation (ISF) hypothesis. The ISF hypothesis proposes that dark matter consists of ordinary baryonic matter on gravitationally coupled but electromagnetically isolated dimensions (psi-branches), each hosting independent cosmic structure.

**Author:** Christopher K. Merrill

---

## Papers

### Paper I: Phase Space Dimensionality: A Unified Framework for Dark Matter, Consciousness, and Quantum Substrate Requirements
- **Zenodo:** https://zenodo.org/records/18578012
- **Summary:** Exploratory brainstorming session proposing speculative connections between dark matter as ordinary matter in adjacent phase space branches, consciousness as a quantum receiver across phase space dimensions, time as a partially opened dimension, and variable physical constants across the multiverse. Builds on earlier work in Radial Dimensionality Theory (RDT). No scripts in this repository; published to document the thinking process and invite collaboration. (No analysis code -- foundational ideas only.)


### Paper II: Testing Radial Dark Matter Fraction Profiles for Quantized Staircase Structure in the Phase Space Dimensionality Framework
- **Zenodo:** https://zenodo.org/records/18604435
- **Summary:** Tests whether radial dark matter fraction profiles f_DM(R) in 76 SPARC galaxies exhibit a "staircase morphology" (discrete plateaus connected by sharp transitions) as predicted by the RDT framework. Five independent statistical tests on 1,830 radial data points find no staircase signature, constraining step heights to Delta f_DM < 0.02.

### Paper III: Dark Matter as Superimposed Cosmic Webs: A Methodology for Testing Independent Structure Formation Across psi-Branches
- **Zenodo:** https://zenodo.org/records/18612335
- **Summary:** Develops the ISF hypothesis and tests it using 84 SPARC galaxies. Detects apparent spatial autocorrelation (Moran's I = 0.67) but finds it entirely attributable to galaxy group membership after deconfounding. Proposes three follow-up tests: spectroscopic survey residual autocorrelation, weak lensing power spectrum suppression, and strong lensing dark substructure.

### Paper IV: Testing Independent Structure Formation Across psi-Branches with 258,000 BOSS Galaxies
- **Zenodo:** https://zenodo.org/records/18627110
- **Summary:** Scales the ISF spatial autocorrelation test to 258,153 BOSS DR12 LOWZ galaxies. Random forest regression explains 91.3% of f_DM variance. Three diagnostic tests identify the residual signal as redshift systematic rather than genuine ISF structure, yielding a clean null result for ISF spatial autocorrelation.

### Paper V: Weak Lensing Constraints on Cross-Branch Correlations in the Independent Structure Formation Framework
- **Zenodo:** https://zenodo.org/records/18634086
- **Summary:** Tests ISF against published weak lensing measurements. The ISF power spectrum modification factor [1 + (N-1)*epsilon(k)]/N is constrained using the S8 tension. ISF models with scale-dependent epsilon(k) are preferred over constant suppression at 1.9 sigma, with best-fit parameters k_t = 0.20 h/Mpc and epsilon_inf = 0.58.

### Paper VI: Galaxy-Mass Dark Substructure from Strong Lensing: A Parameter-Free Test of Independent Structure Formation
- **Zenodo:** https://zenodo.org/records/18637044
- **Summary:** Derives parameter-free predictions for ISF dark perturbers detectable via strong gravitational lensing. Predicts ~1.08 detectable dark perturbers per lens sightline (10^10-10^12 solar masses) with a Schechter-shaped mass function. Provides partial explanation (63%) for the Meneghetti galaxy-galaxy strong lensing anomaly. Forecasts ~4,000 detectable ISF perturbers with Euclid.

### Paper VII: Joint Weak Lensing and Strong Lensing Substructure Constraints on ISF Cross-Branch Correlation Parameters -- A Euclid Forecast for Decisive Falsification
- **Zenodo:** https://zenodo.org/records/18644812
- **Summary:** Presents the first joint constraint from weak lensing (KiDS-1000 band powers) and strong lensing substructure. Joint best-fit: N = 10, k_t = 1.32 h/Mpc apparent (0.52 h/Mpc physical after Limber correction), with Delta chi^2 = -53.6 vs LCDM (~7.3 sigma equivalent). Euclid forecast shows 33 sigma discrimination between N=2 and N=6; if no ISF substructure is detected, all ISF models are excluded at > 62 sigma.

### Paper VIII: Anchoring the Falsification Forecast to Measured Euclid Q1 Strong-Lens Demographics
- **Zenodo:** https://zenodo.org/records/21195841
- **Summary:** Replaces the three assumed survey inputs of Paper VII's Euclid forecast (75,000 grade-A/B lenses, flat eta = 0.05) with quantities measured from the public Q1 Strong Lensing Discovery Engine catalogue (497 grade-A/B candidates at 7.88/deg^2; Einstein-radius median 0.88"). The recalibrated forecast is robust and slightly strengthened: 4,463-5,955 expected fiducial detections (vs 4,050 assumed), with the all-ISF 95% falsification threshold rising from 1,844 to 2,037-2,733 detections. Staging over real release dates shows DR1 falsifies (a null search excludes all ISF at >= 24 sigma) but cannot confirm (an ISF-true count leaves the posterior at N=10; only the full wide survey collapses it to N=6). A perturber search of the 312 already-modelled Q1 systems would exclude N >= 6 at >= 5 sigma if null.

### Paper IX: Black Holes as Branch-Blind Objects -- A Gravitational-Capture Origin for the ISF Cross-Branch Correlation
- **Zenodo:** https://zenodo.org/records/21209225
- **Summary:** Theory paper proposing a physical origin for the fitted correlation function epsilon(k): branches co-locate by falling into shared gravitational wells, with black holes as the unique branch-blind objects (the no-hair theorem erases every branch-distinguishing attribute). The capture mechanism reproduces the sigmoid shape, recovers the fitted transition scale to a factor of 2.1 with no free parameters (k_NL(0) = 0.25 vs fitted k_t = 0.52 h/Mpc), and predicts eps_inf(z=1)/eps_inf(0) = 0.45-0.70 -- testable with Euclid DR1 tomography from 2027. Black-hole growth accounting bounds the cross-branch capture efficiency to gamma < 0.19 at N=6 (Soltan argument) while the maximal early excess reaches the low edge of the JWST z>6 overmassive anomaly at N=10. Revises the redshift-independence discriminant of Papers V-VIII.

### Paper X: The Galaxy-Scale Dark Mass Profile Under Independent Structure Formation -- A Dated Prediction for the Milky Way's Outer Rotation Curve, and a Corrected Threshold for the Euclid Perturber Search
- **Zenodo:** (pending)
- **Summary:** Theory paper deriving the dark mass profile around a single galaxy from the Paper IX capture ontology, motivated by the Gaia DR3 debate over a declining Milky Way rotation curve. Shows that other-branch diffuse gas cannot be the halo (an isothermal atmosphere holds the halo mass only above the virial temperature, where it is too diffuse: 177-179 km/s at the solar radius against 230 measured), so captured other-branch collapsed matter carries it. With M_cap = (N-1) eps_inf(N) M_bar from the series' own numbers, N = 2 is excluded by the inner rotation curve, N = 6 reaches the solar-radius speed with a 20 kpc extent and nothing tuned, and the outer slope follows from N alone: beta(15-30 kpc) = -0.42, -0.30, -0.22, -0.17 for N = 4, 6, 8, 10, with a dip near 100 kpc that external galaxies test. That prediction is already excluded by the external population: run galaxy by galaxy for 20 SPARC spirals with Milky-Way rotation speeds, no galaxy falls at or below its own prediction at N <= 8 (sign-test P < 0.001) and four of twenty do at N = 10 (P = 0.006), a published null for the registered model. The family with other branches holding two to three times our collapsed mass per well survives: (N, m) = (10, 2) and (8, 3) fit the population while the Milky Way's streams prefer (8, 2), so the family spans one step in N, and the Milky Way's outer slope (-0.13 to -0.05 across it) discriminates within it for Gaia DR4 (2 December 2026). For the Euclid Q1 in-principle test the family gives 6.5-8.9 expected detections with the perturber extent held and 15-22 scaled (3.6-6.6 sigma null test).

---

## Repository Structure

```
phase_space_research/
|-- data/
|   |-- sparc/                    # SPARC galaxy rotation curve data (Papers II-III)
|   |   |-- SPARC_Lelli2016c.mrt  # Galaxy catalog
|   |   |-- MassModels_Lelli2016c.mrt  # Mass decomposition models
|   |-- paper7/                   # Precomputed data products (Paper VII)
|       |-- kids1000_bandpowers.npz    # KiDS-1000 tomographic band powers
|       |-- response_operators_tomo.npz # Limber response operators
|       |-- chi2_wl_grid_v2.npz        # Weak lensing chi^2 grid
|       |-- lnL_sub_grid.npz           # Substructure likelihood grid
|       |-- joint_constraint.npz       # Joint constraint results
|       |-- euclid_forecast.npz        # Euclid forecast predictions
|   |-- paper8/                   # Precomputed data products (Paper VIII)
|       |-- q1_demographics.npz        # Q1 SLDE demographics (rates, theta_E)
|       |-- recalibrated_forecast.npz  # Recalibrated forecast grids
|   |-- paper9/                   # Data products (Paper IX)
|       |-- pknl_camb.npz              # Linear P(k,z) input (from Paper VII)
|       |-- eps_evolution.npz          # eps_inf(z), k_NL(z) predictions
|       |-- bh_accounting.npz          # Soltan gamma(N) bounds
|   |-- paper10/                  # Data products (Paper X)
|       |-- hydrostatic_halo.npz        # Step 1: other-branch gas atmosphere grid
|       |-- dissipated_component.npz   # Step 2: captured-component budget and chain
|       |-- combined_halo.npz          # Step 2b: captured + hot remainder to R_200
|       |-- combined_halo_m2.npz       # Step 2b at m = 2 (twice our collapsed mass per branch)
|       |-- perturber_profile.npz      # Step 3: perturber threshold, Paper XI numbers
|       |-- sparc_mw_mass_slopes.npz   # Step 5: SPARC Milky-Way-mass outer slopes
|       |-- sparc_per_galaxy_chain.npz  # Step 5b: the chain per SPARC galaxy (main test)
|       |-- perturber_profile_m2.npz, perturber_profile_m3.npz  # Step 3 at the family's extents
|       |-- joint_constraint_paper7.npz, euclid_forecast_paper7.npz,
|       |   q1_demographics_paper8.npz, paper6_phase2_results.json  # inputs (series products)
|
|-- paper2_radial_profiles/       # Paper II scripts
|   |-- step1_radial.py           # Compute f_DM(R) for all SPARC galaxies
|   |-- step3_radial_eda.py       # Exploratory data analysis of profiles
|   |-- step4_null_models.py      # Fit NFW, pseudo-isothermal, LOWESS null models
|   |-- step5_tests.py            # Five statistical tests for staircase structure
|
|-- paper3_isf_spatial/           # Paper III scripts
|   |-- phaseA_spatial.py         # 3D spatial autocorrelation analysis
|   |-- phaseA_confounds.py       # Confound checks (galaxy groups, systematics)
|   |-- step6_robustness.py       # Robustness checks across analysis choices
|   |-- make_paper_figs.py        # Publication figures for the radial paper
|   |-- make_isf_schematic.py     # Conceptual schematic of superimposed cosmic webs
|   |-- download_sdss_vdisp.py    # Download SDSS DR12 velocity dispersions
|
|-- paper4_boss_galaxies/         # Paper IV scripts
|   |-- phaseB_step1_fdm_proxy2.py        # f_DM proxy from velocity dispersion + color
|   |-- phaseB_step2_residuals.py          # Regress f_DM against galaxy properties
|   |-- phaseB_step3b_zslice_control.py    # Spatial autocorrelation in redshift slices
|   |-- phaseB_step4_clustering_control.py # Galaxy clustering control (Landy-Szalay)
|   |-- phaseB_step5_cosmic_web.py         # Cosmic web classification control
|   |-- phaseB_step6_signal_diagnostics.py # Signal origin diagnostics (anisotropy, plates)
|
|-- paper5_weak_lensing/          # Paper V scripts
|   |-- isf_theory.py             # ISF power spectrum theory module
|   |-- generate_camb_baseline.py # LCDM P(k) baseline from CAMB (Planck 2018)
|   |-- compute_isf_grid_v2.py    # ISF model grid with S8 constraint
|   |-- compile_published_data.py # Compile published lensing measurements
|   |-- make_fig1_isf_predictions.py   # Figure 1: ISF modification factors
|   |-- make_fig2_s8_constrained.py    # Figure 2: S8-constrained predictions
|   |-- make_fig3_data_comparison.py   # Figure 3: ISF vs published data
|   |-- make_fig4_comprehensive.py     # Figure 4: Comprehensive comparison
|   |-- make_fig_paper_draft.py        # Draft 2-panel summary figure
|
|-- paper6_strong_lensing/        # Paper VI scripts
|   |-- paper6_phase2_isf_predictions.py  # ISF subhalo mass function & abundance
|   |-- paper6_phase2c_figures.py         # Figures 1-2: epsilon(k) and mass function
|   |-- paper6_phase2d_profiles.py        # Figure 3: density profiles & convergence
|   |-- paper6_phase3b_figure4.py         # Figure 4: mass function comparison
|   |-- paper6_phase3c_meneghetti.py      # Figure 5: Meneghetti GGSL analysis
|   |-- paper6_phase3d_los_profile.py     # Figure 6: secondary discriminants
|   |-- paper6_phase3e_synthesis.py       # Figure 7: synthesis scorecard
|
|-- paper7_joint_constraints/     # Paper VII scripts
|   |-- phase1/                   # Power spectrum pipeline
|   |   |-- step1a_generate_pknl.py       # LCDM nonlinear P(k,z) from CAMB
|   |   |-- step1b_isf_transfer.py        # Apply ISF transfer function
|   |   |-- step1c_limber_integral.py     # Limber integral for C_kappa(ell)
|   |   |-- step1d_bessel_transform_v2.py # Bessel transform to real-space
|   |   |-- step1e_response_operator.py   # Precompute response operator matrix
|   |   |-- step1[a-e]_plot_diagnostic.py # Diagnostic plots for each step
|   |-- phase2/                   # KiDS-1000 data & weak lensing likelihood
|   |   |-- step2a_explore_kids_data.py   # Explore KiDS-1000 data structure
|   |   |-- step2a_extract_kids_data.py   # Extract band powers & covariance
|   |   |-- step2b_response_operators_tomo.py  # Tomographic response operators
|   |   |-- step2c_wl_likelihood.py       # Weak lensing chi^2 on ISF grid
|   |-- phase2_systematics/       # Systematic checks
|   |   |-- step_A1_audit.py              # Unit audit
|   |   |-- step_B1_ell_calibration.py    # Angular scale calibration
|   |   |-- step_B2_diagonal_cov.py       # Diagonal covariance test
|   |   |-- step_B3_per_pair.py           # Per-pair bin fits
|   |   |-- step_C1_limber_projection.py  # Limber projection effect
|   |   |-- step_C2_paper_vi_consistency.py  # Paper VI consistency check
|   |   |-- step_C3_amplitude_decomposition.py  # Profiled amplitude decomposition
|   |   |-- step_D2_propagation.py        # Error propagation
|   |-- phase3/
|   |   |-- step3a_substructure_likelihood.py  # Strong lensing likelihood
|   |-- phase4/
|   |   |-- step4_joint_constraint.py     # Joint WL + SL constraint
|   |-- phase5/
|       |-- step5a_euclid_forecast.py     # Euclid Fisher forecast
|       |-- step5b_forecast_figure.py     # Forecast visualization
|
|-- paper8_euclid_q1/             # Paper VIII scripts
|   |-- step1_q1_demographics.py  # Q1 SLDE catalogue demographics
|   |-- step2_recalibrated_forecast.py  # Forecast with measured inputs
|   |-- step3_figures.py          # Paper figures
|
|-- paper9_blackholes/            # Paper IX scripts (theory)
|   |-- step1_eps_evolution.py    # Capture mechanism: eps_inf(z), k_t(z)
|   |-- step2_bh_accounting.py    # Soltan gamma(N) bound, JWST headroom
|   |-- step3_figures.py          # Paper figures
|
|-- paper10_galaxy_profile/       # Paper X scripts (theory)
|   |-- step1_hydrostatic_halo.py        # Other-branch gas atmosphere (excluded)
|   |-- step2_dissipated_component.py    # Captured component: budget, N -> slope
|   |-- step2b_combine_hot_remainder.py  # Hot remainder added, curve to R_200
|   |-- step3_perturber_profile.py       # Perturber profile, Paper XI threshold
|   |-- step5_sparc_mw_mass_slopes.py    # SPARC outer slopes, Milky-Way-mass spirals
|   |-- step5b_sparc_per_galaxy_chain.py # The chain per SPARC galaxy (main test)
|   |-- step4_figures.py                 # Paper figures
```

---

## Data Sources

### SPARC (Papers II-III)
The SPARC (Spitzer Photometry and Accurate Rotation Curves) dataset is included in `data/sparc/`. Original source: [Lelli, McGaugh & Schombert (2016)](http://astroweb.cwru.edu/SPARC/).

### SDSS/BOSS DR12 (Papers III-IV)
Velocity dispersions for BOSS LOWZ galaxies are downloaded from SDSS DR12 using `paper3_isf_spatial/download_sdss_vdisp.py`. The script queries the SDSS SkyServer SQL endpoint. You will also need the BOSS LOWZ FITS catalogs:
- `galaxy_DR12v5_LOWZ_North.fits.gz`
- `galaxy_DR12v5_LOWZ_South.fits.gz`

These can be obtained from the [SDSS DR12 BOSS LSS catalogs](https://data.sdss.org/sas/dr12/boss/lss/).

### KiDS-1000 (Paper VII)
The KiDS-1000 cosmic shear band power data used in Paper VII is included as preprocessed `.npz` files in `data/paper7/`. The original KiDS-1000 data release is available from [kids.strw.leidenuniv.nl](https://kids.strw.leidenuniv.nl/DR4/lensing.php).

### Euclid Q1 Strong Lensing Discovery Engine (Paper VIII)
The Q1 SLDE lens catalogue and automated mass models (Walmsley et al. 2025, arXiv:2503.15324) are not redistributed here; download `q1_discovery_engine_lens_catalog.csv` and `modeling_lens_mass.csv` from the Euclid Collaboration's public record at [Zenodo 10.5281/zenodo.15003116](https://doi.org/10.5281/zenodo.15003116). Derived demographics are included as `.npz` files in `data/paper8/`. See `paper8_euclid_q1/README.md` for the expected directory layout.

### Published Lensing Measurements (Paper V)
Compiled from Amon+2023, Leauthaud+2017, Efstathiou+2022, and major weak lensing survey S8 results. See `paper5_weak_lensing/compile_published_data.py` for details and references.

---

## License

This project is released under the MIT License. See [LICENSE.md](LICENSE.md) for details.

### Milky Way baryonic model, stream masses, and SPARC (Paper X)
Paper X's step 5 reads the SPARC tables in `data/sparc/` (Papers II-III). Paper X uses the B2 Milky Way baryonic model of de Salas et al. (2019, JCAP 10, 037) as adopted by Jiao et al. (2026, arXiv:2609.30364), with parameters written into the scripts; no data files are needed. The solar-radius circular speed, the Orphan-Chenab (Koposov et al. 2023) and Sagittarius (Vasiliev et al. 2021) stream masses, and the Gaia-era outer slopes are literature values recorded in the paper as data to be tested.
