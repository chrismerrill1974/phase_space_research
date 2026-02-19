# Installation & Setup

## Prerequisites

- Python 3.9 or later
- pip (Python package manager)

## Python Dependencies

Install all required packages:

```bash
pip install numpy scipy matplotlib pandas astropy requests scikit-learn
```

For Paper V and Paper VII scripts that generate LCDM baseline power spectra, you also need CAMB:

```bash
pip install camb
```

### Full dependency list

| Package | Version | Used by |
|---------|---------|---------|
| `numpy` | >= 1.21 | All papers |
| `scipy` | >= 1.7 | Papers II-VII (optimization, integration, spatial analysis) |
| `matplotlib` | >= 3.5 | All papers (figures) |
| `pandas` | >= 1.3 | Papers III-IV (dataframe operations) |
| `astropy` | >= 5.0 | Papers III-IV (FITS I/O, coordinates) |
| `requests` | >= 2.25 | Paper III (SDSS SkyServer queries) |
| `scikit-learn` | >= 1.0 | Paper IV (Random Forest regression) |
| `camb` | >= 1.4 | Papers V, VII (LCDM power spectra) |

## Quick Start

### Papers II-III: SPARC Rotation Curve Analysis

The SPARC data is included in `data/sparc/`. To run the radial profile analysis:

```bash
cd paper2_radial_profiles
python step1_radial.py          # Compute f_DM(R) profiles
python step3_radial_eda.py      # Exploratory data analysis
python step4_null_models.py     # Fit null models
python step5_tests.py           # Run statistical tests
```

Scripts expect the SPARC data at `../data/sparc/`. Check each script header for exact input/output paths and adjust as needed.

### Paper III: ISF Spatial Analysis

```bash
cd paper3_isf_spatial
python phaseA_spatial.py        # 3D spatial autocorrelation
python phaseA_confounds.py      # Confound checks
python step6_robustness.py      # Robustness analysis
python make_paper_figs.py       # Generate publication figures
python make_isf_schematic.py    # ISF conceptual schematic
```

### Paper IV: BOSS Galaxy Analysis

Before running Paper IV scripts, you need SDSS/BOSS data:

1. **Download BOSS LOWZ catalogs** from [SDSS DR12](https://data.sdss.org/sas/dr12/boss/lss/):
   - `galaxy_DR12v5_LOWZ_North.fits.gz`
   - `galaxy_DR12v5_LOWZ_South.fits.gz`

2. **Download velocity dispersions** (uses the SDSS SkyServer SQL endpoint):
   ```bash
   cd paper3_isf_spatial
   python download_sdss_vdisp.py
   ```
   This queries SDSS DR12 in batches with retry logic. The download may take several hours for the full LOWZ sample. The script supports checkpoint/resume if interrupted.

3. **Run the analysis pipeline:**
   ```bash
   cd paper4_boss_galaxies
   python phaseB_step1_fdm_proxy2.py        # Compute f_DM proxies
   python phaseB_step2_residuals.py          # Regression & residuals
   python phaseB_step3b_zslice_control.py    # Autocorrelation in z-slices
   python phaseB_step4_clustering_control.py # Clustering control
   python phaseB_step5_cosmic_web.py         # Cosmic web classification
   python phaseB_step6_signal_diagnostics.py # Signal diagnostics
   ```

### Paper V: Weak Lensing Constraints

```bash
cd paper5_weak_lensing
python generate_camb_baseline.py    # Generate LCDM P(k) (requires CAMB)
python isf_theory.py                # (module, imported by other scripts)
python compute_isf_grid_v2.py       # Compute ISF model grid
python compile_published_data.py    # Compile published measurements
python make_fig1_isf_predictions.py # Figure 1
python make_fig2_s8_constrained.py  # Figure 2
python make_fig3_data_comparison.py # Figure 3
python make_fig4_comprehensive.py   # Figure 4
```

### Paper VI: Strong Lensing Predictions

```bash
cd paper6_strong_lensing
python paper6_phase2_isf_predictions.py  # Core ISF predictions
python paper6_phase2c_figures.py         # Figures 1-2
python paper6_phase2d_profiles.py        # Figure 3
python paper6_phase3b_figure4.py         # Figure 4 (mass function)
python paper6_phase3c_meneghetti.py      # Figure 5 (Meneghetti)
python paper6_phase3d_los_profile.py     # Figure 6 (LOS profiles)
python paper6_phase3e_synthesis.py       # Figure 7 (synthesis)
```

### Paper VII: Joint Constraints & Euclid Forecast

Paper VII has the most involved pipeline. Precomputed intermediate data products are provided in `data/paper7/` so you can run later steps without recomputing everything.

**Phase 1 -- Power spectrum pipeline** (requires CAMB):
```bash
cd paper7_joint_constraints/phase1
python step1a_generate_pknl.py       # LCDM P_NL(k,z)
python step1b_isf_transfer.py        # ISF transfer function
python step1c_limber_integral.py     # Limber integral -> C_kappa(ell)
python step1d_bessel_transform_v2.py # Bessel transform -> xi+(theta)
python step1e_response_operator.py   # Response operator matrix
```

**Phase 2 -- KiDS-1000 likelihood:**
```bash
cd paper7_joint_constraints/phase2
python step2a_explore_kids_data.py       # Explore data structure
python step2a_extract_kids_data.py       # Extract band powers
python step2b_response_operators_tomo.py # Tomographic operators
python step2c_wl_likelihood.py           # Weak lensing chi^2
```

**Phase 2 systematics** (optional, for reproducing validation checks):
```bash
cd paper7_joint_constraints/phase2_systematics
python step_A1_audit.py                    # Unit audit
python step_B1_ell_calibration.py          # ell calibration
python step_B2_diagonal_cov.py             # Diagonal covariance
python step_B3_per_pair.py                 # Per-pair fits
python step_C1_limber_projection.py        # Limber projection
python step_C2_paper_vi_consistency.py     # Paper VI consistency
python step_C3_amplitude_decomposition.py  # Amplitude decomposition
python step_D2_propagation.py              # Error propagation
```

**Phases 3-5 -- Joint constraints & forecast:**
```bash
cd paper7_joint_constraints/phase3
python step3a_substructure_likelihood.py   # Strong lensing likelihood

cd ../phase4
python step4_joint_constraint.py           # Joint WL + SL constraint

cd ../phase5
python step5a_euclid_forecast.py           # Euclid Fisher forecast
python step5b_forecast_figure.py           # Forecast figure
```

## Data Downloads

### SPARC Data (included)
Already included in `data/sparc/`. Original source: http://astroweb.cwru.edu/SPARC/

### SDSS DR12 BOSS LOWZ Catalogs
Required for Papers III-IV. Download from:
- https://data.sdss.org/sas/dr12/boss/lss/

Files needed:
- `galaxy_DR12v5_LOWZ_North.fits.gz` (~180 MB)
- `galaxy_DR12v5_LOWZ_South.fits.gz` (~75 MB)

### SDSS Velocity Dispersions
Use the included `download_sdss_vdisp.py` script which queries the SDSS SkyServer:
- Endpoint: `https://skyserver.sdss.org/dr12/en/tools/search/x_sql.aspx`
- Uses PLATE/FIBERID/MJD as keys
- Batched queries with retry logic (may take several hours)

### KiDS-1000 Cosmic Shear Data
Preprocessed data is included in `data/paper7/`. The original KiDS-1000 release is at:
- https://kids.strw.leidenuniv.nl/DR4/lensing.php

The band power measurements, redshift distributions, and covariance matrices are from Asgari et al. (2021).

## File Path Configuration

Most scripts use relative paths to locate data files. If you encounter path issues, check the top of each script for path configuration variables and adjust them to match your directory layout.

## Recreating Figures

Each paper directory contains figure-generating scripts. Run the analysis scripts first (they produce intermediate CSV/NPZ files), then run the `make_fig*.py` or `*_figures.py` scripts to produce publication-quality PDF figures.

Figures are saved to the current working directory or a `figures/` subdirectory. Check each script's header docstring for exact output filenames.
