# solopy

Data-reduction pipeline for **SOLO**, the Solar system Object Light-curve Observatory.

## Introduction

SOLO is a RASA 11" (f/2.2) astrograph with an FLI KL4040 camera at Sierra Remote Observatories, California (MPC code
G80). The camera has a GSENSE4040 CMOS sensor of 4096 × 4096 pixels, giving 2.97″ per pixel and a 3.4° × 3.4° field.

Each night SOLO takes repeated, unfiltered exposures (mostly 60 s) of fields near the ecliptic, in the evening
(`dusk…`) and morning (`dawn…`) skies. The goal is light curves of bright asteroids (V ≤ 16.5).

solopy turns one night of raw frames into asteroid photometry calibrated to Gaia G:

| Level | Class | Step |
|---|---|---|
| Lv0 | `FitsLv0` | Decompress `.fits.bz2`; normalize headers (mid-exposure UTC, site, detector). |
| masters | `CombMaster` | Combine master bias and dark frames (the sky flat is made offline). |
| Lv1 | `FitsLv1` | Plate solution with astrometry.net; bias, dark and flat correction; bit mask (bad, saturated, border, …). |
| Lv2 | `soloPSF`, `FitsLv2` | PSF FWHM per 500 px tile; aperture photometry of isolated Gaia stars; zero point with color term. |
| Lv3 | `FitsLv3` | Asteroid positions from kete/skyloc; centroiding and aperture photometry; local zero point; star contamination. |
| light curves | `solopy.lightcurve` | Quality flags, light-time correction, 5 min bins with outliers clipped. |

## Installation

solopy needs Python ≥ 3.10. The production environment is the conda env `solopy`:

```bash
conda create -n solopy python=3.11
conda activate solopy
git clone https://github.com/bumhooLIM/solopy.git
cd solopy
pip install -e ".[lv3]"
```

- The `lv3` extra installs kete (≥ 1.0.8, < 2) and skyloc (from GitHub), which are needed only for asteroid
  photometry. `pip install -e .` is enough for Lv0–Lv2; `import solopy` works either way.
- astrometry.net index files (series 4100, scales 8–10, about 170 MB) download into the cache folder on the first
  plate solution.

To check the installation, run the unit tests. They use synthetic data only and need no network:

```bash
python -m unittest discover -s tests
```

## Usage

### Configure the paths

The nightly driver `notebooks/main.py` reads its paths from a `directory.py` in the folder it runs from. In
production that folder is `~/Desktop/data/solo/notebooks/`.

| Entry in `directory.py` | Content |
|---|---|
| `LV0_DIR` | Raw frames, one folder per night (`lv0/2026_0630/`). |
| `MASTER_DIR` | Master bias and dark (written by Lv0), the master flat `kl4040.flat.clear.comb.20260526.fits` and the bad-pixel mask `kl4040.bpm.20260616.fits`. |
| `GAIA_DIR` | `gaiadr3.npy`: Gaia DR3 down to G = 18.5. |
| `SLOC_DIR` | `orb_sbdb.parq`: JPL SBDB orbital elements, used by skyloc. |
| `LV1_DIR`, `PSF_DIR`, `ZP_DIR`, `RESULT_DIR`, `LOG_DIR` | Output folders. |
| `WORK_DIR` | Base folder. The astrometry.net cache defaults to `WORK_DIR/astrometry_cache` (override with `ASTROMETRY_CACHE_DIR`). |

### Run one night

```bash
cd ~/Desktop/data/solo/notebooks
python main.py -s 2026_0630
```

To redo only the zero points and the asteroid photometry:

```bash
python main.py -s 2026_0630 --levels 2,3
```

| Option | Default | Meaning |
|---|---|---|
| `-s`, `--subdir` | required | Night folder, `YYYY_MMDD`. |
| `--levels` | `0,1,2,3` | 0 = decompress, headers, master bias/dark; 1 = WCS and bias/dark/flat; 2 = PSF and zero point; 3 = asteroid photometry. |
| `--badpix-frac-max` | `0.05` | Flag a measurement (`badphot`) when masked pixels cover more than this fraction of the aperture. Saturated pixels always set the flag. |
| `--rebuild-gaia` | off | Rebuild the nightly Gaia subset. |
| `-d`, `--detector` | `kl4040` | Detector name used in master file names. |

Re-running level 1 replaces the night's earlier Lv1 files, including files renamed because the plate solution
moved the field center, together with their PSF and zero-point tables.

Before calibration, the driver builds a Gaia subset for the night (`GAIA_DIR/nightly/gaiadr3.<night>.npy`). It
covers 4.5° around every pointing (about 1 M stars) and is used instead of the 247 M-star catalog.

### Run all nights

`notebooks/run_solopy.sh` runs the nights listed in the script, one after another:

```bash
./run_solopy.sh
```

To run only some levels:

```bash
LEVELS=2,3 ./run_solopy.sh
```

The script uses the Python of the `solopy` env (set `SOLOPY_PYTHON` to use another one). It saves each night's
terminal output to `../log/run_<night>.out`.

### Outputs

| Product | Location |
|---|---|
| Lv1 frames: `SCI` and `MASK` HDUs; WCS, `PSF_FWHM`, `ZP_G` and `ZP_SUN` in the header | `LV1_DIR/<night>/kl4040.sci.lv1.*.fits` |
| PSF FWHM per tile | `PSF_DIR/<night>/psf.<frame>.csv` |
| Zero-point stars | `ZP_DIR/<night>/zp.<frame>.parquet` |
| Asteroid photometry: `gmag`, `mag_err_tot`, `badphot`, `contam_frac`, … | `RESULT_DIR/solo.summary.<YYYYMMDD>.csv` |
| Nightly Gaia subset | `GAIA_DIR/nightly/gaiadr3.<night>.npy` |
| Log | `LOG_DIR/solopy_<night>.log` |

### Light curves

```python
import pandas as pd
from solopy.lightcurve import add_quality_flags, bin_lightcurve

df = add_quality_flags(pd.read_csv("results/solo.summary.20260630.csv"))  # adds flag_* columns, sun_alt and jd_ltc
bins = bin_lightcurve(df)  # 5 min bins of unflagged points, 3σ-clipped, 0.01 mag systematic floor
```

### As a library

```python
import solopy

lv1 = solopy.FitsLv1(log_file="night.log")
wcs_path = lv1.update_wcs("lv0/frame.fits", outdir="lv1", cache_directory="astrometry_cache")
if wcs_path:  # None when the frame has no plate solution
    lv1.correct_bdf(wcs_path, outdir="lv1", masterdir="calibration_files", ccdmflat=master_flat)

subset, boxes = solopy.GaiaQuery.build_nightly_subset("gaiadr3.npy", pointing_ra, pointing_dec)
solopy.FitsLv2().calculate_zeropoint(lv1_file, subset[subset["iso"]], outdir_zp="zp")
```

## Reference

If you use solopy or SOLO data, please cite:

> Lim et al. 2026, *Journal of the Korean Astronomical Society* (JKAS)

## Documentation

- [`claude_doc/primitive_repo.md`](claude_doc/primitive_repo.md): module reference, workflow and data products.
- [`claude_doc/robustness_review.md`](claude_doc/robustness_review.md): scientific checks of each calibration step.
- [`claude_doc/code_update_log.md`](claude_doc/code_update_log.md): log of code changes.
