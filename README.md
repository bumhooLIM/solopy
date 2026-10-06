# solopy

Data-reduction pipeline for **SOLO**, the Solar system Object Light-curve Observatory: a RASA 11" astrograph with an
FLI KL4040 camera (GSENSE4040, 4096 × 4096, ≈ 2.97″/px) at Sierra Remote Observatories (MPC G80). It turns raw FITS
frames into Gaia-G-calibrated photometry of known asteroids.

| Level | Class | Step |
|---|---|---|
| Lv0 | `FitsLv0` | decompress `.fits.bz2`; normalize headers (mid-exposure UTC, site, detector) |
| masters | `CombMaster` | master bias, dark, and sky flat |
| Lv1 | `FitsLv1` | astrometry.net plate solution; bias, dark, flat; bad-pixel mask |
| Lv2 | `soloPSF`, `FitsLv2`, `GaiaQuery` | tile-wise PSF FWHM; aperture photometry of Gaia stars; zero point `ZP_G` |
| Lv3 | `FitsLv3` | kete/skyloc asteroid prediction; centroid and aperture photometry; Gaia blend check |

## Installation

```bash
conda activate solopy        # Python >= 3.10
pip install -e .             # Lv0–Lv2
pip install -e ".[lv3]"      # adds kete (< 2) and skyloc (from GitHub) for Lv3
```

`import solopy` works without the Lv3 extras; `solopy.FitsLv3` imports them on first use.

## Usage

The nightly driver is `notebooks/main.py`. It runs from a folder that also contains `directory.py`, the path
configuration; in production that folder is `~/Desktop/data/solo/notebooks/`.

```bash
python main.py -s 2026_0630                  # all levels: 0 headers+masters, 1 WCS+BDF, 2 PSF+ZP, 3 asteroids
python main.py -s 2026_0630 --levels 2,3     # re-run zero points and asteroid photometry only
```

Before calibration the driver builds a nightly Gaia subset (`gaia_dr3/nightly/gaiadr3.<night>.npy`) covering every
field of the night, and uses it instead of the full catalog.

Library use:

```python
import solopy

lv1 = solopy.FitsLv1(log_file="night.log")
wcs_path = lv1.update_wcs("lv0/dawn_001.fits", outdir="lv1", cache_directory="astrometry_cache")
if wcs_path:  # None when the frame has no astrometric solution
    lv1.correct_bdf(wcs_path, outdir="lv1", masterdir="calibration_files", ccdmflat=master_flat)

subset, boxes = solopy.GaiaQuery.build_nightly_subset("gaiadr3.npy", pointing_ra, pointing_dec)
solopy.FitsLv2().calculate_zeropoint(lv1_file, subset[subset["iso"]], outdir_zp="zp")
```

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests use synthetic data only; they do not need the observatory data or network access.

## Documentation

- [`claude_doc/primitive_repo.md`](claude_doc/primitive_repo.md): module reference, workflow, data products,
  and how the pipeline is used on the SOLO data.
- [`claude_doc/code_update_log.md`](claude_doc/code_update_log.md): log of code changes.
