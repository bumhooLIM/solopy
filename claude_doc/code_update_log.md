# solopy — Code Update Log

Chronological record of code changes, newest last. Every code update adds an entry here **in the same commit**.
Each entry has an ID (`CU-###`) that also starts the commit subject, so `git log --grep CU-001` finds it.

Each entry lists:

- **Issue:** the problem, with a reference to `claude_doc/primitive_repo.md` §8 where applicable
- **Change:** what was changed, file by file
- **Verification:** how the change was checked
- **Effect on products:** whether existing data products in `data/solo` are affected and must be regenerated

Tests run with the stdlib runner from the repo root: `python -m unittest discover -s tests -v`.

---

## 2026-10-06 · Section 8 bug fixes (branch `fix-section8`)

### CU-001 · Lv3: pass TDB, not UTC, to kete (§8 #1)

- **Issue:** `FitsLv3.predict_targets` passed the header `JD` (mid-exposure **UTC**) to
  `kete.spice.earth_pos_to_ecliptic`, which takes **TDB**. As a result, skyloc's output column `jd_tdb` actually held
  UTC, and `jd_utc` was about 69.2 s early. Light-curve times (`jd_utc`, and `jd_ltc` derived from it) inherited the
  69 s offset. Ephemerides were also evaluated 69 s late (≈ 1″ of main-belt motion).
- **Change:**
  - `solopy/_timeutil.py` (new): `utc_jd_to_tdb()`.
  - `solopy/fitslv3.py`: convert the header JD to TDB before building the observer state.
- **Verification:** `tests/test_timeutil.py`: (a) TDB − UTC = 69.184 s at the 2026-06-30 epoch; (b) with kete and
  skyloc mocked, `predict_targets` requests the observer state at JD(UTC) + 69.184 s. Test (b) fails on the old code.
- **Effect on products:** all Lv3 outputs (`results/solo.summary.*.csv`) and the cleaned light curves have `jd_tdb`
  and `jd_utc` mislabeled or shifted by 69.2 s. **Lv3 must be re-run**; Lv1/Lv2 are unaffected.

### CU-002 · One shared logger helper; no duplicated log lines (§8 #5)

- **Issue:** `FitsLv2.__init__` and `CombMaster.__init__` added a console and a file handler on every instantiation.
  `FitsLv3` builds its own `FitsLv2`, and the `FitsLv3` logger propagated to the root logger configured by the driver.
  Together these wrote many log lines twice. `FitsLv0`/`FitsLv1` avoided duplicates, but only by ignoring any
  `log_file` passed after the first instance.
- **Change:**
  - `solopy/_logutil.py` (new): `get_logger(name, log_file)`. It keeps exactly one console handler and at most one
    file handler, sets `propagate = False`, and swaps the file handler when a new `log_file` is given.
  - `FitsLv0`, `FitsLv1`, `FitsLv2`, `FitsLv3`, `CombMaster` now call `get_logger`. Format unchanged.
- **Verification:** `tests/test_logutil.py`: repeated calls keep one handler of each kind; switching files redirects
  output; `log_file=None` keeps the current file; two `FitsLv2` instances write a message once. The last test fails on
  the old code.
- **Effect on products:** none (log formatting only). Future logs no longer repeat lines.

### CU-003 · Ignore macOS AppleDouble (`._*`) files everywhere (§8 #6)

- **Issue:** on the exFAT T7 drive macOS creates `._<name>` companions. `FitsLv0.batch_decompress` tried to
  decompress them, and `FitsLv1._select_master` and the `CombMaster` flat temp-dir scan read them as FITS. Together
  they caused most of the 929 `ERROR` and 5,655 `WARNING` log lines.
- **Change:**
  - `solopy/_fileutil.py`: add `APPLEDOUBLE_GLOB = "._*"` and `is_appledouble()`.
  - `fitslv0.py`: `batch_decompress` skips AppleDouble files.
  - `fitslv1.py`: `_select_master` excludes them.
  - `combmaster.py`: the master searches use the shared pattern (previously `*._*.fits`), and the flat `tmp/` scan
    excludes them.
  - ccdproc matches `glob_exclude` against bare file names, so `._*` is sufficient.
- **Verification:** `tests/test_fileutil.py`:
  - a real `.bz2` FITS beside a fake `._` companion decompresses with no `ERROR` log, and the companion is left alone;
  - `_select_master` picks the bias with no ccdproc `WARNING`. The old scan reproduces the production warning
    `unable to get FITS header … No SIMPLE card found`.
- **Effect on products:** none (log noise and robustness only). `clean_double.py` is no longer required before a run.

### CU-004 · `update_wcs` returns `None` for unsolved frames (§8 #7)

- **Issue:** when astrometry.net found no solution, `FitsLv1.update_wcs` still wrote `<stem>.wcs.fits` (without WCS)
  and returned its path. The driver's `if not fpath_wcs` guard therefore never fired. The frame was dropped one step
  later by `correct_bdf` with `'NoneType' object has no attribute 'to_header'` (80 frames in the 2026 logs).
- **Change:** `fitslv1.py`: return `None` before computing center coordinates when there is no solution; nothing is
  written. The docstring now documents the return value.
- **Verification:** `tests/test_fitslv1_wcs.py` replaces astrometry.net with a fake solver. Unsolved: `None` and no
  file (this test fails on the old code). Solved: a path whose header has `RACEN`, `ALTCEN`, `PIXSCALE`, `LV0FILE`.
- **Effect on products:** none. The same frames are dropped, now at the intended place with a clear log message.

### CU-005 · Header metadata fixes (§8 #8)

- **Issue:**
  - The camera writes the aperture diameter as `APDIA` (279.4 mm), but `FitsLv0.update_header` read `APTDIA`, so
    every frame got `APTDIA = 0.0`.
  - Master darks kept `BIASCORR = False` although `comb_master_dark` subtracts a master bias.
  - The `comb_master_flat` docstring documented a `filter_name` parameter that does not exist, and the method
    assigned `IMAGETYP` twice.
- **Change:**
  - `fitslv0.py`: `APTDIA` falls back to `APDIA` when missing or ≤ 0. This also repairs files the old code set to
    0.0, because the header update is re-runnable.
  - `combmaster.py`: master darks record `BIASCORR = True` and `BIASNAME`. The flat docstring now describes the real
    parameters and the naming (filter from `FILTER`, date = creation day); the duplicate assignment is removed.
- **Verification:** `tests/test_fitslv0.py` (5 tests: mid-exposure timestamps, normalization, `APDIA` fallback,
  repair of `APTDIA = 0`, idempotency). `tests/test_combmaster.py`: a master dark built from synthetic frames carries
  `BIASCORR = True` and the bias name, and its level equals dark − bias.
- **Effect on products:** metadata only. Existing Lv0/Lv1 headers keep `APTDIA = 0.0`, and existing master darks keep
  `BIASCORR = False`, until those steps are re-run. Pixel data and photometry are unaffected.

### CU-006 · Nightly Gaia subset instead of the full catalog (§8 #4)

- **Issue:** the Lv3 blend check passed the full 247.5 M-row `gaiadr3.npy` to `query_nearest_gaia`, which turns it
  into a DataFrame and SkyCoord (several GB, about 2.3 min per night). Zero points scanned the 61.5 M-row
  `gaiadr3_20arcsec.npy` for every frame.
- **Change** (`solopy/gaia.py`, new `GaiaQuery` methods; existing methods unchanged):
  - `build_nightly_subset(gaia_data, ra, dec, radius_deg=4.5)` is called *before* a night's calibration with the
    Lv0 telescope pointings. Each pointing gets an exact spherical-cap box (RA-wrap and pole safe), and the full
    catalog is read in one chunked pass.
  - Each row gains an `iso` flag (no other source within 20″), using the same KD-tree criterion that built
    `gaiadr3_20arcsec.npy`. So `subset[subset['iso']]` replaces that file for zero points, and the whole subset
    replaces `gaiadr3.npy` for the blend check.
  - Radius 4.5° = FoV half-diagonal 2.40° + astrometry.net search radius 2.0° + 0.1° buffer: any plate-solved field
    lies inside.
  - `footprint_covered(boxes, wcs)` and `wcs_boxes(wcs)` verify, and if needed extend, coverage after plate solving.
    The margin is 60 px, covering `query_gaia`'s 10 px edge buffer and 50 px bright-star radius.
  - `save_subset` / `load_subset` store `.npy` plus a `.json` sidecar (boxes, counts, provenance).
  - Helpers: `cap_boxes`, `in_boxes`, `isolation_flags`, `build_subset`.
  - `solopy/fitslv3.py`: `FitsLv3(orb_path, gaia_path)` also accepts an in-memory catalog (the subset), and warns
    when given a catalog above 20 M rows.
- **Verification:**
  - `tests/test_gaia.py` (10 tests, synthetic sky with RA-wrap and polar cases): caps contain random points; the
    subset equals a brute-force selection; chunked and single passes agree; `iso` equals the all-sky flag inside
    fields; `query_gaia` returns the identical star list from the subset and from an all-sky isolated catalog;
    coverage detection and extension; save/load round trip.
  - Real data, night 2026_0630: 198 science pointings (11 unique) → 1,119,712 rows (956,235 isolated, 41 MB) in
    18.8 s. 195/195 Lv1 footprints covered. For 3 frames, zero-point star lists are identical to those from
    `gaiadr3_20arcsec.npy` (1,750 / 6,292 / 5,975 stars), with query time 0.01 s instead of 0.2–3.5 s.
- **Effect on products:** none by itself. Zero-point inputs are identical by construction (verified). The driver
  must call it (see the driver entry).

### CU-007 · **Critical:** photometry crashed on every call since `82036f8` (found while testing)

- **Issue:** commit `82036f8` ("fix the photometric bug", 2026-07-04) set `ap_area = ApertureStats(...).sum_aper_area`.
  In photutils 2.3 that is a `Quantity` in pix², so `aperture_sum - ap_area * msky` raised
  `Can only apply 'subtract' function to dimensionless quantities …`.
  - `FitsLv2.perform_photometry` caught the error and returned `None` for **every** call, so on the current `main` no
    zero point (Lv2) and no asteroid photometry (Lv3) could be produced.
  - Nobody noticed because the production run (2026-06-21 → 07-03) predates that commit.
  - The same commit made `ap_area` an array but squared it without `[valid_nsky]`. That raises a broadcasting error
    whenever one source of a batch has a fully masked sky annulus.
- **Change:** `fitslv2.py`:
  - new `_as_float_array()` converts photutils outputs (`sum_aper_area`, `median`, `std`) to plain float arrays;
  - the sky-mean error term indexes `ap_area[valid_nsky]`.
  - The intended behavior of `82036f8` is kept: the background is subtracted over the *unmasked* aperture area,
    matching the masked aperture sum.
- **Verification:** `tests/test_photometry.py` (3 tests; **all 3 fail on `main` @ `82036f8`**):
  - a synthetic star of known flux is recovered within 1 %;
  - a masked pixel reduces `aperture_area` by exactly 1 px²;
  - a batch with one fully masked annulus returns both rows and flags the bad one.
- **Effect on products:** none of the existing products were made with the broken code. Without this fix, the next
  run of Lv2/Lv3 would have produced nothing.

### CU-008 · `badphot` from the masked fraction of the aperture (new requirement)

- **Issue:** `badphot` was set when *any* masked pixel touched the aperture, discarding 12.5 % of all asteroid
  measurements. Most of these had only one or two hot or bad pixels in a ~40–70 px² aperture.
- **Change:**
  - `FitsLv2.perform_photometry(..., badpix_frac_max=0.05)`: `badpix_frac` = masked area inside the aperture
    (exact-overlap weights) ÷ geometric aperture area. `badphot = badpix_frac > badpix_frac_max` or flux ≤ 0.
  - New output column `badpix_frac`; `nbadpix` is kept. `badpix_frac_max=0` reproduces the old rule.
  - `calculate_zeropoint` and `FitsLv3.extract_sso_photometry` pass the parameter through; the default is 5 %
    everywhere.
- **Verification:** `tests/test_photometry.py`: 1 masked px (2.3 %) is not flagged; 3 px (6.8 %) is flagged; a
  threshold of 0 reproduces the old behavior; unmasked sources have fraction 0. Full suite passes.
- **Effect on products:** applied to the existing 14,576 measurements, `badphot` falls from 12.5 % to 5.1 %, and
  1,081 rows (median masked fraction 1.6 %) become usable. Takes effect when Lv2/Lv3 are re-run.
  - **Caveat for the robustness review:** masked pixels are excluded, not repaired, so a "good" source can still lose
    the flux of up to 5 % of its aperture.
  - 337 of the newly usable rows are bright (V 11–13), where the masked pixels may be saturated cores.

### CU-009 · Remove dead code; keep warning filters local (§8 #11, part 1)

- **Issue:** about 320 commented-out lines in `fitslv2.py` (an old `find_centroid`, an old vectorized
  `perform_photometry`, a SkyBoT `find_asteroids_in_fov`, and commented blocks inside `calculate_zeropoint`). Plus
  unused modules (`_utils.py`, `_ccdutil.py`), unused helpers in `_fileutil.py`, and unused imports, including
  `astroquery` for the dead SkyBoT code. `psf.py` silenced `AstropyUserWarning` and `RuntimeWarning` for the whole
  Python process as soon as `solopy` was imported. `fitslv1.py` used `ndarray.newbyteorder()`, which NumPy 2 removed
  (in a branch that is never reached today).
- **Change:**
  - `fitslv2.py`: commented blocks removed (318 comment-only lines; no executable line changed). Unused imports
    removed: `astroquery`/`Skybot`, `CCDData`, `Cutout2D`, `SkyCoord`, `Time`, `units`, the centroid functions,
    `_utils`, `logging`.
  - Deleted `solopy/_utils.py` (a duplicate of `FitsLv0.batch_decompress`) and `solopy/_ccdutil.py` (never imported;
    a copy lives in `data/solo/notebooks/ccdutil.py`).
  - `_fileutil.py` keeps only `clear_dir` and the AppleDouble helpers.
  - `psf.py`: unused imports removed; the warning filters apply only inside `soloPSF.process_ccd`; `__all__` added.
  - `fitslv1.py`: NumPy-2-safe byte swap. `fitslv3.py`: unused `logging` import removed.
  - The stale `notebooks/main.py` is replaced by the updated driver later in this branch.
- **Verification:** a usage search over `data/solo/notebooks` (all `*.py`/`*.ipynb`) found no reference to the
  removed helpers. The only `solopy.clear_dir` calls are two old notebook appendices that already fail.
  - New `tests/test_region_psf.py`: 4096 px tiling, 8×8 tiles, last tile 596 px; `soloPSF` recovers an injected
    2.6 px FWHM in all 4 tiles within 0.15 px; importing no longer installs a global `RuntimeWarning` filter.
  - The full suite (39 tests) passes with `-W default` and emits no warnings.
- **Effect on products:** none.

### CU-010 · Packaging and import-time dependencies (§8 #3) + README (§8 #11, part 2)

- **Issue:**
  - `__init__.py` star-imported every module, so `import solopy` required kete and skyloc (via `fitslv3`) and
    leaked names such as `np` and `fits` into the namespace.
  - `pyproject.toml` omitted pandas, scipy, pyarrow, tqdm, kete, and skyloc, but listed the unused astroalign and
    matplotlib.
  - `requires-python >= 3.9` was too low (the code uses `str | None`).
  - `fitslv3` did `import solopy` (a circular import).
  - The README was empty.
- **Change:**
  - `solopy/__init__.py`: explicit imports; `FitsLv3` loaded lazily through module `__getattr__`; `__version__ = "1.1.0"`.
  - Every module now has `__all__`. `fitslv3.py` uses relative imports.
  - `pyproject.toml`: version 1.1.0, Python ≥ 3.10, the real core dependencies, extras `lv3`
    (`kete>=1.0.8,<2`, the version skyloc requires; skyloc from GitHub, since it is not on PyPI) and `notebooks`
    (matplotlib). astroalign removed.
  - `README.md`: overview, installation, usage, tests, documentation links.
- **Verification:** `tests/test_package.py`, using fresh interpreters:
  - `import solopy` succeeds with kete and skyloc blocked, and `solopy.FitsLv3` then raises `ImportError`;
  - no leaked names, and `from solopy import *` does not need Lv3;
  - `FitsLv3` is reachable when the extras are installed;
  - `pyproject.toml` is valid and matches `__version__`. The setuptools schema validation also passes.
  - Full suite: 43 tests OK.
- **Effect on products:** none. The installed editable metadata still reports 1.0.0 until
  `pip install -e . --no-deps` is re-run.

### CU-011 · Versioned nightly driver: nightly Gaia subset, `--levels`, robust file lists (§8 #4, #7, #11; enables #2)

- **Issue:**
  - The production driver (`~/Desktop/data/solo/notebooks/main.py`) is not under version control, and the repo's
    `notebooks/main.py` was a stale 2025 example.
  - Re-running only Lv2/Lv3 (needed to regenerate products, §8 #2) required re-running Lv0/Lv1.
  - The driver loaded the full Gaia catalogs.
- **Change:** `notebooks/main.py` is replaced by the production driver with these updates (parameters unchanged:
  flat `…20260526`, BPM `…20260616`, ZP stars G 13–15, apertures 1.5/3/4 × FWHM):
  - **Nightly Gaia subset** (#4): built *before* calibration from the Lv0 pointings (`GaiaQuery.build_nightly_subset`)
    and saved to `GAIA_NIGHTLY_DIR/gaiadr3.<night>.npy` (default `gaia_dr3/nightly/`). It is reloaded on re-runs
    (`--rebuild-gaia` forces a rebuild). After Lv1, every solved footprint is checked, and the subset is extended
    if any falls outside. Lv2 uses `subset[iso]`; Lv3 uses the whole subset.
  - `--levels` (default `0,1,2,3`): for example `--levels 2,3` regenerates zero points and asteroid photometry from
    existing Lv1 frames.
  - `--badpix-frac-max` (default 0.05) is passed to Lv2 and Lv3.
  - Results CSVs gain a `badpix_frac` column. The `update_wcs` → `None` guard now works (#7).
  - Frame lists come from one helper, `read_summary`, which always yields absolute paths. The old code got them as a
    side effect of a bare `.filter()` call; a first version of this rewrite lost that and could not open frames,
    which the smoke test caught.
  - The driver logs an error, instead of crashing, when a night has no Lv0 or Lv1 science frames.
  - Optional `directory.py` entries `GAIA_NIGHTLY_DIR` and `ASTROMETRY_CACHE_DIR`; defaults keep the current layout.
  - `PSFFILE` (driver) and `ZPFILE` (`fitslv2.py`) comments shortened so the cards fit in 80 characters. These
    truncation warnings were previously hidden by `psf.py`'s global warning filter (removed in CU-009).
- **Verification:** smoke test on 6 real 2026_0630 frames in a scratch tree (production inputs read only;
  `--levels 1,2,3`, then `--levels 2,3`): exit 0, no `ERROR` or `WARNING` lines, no duplicated log lines; the subset
  is built, reused, and covers 6/6 footprints.
  - Compared with production: Lv1 pixels and masks are bit-identical; `ZP_G` agrees within 0.002 mag (N rises by
    5–10 % from the 5 % rule); `jd_utc`/`jd_tdb` move by +69.18 s (CU-001); asteroid positions agree to 10⁻⁴ px;
    non-flagged magnitudes change by a median of −0.0015 mag.
  - The only large change is (192) Nausikaa (V = 11.1; 12 masked px, 16–26 % of the aperture, still `badphot`):
    +0.4 to +0.5 mag brighter, because sky is no longer subtracted over masked pixels. This exposes the
    saturated-core issue raised in CU-008.
- **Effect on products:** none until deployed. Deploying means copying this file to
  `~/Desktop/data/solo/notebooks/main.py`, which needs user confirmation.

---

## 2026-10-06 · Robustness fixes confirmed by the user (`claude_doc/robustness_review.md`)

### CU-012 · WCS was shifted by one pixel (review R4)

- **Issue:** `update_wcs` passed 0-based SEP positions to astrometry.net, which works in FITS 1-based pixels. Every
  Lv1 WCS was therefore offset by (+1, +1) px (≈ 4″). Gaia − SEP residuals were (+0.68…+0.95, +0.90…+1.32) px in
  all 811 frames checked.
- **Change:** `fitslv1.py`: the star list for the solver is `x + 1, y + 1`.
- **Verification:**
  - `tests/test_fitslv1_wcs.py`: the solver receives the brightest SEP star at (x + 1, y + 1).
  - Real solve of `dawn_field1_001_20260630010758`: the median Gaia residual moves from (+0.61, +0.87) px
    (production WCS) to (−0.39, −0.13) px, a shift of exactly (−1, −1); the rms (0.24, 0.34 px) is unchanged.
  - The remaining ~0.4 px (≈ 1.2″) comes from astrometry.net's fit to ~40 J2000 index stars. It is well below the
    FWHM, so asteroid recentering absorbs it; a Gaia-based WCS refinement could remove it later.
- **Effect on products:** Lv1 WCS (and everything positioned with it) needs Lv1 regeneration.

### CU-013 · Lv1 bit mask; saturated pixels always flag photometry (review R1)

- **Issue:** the Lv1 `MASK` was a plain 0/1 array, so photometry could not tell a saturated core from a hot pixel.
  Under the 5 % rule (CU-008), 24 of 104 saturated asteroid measurements (all V ≤ 11.5, and 70 % at V 12–12.5)
  would have been accepted with ≥ 13 % of their flux missing.
- **Change:**
  - New public module `solopy/maskbits.py`, also exported as `solopy.maskbits`, defining the bits: 1 `BADPIX` (BPM,
    flat defect, NaN/Inf, pre-existing), 2 `SATURATED` (raw ≥ 3800 ADU), 4 `BORDER` (100 px), 8 `NONPOSITIVE`
    (≤ 0 after dark or flat), 16 `BRIGHT_STAR`, 32 `TRAIL`. It also holds `SATURATION_ADU`, `BORDER_PIX`,
    `header_cards()` and `has_bits()`.
  - `FitsLv1.correct_bdf` writes the `MASK` extension as a `uint8` bit mask. The same pixels are masked as before;
    only the reason is now kept. Headers gain `MASKVER = 2`, `MASKB1…MASKB32`, and `NSATPIX`.
    `_mask_source(..., return_parts=True)` returns the bright-star and trail masks separately.
  - `FitsLv2.perform_photometry` accepts a boolean or a bit mask. Any saturated pixel touching the aperture sets
    `saturated` and `badphot`, whatever the masked fraction. New columns: `saturated`, `nsatpix`.
  - `calculate_zeropoint` and `FitsLv3.extract_sso_photometry` pass the bit mask through (boolean only to SEP).
    For an old 0/1 mask, Lv2 logs that saturation cannot be recognized.
- **Verification:**
  - `tests/test_fitslv1_bdf.py`: a synthetic `correct_bdf` run gives `SATURATED` for a raw 4000 ADU pixel, `BADPIX`
    for the BPM hot pixel and for the flat defect, `BORDER` at the edge, and 0 elsewhere. Both headers document the
    bits, and `NSATPIX = 1`.
  - `tests/test_photometry.py`: one saturated core pixel (2.3 % of the aperture) flags the source; one hot pixel
    there does not. Full suite passes.
- **Effect on products:** needs Lv1 regeneration, so the masks carry the bits, then Lv2/Lv3.
