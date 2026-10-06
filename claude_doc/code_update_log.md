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
