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
