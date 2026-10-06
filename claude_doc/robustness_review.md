# Scientific robustness review of the SOLO calibration (2026-10-06)

**Scope:** every calibration stage and the derived asteroid photometry, checked quantitatively on real data:

- Zero-point tables of three nights spread over the campaign (2026-05-25, 06-11, 06-30): 811 frames and 621,508
  star measurements, each matched to Gaia DR3 colors.
- The 1,708 asteroid measurements of the same nights, checked against raw Lv0 pixels.
- 12 frames re-photometered over 10.5 < G < 17.5.

All checks were read-only; no product was modified. The bug fixes `CU-001`–`CU-011` are a separate topic, recorded
in `code_update_log.md`.

## Summary

| Stage | Verdict | Key number |
|---|---|---|
| Time stamps | OK (after CU-001) | mid-exposure UTC; TDB now passed to kete |
| Master bias and dark | OK; one minor issue (R9) | 2 nights calibrated with masters taken at another CCD temperature |
| Master flat | Stable in time | 2026-05-26 vs 06-21 flats: < 2 % large scale, 0.3 % pixel rms |
| Masking under the 5 % rule | **Not robust (R1, R6)** | 24 saturated asteroid points would pass, ≥ 0.15 mag too faint |
| Astrometry (Lv1 WCS) | **Systematic offset (R4)** | every frame shifted by (+1, +1) px = 4″ |
| PSF / FWHM per tile | OK | injected FWHM recovered within 0.15 px |
| Zero point | **Not robust across the field (R2)** | 0.10–0.13 mag gradient at airmass > 2; ±40 mmag detector pattern |
| Color term | Small systematic (R3) | −0.063 mag/mag; +0.012 mag at solar color |
| Photometric errors | Mis-modeled (R5) | sky variance counted twice, yet errors underestimate scatter by up to 55 % |
| Asteroid centroiding | OK | median 0.5–0.6 px from prediction; > 2 px only at SNR < 3 |
| Light-curve cleaning (notebook) | **Not robust (R7)** | blend flag compares arcsec with pixels; no outlier rejection |

## Findings and proposed fixes

### R1 · Saturated pixels must always flag a measurement (high)

- **Evidence:**
  - 104 of 1,708 asteroid measurements (6.1 %) contain raw pixels ≥ 3800 ADU inside the aperture: every
    measurement at V ≤ 11.5 (55/55), 70 % at V 12–12.5 (49/70), and none fainter than V 12.5.
  - The Lv1 `MASK` cannot tell these apart from hot pixels.
  - Under the 5 % rule (CU-008), 24 of them would count as good, with 1–3 saturated core pixels and ≥ 13 % (median;
    up to 27 %) of the flux missing, i.e. ≥ 0.15 mag too faint. This is a lower bound, because the saturated pixels
    clip the peak.
  - Zero-point stars (G 13–15) are never saturated.
- **Fix:** make the Lv1 `MASK` a bit mask: 1 bad pixel (BPM, flat defect), 2 saturated, 4 border, 8 negative after
  dark, 16 bright-star halo, 32 trail. Bits are documented in the header (`MASKBITS`). Any saturated pixel in the
  aperture sets `saturated` and `badphot` whatever the masked fraction; the 5 % rule applies to the other bits.
  Readers that treat the mask as boolean keep working.
- **Re-run:** Lv1, Lv2, Lv3.

### R2 · Use a local zero point instead of one value per frame (high)

- **Evidence:**
  - *Extinction across the field.* Within-frame gradients give k = 0.31 mag/airmass, consistent with the
    frame-to-frame trend (0.19–0.35). At airmass > 2 (45 % of frames) the zero point changes by 0.10–0.13 mag across
    the 3.4° field.
  - *Fixed detector pattern.* After removing the gradient and color term, the field center reads +40 mmag and the
    edges −15 to −25 mmag (8 × 8 map). The pattern is stable night to night (correlation 0.67–0.93). It is consistent
    with the night-sky flat over-correcting vignetting (the flat falls to 0.64 in the corners) and/or with aperture
    corrections varying across the field.
  - *Leave-one-out test.* Predicting each star's zero point: the global `ZP_G` has 47.1 mmag scatter; the local zero
    point (3σ-clipped median of ≥ 10 stars within 500 px, median 37 neighbors) has 27.8 mmag. Stars off by more than
    50 mmag drop from 30 % to 17 %.
  - Asteroid magnitudes would change by −64 to +56 mmag (5–95 %).
  - A global polynomial surface was tested and **rejected**: it extrapolates by more than 1 mag in sparse
    low-altitude frames.
- **Fix:** for each asteroid, `zp_local` is computed from its own frame's zero-point table (stars within 500 px,
  ≥ 10 stars, otherwise the global value plus a flag), together with `zperr_local` and `n_zp_local`. `gmag` then
  uses `zp_local`. A frame-quality flag is set when the local values spread more than 0.1 mag: 45 frames (5.5 %)
  have zero-point stars spanning more than 0.5 mag, and 15 of those pass today's `zperr > 0.2` cut.
- **Re-run:** Lv2, Lv3.

### R3 · Color term, evaluated at solar color (medium-low)

- **Evidence:**
  - `ZP` changes by −0.063 mag per mag of BP−RP, weaker at high airmass (−0.094 at X = 1, −0.038 at X = 2.5).
  - Zero-point stars average BP−RP = 1.01, while the Sun is 0.82, so asteroid magnitudes are off by +0.012 mag
    (0.005 mag frame-to-frame).
  - The color term explains most of the apparent faint-end "non-linearity" of stars: −35 → −13 mmag at
    G 16.5–17 once it is removed.
- **Fix:** fit the color slope per frame alongside the zero point (header `ZPCOLOR`) and evaluate asteroid zero
  points at BP−RP = 0.82.
- **Re-run:** Lv2, Lv3.

### R4 · WCS shifted by one pixel (medium)

- **Evidence:**
  - SEP positions minus Gaia positions through the Lv1 WCS are (+0.68…+0.95, +0.90…+1.32) px in every frame.
  - Adding 1 to `CRPIX` reduces this to (−0.26…−0.04, −0.03…+0.19) px.
  - Cause: astrometry.net works in FITS 1-based pixels, but receives 0-based SEP coordinates.
  - Photometry is barely affected, thanks to recentering and the 3 px matching tolerance. But predicted asteroid
    positions and any astrometric use are off by 1.4 px (≈ 4″).
- **Fix:** pass `x + 1, y + 1` to the solver. Existing Lv1 files are corrected by the Lv1 re-run, or in place
  (`CRPIX += 1`, plus recomputed `RACEN`/`DECCEN`/`ALTCEN`/`AZCEN`).
- **Re-run:** Lv1, or a header-only patch.

### R5 · Photometric error model (medium)

- **Evidence:**
  - `aperture_sum_err²` already contains sky Poisson noise and read noise; adding `area·σ_sky²` counts the sky
    twice, which makes errors 16–20 % too large at G 14–15.
  - Still, the repeatability of zero-point stars (12,284 stars with ≥ 8 visits per field and night) exceeds the
    reported `mag_err` by 3–55 %, worst for bright stars. That is a systematic floor of ≈ 13 mmag at G 13, mostly
    from R2.
- **Fix:** statistical error σ² = F/g + A·σ_sky²·(1 + A/N_sky) (`mag_err`), plus
  `mag_err_tot` = √(mag_err² + zperr_local² + σ_floor²). σ_floor is re-measured from repeatability after R2. Use
  `mag_err_tot` for weights and bins.
- **Re-run:** Lv2, Lv3.

### R6 · Flux lost to masked (unsaturated) pixels (low-medium)

- **Evidence:** 61 of 1,708 measurements (3.6 %) have a masked fraction between 0 and 5 %. The median loss is
  0.001 mag (pixels at the aperture edge), but 12 measurements lose 0.01–0.09 mag (pixels near the core).
- **Fix:** correct the flux by the PSF-weighted masked fraction (Gaussian with the local FWHM),
  F / (1 − f_PSF). Store `psf_lost_frac`, and flag when f_PSF > 5 %.
- **Re-run:** Lv2, Lv3.

### R7 · Light-curve cleaning in `summary_results.ipynb` (medium)

- **Evidence:**
  - (a) The blend flag compares `nearest_gaia_dist_arcsec` with `5*r_ap_pixel` (pixels). The effective radius is
    therefore ~18″ (1.7 aperture radii) instead of 5 radii (~53″).
  - (b) Only the nearest star is considered, and only if G ≤ V + 2.5.
  - (c) The Gaia file stops at G = 18.5. A G = 19 star inside the aperture of a V = 16.5 target adds 10 % of the
    flux and is never flagged.
  - (d) 5-min bins have no outlier rejection and no systematic error floor.
- **Fix:** a versioned, tested `solopy.lightcurve` module, used by the notebook, with:
  - contamination = PSF-weighted flux of *all* Gaia stars within r_ap + 2 FWHM, divided by the asteroid flux
    (unit-correct);
  - the frame-quality flag from R2;
  - σ-clipping inside bins;
  - `mag_err_tot`.
  - A deeper star catalog (to G ≈ 20) would need a new Gaia download and is left as an option.

### R8 · Provenance (low)

- **Fix:** write `SOLOPYV` (version and git commit) into Lv1/Lv2 headers and result tables. The driver is now
  versioned (CU-011).

### R9 · Dark temperature matching (low)

- **Evidence:** nights 0602 and 0603 have no darks. They were corrected with masters taken at −10 °C and −5 °C,
  while the frames were at −8.5 °C and −7.3 °C. The photometric impact is negligible: the dark adds ≈ 2 ADU per
  60 s against ≈ 800 ADU of sky, and the local sky removes it.
- **Fix:** prefer masters within 1 °C; otherwise warn and write `DARKDT`.

## Verified as robust (no change needed)

- **Timing:** `UTC-END − EXPTIME/2` gives mid-exposure UTC, and re-running the header update is idempotent.
- **Masters:** bias and dark combination (median, 5σ MAD clipping).
- **Flat:** stable between May and June (< 2 % large scale).
- **PSF:** tile FWHM recovered within 0.15 px in an injection test.
- **Aperture:** 1.5 FWHM holds 99.8 % of a Gaussian's flux; the zero point absorbs the remainder.
- **Nightly Gaia subset:** yields zero-point star lists identical to the all-sky catalog.
- **Recentering:** asteroid centroids are reliable at SNR > 3.
- **Zero-point precision:** median `ZPERR_G` = 0.043 mag with N ≈ 760 stars.

## Reprocessing plan (fix #2)

1. Validate the code on one night (2026_0630) in a scratch tree and compare with the current products.
2. Reprocess all 33 nights:
   - `--levels 1,2,3` if R1 or R4 is accepted: about 27 min per night, about 15 h in total. This overwrites the Lv1
     files on T7 and the PSF/ZP/results tables.
   - Otherwise `--levels 2,3`: about 9 h.
   - Lv0 is not touched.
3. Before the production run, investigate the two nights that stopped during Lv2 (0619, 0626).

## Implementation status (confirmed by the user on 2026-10-06)

| Finding | Implemented in | Notes |
|---|---|---|
| R1 saturation | CU-013 | `solopy.maskbits`; Lv1 bit mask; `saturated` always sets `badphot` |
| R2 local zero point | CU-016 | `zp_local` from stars within 500 px (≥ 10), else the frame value; `zp_local_spread` |
| R3 color term | CU-016 | `ZP_SUN`/`ZPCOLOR` per frame; asteroids at BP−RP = 0.82 |
| R4 WCS off-by-one | CU-012 | real re-solve: offset (+0.61, +0.87) → (−0.39, −0.13) px |
| R5 error model | CU-014, CU-016 | DAOPHOT error (simulation ratio 0.77 → 1.0); `mag_err_tot` with a 0.01 mag floor |
| R6 masked flux | CU-015 | PSF-weighted restoration; flag above 5 % of the PSF flux |
| R7 light curves | CU-019 | `solopy.lightcurve`; contamination from all Gaia stars; notebook switch pending review |
| R8 provenance | CU-018 | `SOLOPYV*`, `solopy_version` |
| R9 dark temperature | CU-017 | ±1 °C preference, `DARKDT` |
| Crashed nights 0619/0626 | CU-020 | diverging PSF fit no longer fatal; per-frame errors logged |
