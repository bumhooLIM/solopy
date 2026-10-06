"""
Light-curve products from Lv3 asteroid photometry (robustness review R7).

- `aperture_contamination`: flux that catalogued stars put into an aperture (Gaussian PSF).
- `add_quality_flags`: per-measurement flags (zero point, SNR, masking/saturation, contamination,
  altitude, twilight), Sun altitude, and light-time-corrected times.
- `bin_lightcurve`: groups consecutive measurements of one asteroid in one night into short
  bins, rejects outliers, and returns inverse-variance weighted means.

These reproduce the analysis previously done in data/solo/notebooks/summary_results.ipynb,
with the blend test unit-correct, outlier rejection inside bins, and a systematic floor that is
added once per bin rather than averaged down.
"""
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.constants import c
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time
from scipy.stats import ncx2

from .zeropoint import robust_std

__all__ = ["SITE", "aperture_contamination", "add_quality_flags", "bin_lightcurve"]

SITE = EarthLocation(lat=37.07 * u.deg, lon=-119.4 * u.deg, height=1.405 * u.km)  # SRO, MPC G80


def aperture_contamination(distance_pix, star_flux, r_ap, fwhm):
    """
    Flux that a star at `distance_pix` from the aperture centre puts inside a circular aperture of
    radius `r_ap`, for a circular Gaussian PSF of FWHM `fwhm` (all in pixels). The enclosed fraction
    is exact: P(|X| <= r_ap) with X ~ N(d, sigma^2 I), a noncentral chi-square with 2 dof.
    Inputs broadcast; returns star_flux * fraction.
    """
    sigma = np.asarray(fwhm, dtype=float) / 2.3548
    d = np.asarray(distance_pix, dtype=float)
    frac = ncx2.cdf((np.asarray(r_ap, dtype=float) / sigma) ** 2, df=2, nc=(d / sigma) ** 2)
    return np.asarray(star_flux, dtype=float) * frac


def add_quality_flags(df, zperr_max=0.2, zp_spread_max=0.1, snr_min=3.0, contam_max=0.02,
                      alt_min_dusk=22.0, alt_min_dawn=20.0, sun_alt_max=-11.0, pixscale_arcsec=2.974,
                      site=SITE):
    """
    Add Sun altitude, a light-time-corrected time `jd_ltc`, and boolean flags to Lv3 results.

    Flags (True = do not use):
      flag_zperr      frame zero point scatter `zperr_global` > zperr_max
      flag_zpspread   local zero-point stars scatter `zp_local_spread` > zp_spread_max (cloud, twilight)
      flag_lowsnr     snr < snr_min
      flag_badphot    badphot (masked fraction, saturation, masked PSF flux, non-positive flux)
      flag_contam     catalogued stars add more than contam_max of the asteroid flux. Uses
                      `contam_frac` (all Gaia stars, Lv3 >= 1.1); for older result files it is
                      estimated from the nearest Gaia star alone and stored in `contam_frac`.
      flag_lowalt     field altitude below alt_min_dusk (dusk fields) / alt_min_dawn (dawn fields)
      flag_twilight   Sun altitude above sun_alt_max
      flag_any        any of the above
    Informational only (not part of flag_any):
      flag_neargaia   nearest Gaia star within 5 aperture radii (arcsec) and G <= V + 2.5 — the
                      notebook's rule with units fixed; it is much stricter than the measured
                      contamination and kept for comparison.
    Other columns missing from older result files leave their flag False.
    """
    out = df.copy()
    t = Time(out["jd_utc"].to_numpy(dtype=float), format="jd", scale="utc")
    out["sun_alt"] = get_sun(t).transform_to(AltAz(obstime=t, location=site)).alt.deg
    out["jd_ltc"] = out["jd_utc"] - (out["r_obs"].to_numpy(dtype=float) * u.au / c).to(u.day).value

    def col(name, default=np.nan):
        return out[name] if name in out.columns else pd.Series(default, index=out.index)

    pixscale = col("pixscale", pixscale_arcsec).fillna(pixscale_arcsec)
    obj = col("object", "").astype(str)

    if "contam_frac" not in out.columns:
        # Older result files: estimate the contamination from the nearest Gaia star only
        zp = col("zp_local").fillna(col("zp_global")) if "zp_local" in out.columns else col("zp_global")
        fwhm = col("psf_fwhm").fillna(col("mapped_fwhm")) if "psf_fwhm" in out.columns else col("mapped_fwhm")
        star_flux = col("exptime") * 10 ** (-0.4 * (col("nearest_gaia_gmag") - zp))
        inside = aperture_contamination(col("nearest_gaia_dist_arcsec") / pixscale, star_flux,
                                        col("r_ap_pixel"), fwhm)
        source = col("source_sum")
        out["contam_frac"] = np.where(source > 0, inside / source.where(source > 0, 1.0), np.nan)

    flags = {
        "flag_zperr": col("zperr_global") > zperr_max,
        "flag_zpspread": col("zp_local_spread") > zp_spread_max,
        "flag_lowsnr": col("snr") < snr_min,
        "flag_badphot": col("badphot", False).astype(bool),
        "flag_contam": col("contam_frac") > contam_max,
        "flag_lowalt": ((col("altcen") < alt_min_dusk) & obj.str.contains("dusk"))
                       | ((col("altcen") < alt_min_dawn) & obj.str.contains("dawn")),
        "flag_twilight": out["sun_alt"] > sun_alt_max,
    }
    for name, flag in flags.items():
        out[name] = flag.fillna(False).astype(bool)
    out["flag_any"] = np.logical_or.reduce([out[name].to_numpy() for name in flags])
    out["flag_neargaia"] = ((col("nearest_gaia_dist_arcsec") < 5 * col("r_ap_pixel") * pixscale)
                            & (col("nearest_gaia_gmag") <= col("vmag") + 2.5)).fillna(False).astype(bool)
    return out


def _time_groups(obsdate, desig, times, window_days):
    """Anchor-based groups: a new group starts on a new night/object or `window_days` after the anchor."""
    group = np.zeros(len(times), dtype=int)
    anchor = times[0] if len(times) else 0.0
    for i in range(1, len(times)):
        if obsdate[i] != obsdate[i - 1] or desig[i] != desig[i - 1] or times[i] - anchor > window_days:
            group[i] = group[i - 1] + 1
            anchor = times[i]
        else:
            group[i] = group[i - 1]
    return group


def bin_lightcurve(df, window_min=5.0, mag_col="gmag_distcorr", stat_err_cols=("mag_err", "zperr_local"),
                   floor_mag=0.01, clip_sigma=3.0, flag_col="flag_any"):
    """
    Bin unflagged measurements of each asteroid per night into windows of `window_min` minutes.

    Inside a bin, points deviating more than `clip_sigma` robust sigmas from the median are dropped
    (bins of 3+ points). The bin value is the inverse-variance weighted mean with per-point errors
    sqrt(sum of `stat_err_cols`^2); its error is sqrt(1/sum(w) + floor_mag^2), so a systematic floor
    is added once and not averaged down.
    """
    d = df.copy()
    d["desig"] = d["desig"].astype(str).str.strip()
    d["obsdate"] = d["obsdate"].astype(str).str.strip()
    tcol = "jd_ltc" if "jd_ltc" in d.columns else "jd_utc"
    d = d.sort_values(["obsdate", "desig", tcol]).reset_index(drop=True)
    d["time_group"] = _time_groups(d["obsdate"].to_numpy(), d["desig"].to_numpy(), d[tcol].to_numpy(dtype=float),
                                   window_min / 1440.0)
    err_cols = [c for c in stat_err_cols if c in d.columns]
    d["_stat_err"] = np.sqrt(np.sum([d[c].to_numpy(dtype=float) ** 2 for c in err_cols], axis=0))
    usable = np.isfinite(d[mag_col]) & np.isfinite(d["_stat_err"]) & (d["_stat_err"] > 0)
    if flag_col in d.columns:
        usable &= ~d[flag_col].astype(bool)

    rows = []
    for (obsdate, desig, _), g in d[usable].groupby(["obsdate", "desig", "time_group"], sort=False):
        mag = g[mag_col].to_numpy(dtype=float)
        keep = np.ones(len(g), dtype=bool)
        if len(g) >= 3:
            s = robust_std(mag)
            if np.isfinite(s) and s > 0:
                keep = np.abs(mag - np.median(mag)) <= clip_sigma * s
        gk = g[keep]
        w = 1.0 / gk["_stat_err"].to_numpy() ** 2
        rows.append({
            "obsdate": obsdate, "desig": desig,
            "jd_utc_mean": gk["jd_utc"].mean(),
            f"{tcol}_mean": gk[tcol].mean(),
            "r_hel_mean": gk["r_hel"].mean() if "r_hel" in gk else np.nan,
            "r_obs_mean": gk["r_obs"].mean() if "r_obs" in gk else np.nan,
            "vmag_mean": gk["vmag"].mean() if "vmag" in gk else np.nan,
            "alpha_mean": gk["alpha"].mean() if "alpha" in gk else np.nan,
            f"{mag_col}_wmean": float(np.sum(w * gk[mag_col].to_numpy(dtype=float)) / np.sum(w)),
            "mag_err_wmean": float(np.sqrt(1.0 / np.sum(w) + floor_mag ** 2)),
            "n_obs": int(len(gk)),
            "n_clipped": int((~keep).sum()),
        })
    return pd.DataFrame(rows)
