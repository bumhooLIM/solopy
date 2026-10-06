import unittest

import numpy as np
import pandas as pd

from solopy.lightcurve import add_quality_flags, aperture_contamination, bin_lightcurve


class TestLv3Contamination(unittest.TestCase):
    def test_star_near_asteroid_is_counted(self):
        import importlib.util
        if not all(importlib.util.find_spec(m) for m in ("kete", "skyloc")):
            self.skipTest("kete/skyloc not installed")
        from unittest import mock
        from astropy.wcs import WCS
        from solopy.fitslv3 import FitsLv3

        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]; wcs.wcs.crval = [335.0, -10.0]
        wcs.wcs.crpix = [2048.5, 2048.5]; wcs.wcs.cdelt = [-2.97 / 3600, 2.97 / 3600]
        ra_star, dec_star = wcs.pixel_to_world_values(2049.0, 2047.0)      # 2 px from the asteroid
        gaia = np.zeros(2, dtype=[("source_id", "<i8"), ("phot_g_mean_mag", "<f4"), ("ra", "<f8"), ("dec", "<f8")])
        gaia["phot_g_mean_mag"] = [16.0, 12.0]
        gaia["ra"], gaia["dec"] = [ra_star, 336.0], [dec_star, -10.0]      # the bright star is far away
        proc = FitsLv3.__new__(FitsLv3)
        proc.logger, proc.gaia_all = mock.Mock(), gaia
        ra_ast, dec_ast = wcs.pixel_to_world_values(2047.0, 2047.0)
        phot = pd.DataFrame({"ra": [ra_ast], "dec": [dec_ast], "x_winpos": [2047.0], "y_winpos": [2047.0],
                             "r_ap_pixel": [3.75], "mapped_fwhm": [2.5], "zp_local": [18.6], "source_sum": [5000.0]})
        flux, frac, n_near = proc._aperture_contamination(phot, wcs, exptime=60.0)
        expected = aperture_contamination(2.0, 60.0 * 10 ** (-0.4 * (16.0 - 18.6)), 3.75, 2.5)
        self.assertAlmostEqual(flux[0], expected, delta=1e-6 * expected)
        self.assertAlmostEqual(frac[0], expected / 5000.0, places=9)
        self.assertEqual(n_near[0], 1)


class TestApertureContamination(unittest.TestCase):
    def test_enclosed_fraction_matches_a_sampled_gaussian(self):
        fwhm, r_ap = 2.5, 3.75
        sigma = fwhm / 2.3548
        rng = np.random.default_rng(0)
        for d in (0.0, 2.0, 4.0, 6.0):
            pts = rng.normal([d, 0.0], sigma, size=(400_000, 2))
            sampled = np.mean(np.hypot(pts[:, 0], pts[:, 1]) <= r_ap)
            self.assertAlmostEqual(aperture_contamination(d, 1.0, r_ap, fwhm), sampled, delta=0.003)

    def test_star_on_the_aperture_is_fully_counted(self):
        self.assertAlmostEqual(aperture_contamination(0.0, 1000.0, 1.5 * 2.5, 2.5), 998.0, delta=1.0)

    def test_broadcasts_over_stars(self):
        out = aperture_contamination(np.array([0.0, 10.0]), np.array([100.0, 100.0]), 3.75, 2.5)
        self.assertEqual(out.shape, (2,))
        self.assertLess(out[1], 1e-3)


def _results(n=12, seed=0):
    rng = np.random.default_rng(seed)
    jd0 = 2461221.97
    df = pd.DataFrame({
        "desig": ["1082"] * n, "obsdate": ["20260630"] * n, "object": ["dawn_field1"] * n,
        "jd_utc": jd0 + np.arange(n) * 60 / 86400,                     # one frame per minute
        "r_obs": 2.0, "r_hel": 2.6, "vmag": 15.0, "alpha": 22.0,
        "gmag_distcorr": 10.0 + rng.normal(0, 0.02, n), "mag_err": 0.02, "zperr_local": 0.005,
        "snr": 50.0, "badphot": False, "zperr_global": 0.04, "zp_local_spread": 0.03, "contam_frac": 0.0,
        "nearest_gaia_dist_arcsec": 200.0, "nearest_gaia_gmag": 15.0, "r_ap_pixel": 3.6, "altcen": 40.0,
    })
    return df


def _old_results(n=6, seed=1):
    """Lv3 rows as written before 1.1 (no zperr_local, contam_frac, zp_local_spread), 18 nights earlier."""
    df = _results(n, seed).drop(columns=["zperr_local", "contam_frac", "zp_local_spread"])
    df["obsdate"], df["jd_utc"] = "20260612", df["jd_utc"] - 18.0
    df["exptime"], df["zp_global"], df["psf_fwhm"], df["source_sum"] = 60.0, 18.6, 2.5, 5000.0
    return df


class TestQualityFlags(unittest.TestCase):
    def test_near_gaia_rule_uses_arcsec(self):
        df = _results(2)
        df.loc[0, ["nearest_gaia_dist_arcsec", "nearest_gaia_gmag"]] = [30.0, 15.0]  # 30" < 5*3.6 px*2.97"
        out = add_quality_flags(df)
        # The notebook compared 30 (arcsec) with 5*3.6 = 18 (pixels) and missed this star.
        self.assertTrue(out.flag_neargaia.iloc[0])
        self.assertFalse(out.flag_neargaia.iloc[1])

    def test_flags_and_light_time(self):
        df = _results(4)
        df.loc[0, "contam_frac"] = 0.05
        df.loc[1, "badphot"] = True
        df.loc[2, "zp_local_spread"] = 0.3
        out = add_quality_flags(df)
        self.assertEqual(out.flag_any.tolist(), [True, True, True, False])
        ltt_s = (out.jd_utc - out.jd_ltc).iloc[0] * 86400
        self.assertAlmostEqual(ltt_s, 2.0 * 499.005, delta=0.1)          # 2 au of light time
        self.assertIn("sun_alt", out)

    def test_neargaia_is_informational(self):
        df = _results(2)
        df.loc[0, ["nearest_gaia_dist_arcsec", "nearest_gaia_gmag"]] = [40.0, 14.0]  # far outside the aperture
        out = add_quality_flags(df)
        self.assertTrue(out.flag_neargaia.iloc[0])
        self.assertFalse(out.flag_any.iloc[0])

    def test_old_results_estimate_contamination_from_nearest_star(self):
        df = _results(2).drop(columns=["contam_frac", "zp_local_spread"])
        df["exptime"], df["zp_global"], df["psf_fwhm"], df["source_sum"] = 60.0, 18.6, 2.5, 5000.0
        df.loc[0, ["nearest_gaia_dist_arcsec", "nearest_gaia_gmag"]] = [6.0, 15.0]   # 2 px away, bright
        out = add_quality_flags(df)
        star_flux = 60.0 * 10 ** (-0.4 * (15.0 - 18.6))
        expected = aperture_contamination(6.0 / 2.974, star_flux, 3.6, 2.5) / 5000.0   # ~0.30
        self.assertAlmostEqual(out.contam_frac.iloc[0], expected, places=9)
        self.assertTrue(out.flag_contam.iloc[0])
        self.assertFalse(out.flag_contam.iloc[1])
        self.assertFalse(out.flag_zpspread.any())

    def test_mixed_old_and_new_results_estimate_contamination_for_old_rows(self):
        # Regression: after concatenating 1.0 and 1.1 result files, contam_frac was NaN for the
        # 1.0 rows, so they were never flagged as contaminated.
        old = _old_results(2)
        old.loc[0, ["nearest_gaia_dist_arcsec", "nearest_gaia_gmag"]] = [6.0, 15.0]   # 2 px away, bright
        new = _results(2)
        new.loc[0, "contam_frac"] = 0.05
        out = add_quality_flags(pd.concat([new, old], ignore_index=True))
        self.assertEqual(out.flag_contam.tolist(), [True, False, True, False])
        self.assertEqual(out.contam_frac.iloc[1], 0.0)                     # measured values are kept


class TestBinning(unittest.TestCase):
    def test_bins_clip_outliers_and_add_floor_once(self):
        df = add_quality_flags(_results(12))
        df.loc[3, "gmag_distcorr"] += 0.5                                 # a passing star in one frame
        out = bin_lightcurve(df, window_min=5.0, floor_mag=0.01)
        self.assertEqual(len(out), 2)                                     # 12 one-minute frames -> 2 bins
        self.assertEqual(out.n_clipped.sum(), 1)
        self.assertTrue(np.all(np.abs(out.gmag_distcorr_wmean - 10.0) < 0.03))
        n0 = out.n_obs.iloc[0]
        expected = np.sqrt((0.02**2 + 0.005**2) / n0 + 0.01**2)
        self.assertAlmostEqual(out.mag_err_wmean.iloc[0], expected, places=6)

    def test_three_point_bin_keeps_points_within_their_errors(self):
        # Regression: the MAD of 3 points is often tiny. Two nearly equal points made a 2-sigma
        # third point an "outlier"; on the real data this clipped 8 % of all points.
        df = add_quality_flags(_results(3))
        df["gmag_distcorr"] = [10.000, 10.001, 10.040]                   # errors 0.02 mag
        self.assertEqual(bin_lightcurve(df, window_min=5.0).n_clipped.sum(), 0)
        df.loc[2, "gmag_distcorr"] = 10.5                                 # a real outlier is still clipped
        self.assertEqual(bin_lightcurve(df, window_min=5.0).n_clipped.sum(), 1)

    def test_flagged_points_are_excluded(self):
        df = add_quality_flags(_results(6))
        df.loc[0:2, "flag_any"] = True
        out = bin_lightcurve(df, window_min=10.0)
        self.assertEqual(out.n_obs.sum(), 3)

    def test_mixed_old_and_new_results_are_all_binned(self):
        # Regression: zperr_local is NaN for 1.0 rows after concatenation; their errors became NaN
        # and every 1.0 point was dropped from the bins.
        mixed = add_quality_flags(pd.concat([_results(6), _old_results(6)], ignore_index=True))
        out = bin_lightcurve(mixed, window_min=10.0, floor_mag=0.01).set_index("obsdate")
        self.assertEqual(sorted(out.index), ["20260612", "20260630"])
        self.assertEqual(out.n_obs.sum() + out.n_clipped.sum(), 12)
        old, new = out.loc["20260612"], out.loc["20260630"]
        self.assertAlmostEqual(old.mag_err_wmean, np.sqrt(0.02**2 / old.n_obs + 0.01**2), places=6)
        self.assertAlmostEqual(new.mag_err_wmean, np.sqrt((0.02**2 + 0.005**2) / new.n_obs + 0.01**2), places=6)


if __name__ == "__main__":
    unittest.main()
