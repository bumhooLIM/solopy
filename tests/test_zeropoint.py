import unittest

import numpy as np

from solopy.zeropoint import SOLAR_BP_RP, fit_color_term, local_zero_points, robust_std


class TestColorTerm(unittest.TestCase):
    def test_recovers_slope_and_solar_zero_point_with_outliers(self):
        rng = np.random.default_rng(1)
        color = rng.uniform(0.5, 1.8, 800)
        zp = 18.60 - 0.063 * (color - SOLAR_BP_RP) + rng.normal(0, 0.02, 800)
        zp[:40] += rng.uniform(0.3, 1.0, 40)                     # 5 % outliers (blends, variables)
        zp_sun, slope, n = fit_color_term(zp, color)
        self.assertAlmostEqual(slope, -0.063, delta=0.006)
        self.assertAlmostEqual(zp_sun, 18.60, delta=0.004)
        self.assertLess(n, 800)

    def test_without_colors_returns_clipped_median(self):
        zp = np.r_[18.5 + np.random.default_rng(4).normal(0, 0.02, 50), 25.0]
        zp_sun, slope, n = fit_color_term(zp, np.full(51, np.nan))
        self.assertAlmostEqual(zp_sun, 18.5, delta=0.01)
        self.assertEqual((slope, n), (0.0, 50))   # outlier clipped, no color slope


class TestLocalZeroPoints(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        self.x, self.y = rng.uniform(100, 3996, (2, 800))
        # 0.10 mag gradient across the field (extinction at airmass ~2.5) + noise + outliers
        self.zp = 18.6 + 0.10 * (self.x / 4096) + rng.normal(0, 0.03, 800)
        self.zp[:20] -= 1.0

    def test_follows_a_gradient_the_global_value_misses(self):
        out = local_zero_points(self.x, self.y, self.zp, [500, 3500], [2000, 2000], radius=500, min_stars=10)
        truth = 18.6 + 0.10 * np.array([500, 3500]) / 4096
        np.testing.assert_allclose(out.zp_local, truth, atol=0.012)
        self.assertFalse(out.zp_local_fallback.any())
        global_zp = np.median(self.zp)
        self.assertGreater(np.max(np.abs(global_zp - truth)), 0.03)
        self.assertTrue((out.zperr_local < 0.01).all())

    def test_spread_reflects_local_scatter(self):
        out = local_zero_points(self.x, self.y, self.zp, [2000], [2000], radius=500)
        self.assertAlmostEqual(out.zp_local_spread.iloc[0], 0.03, delta=0.01)

    def test_falls_back_with_too_few_stars(self):
        left = self.x < 2000
        out = local_zero_points(self.x[left], self.y[left], self.zp[left], [3800], [2000],
                                radius=500, min_stars=10, fallback_zp=18.65, fallback_err=0.002)
        row = out.iloc[0]
        self.assertTrue(row.zp_local_fallback)
        self.assertEqual((row.zp_local, row.zperr_local), (18.65, 0.002))

    def test_empty_star_list_falls_back(self):
        out = local_zero_points([], [], [], [100.0, 200.0], [100.0, 200.0], fallback_zp=18.6)
        self.assertTrue(out.zp_local_fallback.all())
        self.assertTrue((out.zp_local == 18.6).all())

    def test_robust_std_of_gaussian(self):
        self.assertAlmostEqual(robust_std(np.random.default_rng(3).normal(0, 0.05, 20000)), 0.05, delta=0.002)


if __name__ == "__main__":
    unittest.main()
