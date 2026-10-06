import unittest
import warnings

import numpy as np

from solopy.psf import soloPSF
from solopy.region import SOLORegion


class TestSOLORegion(unittest.TestCase):
    def test_4096_frame_has_8x8_tiles_with_wide_last_tile(self):
        reg = SOLORegion((4096, 4096), base_tile_size=500)
        self.assertEqual((reg.num_tiles_x, reg.num_tiles_y), (8, 8))
        last = reg.get_tile_info(7, 7)
        self.assertEqual((last["x_start"], last["x_end"], last["size_x"]), (3500, 4096, 596))
        self.assertEqual(reg.get_tile_info(0, 0)["x_center"], 250.0)

    def test_find_region_caps_at_last_tile(self):
        reg = SOLORegion((4096, 4096), base_tile_size=500)
        self.assertEqual(reg.find_region(4095.9, 10.0), (7, 0))
        self.assertEqual(reg.find_region(499.9, 500.0), (0, 1))


class TestSoloPSF(unittest.TestCase):
    def test_recovers_injected_fwhm_per_tile(self):
        rng = np.random.default_rng(7)
        shape, fwhm = (1000, 1000), 2.6
        sigma = fwhm / 2.3548
        yy, xx = np.indices(shape)
        img = rng.normal(800.0, 6.5, shape)
        # stars on a jittered grid, peaks between peakmin (300) and peakmax (3000) above sky
        for x0 in np.arange(40, 980, 60):
            for y0 in np.arange(40, 980, 60):
                x, y = x0 + rng.uniform(-5, 5), y0 + rng.uniform(-5, 5)
                stamp = slice(int(y) - 12, int(y) + 13), slice(int(x) - 12, int(x) + 13)
                img[stamp] += rng.uniform(800, 2500) * np.exp(
                    -((xx[stamp] - x) ** 2 + (yy[stamp] - y) ** 2) / (2 * sigma ** 2))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            table = soloPSF(init_fwhm=2.5, base_tile_size=500).process_ccd(img.astype(np.float32))
        self.assertEqual(len(table), 4)  # 2 x 2 tiles
        np.testing.assert_allclose(table["fwhm_avg"], fwhm, atol=0.15)
        self.assertTrue((table["n_star"] >= 10).all())
        # fitting warnings stay inside process_ccd; global filters are untouched afterwards
        self.assertFalse(any(issubclass(w.category, RuntimeWarning) for w in caught))
        self.assertNotIn(("ignore", None, RuntimeWarning, None, 0), warnings.filters)


if __name__ == "__main__":
    unittest.main()
