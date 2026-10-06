import unittest

import numpy as np
import pandas as pd

from solopy.fitslv2 import FitsLv2

GAIN, RDNOISE = 18.69, 3.7


def _scene(flux=20000.0, fwhm=2.5, x0=50.3, y0=50.7, sky=800.0, shape=(101, 101), seed=3):
    """Gaussian star on a flat sky with Poisson + read noise (ADU), like a 60 s SOLO frame."""
    rng = np.random.default_rng(seed)
    yy, xx = np.indices(shape)
    sigma = fwhm / 2.3548
    star = flux / (2 * np.pi * sigma ** 2) * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2))
    model = sky + star
    data = rng.normal(model, np.sqrt(model / GAIN + (RDNOISE / GAIN) ** 2)).astype(np.float32)
    err = np.sqrt(np.maximum(data, 0) / GAIN + (RDNOISE / GAIN) ** 2)
    return data, err, pd.DataFrame({"x": [x0], "y": [y0]})


class TestPerformPhotometry(unittest.TestCase):
    def setUp(self):
        self.lv2 = FitsLv2()
        self.data, self.err, self.src = _scene()

    def _phot(self, sources=None, mask=None, **kwargs):
        kwargs.setdefault("ap_in_out", (1.5, 3.0, 4.0))
        out = self.lv2.perform_photometry(self.data, self.src if sources is None else sources, exptime=60.0,
                                          err=self.err, mask=mask, fwhm=2.5, **kwargs)
        self.assertIsNotNone(out, "perform_photometry returned None (an exception was logged)")
        return out

    def _mask(self, offsets):
        mask = np.zeros(self.data.shape, dtype=bool)
        for dx, dy in offsets:  # pixels fully inside the r = 3.75 px aperture
            mask[int(round(50.7 + dy)), int(round(50.3 + dx))] = True
        return mask

    def test_flux_is_recovered(self):
        # Regression for 82036f8: sum_aper_area is a Quantity [pix2] and the subtraction raised.
        row = self._phot().iloc[0]
        # A Gaussian keeps 99.8 % of its flux inside r = 1.5 FWHM.
        self.assertAlmostEqual(row["source_sum"] / (20000.0 * 0.998), 1.0, delta=0.01)
        self.assertAlmostEqual(row["aperture_area"], np.pi * 3.75 ** 2, places=6)
        self.assertAlmostEqual(row["mag_inst"], -2.5 * np.log10(row["source_sum"] / 60.0), places=9)

    def test_masked_pixels_leave_aperture_area(self):
        row = self._phot(mask=self._mask([(3, 0)])).iloc[0]
        self.assertAlmostEqual(row["aperture_area"], np.pi * 3.75 ** 2 - 1.0, places=6)
        self.assertEqual(row["nbadpix"], 1.0)

    def test_fully_masked_annulus_does_not_break_the_batch(self):
        # Two sources of equal FWHM share one photutils call; the second one's sky annulus is masked.
        sources = pd.DataFrame({"x": [50.3, 20.0], "y": [50.7, 20.0]})
        mask = np.zeros(self.data.shape, dtype=bool)
        yy, xx = np.indices(mask.shape)
        r = np.hypot(xx - 20.0, yy - 20.0)
        mask[(r >= 7.0) & (r <= 10.5)] = True
        out = self._phot(sources=sources, mask=mask)
        self.assertEqual(len(out), 2)
        self.assertFalse(out.iloc[0]["badphot"])
        self.assertTrue(out.iloc[1]["badphot"])  # no sky estimate -> no valid flux

    # --- badphot = masked fraction of the aperture above badpix_frac_max (default 5 %) ---
    def test_small_masked_fraction_is_not_flagged(self):
        row = self._phot(mask=self._mask([(3, 0)])).iloc[0]           # 1 px of 44.2 px = 2.3 %
        self.assertAlmostEqual(row["badpix_frac"], 1.0 / (np.pi * 3.75 ** 2), places=6)
        self.assertFalse(row["badphot"])

    def test_large_masked_fraction_is_flagged(self):
        row = self._phot(mask=self._mask([(3, 0), (-3, 0), (0, 3)])).iloc[0]  # 3 px = 6.8 %
        self.assertGreater(row["badpix_frac"], 0.05)
        self.assertTrue(row["badphot"])

    def test_zero_threshold_reproduces_old_any_pixel_rule(self):
        row = self._phot(mask=self._mask([(3, 0)]), badpix_frac_max=0.0).iloc[0]
        self.assertTrue(row["badphot"])

    def test_unmasked_source_has_zero_fraction(self):
        row = self._phot(mask=np.zeros(self.data.shape, dtype=bool)).iloc[0]
        self.assertEqual(row["badpix_frac"], 0.0)
        self.assertFalse(row["badphot"])

    # --- saturation from the Lv1 bit mask (robustness review R1) ---
    def test_single_saturated_pixel_always_flags(self):
        from solopy import maskbits
        bits = np.zeros(self.data.shape, dtype=np.uint8)
        bits[51, 50] = maskbits.SATURATED                       # 1 core pixel = 2.3 % of the aperture
        row = self._phot(mask=bits).iloc[0]
        self.assertLess(row["badpix_frac"], 0.05)
        self.assertTrue(row["saturated"])
        self.assertTrue(row["badphot"])

    def test_single_hot_pixel_does_not_flag(self):
        from solopy import maskbits
        bits = np.zeros(self.data.shape, dtype=np.uint8)
        bits[51, 50] = maskbits.BADPIX
        row = self._phot(mask=bits).iloc[0]
        self.assertFalse(row["saturated"])
        self.assertFalse(row["badphot"])

    def test_aperture_scales_with_tile_fwhm(self):
        psf_table = pd.DataFrame({"region_i": [0], "region_j": [0], "fwhm_avg": [3.0]})
        row = self._phot(psf_table=psf_table, base_tile_size=500).iloc[0]
        self.assertAlmostEqual(row["mapped_fwhm"], 3.0)
        self.assertAlmostEqual(row["r_ap_pixel"], 4.5)


if __name__ == "__main__":
    unittest.main()
