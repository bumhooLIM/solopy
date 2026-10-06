import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time

from solopy.fitslv0 import FitsLv0


def _raw_header(**overrides):
    hdr = fits.Header()
    hdr["EXPTIME"] = 60.0
    hdr["UTC"] = "2026-06-30T08:07:57.444"  # camera writes the end of the exposure
    hdr["RA"] = "22:18:20.1"
    hdr["DEC"] = "-09:49:48.3"
    hdr["ALT"] = "20:30:58"
    hdr["AZ"] = "120:44:35"
    hdr["APDIA"] = 279.4
    hdr["OBJECT"] = "C:\\obs\\dawn_field1"
    hdr["IMAGETYP"] = "Light"
    hdr["CCDTEMP"] = -5.0
    hdr["HISTORY"] = "camera history"
    for key, value in overrides.items():
        hdr[key] = value
    return hdr


class TestUpdateHeader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "dawn_field1_001_20260630010758.fits"

    def tearDown(self):
        self.tmp.cleanup()

    def _update(self, hdr):
        fits.writeto(self.path, np.zeros((4, 4), np.int16), hdr, overwrite=True)
        FitsLv0().update_header(self.path)
        return fits.getheader(self.path)

    def test_times_are_mid_exposure(self):
        hdr = self._update(_raw_header())
        self.assertAlmostEqual(hdr["JD"], Time("2026-06-30T08:07:27.444").jd, places=8)
        self.assertEqual(hdr["DATE-OBS"], "2026-06-30T08:07:27.444")
        self.assertEqual(hdr["UTC-STA"], "2026-06-30T08:06:57.444")
        self.assertEqual(hdr["UTC-END"], "2026-06-30T08:07:57.444")
        self.assertEqual(hdr["OBSDATE"], "20260630")

    def test_normalization(self):
        hdr = self._update(_raw_header())
        self.assertAlmostEqual(hdr["RA"], 334.58375, places=5)
        self.assertEqual(hdr["OBJECT"], "dawn_field1")
        self.assertEqual(hdr["IMAGETYP"], "LIGHT")
        self.assertNotIn("camera history", str(hdr.get("HISTORY", "")))

    def test_aperture_read_from_apdia(self):
        # Regression for primitive_repo.md §8 #8 (APTDIA was always written as 0.0).
        self.assertEqual(self._update(_raw_header())["APTDIA"], 279.4)

    def test_zero_aptdia_from_old_runs_is_repaired(self):
        self.assertEqual(self._update(_raw_header(APTDIA=0.0))["APTDIA"], 279.4)

    def test_idempotent(self):
        first = self._update(_raw_header())
        FitsLv0().update_header(self.path)
        second = fits.getheader(self.path)
        for key in ("JD", "DATE-OBS", "UTC-STA", "UTC-END", "APTDIA", "RA", "DEC"):
            self.assertEqual(first[key], second[key], key)


if __name__ == "__main__":
    unittest.main()
