import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from solopy.fitslv1 import FitsLv1


def _dark(path, ccdtemp, jd, level):
    hdr = fits.Header({"IMAGETYP": "DARK", "EXPTIME": 60.0, "JD": jd, "CCDTEMP": ccdtemp, "BUNIT": "adu"})
    fits.writeto(path, np.full((4, 4), level, np.float32), hdr)


class TestDarkTemperatureMatching(unittest.TestCase):
    """Robustness review R9: prefer master darks taken at the frame's CCD temperature."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        _dark(self.dir / "kl4040.dark.60s.comb.20260601.fits", -10.0, 2461193.0, 1.0)  # close in time, cold
        _dark(self.dir / "kl4040.dark.60s.comb.20260604.fits", -5.0, 2461196.0, 2.0)   # later, warmer

    def tearDown(self):
        self.tmp.cleanup()

    def test_prefers_master_within_one_degree(self):
        lv1 = FitsLv1()
        master = lv1._select_master(self.dir, "DARK", 2461193.2, 60.0, ccdtemp=-5.3)
        self.assertEqual(float(master.data.mean()), 2.0)            # the -5 C master, despite being 3 days later
        self.assertAlmostEqual(lv1.master_dtemp["DARK"], -0.3, places=6)

    def test_falls_back_to_closest_in_time_with_warning(self):
        lv1 = FitsLv1()
        with self.assertLogs("FitsLv1", level="WARNING"):
            master = lv1._select_master(self.dir, "DARK", 2461193.2, 60.0, ccdtemp=-7.5)
        self.assertEqual(float(master.data.mean()), 1.0)            # nearest in time
        self.assertAlmostEqual(lv1.master_dtemp["DARK"], 2.5, places=6)

    def test_without_temperature_behaves_as_before(self):
        lv1 = FitsLv1()
        self.assertEqual(float(lv1._select_master(self.dir, "DARK", 2461193.2, 60.0).data.mean()), 1.0)
        self.assertTrue(np.isnan(lv1.master_dtemp["DARK"]))


if __name__ == "__main__":
    unittest.main()
