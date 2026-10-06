import bz2
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from solopy._fileutil import is_appledouble


def _write_appledouble(path):
    # AppleDouble files are small binary blobs that are not FITS (and not bz2).
    Path(path).write_bytes(b"\x00\x05\x16\x07" + os.urandom(4092))


class TestAppleDouble(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_is_appledouble(self):
        self.assertTrue(is_appledouble("/Volumes/T7/x/._dawn_001.fits"))
        self.assertTrue(is_appledouble(Path("._a.fits.bz2")))
        self.assertFalse(is_appledouble("/Volumes/T7/x/dawn_001.fits"))
        self.assertFalse(is_appledouble("kl4040._odd.fits"))

    def test_batch_decompress_skips_appledouble(self):
        from solopy.fitslv0 import FitsLv0

        fits_bytes = self.dir / "frame.fits"
        fits.writeto(fits_bytes, np.zeros((4, 4), np.int16))
        (self.dir / "frame.fits.bz2").write_bytes(bz2.compress(fits_bytes.read_bytes()))
        fits_bytes.unlink()
        _write_appledouble(self.dir / "._frame.fits.bz2")

        with self.assertNoLogs("FitsLv0", level="ERROR"):
            FitsLv0().batch_decompress(self.dir, self.dir, delete_source=True)

        self.assertTrue((self.dir / "frame.fits").exists())
        self.assertFalse((self.dir / "._frame.fits").exists())
        self.assertFalse((self.dir / "frame.fits.bz2").exists())   # source deleted after success
        self.assertTrue((self.dir / "._frame.fits.bz2").exists())  # AppleDouble left untouched

    def test_select_master_ignores_appledouble(self):
        from solopy.fitslv1 import FitsLv1

        hdr = fits.Header({"IMAGETYP": "BIAS", "JD": 2461221.99, "BUNIT": "adu"})
        fits.writeto(self.dir / "kl4040.bias.comb.20260630.fits", np.full((4, 4), 69.0, np.float32), hdr)
        _write_appledouble(self.dir / "._kl4040.bias.comb.20260630.fits")

        with self.assertNoLogs("ccdproc", level="WARNING"):
            master = FitsLv1()._select_master(self.dir, "BIAS", 2461222.0)
        self.assertEqual(float(master.data.mean()), 69.0)


if __name__ == "__main__":
    unittest.main()
