import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.wcs import WCS

from solopy import maskbits
from solopy.fitslv1 import FitsLv1

N = 300  # small frame: the 100 px border leaves a 100 x 100 interior


def _master(path, imagetyp, level, exptime):
    hdr = fits.Header({"IMAGETYP": imagetyp, "EXPTIME": exptime, "JD": 2461221.97, "BUNIT": "adu",
                       "FILENAME": path.name})
    fits.writeto(path, np.full((N, N), level, np.float32), hdr)


class TestCorrectBdfBitMask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        masters = root / "masters"; masters.mkdir()
        _master(masters / "kl4040.bias.comb.20260630.fits", "BIAS", 69.0, 0.01)
        _master(masters / "kl4040.dark.60s.comb.20260630.fits", "DARK", 2.0, 60.0)

        rng = np.random.default_rng(0)
        raw = rng.normal(800.0 + 71.0, 8.0, (N, N)).astype(np.float32)
        raw[170, 170] = 4000.0                      # saturated raw pixel
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]; wcs.wcs.crval = [335.0, -9.9]
        wcs.wcs.crpix = [N / 2, N / 2]; wcs.wcs.cdelt = [-2.97 / 3600, 2.97 / 3600]
        hdr = wcs.to_header()
        hdr.update({"BUNIT": "adu", "IMAGETYP": "LIGHT", "EXPTIME": 60.0, "JD": 2461221.97,
                    "DATE-OBS": "2026-06-30T08:07:27.444", "RACEN": 335.0, "DECCEN": -9.9})
        frame = root / "dawn_field1_001_20260630010758.wcs.fits"
        fits.writeto(frame, raw, hdr)

        flat = CCDData(np.ones((N, N), np.float32), unit="adu", meta={"FILENAME": "flat.fits"})
        flat.data[150, 150] = 0.3                   # flat defect
        bpm = CCDData(np.zeros((N, N), np.uint8), unit="adu", meta={"FILENAME": "bpm.fits"})
        bpm.data[160, 160] = 1                      # hot pixel from the BPM

        cls.out = FitsLv1().correct_bdf(frame, root / "lv1", masters, ccdmflat=flat, ccdmask=bpm)
        with fits.open(cls.out) as hdul:
            cls.mask, cls.hdr, cls.mhdr = hdul["MASK"].data, hdul[0].header, hdul["MASK"].header

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_each_reason_has_its_bit(self):
        self.assertTrue(self.mask[170, 170] & maskbits.SATURATED)
        self.assertTrue(self.mask[160, 160] & maskbits.BADPIX)
        self.assertTrue(self.mask[150, 150] & maskbits.BADPIX)
        self.assertTrue(self.mask[10, 10] & maskbits.BORDER)
        self.assertEqual(self.mask[120, 130], 0)
        self.assertFalse(self.mask[160, 160] & maskbits.SATURATED)

    def test_headers_document_the_bits(self):
        self.assertEqual(self.mask.dtype, np.uint8)
        for header in (self.hdr, self.mhdr):
            self.assertTrue(maskbits.has_bits(header))
            self.assertEqual(header["MASKB2"], "SATURATED")
        self.assertEqual(self.hdr["NSATPIX"], 1)
        self.assertTrue(self.hdr["SOLOPYV1"].startswith("1.1"))       # provenance (review R8)
        self.assertEqual(self.hdr["NBADPIX"], int(np.count_nonzero(self.mask)))


if __name__ == "__main__":
    unittest.main()
