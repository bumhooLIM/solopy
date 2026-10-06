import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from solopy.combmaster import CombMaster


def _write(path, level, imagetyp, exptime, seed):
    rng = np.random.default_rng(seed)
    hdr = fits.Header({"IMAGETYP": imagetyp, "EXPTIME": exptime, "JD": 2461221.99,
                       "OBSDATE": "20260630", "BUNIT": "adu"})
    fits.writeto(path, (level + rng.normal(0, 0.5, (8, 8))).astype(np.float32), hdr)
    return path


class TestMasterDark(unittest.TestCase):
    def test_master_dark_records_bias_subtraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, masters = Path(tmp) / "raw", Path(tmp) / "masters"
            raw.mkdir()
            bias = [_write(raw / f"bias_{i}.fits", 69.0, "BIAS", 0.01, i) for i in range(5)]
            dark = [_write(raw / f"dark60_{i}.fits", 71.0, "DARK", 60.0, 10 + i) for i in range(5)]

            comb = CombMaster()
            fpath_bias = comb.comb_master_bias(bias, masters, outname="kl4040")
            (fpath_dark,) = comb.comb_master_dark(dark, masters, outname="kl4040")

            hdr = fits.getheader(fpath_dark)
            # Regression for primitive_repo.md §8 #8: master darks claimed BIASCORR = False.
            self.assertIs(hdr["BIASCORR"], True)
            self.assertEqual(hdr["BIASNAME"], fpath_bias.name)
            self.assertEqual(fpath_dark.name, "kl4040.dark.60s.comb.20260630.fits")
            self.assertAlmostEqual(float(np.median(fits.getdata(fpath_dark))), 2.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()
