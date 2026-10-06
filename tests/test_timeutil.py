import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

from solopy._timeutil import utc_jd_to_tdb

HAS_LV3_DEPS = all(importlib.util.find_spec(m) is not None for m in ("kete", "skyloc"))
JD_UTC = 2461221.9742215974  # mid-exposure of a 2026-06-30 frame


class TestUtcToTdb(unittest.TestCase):
    def test_offset_in_2026(self):
        # TDB - UTC = 37 leap seconds + 32.184 s (+ a periodic term below 2 ms)
        dt_sec = (utc_jd_to_tdb(JD_UTC) - JD_UTC) * 86400.0
        self.assertAlmostEqual(dt_sec, 69.184, delta=0.01)

    def test_accepts_arrays(self):
        out = utc_jd_to_tdb(np.array([JD_UTC, JD_UTC + 1.0]))
        self.assertEqual(out.shape, (2,))


@unittest.skipUnless(HAS_LV3_DEPS, "kete/skyloc not installed")
class TestPredictTargetsUsesTdb(unittest.TestCase):
    def test_observer_state_requested_in_tdb(self):
        from solopy import fitslv3

        with tempfile.TemporaryDirectory() as tmp:
            wcs = WCS(naxis=2)
            wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            wcs.wcs.crval = [335.0, -9.9]
            wcs.wcs.crpix = [2048.0, 2048.0]
            wcs.wcs.cdelt = [-2.97 / 3600, 2.97 / 3600]
            hdr = wcs.to_header()
            hdr["JD"] = JD_UTC
            hdr["LAT"], hdr["LON"], hdr["ELEVAT"] = 37.07, -119.4, 1.405
            fpath = Path(tmp) / "frame.fits"
            fits.writeto(fpath, np.zeros((8, 8), np.float32), hdr)

            proc = fitslv3.FitsLv3.__new__(fitslv3.FitsLv3)  # skip loading orbit/Gaia files
            proc.logger = mock.Mock()
            proc.orb = None
            proc.gaia_all = None

            seen = {}

            def fake_observer(jd, *args, **kwargs):
                seen["jd"] = jd
                return object()

            empty = mock.Mock(eph=pd.DataFrame({"vmag": []}))
            with mock.patch.object(fitslv3.kete.spice, "earth_pos_to_ecliptic", side_effect=fake_observer), \
                 mock.patch.object(fitslv3.kete.fov.RectangleFOV, "from_wcs", return_value=object()), \
                 mock.patch.object(fitslv3.sloc, "FOVCollection", side_effect=lambda f: f), \
                 mock.patch.object(fitslv3.sloc, "locator_twice", return_value=(None, empty)):
                proc.predict_targets(pd.DataFrame({"file": [str(fpath)]}))

        self.assertAlmostEqual((seen["jd"] - JD_UTC) * 86400.0, 69.184, delta=0.01)


if __name__ == "__main__":
    unittest.main()
