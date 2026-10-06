import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from solopy import fitslv1


def _star_field(shape=(256, 256), n_star=25, seed=1):
    rng = np.random.default_rng(seed)
    yy, xx = np.indices(shape)
    img = rng.normal(800.0, 8.0, shape)
    for x0, y0, amp in zip(rng.uniform(20, shape[1] - 20, n_star),
                           rng.uniform(20, shape[0] - 20, n_star),
                           rng.uniform(300, 3000, n_star)):
        img += amp * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * 1.1 ** 2))
    return img.astype(np.float32)


def _tan_wcs(shape):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [335.0, -9.9]
    wcs.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    wcs.wcs.cdelt = [-2.97 / 3600, 2.97 / 3600]
    return wcs


class _FakeSolver:
    """Stands in for astrometry.Solver: a context manager whose solve() succeeds or fails on demand."""

    def __init__(self, solved, wcs):
        self.solved, self.wcs = solved, wcs

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def solve(self, **kwargs):
        match = mock.Mock(center_ra_deg=335.0, center_dec_deg=-9.9, scale_arcsec_per_pixel=2.97)
        match.astropy_wcs.return_value = self.wcs
        return mock.Mock(has_match=mock.Mock(return_value=self.solved), best_match=mock.Mock(return_value=match))


class TestUpdateWcs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        data = _star_field()
        hdr = fits.Header({"BUNIT": "adu", "RA": 335.0, "DEC": -9.9, "JD": 2461221.97,
                           "LAT": 37.07, "LON": -119.4, "ELEVAT": 1.405})
        self.fpath = self.dir / "dawn_field1_001_20260630010758.fits"
        fits.writeto(self.fpath, data, hdr)
        self.wcs = _tan_wcs(data.shape)

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, solved):
        with mock.patch.object(fitslv1.astrometry, "Solver", lambda files: _FakeSolver(solved, self.wcs)), \
             mock.patch.object(fitslv1.astrometry.series_4100, "index_files", return_value=[]):
            return fitslv1.FitsLv1().update_wcs(self.fpath, self.dir / "out")

    def test_unsolved_frame_returns_none_and_writes_nothing(self):
        # Regression for primitive_repo.md §8 #7: the old code wrote <stem>.wcs.fits and returned its path.
        self.assertIsNone(self._run(solved=False))
        self.assertEqual(list((self.dir / "out").glob("*.wcs.fits")), [])

    def test_solved_frame_returns_path_with_wcs(self):
        out = self._run(solved=True)
        self.assertIsNotNone(out)
        hdr = fits.getheader(out)
        self.assertAlmostEqual(hdr["RACEN"], 335.0, places=2)
        self.assertIn("ALTCEN", hdr)
        self.assertEqual(hdr["LV0FILE"], self.fpath.name)
        self.assertEqual(hdr["PIXSCALE"], 2.97)


if __name__ == "__main__":
    unittest.main()
