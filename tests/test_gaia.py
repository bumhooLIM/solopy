import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.wcs import WCS

from solopy.gaia import GaiaQuery

DTYPE = [("source_id", "<i8"), ("phot_bp_mean_mag", "<f4"), ("phot_rp_mean_mag", "<f4"),
         ("phot_g_mean_mag", "<f4"), ("ra", "<f8"), ("dec", "<f8")]


def _cap_points(rng, ra0, dec0, radius_deg, n):
    """Points distributed uniformly inside a spherical cap."""
    t = np.arccos(rng.uniform(np.cos(np.radians(radius_deg)), 1.0, n))
    b = rng.uniform(0.0, 2.0 * np.pi, n)
    lat1, lon1 = np.radians(dec0), np.radians(ra0)
    lat2 = np.arcsin(np.sin(lat1) * np.cos(t) + np.cos(lat1) * np.sin(t) * np.cos(b))
    lon2 = lon1 + np.arctan2(np.sin(b) * np.sin(t) * np.cos(lat1), np.cos(t) - np.sin(lat1) * np.sin(lat2))
    return np.degrees(lon2) % 360.0, np.degrees(lat2)


def _sep_deg(ra1, dec1, ra2, dec2):
    ra1, dec1, ra2, dec2 = map(np.radians, (ra1, dec1, ra2, dec2))
    h = np.sin((dec2 - dec1) / 2) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin((ra2 - ra1) / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(h)))


def _catalog(seed=0):
    rng = np.random.default_rng(seed)
    ra_bg = rng.uniform(0, 360, 200_000)
    dec_bg = np.degrees(np.arcsin(rng.uniform(-1, 1, 200_000)))
    ra_f, dec_f = _cap_points(rng, 335.0, -10.0, 6.0, 60_000)   # dense ecliptic-like field
    ra_w, dec_w = _cap_points(rng, 0.5, 2.0, 6.0, 20_000)       # field straddling RA = 0
    # close companions (5-15") to 800 field stars, so isolation flags are exercised
    pick = rng.choice(60_000, 800, replace=False)
    ra_c, dec_c = [], []
    for k in pick:
        r, d = _cap_points(rng, ra_f[k], dec_f[k], 15.0 / 3600.0, 1)
        ra_c.append(r[0]); dec_c.append(d[0])
    ra = np.concatenate([ra_bg, ra_f, ra_w, ra_c])
    dec = np.concatenate([dec_bg, dec_f, dec_w, dec_c])
    cat = np.zeros(ra.size, dtype=DTYPE)
    cat["source_id"] = np.arange(ra.size)
    cat["ra"], cat["dec"] = ra, dec
    cat["phot_g_mean_mag"] = rng.uniform(10.0, 18.5, ra.size)
    cat["phot_bp_mean_mag"] = cat["phot_g_mean_mag"] + 0.4
    cat["phot_rp_mean_mag"] = cat["phot_g_mean_mag"] - 0.6
    return cat


def _tan(ra0, dec0, n=4096, scale_arcsec=2.98):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [ra0, dec0]
    wcs.wcs.crpix = [n / 2 + 0.5, n / 2 + 0.5]
    wcs.wcs.cdelt = [-scale_arcsec / 3600.0, scale_arcsec / 3600.0]
    wcs.pixel_shape = (n, n)
    return wcs


class TestCapBoxes(unittest.TestCase):
    def test_boxes_contain_the_cap(self):
        rng = np.random.default_rng(1)
        for ra0, dec0 in [(335.0, -10.0), (0.5, 2.0), (359.8, -40.0), (120.0, 80.0), (10.0, 87.0)]:
            boxes = GaiaQuery.cap_boxes(ra0, dec0, 4.5)
            ra, dec = _cap_points(rng, ra0, dec0, 4.5 - 1e-6, 20_000)
            self.assertTrue(GaiaQuery.in_boxes(boxes, ra, dec).all(), (ra0, dec0))

    def test_ra_wrap_is_split(self):
        boxes = GaiaQuery.cap_boxes(0.5, 2.0, 4.5)
        self.assertEqual(len(boxes), 2)
        self.assertTrue(all(0.0 <= b[0] <= b[1] <= 360.0 for b in boxes))

    def test_cap_containing_pole_spans_all_ra(self):
        self.assertEqual(GaiaQuery.cap_boxes(10.0, 87.0, 4.5), [[0.0, 360.0, 82.5, 90.0]])


class TestNightlySubset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.cat = _catalog()
        cls.path = Path(cls.tmp.name) / "gaia_full.npy"
        np.save(cls.path, cls.cat)
        cls.full_iso = GaiaQuery.isolation_flags(cls.cat["ra"], cls.cat["dec"], 20.0)
        # two dawn pointings and one field across RA = 0, as read from Lv0 headers
        cls.subset, cls.boxes = GaiaQuery.build_nightly_subset(cls.path, [335.0, 335.3, 0.5], [-10.0, -9.8, 2.0])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_rows_equal_brute_force_selection(self):
        merged = GaiaQuery._merge_bounding_boxes([list(b) for b in self.boxes])
        expected = self.cat["source_id"][GaiaQuery.in_boxes(merged, self.cat["ra"], self.cat["dec"])]
        np.testing.assert_array_equal(np.sort(self.subset["source_id"]), np.sort(expected))
        inside_raw = GaiaQuery.in_boxes(self.boxes, self.cat["ra"], self.cat["dec"])
        self.assertTrue(np.isin(self.cat["source_id"][inside_raw], self.subset["source_id"]).all())

    def test_chunked_pass_equals_single_pass(self):
        chunked = GaiaQuery.build_subset(self.path, self.boxes, chunk_rows=7_919)
        for name in self.subset.dtype.names:
            np.testing.assert_array_equal(chunked[name], self.subset[name], name)

    def test_isolation_equals_all_sky_flag_inside_fields(self):
        near = _sep_deg(self.subset["ra"], self.subset["dec"], 335.0, -10.0) < 4.0
        ids = self.subset["source_id"][near]
        np.testing.assert_array_equal(self.subset["iso"][near], self.full_iso[ids])
        self.assertLess(self.subset["iso"][near].mean(), 1.0)  # planted close pairs are flagged

    def test_query_gaia_identical_to_all_sky_isolated_catalog(self):
        # The zero-point star list must not change when the nightly subset replaces gaiadr3_20arcsec.npy.
        wcs = _tan(335.4, -9.6)
        full = GaiaQuery.query_gaia(wcs, self.cat[self.full_iso], 18.0, 12.0)
        night = GaiaQuery.query_gaia(wcs, self.subset[self.subset["iso"]], 18.0, 12.0)
        self.assertGreater(len(full), 100)
        np.testing.assert_array_equal(np.sort(full["source_id"]), np.sort(night["source_id"]))

    def test_footprint_coverage(self):
        self.assertTrue(GaiaQuery.footprint_covered(self.boxes, _tan(335.0, -11.9)))
        far = _tan(335.0, -13.0)
        self.assertFalse(GaiaQuery.footprint_covered(self.boxes, far))
        self.assertTrue(GaiaQuery.footprint_covered(self.boxes + GaiaQuery.wcs_boxes(far), far))

    def test_field_across_ra_zero_is_complete(self):
        near = _sep_deg(self.cat["ra"], self.cat["dec"], 0.5, 2.0) < 4.4
        self.assertTrue(np.isin(self.cat["source_id"][near], self.subset["source_id"]).all())
        self.assertTrue((self.subset["ra"] > 355).any() and (self.subset["ra"] < 5).any())

    def test_save_and_load(self):
        out = Path(self.tmp.name) / "nightly" / "gaiadr3.2026_0630.npy"
        GaiaQuery.save_subset(out, self.subset, self.boxes, night="2026_0630")
        subset, boxes, meta = GaiaQuery.load_subset(out)
        np.testing.assert_array_equal(subset, self.subset)
        self.assertEqual(boxes, [[float(v) for v in b] for b in self.boxes])
        self.assertEqual(meta["night"], "2026_0630")
        self.assertEqual(meta["n_isolated"], int(self.subset["iso"].sum()))


if __name__ == "__main__":
    unittest.main()
