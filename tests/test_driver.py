import importlib.util
import logging
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parent.parent


def load_driver():
    """Import notebooks/main.py; its path configuration module `directory` is only used inside main()."""
    # Stub only the `directory` entry: patching all of sys.modules would also drop the astropy and
    # ccdproc modules first imported here, and their re-import would create duplicate classes.
    saved = sys.modules.get("directory")
    sys.modules["directory"] = types.ModuleType("directory")
    try:
        spec = importlib.util.spec_from_file_location("solopy_driver", ROOT / "notebooks" / "main.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if saved is None:
            del sys.modules["directory"]
        else:
            sys.modules["directory"] = saved
    return module


class TestStaleLv1Products(unittest.TestCase):
    # Regression: after the WCS fix one 2026_0630 frame was written as ...346.n09... while the
    # earlier run had named it ...346.n10...; a re-run of level 1 left both, and Lv2/Lv3 used both.

    @classmethod
    def setUpClass(cls):
        cls.driver = load_driver()
        cls.logger = logging.getLogger("test_driver")
        cls.logger.addHandler(logging.NullHandler())
        cls.logger.propagate = False

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.lv1, self.psf, self.zp = base / "lv1", base / "psf", base / "zp"
        for folder in (self.lv1, self.psf, self.zp):
            folder.mkdir()
        self.renamed = self._lv1_file("kl4040.sci.lv1.346.n10.060.20260630084740.fits",
                                      "dawn_field6_003_20260630014810.fits")
        self.other = self._lv1_file("kl4040.sci.lv1.001.n04.060.20260630094051.fits",
                                    "dawn_field1_001_20260630010758.fits")
        (self.lv1 / "._kl4040.sci.lv1.346.n10.060.20260630084740.fits").write_bytes(b"\x00" * 64)

    def tearDown(self):
        self.tmp.cleanup()

    def _lv1_file(self, name, lv0_name):
        hdr = fits.Header({"IMAGETYP": "LIGHT", "LV0FILE": lv0_name})
        fits.PrimaryHDU(np.zeros((4, 4), dtype=np.float32), header=hdr).writeto(self.lv1 / name)
        stem = Path(name).stem
        (self.psf / f"psf.{stem}.csv").write_text("fwhm_avg\n2.5\n")
        (self.zp / f"zp.{stem}.parquet").write_bytes(b"PAR1")
        return self.lv1 / name

    def test_groups_lv1_files_by_their_lv0_frame(self):
        previous = self.driver.previous_lv1_products(self.lv1)
        self.assertEqual(previous, {"dawn_field6_003_20260630014810.fits": [self.renamed],
                                    "dawn_field1_001_20260630010758.fits": [self.other]})

    def test_missing_folder_has_no_previous_products(self):
        self.assertEqual(self.driver.previous_lv1_products(self.lv1.parent / "absent"), {})

    def test_removes_only_that_frame_and_its_tables(self):
        self.driver.remove_lv1_products(self.renamed, self.psf, self.zp, self.logger)
        stem, other = self.renamed.stem, self.other.stem
        self.assertFalse(self.renamed.exists())
        self.assertFalse((self.psf / f"psf.{stem}.csv").exists())
        self.assertFalse((self.zp / f"zp.{stem}.parquet").exists())
        self.assertTrue(self.other.exists())
        self.assertTrue((self.psf / f"psf.{other}.csv").exists())
        self.assertTrue((self.zp / f"zp.{other}.parquet").exists())
        # tables already gone are not an error
        self.driver.remove_lv1_products(self.renamed, self.psf, self.zp, self.logger)


if __name__ == "__main__":
    unittest.main()
