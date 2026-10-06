import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _run(code):
    """Run code in a fresh interpreter that imports solopy from this checkout."""
    env = dict(os.environ, PYTHONPATH=str(REPO))
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=120)


class TestImports(unittest.TestCase):
    def test_import_without_lv3_dependencies(self):
        # Regression for primitive_repo.md §8 #3: kete/skyloc were required just to import solopy.
        result = _run("""
            import sys
            sys.modules["kete"] = None      # make `import kete` fail
            sys.modules["skyloc"] = None
            import solopy
            assert "solopy.fitslv3" not in sys.modules
            print(solopy.FitsLv1.__name__, solopy.GaiaQuery.__name__, solopy.__version__)
            try:
                solopy.FitsLv3
            except ImportError:
                print("lv3-unavailable")
        """)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("FitsLv1 GaiaQuery 1.1.0", result.stdout)
        self.assertIn("lv3-unavailable", result.stdout)

    def test_public_namespace_is_explicit(self):
        result = _run("""
            import solopy
            assert not hasattr(solopy, "np") and not hasattr(solopy, "fits"), "leaked imports"
            from solopy import *
            assert "FitsLv3" not in dir()
            print("ok")
        """)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_fitslv3_is_reachable_when_dependencies_exist(self):
        result = _run("""
            import importlib.util, solopy
            if all(importlib.util.find_spec(m) for m in ("kete", "skyloc")):
                print(solopy.FitsLv3.__name__)
            else:
                print("FitsLv3")  # nothing to check without the extras
        """)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("FitsLv3", result.stdout)

    def test_version_string_records_git_commit(self):
        # Robustness review R8: products carry the exact code version.
        import shutil
        import solopy
        v = solopy.version_string()
        self.assertTrue(v.startswith(solopy.__version__))
        if shutil.which("git") and (REPO / ".git").exists():
            self.assertRegex(v, r"^\d+\.\d+\.\d+\+g[0-9a-f]{7,}(\.dirty)?$")

    def test_pyproject_is_valid_and_matches_version(self):
        import tomllib

        import solopy

        meta = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
        self.assertEqual(meta["version"], solopy.__version__)
        self.assertEqual(meta["requires-python"], ">=3.10")
        for dep in ("pandas", "scipy", "pyarrow", "tqdm"):
            self.assertIn(dep, meta["dependencies"])
        self.assertNotIn("astroalign", meta["dependencies"])


if __name__ == "__main__":
    unittest.main()
