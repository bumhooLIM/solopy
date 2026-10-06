import subprocess
from functools import lru_cache
from pathlib import Path

__version__ = "1.1.0"


@lru_cache(maxsize=1)
def version_string():
    """
    Package version, plus '+g<commit>' (and '.dirty' for uncommitted changes) when solopy runs
    from a git checkout, e.g. '1.1.0+g6cdcccc'. Written into products for provenance.
    """
    root = Path(__file__).resolve().parents[1]
    try:
        def git(*args):
            return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                                  timeout=5, check=True).stdout.strip()
        sha = git("rev-parse", "--short", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
        return f"{__version__}+g{sha}{'.dirty' if dirty else ''}"
    except Exception:
        return __version__
