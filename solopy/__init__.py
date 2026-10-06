"""solopy: data reduction for SOLO (Solar system Object Light-curve Observatory)."""

from .fitslv0 import FitsLv0
from .fitslv1 import FitsLv1
from .fitslv2 import FitsLv2
from .combmaster import CombMaster
from .gaia import GaiaQuery, NIGHTLY_SUBSET_RADIUS_DEG
from .psf import soloPSF
from .region import SOLORegion

__version__ = "1.1.0"

# FitsLv3 is left out of __all__ on purpose: `from solopy import *` should not require
# the optional Level-3 dependencies.
__all__ = [
    "FitsLv0", "FitsLv1", "FitsLv2", "CombMaster",
    "GaiaQuery", "NIGHTLY_SUBSET_RADIUS_DEG", "soloPSF", "SOLORegion",
]


def __getattr__(name):
    # FitsLv3 needs the optional Level-3 dependencies (kete, skyloc). Import it on first
    # access so that `import solopy` works without them.
    if name == "FitsLv3":
        from .fitslv3 import FitsLv3
        return FitsLv3
    raise AttributeError(f"module 'solopy' has no attribute {name!r}")
