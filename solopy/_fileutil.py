from pathlib import Path
from datetime import datetime

# macOS writes AppleDouble companions ("._<name>") on exFAT/FAT volumes such as the T7 SSD.
# They are not FITS files; use this pattern as `glob_exclude` for ccdproc.ImageFileCollection.
APPLEDOUBLE_GLOB = "._*"


def is_appledouble(path) -> bool:
    """True for macOS AppleDouble metadata files (names starting with '._')."""
    return Path(path).name.startswith("._")


def clear_dir(fpath):
    '''
    Clear all the files inside the fpath. `fpath` should be directory.
    '''

    directory = Path(fpath)

    if directory.exists() and directory.is_dir():
        for item in directory.rglob('*'):
            item.unlink() if item.is_file() else item.rmdir()

        print(f"{directory} has been cleared out. ({datetime.now()})")

    else:
        print(f"{directory} does not exist.")
