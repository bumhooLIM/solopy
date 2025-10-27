import bz2
import io
from pathlib import Path
from astropy.io import fits
from astropy.io.fits import HDUList, PrimaryHDU # <-- 수정
from typing import Union # <-- 추가

def is_bz2(path: Path) -> bool:
    """Check if a file path has a .bz2 extension."""
    return ".bz2" in "".join(path.suffixes)

def open_fits_any(path: Path):
    """
    Open .fits or .fits.bz2.

    Returns
    -------
    hdul : fits.HDUList
    was_bz2 : bool
    fobj : file-like or None   # keep this open until after reading data
    """
    path = Path(path) # Ensure it's a Path object
    if is_bz2(path):
        fobj = bz2.BZ2File(path, "rb")
        hdul = fits.open(fobj, memmap=False)
        return hdul, True, fobj
    else:
        hdul = fits.open(path, memmap=False)
        return hdul, False, None

# def write_fits_any(path: Path, hdu: fits.PrimaryHDU, as_bz2: bool):
#     """
#     Write a PrimaryHDU to .fits or .fits.bz2.
#     If as_bz2 is True, .fits.bz2 extension is enforced.
#     """
#     path = Path(path)
    
#     if as_bz2:
#         # ".fits.bz2" 확장자 보장
#         if not path.name.endswith(".fits.bz2"):
#             # ".fits" 확장자라면 ".bz2" 추가
#             if path.name.endswith(".fits"):
#                 path = path.with_suffix(path.suffix + '.bz2')
#             # 확장자가 없다면 ".fits.bz2" 추가
#             else:
#                 path = path.with_suffix(path.suffix + '.fits.bz2')
        
#         buf = io.BytesIO()
#         hdu.writeto(buf, overwrite=True)
#         with open(path, "wb") as f:
#             f.write(bz2.compress(buf.getvalue()))
#     else:
#         # ".fits.bz2" 확장자가 있다면 제거 (예: ".fits"로 저장)
#         if path.name.endswith(".fits.bz2"):
#             path = path.with_suffix('').with_suffix('.fits')
            
#         hdu.writeto(path, overwrite=True)

def write_fits_any(path: Path, hdu_or_hdulist: Union[PrimaryHDU, HDUList], as_bz2: bool): # <-- 타입 힌트 수정
    """
    Write a PrimaryHDU or HDUList to .fits or .fits.bz2.
    If as_bz2 is True, .fits.bz2 extension is enforced.
    """
    path = Path(path)
    
    if as_bz2:
        # ".fits.bz2" 확장자 보장
        if not path.name.endswith(".fits.bz2"):
            if path.name.endswith(".fits"):
                path = path.with_suffix(path.suffix + '.bz2')
            else:
                path = path.with_suffix(path.suffix + '.fits.bz2')
        
        buf = io.BytesIO()
        hdu_or_hdulist.writeto(buf, overwrite=True) # <-- 이 코드는 이미 둘 다 지원함
        with open(path, "wb") as f:
            f.write(bz2.compress(buf.getvalue()))
    else:
        if path.name.endswith(".fits.bz2"):
            path = path.with_suffix('').with_suffix('.fits')
            
        hdu_or_hdulist.writeto(path, overwrite=True) # <-- 이 코드는 이미 둘 다 지원함
        
def get_true_stem(path: Path) -> str:
    """
    파일 경로에서 '.fits' 또는 '.fits.bz2' 확장자를 제거한 순수 파일 이름을 반환합니다.
    e.g., 'image.fits.bz2' -> 'image'
    e.g., 'image.fits' -> 'image'
    """
    name = path.name
    if name.endswith('.fits.bz2'):
        return name[:-9] # '.fits.bz2' (9글자) 제거
    if name.endswith('.fits'):
        return name[:-5] # '.fits' (5글자) 제거
    return path.stem # Fallback