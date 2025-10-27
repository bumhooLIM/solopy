import bz2
import io
import shutil
from pathlib import Path
from astropy.io import fits
from typing import List
from tqdm import tqdm # Progress bar
import logging

# Set up module-level logger
logger = logging.getLogger(__name__)

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

def batch_decompress(source_dir: Path, dest_dir: Path, delete_source=False):
    """
    source_dir의 모든 .fits.bz2 파일을 dest_dir에 .fits로 압축 해제합니다.

    Parameters
    ----------
    source_dir : Path
        '.fits.bz2' 파일이 있는 원본 디렉터리
    dest_dir : Path
        압축 해제된 '.fits' 파일을 저장할 대상 디렉터리
    delete_source : bool
        압축 해제 성공 시 원본 .fits.bz2 파일을 삭제할지 여부
    """
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    bz2_files = list(source_dir.glob("*.fits.bz2"))
    if not bz2_files:
        logger.warning(f"No .fits.bz2 files found in {source_dir}")
        return
        
    logger.info(f"Decompressing {len(bz2_files)} files from {source_dir} to {dest_dir}...")
    
    for f_bz2 in tqdm(bz2_files, desc="Decompressing"):
        f_fits = dest_dir / (f_bz2.stem) # 'image.fits.bz2' -> 'image.fits'
        try:
            with bz2.BZ2File(f_bz2, 'rb') as fin:
                with open(f_fits, 'wb') as fout:
                    shutil.copyfileobj(fin, fout)
            if delete_source:
                f_bz2.unlink()
        except Exception as e:
            logger.error(f"Failed to decompress {f_bz2.name}: {e}")

def batch_compress(source_dir: Path, dest_dir: Path, delete_source=False):
    """
    source_dir의 모든 .fits 파일을 dest_dir에 .fits.bz2로 압축합니다.

    Parameters
    ----------
    source_dir : Path
        '.fits' 파일이 있는 원본 디렉터리
    dest_dir : Path
        압축된 '.fits.bz2' 파일을 저장할 대상 디렉터리
    delete_source : bool
        압축 성공 시 원본 .fits 파일을 삭제할지 여부
    """
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    fits_files = list(source_dir.glob("*.fits"))
    if not fits_files:
        logger.warning(f"No .fits files found in {source_dir}")
        return

    logger.info(f"Compressing {len(fits_files)} files from {source_dir} to {dest_dir}...")

    for f_fits in tqdm(fits_files, desc="Compressing"):
        f_bz2 = dest_dir / (f_fits.name + ".bz2") # 'image.fits' -> 'image.fits.bz2'
        try:
            with open(f_fits, 'rb') as fin:
                with bz2.BZ2File(f_bz2, 'wb') as fout:
                    shutil.copyfileobj(fin, fout)
            if delete_source:
                f_fits.unlink()
        except Exception as e:
            logger.error(f"Failed to compress {f_fits.name}: {e}")