import logging
from pathlib import Path
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import FITSFixedWarning
import astropy.units as u
from astropy.nddata import CCDData
from astropy.stats import mad_std
import ccdproc as ccdp
import numpy as np
import warnings
from datetime import datetime
# from tqdm import tqdm
from . import _fileutil as fileutil

warnings.filterwarnings("ignore", category=FITSFixedWarning)

class CombMaster:
    """
    Class to create master calibration frames (bias, dark, flat) for RASA lcpy KL4040.

    Provides methods to combine raw calibration frames into master frames,
    with automated metadata handling, logging, and temporary file management.
    """

    def __init__(self, log_file: str = None):
        """
        Initialize CombMaster with optional file logging.

        Parameters
        ----------
        log_file : str, optional
            Path to a log file where processing messages are appended. If None,
            logging will only be output to stdout.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        self.logger.addHandler(handler)
        if log_file:
            file_h = logging.FileHandler(log_file)
            file_h.setFormatter(handler.formatter)
            self.logger.addHandler(file_h)

    def make_mbias(self, bias_frames, master_dir):
        """
        Combine multiple bias frames into a single master bias frame.

        Parameters
        ----------
        bias_frames : list of str or pathlib.Path
            Paths to individual raw bias FITS frames to combine.
        master_dir : str or pathlib.Path
            Directory where the master bias file will be saved. Created if it does not exist.

        Returns
        -------
        pathlib.Path or None
            Path to the generated master bias FITS file, or None if no bias_frames supplied.

        Side Effects
        ------------
        - Writes a FITS file named 'kl4040.bias.comb.<obsdate>.fits' in master_dir.
        - Logs progress and any errors.
        """
        self.logger.info("Starting master bias combination...")
        master_dir = Path(master_dir)
        master_dir.mkdir(parents=True, exist_ok=True)
        if not bias_frames:
            self.logger.warning("No bias frames provided.")
            return None
        hdr0 = fits.getheader(bias_frames[0])
        obsdate = hdr0.get('OBSDATE', Time(hdr0['JD'], format='jd').to_value('iso'))
        out_name = master_dir / f"kl4040.bias.comb.{obsdate}.fits"
        mbias = ccdp.combine(
            bias_frames,
            method='median',
            sigma_clip=True,
            sigma_clip_low_thresh=5,
            sigma_clip_high_thresh=5,
            sigma_clip_func=np.ma.median,
            sigma_clip_dev_func=mad_std,
            mem_limit=500e6,
            dtype=np.float32
        )
        mbias.meta.update({
            'COMBINED': True,
            'NCOMBINE': len(bias_frames),
            'IMAGETYP': 'BIAS',
            'HISTORY': f"({datetime.now().isoformat()}) Combined {len(bias_frames)} bias frames." 
        })
        fits.PrimaryHDU(data=mbias.data, header=mbias.meta).writeto(out_name, overwrite=True)
        self.logger.info(f"Master bias saved to {out_name}")
        return out_name

    def make_mdark(self, dark_frames, master_dir, key_exptime='exptime'):
        """
        Create master dark frames by subtracting a master bias and combining per exposure time.

        Parameters
        ----------
        dark_frames : list of str or pathlib.Path
            Paths to raw dark FITS frames for processing.
        master_dir : str or pathlib.Path
            Directory containing master bias and where master dark files will be saved.
        key_exptime : str, optional
            Header keyword for exposure time in dark frames (default 'exptime').

        Returns
        -------
        list of pathlib.Path
            Paths to generated master dark FITS files for each unique exposure time.

        Side Effects
        ------------
        - Reads master bias from master_dir (closest in JD to first dark frame).
        - Writes master dark FITS files named 'kl4040.dark.<exp>s.comb.<obsdate>.fits'.
        - Cleans up temporary bias-subtracted dark files.
        - Logs progress and errors.
        """
        self.logger.info("Starting master dark creation...")
        master_dir = Path(master_dir)
        mbias_coll = ccdp.ImageFileCollection(master_dir, glob_include='*.fits').filter(imagetyp='BIAS')
        if not mbias_coll.files:
            self.logger.error("No master bias found.")
            return []
        if not dark_frames:
            self.logger.warning("No dark frames provided.")
            return []
        hdr0 = fits.getheader(dark_frames[0])
        obs_jd = hdr0['JD']
        jd_series = mbias_coll.summary['jd'].astype(float)
        idx = (abs(jd_series - obs_jd)).idxmin()
        bias_path = Path(mbias_coll.files_filtered(include_path=True)[idx])
        mbias = CCDData.read(bias_path, unit='adu')
        tmp = master_dir / 'tmp'
        fileutil.clear_dir(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        # Subtract bias
        for d in dark_frames:
            dark = CCDData.read(d, unit='adu')
            bdark = ccdp.subtract_bias(dark, mbias)
            bdark.meta['history'] = f"Bias: {bias_path.name}"
            bdark.write(tmp / Path(d).name, overwrite=True)
        bdcoll = ccdp.ImageFileCollection(tmp, glob_include='*.fits')
        exposures = set(bdcoll.summary[key_exptime])
        out_files = []
        for exp in exposures:
            group = [Path(f) for f in bdcoll.files_filtered(**{key_exptime:exp}, include_path=True)]
            out_dark = master_dir / f"kl4040.dark.{int(round(exp))}s.comb.{hdr0['OBSDATE']}.fits"
            mdark = ccdp.combine(group, method='median', sigma_clip=True,
                                 sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                                 sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std,
                                 mem_limit=500e6, dtype=np.float32)
            mdark.meta.update({'COMBINED': True,'NCOMBINE': len(group),'IMAGETYP':'DARK',
                               'HISTORY':f"Combined {len(group)} darks at {datetime.now().isoformat()}"})
            fits.PrimaryHDU(data=mdark.data, header=mdark.meta).writeto(out_dark, overwrite=True)
            self.logger.info(f"Master dark saved to {out_dark}")
            out_files.append(out_dark)
        fileutil.clear_dir(tmp)
        tmp.rmdir()
        self.logger.info("Temporary files cleaned.")
        return out_files