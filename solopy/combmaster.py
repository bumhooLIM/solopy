import logging
from pathlib import Path
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import FITSFixedWarning
import astropy.units as u
from astropy.nddata import CCDData
from astropy.stats import mad_std
import ccdproc
import numpy as np
import warnings
from datetime import datetime
from . import _utils  # _fileutil 대신 _utils 임포트

warnings.filterwarnings("ignore", category=FITSFixedWarning)

class CombMaster:
    """
    Class to create master calibration frames (bias, dark, flat) for RASA lcpy KL4040.
    Handles .fits and .fits.bz2 inputs. Masters are saved as uncompressed .fits.
    """

    def __init__(self, log_file: str = None):
        """
        Initialize CombMaster with optional file logging.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        self.logger.addHandler(handler)
        if log_file:
            file_h = logging.FileHandler(log_file)
            file_h.setFormatter(handler.formatter)
            self.logger.addHandler(file_h)

    def _load_ccd_list(self, file_list):
        """Helper to load a list of .fits paths into CCDData objects."""
        ccd_list = []
        if not file_list:
            return ccd_list
            
        self.logger.info(f"Loading {len(file_list)} frames...")
        for fpath in file_list:
            fpath = Path(fpath)
            try:
                ccd = CCDData.read(fpath)
                ccd_list.append(ccd)
            except Exception as e:
                self.logger.error(f"Failed to load {fpath.name}: {e}")
        return ccd_list

    def make_mbias(self, bias_frames_paths, master_dir):
        """
        Combine multiple bias frames into a single master bias frame.
        Input: List of paths (.fits or .fits.bz2)
        Output: Master Bias (.fits)
        """
        self.logger.info("Starting master bias combination...")
        master_dir = Path(master_dir)
        master_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Load data from .fits.bz2 files into CCDData objects
        bias_ccds = self._load_ccd_list(bias_frames_paths)
        if not bias_ccds:
            self.logger.error("No bias frames loaded. Aborting make_mbias.")
            return None

        hdr0 = bias_ccds[0].header
        obsdate = hdr0.get('OBSDATE', Time(hdr0['JD'], format='jd').to_datetime().strftime('%Y%m%d'))
        out_name = master_dir / f"kl4040.bias.comb.{obsdate}.fits"

        # 2. Combine the list of CCDData objects
        self.logger.info(f"Combining {len(bias_ccds)} bias CCDData objects...")
        mbias = ccdproc.combine(
            bias_ccds,
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
            'COMBINED': (True, "Combined frame?"),
            'NCOMBINE': (len(bias_ccds), "Number of combined frames"),
            'IMAGETYP': 'BIAS',
            'JD': (hdr0.get('JD', 'NaN'), "Julian Date of first bias"),
            'OBSDATE': (obsdate, "YYYYMMDD observation date"),
            'HISTORY': f"({datetime.now().isoformat()}) Combined {len(bias_ccds)} bias frames."
        })

        # 3. Save as uncompressed .fits
        new_hdu = fits.PrimaryHDU(data=mbias.data, header=mbias.meta)
        new_hdu.writeto(out_name, overwrite=True)
        self.logger.info(f"Master bias saved to {out_name.name}")
        
        return out_name

    def make_mdark(self, dark_frames_paths, master_dir, key_exptime='EXPTIME'):
        """
        Create master dark frames by subtracting a master bias and combining per exposure time.
        Input: List of paths (.fits or .fits.bz2)
        Output: Master Darks (.fits)
        """
        self.logger.info("Starting master dark creation...")
        master_dir = Path(master_dir)
        
        # 1. Load Master Bias (uncompressed .fits)
        mbias_coll = ccdproc.ImageFileCollection(master_dir, glob_include='*.fits').filter(imagetyp='BIAS')
        if not mbias_coll.files:
            self.logger.error("No master bias found in {master_dir}. Run make_mbias first.")
            return []
            
        # 2. Load all dark frames
        dark_ccds = self._load_ccd_list(dark_frames_paths)
        if not dark_ccds:
            self.logger.error("No dark frames loaded. Aborting make_mdark.")
            return []
        
        # 3. Find closest bias
        hdr0_dark = dark_ccds[0].header
        obs_jd = hdr0_dark['JD']
        df_bias = mbias_coll.summary.to_pandas()
        jd_key = 'jd' if 'jd' in df_bias.columns else 'JD'
        jd_series = df_bias[jd_key].astype(float)
        idx = (abs(jd_series - obs_jd)).idxmin()
        bias_path = Path(mbias_coll.files_filtered(include_path=True)[idx])
        mbias = CCDData.read(bias_path, unit='adu')
        self.logger.info(f"Using master bias: {bias_path.name}")
        
        # 4. Group darks by exposure time and subtract bias
        grouped_darks = {}
        for ccd in dark_ccds:
            try:
                # EXPTIME 또는 exptime 키 확인
                if key_exptime.upper() in ccd.header:
                    exptime = float(ccd.header[key_exptime.upper()])
                elif key_exptime.lower() in ccd.header:
                    exptime = float(ccd.header[key_exptime.lower()])
                else:
                    self.logger.warning(f"Cannot find key {key_exptime} in {ccd.meta['FILENAME']}. Skipping.")
                    continue
                
                bdark = ccdproc.subtract_bias(ccd, mbias)
                bdark.meta['history'] = f"Bias: {bias_path.name}"
                if exptime not in grouped_darks:
                    grouped_darks[exptime] = []
                grouped_darks[exptime].append(bdark)
            except Exception as e:
                self.logger.error(f"Failed to bias-subtract {ccd.meta.get('FILENAME', 'unknown file')}: {e}")

        # 5. Combine groups and save
        out_files = []
        for exp, ccd_list in grouped_darks.items():
            self.logger.info(f"Combining {len(ccd_list)} darks for exptime {exp}s...")
            obsdate = hdr0_dark.get('OBSDATE', Time(hdr0_dark['JD'], format='jd').to_datetime().strftime('%Y%m%d'))
            out_dark = master_dir / f"kl4040.dark.{int(round(exp))}s.comb.{obsdate}.fits"
            
            mdark = ccdproc.combine(ccd_list, method='median', sigma_clip=True,
                                 sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                                 sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std,
                                 mem_limit=500e6, dtype=np.float32)
            
            mdark.meta.update({'COMBINED': True,'NCOMBINE': len(ccd_list),'IMAGETYP':'DARK',
                               'EXPTIME': (exp, "Exposure Time (sec)"),
                               'JD': (hdr0_dark.get('JD', 'NaN'), "Julian Date of first dark"),
                               'OBSDATE': (obsdate, "YYYYMMDD observation date"),
                               'HISTORY':f"Combined {len(ccd_list)} darks at {datetime.now().isoformat()}"})
            
            new_hdu = fits.PrimaryHDU(data=mdark.data, header=mdark.meta)
            new_hdu.writeto(out_dark, overwrite=True)
            self.logger.info(f"Master dark saved to {out_dark.name}")
            out_files.append(out_dark)
            
        return out_files

    # def make_mflat(self, flat_frames_paths, master_dir, filter_name, key_exptime='EXPTIME'):
    #     """
    #     Create a master flat frame by subtracting bias and dark.
    #     Input: List of paths (.fits or .fits.bz2)
    #     Output: Master Flat (.fits)
    #     """
    #     self.logger.info(f"Starting master flat creation for filter {filter_name}...")
    #     master_dir = Path(master_dir)
        
    #     # 1. Load flats
    #     flat_ccds = self._load_ccd_list(flat_frames_paths)
    #     if not flat_ccds:
    #         self.logger.error("No flat frames loaded. Aborting make_mflat.")
    #         return None

    #     # 2. Load Master Bias
    #     mbias_coll = ccdproc.ImageFileCollection(master_dir, glob_include='*.fits').filter(imagetyp='BIAS')
    #     if not mbias_coll.files:
    #         self.logger.error(f"No master bias found in {master_dir}. Run make_mbias first.")
    #         return None
        
    #     # 3. Load Master Darks (dict by exptime)
    #     mdark_coll = ccdproc.ImageFileCollection(master_dir, glob_include='*.fits').filter(imagetyp='DARK')
    #     if not mdark_coll.files:
    #         self.logger.warning(f"No master darks found in {master_dir}. Proceeding without dark subtraction for flats.")
    #         mdarks = {}
    #     else:
    #         mdarks = {float(h['EXPTIME']): ccdproc.CCDData.read(p, unit='adu') 
    #                   for p, h in mdark_coll.headers(include_path=True).items()}

    #     # 4. Find closest bias
    #     hdr0_flat = flat_ccds[0].header
    #     obs_jd = hdr0_flat['JD']
    #     jd_series = mbias_coll.summary['jd'].astype(float)
    #     idx = (abs(jd_series - obs_jd)).idxmin()
    #     bias_path = Path(mbias_coll.files_filtered(include_path=True)[idx])
    #     mbias = CCDData.read(bias_path, unit='adu')
    #     self.logger.info(f"Using master bias: {bias_path.name}")

    #     # 5. Process flats (bias and dark subtraction)
    #     processed_flats = []
    #     for ccd in flat_ccds:
    #         try:
    #             bflat = ccdproc.subtract_bias(ccd, mbias)
                
    #             # EXPTIME 또는 exptime 키 확인
    #             if key_exptime.upper() in ccd.header:
    #                 exptime = float(ccd.header[key_exptime.upper()])
    #             elif key_exptime.lower() in ccd.header:
    #                 exptime = float(ccd.header[key_exptime.lower()])
    #             else:
    #                 self.logger.warning(f"Cannot find key {key_exptime} in {ccd.meta['FILENAME']}. Skipping dark subtraction.")
    #                 processed_flats.append(bflat)
    #                 continue

    #             # Find closest dark
    #             if not mdarks:
    #                 processed_flats.append(bflat) # No darks available
    #                 continue
                    
    #             closest_exp_dark = min(mdarks.keys(), key=lambda k: abs(k - exptime))
    #             mdark = mdarks[closest_exp_dark]
    #             if abs(closest_exp_dark - exptime) > 1.0: # 1초 이상 차이나면 경고
    #                 self.logger.warning(f"Using dark {closest_exp_dark}s for flat {exptime}s.")
                
    #             bdflat = ccdproc.subtract_dark(bflat, mdark, exposure_time=key_exptime, exposure_unit=u.second)
    #             processed_flats.append(bdflat)
                
    #         except Exception as e:
    #             self.logger.error(f"Failed to process flat {ccd.meta.get('FILENAME', 'unknown file')}: {e}")

    #     # 6. Combine processed flats
    #     if not processed_flats:
    #         self.logger.error("No flat frames were successfully processed.")
    #         return None
            
    #     self.logger.info(f"Combining {len(processed_flats)} processed flat frames...")
        
    #     def inv_median_scale(a):
    #         return 1.0 / np.nanmedian(a)

    #     mflat = ccdproc.combine(processed_flats,
    #                          method='median',
    #                          scale=inv_median_scale, # 스케일링
    #                          sigma_clip=True,
    #                          sigma_clip_low_thresh=5,
    #                          sigma_clip_high_thresh=5,
    #                          sigma_clip_func=np.ma.median,
    #                          sigma_clip_dev_func=mad_std,
    #                          mem_limit=500e6,
    #                          dtype=np.float32
    #                          )
        
    #     obsdate = hdr0_flat.get('OBSDATE', Time(hdr0_flat['JD'], format='jd').to_datetime().strftime('%Y%m%d'))
    #     out_flat = master_dir / f"kl4040.flat.{filter_name}.{obsdate}.fits"
        
    #     mflat.meta.update({'COMBINED': True, 'NCOMBINE': len(processed_flats),
    #                        'FILTER': filter_name, 'IMAGETYP':'FLAT',
    #                        'HISTORY':f"Combined {len(processed_flats)} flats at {datetime.now().isoformat()}"})

    #     new_hdu = fits.PrimaryHDU(data=mflat.data, header=mflat.meta)
    #     _utils.write_fits_any(out_flat, new_hdu, as_bz2=False) # Save as .fits
    #     self.logger.info(f"Master flat saved to {out_flat.name}")
    #     return out_flat