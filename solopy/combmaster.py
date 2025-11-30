import logging
from pathlib import Path
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import FITSFixedWarning
from astropy.nddata import CCDData
from astropy.stats import mad_std
import astropy.units as u
import ccdproc
import numpy as np
import warnings
import sep
from datetime import datetime
from . import _fileutil

warnings.filterwarnings("ignore", category=FITSFixedWarning)
class CombMaster:
    """
    Class to create master calibration frames (i.e., bias, dark, flat) 
    for astronomical ccd frame.
    Handles FITS-like nddata inputs. Masters are saved as .fits.
    """

    def __init__(self, log_file: str = None):
        """
        Initializes the CombMaster class and configures logging.

        Parameters
        ----------
        log_file : str, optional
            Path to a file where logs should be saved, in addition to
            streaming to the console.
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
        """
        Helper to load a list of FITS file paths into CCDData objects.

        Parameters
        ----------
        file_list : list[str or Path]
            A list of file paths to be loaded.

        Returns
        -------
        list[tuple(CCDData, Path)]
            A list of (CCDData, Path) tuples. Frames that fail to load
            are logged and skipped.
        """
        ccd_list_with_paths = []
        if not file_list:
            return ccd_list_with_paths
            
        self.logger.info(f"Loading {len(file_list)} frames...")
        for fpath in file_list:
            fpath = Path(fpath)
            try:
                ccd = CCDData.read(fpath)
                ccd_list_with_paths.append((ccd, fpath))
            except ValueError:
                ccd = CCDData.read(fpath, unit='adu')
                ccd_list_with_paths.append((ccd, fpath))
            except Exception as e:
                self.logger.error(f"Failed to load {fpath.name}: {e}")
        return ccd_list_with_paths

    def _find_closest_bias(self, master_dir, target_jd, verbose=True):
        """
        Find the master bias frame closest in time (JD) to the target.

        Parameters
        ----------
        master_dir : Path
            Directory to search for master bias files.
        target_jd : float
            The Julian Date of the target frame (e.g., dark or flat).

        Returns
        -------
        tuple (CCDData, Path) or (None, None)
            A tuple containing the loaded CCDData object of the master bias
            and its Path, or (None, None) if not found or an error occurs.
        """
        if verbose:
            self.logger.info(f"Searching for closest master bias to JD={target_jd:.1f}...")
    
        try:
            mbias_coll = ccdproc.ImageFileCollection(master_dir, glob_include='*.fits').filter(imagetyp='BIAS')
            if not mbias_coll.files:
                self.logger.error(f"No master bias found in {master_dir}.")
                return None, None
            
            df_bias = mbias_coll.summary.to_pandas()
            
            # Find JD column
            if 'jd' in df_bias.columns:
                jd_key = 'jd'
            elif 'JD' in df_bias.columns:
                jd_key = 'JD'
            else:
                self.logger.error("Could not find 'jd' or 'JD' column in master bias summary.")
                return None, None
                
            df_bias[jd_key] = df_bias[jd_key].astype(float)
            df_bias['diff'] = (df_bias[jd_key] - target_jd).abs()
            idx = df_bias['diff'].idxmin()
            
            fpath_mbias = Path(mbias_coll.files_filtered(include_path=True)[idx])
            
            try:
                mbias = CCDData.read(fpath_mbias)
            except ValueError:
                mbias = CCDData.read(fpath_mbias, unit='adu')
            
            return mbias, fpath_mbias
            
        except Exception as e:
            self.logger.error(f"Failed to find or load closest master bias: {e}")
            return None, None

    def _find_closest_dark(self, master_dir, target_jd, target_exptime, verbose=True):
        """
        Find the master dark frame closest in exposure time, then closest in time (JD).

        Parameters
        ----------
        master_dir : Path
            Directory to search for master dark files.
        target_jd : float
            The Julian Date of the target frame (e.g., flat).
        target_exptime : float
            The exposure time (in seconds) of the target frame.

        Returns
        -------
        tuple (CCDData, Path) or (None, None)
            A tuple containing the loaded CCDData object of the master dark
            and its Path, or (None, None) if not found or an error occurs.
        """
        if verbose:
            self.logger.info(f"Searching for closest master dark to JD={target_jd:.1f}, exp={target_exptime:.1f}s...")
        
        try:
            mdark_coll = ccdproc.ImageFileCollection(master_dir, glob_include='*.fits').filter(imagetyp='DARK')
            if not mdark_coll.files:
                self.logger.warning(f"No master darks found in {master_dir}.")
                return None, None
            
            df_darks = mdark_coll.summary.to_pandas()

            # Find JD column
            if 'jd' in df_darks.columns:
                jd_key = 'jd'
            elif 'JD' in df_darks.columns:
                jd_key = 'JD'
            else:
                self.logger.error("Could not find 'jd' or 'JD' column in master dark summary.")
                return None, None
                
            # Find EXPTIME column
            if 'exptime' in df_darks.columns:
                exp_key = 'exptime'
            elif 'EXPTIME' in df_darks.columns:
                exp_key = 'EXPTIME'
            else:
                self.logger.error("Could not find 'exptime' or 'EXPTIME' column in master dark summary.")
                return None, None
                
            df_darks[jd_key] = df_darks[jd_key].astype(float)
            df_darks[exp_key] = df_darks[exp_key].astype(float)

            # Step 1: Find closest exposure time
            available_exptimes = df_darks[exp_key].unique()
            if len(available_exptimes) == 0:
                 self.logger.warning(f"No master darks with valid exposure times found in {master_dir}.")
                 return None, None
                 
            closest_exptime = min(available_exptimes, key=lambda x: abs(x - target_exptime))
            if abs(closest_exptime - target_exptime) > 1.0: # 1초 이상 차이나면 경고
                self.logger.warning(f"Target exptime {target_exptime}s, using closest master dark exptime {closest_exptime}s.")

            # Step 2: Filter by that exptime
            df_filtered = df_darks[df_darks[exp_key] == closest_exptime].copy()
            if df_filtered.empty:
                self.logger.error(f"Logic error: No darks found after filtering for closest exptime {closest_exptime}s.")
                return None, None

            # Step 3: Find closest JD from the filtered set
            df_filtered['jd_diff'] = (df_filtered[jd_key] - target_jd).abs()
            idx = df_filtered['jd_diff'].idxmin() # This is the index from the original df_darks/collection
            
            fpath_mdark = Path(mdark_coll.files_filtered(include_path=True)[idx])
            
            try:
                mdark = CCDData.read(fpath_mdark)
            except ValueError:
                mdark = CCDData.read(fpath_mdark, unit='adu')
            
            return mdark, fpath_mdark

        except Exception as e:
            self.logger.error(f"Failed to find or load closest master dark: {e}")
            return None, None

    def _ccd_sigmaclip(self, ccd, nsigma=3.0):
        
        data = ccd.data.astype(np.float32)
        bkg = sep.Background(data)    
        thresh = nsigma * bkg.globalrms
        mask_source = (ccd.data > (bkg.back() + thresh)) | (ccd.data < (bkg.back() - thresh))
        
        ccd_masked = ccd.copy()
        ccd_masked.data[mask_source] = np.nan
        ccd_masked.meta['BKGMED']  = (np.median(bkg.back())/ccd.meta.get('EXPTIME', 1),"[adu/s] Median background level")
        ccd_masked.meta['BKGRMS']  = (bkg.globalrms/ccd.meta.get('EXPTIME', 1),"[adu/s] Background RMS")
        ccd_masked.meta['HISTORY'] = f"({datetime.now().isoformat()}) Sources (>BKGMED + {nsigma}*BKGRMS) masked using SEP."

        return ccd_masked

    def comb_master_bias(self, bias_frames, master_dir, outname):
        """
        Combine multiple bias frames into a single master bias frame.

        This method loads all specified bias frames, combines them using a
        median, sigma-clipped algorithm, and saves the result as a
        master bias FITS file.

        Parameters
        ----------
        bias_frames : list[str or Path]
            List of file paths to the raw bias frames.
        master_dir : str or Path
            Directory where the master bias frame will be saved.
        outname : str
            Base name for the output file (e.g., 'camera_serial').

        Returns
        -------
        Path or None
            The Path to the created master bias file, or None if
            the combination failed.
        """
        self.logger.info("Starting master bias combination...")
        master_dir = Path(master_dir)
        master_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load data from fits files into CCDData objects
        bias_ccds_with_paths = self._load_ccd_list(bias_frames)
        if not bias_ccds_with_paths:
            self.logger.error("No bias frames loaded. Aborting comb_master_bias.")
            return None
        
        # Extract CCDs for combination
        bias_ccds = [ccd for ccd, fpath in bias_ccds_with_paths]
        hdr0 = bias_ccds[0].header # load first header for metadata

        obsdate = hdr0.get('OBSDATE', Time(hdr0['JD'], format='jd').to_datetime().strftime('%Y%m%d'))
        fpath_mbias = master_dir / f"{outname}.bias.comb.{obsdate}.fits"

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

        # Update metadata
        mbias.meta.update({
            'COMBINED': True,
            'NCOMBINE': (len(bias_ccds), "Number of combined frames"),
            'IMAGETYP': 'BIAS',
            'OBSDATE': (obsdate, "YYYYMMDD (UTC)"),
            'OBJECT': fpath_mbias.stem,
            'FILENAME': fpath_mbias.name
        })

        # Add HISTORY entries
        mbias.meta['HISTORY'] = f"({datetime.now().isoformat()}) Combined {len(bias_ccds)} bias frames. (solopy.CombMaster.comb_master_bias)"
        mbias.meta['HISTORY'] = f"({datetime.now().isoformat()}) IMAGETYP set to {hdr0['IMAGETYP']} --> BIAS."

        # 3. Save as fits
        hdu_mbias = fits.PrimaryHDU(data=mbias.data, header=mbias.meta)
        hdu_mbias.writeto(fpath_mbias, overwrite=True)
        self.logger.info(f"Master bias saved to {fpath_mbias.name}")
        
        return fpath_mbias

    def comb_master_dark(self, dark_frames, master_dir, outname, key_exptime='EXPTIME'):
        """
        Create master dark frames from raw darks, grouped by exposure time.

        This method finds the closest master bias, subtracts it from all
        raw darks, groups the bias-subtracted darks by their exposure time,
        combines each group, and saves them as separate master dark files.

        Parameters
        ----------
        dark_frames : list[str or Path]
            List of file paths to the raw dark frames.
        master_dir : str or Path
            Directory to search for master bias and to save master darks.
        outname : str
            Base name for the output files (e.g., 'camera_serial').
        key_exptime : str, optional
            The FITS header keyword for exposure time (default 'EXPTIME').

        Returns
        -------
        list[Path]
            A list of Paths to the created master dark files (one for each
            exposure time). Returns an empty list on failure.
        """
        self.logger.info("Starting master dark creation...")
        master_dir = Path(master_dir)
            
        # 1. Load all dark frames
        dark_ccds_with_paths = self._load_ccd_list(dark_frames)
        if not dark_ccds_with_paths:
            self.logger.error("No dark frames loaded. Aborting comb_master_dark.")
            return []
        
        # 2. Find closest master bias
        hdr0 = dark_ccds_with_paths[0][0].header # Get header from first CCD
        obs_jd = hdr0['JD']
        mbias, fpath_mbias = self._find_closest_bias(master_dir, obs_jd)
        
        if mbias is None:
            # The helper logged the error. We can't proceed.
            self.logger.error(f"Master bias not found. Aborting comb_master_dark.")
            return []

        self.logger.info(f"Using master bias: {fpath_mbias.name}")
        
        # 3. Group darks by exposure time and subtract bias
        grouped_darks = {}
        for ccd, fpath in dark_ccds_with_paths:
            try:
                # Check exptime key
                if key_exptime.upper() in ccd.header:
                    exptime = float(ccd.header[key_exptime.upper()])
                elif key_exptime.lower() in ccd.header:
                    exptime = float(ccd.header[key_exptime.lower()])
                else:
                    self.logger.warning(f"Cannot find key {key_exptime} in {fpath.name}. Skipping.")
                    continue
                
                bdark = ccdproc.subtract_bias(ccd, mbias)
                bdark.meta['HISTORY'] = f"({datetime.now().isoformat()}) Master bias subtracted: {fpath_mbias.name}"
                if exptime not in grouped_darks:
                    grouped_darks[exptime] = []
                grouped_darks[exptime].append(bdark)
            except Exception as e:
                self.logger.error(f"Failed to bias-subtract {fpath.name}: {e}")

        # 4. Combine groups and save
        mdark_frames = []
        for exp, bdark_ccds in grouped_darks.items():
            
            self.logger.info(f"Combining {len(bdark_ccds)} darks for exptime {exp}s...")
            obsdate = hdr0.get('OBSDATE', Time(hdr0['JD'], format='jd').to_datetime().strftime('%Y%m%d'))
            fpath_mdark = master_dir / f"{outname}.dark.{int(round(exp))}s.comb.{obsdate}.fits"
            
            mdark = ccdproc.combine(
                bdark_ccds,
                method='median',
                sigma_clip=True,
                sigma_clip_low_thresh=5,
                sigma_clip_high_thresh=5,
                sigma_clip_func=np.ma.median,
                sigma_clip_dev_func=mad_std,
                mem_limit=500e6,
                dtype=np.float32
            )

            # Update metadata
            mdark.meta.update({
                'COMBINED': True,
                'NCOMBINE': (len(bdark_ccds), "Number of combined frames"),
                'IMAGETYP': 'DARK',
                'OBSDATE': (obsdate, "YYYYMMDD (UTC)"),
                'OBJECT': fpath_mdark.stem,
                'FILENAME': fpath_mdark.name
            })
            
            # Add HISTORY entries
            mdark.meta['HISTORY'] = f"({datetime.now().isoformat()}) Combined {len(bdark_ccds)} dark frames."
            mdark.meta['HISTORY'] = f"({datetime.now().isoformat()}) IMAGETYP set to {hdr0['IMAGETYP']} --> DARK."
            
            # 5. Save as fits
            hdu_mdark = fits.PrimaryHDU(data=mdark.data, header=mdark.meta)
            hdu_mdark.writeto(fpath_mdark, overwrite=True)
            self.logger.info(f"Master dark saved to {fpath_mdark.name}")
            mdark_frames.append(fpath_mdark)
            
        return mdark_frames

    def comb_master_flat(self, flat_frames, master_dir, outname, key_exptime='EXPTIME'):
        """
        Create a master flat frame for a specific filter.

        This method processes raw flat frames by:
        1. Finding and subtracting the closest master bias (by JD).
        2. Finding and subtracting the closest master dark (by exptime, then JD).
        3. Combining the processed flats using a median, sigma-clipped
           algorithm with INVERSE MEDIAN SCALING (1/median) applied
           to each frame before combination.
        4. Saving the result as a single master flat FITS file.

        Parameters
        ----------
        flat_frames : list[str or Path]
            List of file paths to the raw flat frames for this filter.
        master_dir : str or Path
            Directory to search for master bias/darks and to save the master flat.
        outname : str
            Base name for the output file (e.g., 'camera_serial').
        filter_name : str
            The name of the filter (e.g., 'R', 'G', 'B') for this flat.
            This will be included in the output filename and metadata.
        key_exptime : str, optional
            The FITS header keyword for exposure time (default 'EXPTIME').

        Returns
        -------
        Path or None
            The Path to the created master flat file, or None if
            the combination failed.
        """
        self.logger.info(f"Starting master flat creation...")
        master_dir = Path(master_dir)
        
        TMPDIR = master_dir / "tmp"
        TMPDIR.mkdir(exist_ok=True)
        _fileutil.clear_dir(TMPDIR)

        # Process flats
        # Bias and dark subtraction, masking outliers (i.e., sources)
        self.logger.info(f"Processing {len(flat_frames)} flat frames...")
        # processed_flats = []
        for fpath in flat_frames:
            try:
                fpath = Path(fpath)
                
                try:
                    flat = CCDData.read(fpath)
                except ValueError:
                    flat = CCDData.read(fpath, unit="adu")
                except Exception as e:
                    self.logger.error(f"Failed to load flat frame: {fpath.name}: {e}")
                    continue
                
                self.logger.info(f"Processing flat frame: {fpath.name}")
                
                # Bias subtraction
                mbias, fpath_mbias = self._find_closest_bias(master_dir, flat.header['JD'], verbose=False)
                if mbias is None:
                    self.logger.warning(f"Master bias not found. Skip the frame.")
                    continue                
                bflat = ccdproc.subtract_bias(flat, mbias)
                bflat.meta['BIASNAME'] = (fpath_mbias.name, "Master bias file used")
                bflat.meta['BIASCORR'] = True
                bflat.meta['HISTORY'] = f"({datetime.now().isoformat()}) Master bias subtracted."
                
                # Dark subtraction
                if (key_exptime.upper() in flat.header):
                    exptime = float(flat.header[key_exptime.upper()])
                else:
                    self.logger.warning(f"Cannot find exptime key={key_exptime}. Skipping dark subtraction.")
                    exptime = None
                
                mdark, fpath_mdark = self._find_closest_dark(master_dir, flat.header['JD'], exptime, verbose=False)
                if mdark is None:
                    self.logger.warning(f"Master dark not found. Skipping dark subtraction.")
                    bdflat = bflat.copy()
                else:
                    bdflat = ccdproc.subtract_dark(bflat, mdark, exposure_time=key_exptime, exposure_unit=u.second)
                    bdflat.meta['DARKCORR'] = True
                    bdflat.meta['DARKNAME'] = (fpath_mdark.name, "Master dark file used")
                    bdflat.meta['HISTORY'] = f"({datetime.now().isoformat()}) Master dark subtracted."
                
                # Mask sources using sigma-clipping
                bdmflat = self._ccd_sigmaclip(bdflat, nsigma=2.5)
                
                # Add to processed list
                bdmflat.meta['IMAGETYP'] = fpath.name
                bdmflat.meta['IMAGETYP'] = 'FLAT'
                bdmflat.write(TMPDIR / fpath.name, overwrite=True)
                
                # processed_flats.append(bdsci_masked)
                
            except Exception as e:
                self.logger.error(f"Failed to process: {e}")

        # Combine processed flats
        processed_flats = ccdproc.ImageFileCollection(TMPDIR).filter(imagetyp='FLAT').files
        if not processed_flats:
            self.logger.error("No flat frames were successfully processed.")
            return None  
        self.logger.info(f"Combining {len(processed_flats)} processed flat frames...")
        
        # Define scaling function: 1 / median
        def inv_median_scale(a):
            return 1.0 / np.nanmedian(a)

        mflat = ccdproc.combine(processed_flats,
                             method='median',
                             scale=inv_median_scale, # Scale by inverse median
                             sigma_clip=True,
                             sigma_clip_low_thresh=3,
                             sigma_clip_high_thresh=3,
                             sigma_clip_func=np.ma.median,
                             sigma_clip_dev_func=mad_std,
                             mem_limit=500e6,
                             dtype=np.float32
                             )
        
        hdr0_flat = CCDData.read(processed_flats[0]).header
        filter_name = hdr0_flat.get('FILTER', 'UNKNOWN').upper()
        obsdate = hdr0_flat.get('OBSDATE', Time(hdr0_flat['JD'], format='jd').to_datetime().strftime('%Y%m%d'))
        fpath_mflat = master_dir / f"{outname}.flat.{filter_name}.comb.{obsdate}.fits"
        
        mflat.meta.update({
            'COMBINED': True,
            'NCOMBINE': len(processed_flats),
            'IMAGETYP': 'FLAT',
            'OBSDATE': (obsdate, "YYYYMMDD (UTC)"),
            'OBJECT': fpath_mflat.stem,
            'FILENAME': fpath_mflat.name
        })
        
        mflat.meta['HISTORY'] = f"({datetime.now().isoformat()}) Combined {len(processed_flats)} flat frames. (solopy.CombMaster.comb_master_flat)"
        mflat.meta['HISTORY'] = f"({datetime.now().isoformat()}) IMAGETYP set to {hdr0_flat['IMAGETYP']} --> FLAT."

        hdu_mflat = fits.PrimaryHDU(data=mflat.data, header=mflat.meta)
        hdu_mflat.writeto(fpath_mflat, overwrite=True) # Save as .fits
        self.logger.info(f"Master flat saved to {fpath_mflat.name}")
        
        return fpath_mflat