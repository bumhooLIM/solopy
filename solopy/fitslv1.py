import logging
import sep
import numpy as np
from pathlib import Path
from datetime import datetime
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz
import astropy.units as u
import astrometry
import ccdproc

class FitsLv1:
    """
    Class for Level-1 processing of RASA lcpy KL4040 science frames.

    Provides methods to solve and update WCS, and to apply bias, dark, and flat corrections.
    Handles .fits and .fits.bz2 file I/O.
    """

    def __init__(self, log_file: str = None):
        """
        Configure logging to console and optional file (Jupyter-safe).
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        
        # Defense: Prevent duplicate loggers in Jupyter Notebooks
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self.logger.addHandler(handler)

            if log_file:
                file_h = logging.FileHandler(log_file)
                file_h.setFormatter(handler.formatter)
                self.logger.addHandler(file_h)

    def update_wcs(self,
                   fpath_fits,
                   outdir,
                   cache_directory="astrometry_cache",
                   verbose=False,
                   return_fpath=True):
        """
        Solve for WCS and update FITS header.
        Reads .fits and writes to .wcs.fits.
        """
        fpath_fits = Path(fpath_fits)
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        astrometry_logger = logging.getLogger("astrometry")
        astrometry_logger.setLevel(logging.INFO if verbose else logging.WARNING)
        
        self.logger.info(f"Find WCS solution: {fpath_fits.name}")

        # 1. Read input file safely
        try:
            sci = CCDData.read(fpath_fits) 
        except ValueError:
            sci = CCDData.read(fpath_fits, unit='adu')
        except Exception as e:
            self.logger.error(f"Failed to open FITS file {fpath_fits.name}: {e}")
            return None

        sci.header["LV0FILE"] = (fpath_fits.name, "Original Level-0 file name")
        
        # 2. SEP Data Preparation (Byte-swap defense for C-code compatibility)
        data_sep = sci.data.astype(np.float32)
        if data_sep.dtype.byteorder == '>':
            data_sep = data_sep.byteswap().newbyteorder()

        # 3. Detect sources using SEP
        try:
            bkg = sep.Background(data_sep)
            objs = sep.extract(data_sep - bkg.back(), thresh=3.0 * bkg.globalrms, minarea=5)
            # Only take the top 50 brightest stars to speed up the KD-Tree WCS matching
            bright = objs[np.argsort(objs['flux'])[::-1][:50]]
            coords = np.vstack((bright['x'], bright['y'])).T
        except Exception as e:
            self.logger.error(f"Source detection failed for {fpath_fits.name}: {e}")
            return None

        # 4. Solve WCS
        wcs_solved = False
        try:
            # Defense: Cast RA/DEC to strict floats to prevent Astrometry TypeError
            guess_ra = float(sci.header.get("RA", 0.0))
            guess_dec = float(sci.header.get("DEC", 0.0))

            with astrometry.Solver(
                astrometry.series_4100.index_files(cache_directory=cache_directory, scales={8,9,10})
            ) as solver:
                sol = solver.solve(
                    stars=coords,
                    size_hint=astrometry.SizeHint(lower_arcsec_per_pixel=2.90, upper_arcsec_per_pixel=3.00),
                    position_hint=astrometry.PositionHint(
                        ra_deg=guess_ra, dec_deg=guess_dec, radius_deg=2.0
                    ),
                    solution_parameters=astrometry.SolutionParameters(
                        logodds_callback=lambda l: astrometry.Action.STOP if len(l)>=10 else astrometry.Action.CONTINUE,
                        sip_order=3
                    ),
                )
                
                if sol.has_match():
                    wcs_solved = True
                    best = sol.best_match()
                    self.logger.info(f"WCS match: RA={best.center_ra_deg:.2f}, DEC={best.center_dec_deg:.2f}, scale={best.scale_arcsec_per_pixel:.2f}" )
                    
                    # Update Astropy WCS and push to Header
                    sci.wcs = best.astropy_wcs()
                    hdr = sci.wcs.to_header(relax=True)
                    sci.header.extend(hdr, update=True)
                    
                    sci.header['PIXSCALE'] = (best.scale_arcsec_per_pixel, "[arcsec/pixel] Pixel scale")
                    sci.header['HISTORY'] = f"({datetime.now().isoformat()}) WCS updated. (solopy.FitsLv1.update_wcs)"
                else:
                    self.logger.warning(f"No WCS solution found for {fpath_fits.name}.")
                    
        except Exception as e:
            self.logger.warning(f"Astrometry.net solver failed for {fpath_fits.name}: {e}")

        # 5. Compute center coordinates safely
        if wcs_solved:
            try:
                # 5a. Time-Independent Coordinates (RA/DEC)
                naxis1 = int(sci.header.get('NAXIS1', 4096))
                naxis2 = int(sci.header.get('NAXIS2', 4096))
                cen = (naxis1 // 2, naxis2 // 2)
                
                sky = sci.wcs.pixel_to_world(*cen)
                sci.header['RACEN']  = (sky.ra.value, "[deg] Center Right Ascension")
                sci.header['DECCEN'] = (sky.dec.value, "[deg] Center Declination")
                
                # 5b. Time-Dependent Coordinates (Alt/Az)
                if 'JD' not in sci.header:
                    self.logger.warning(f"Missing 'JD' in header for {fpath_fits.name}. Skipping Alt/Az calculation.")
                else:
                    obs_lat = float(sci.header.get('LAT', 37.07))
                    obs_lon = float(sci.header.get('LON', -119.4))
                    obs_el  = float(sci.header.get('ELEVAT', 1.405))
                    obs_jd  = float(sci.header['JD']) # Strictly use the header's Julian Date

                    loc = EarthLocation(lat=obs_lat*u.deg, lon=obs_lon*u.deg, height=obs_el*u.km)
                    altaz = sky.transform_to(AltAz(obstime=Time(obs_jd, format='jd'), location=loc))
                    
                    sci.header['ALTCEN'] = (altaz.alt.value, "[deg] Center Altitude")
                    sci.header['AZCEN']  = (altaz.az.value, "[deg] Center Azimuth")
                
            except Exception as e:
                self.logger.warning(f"Center coordinate calculation failed for {fpath_fits.name}: {e}")
        else:
            self.logger.warning(f"Skipping center coord calculation (no WCS) for {fpath_fits.name}.")

        # 6. Save output file
        out_name = f"{fpath_fits.stem}.wcs.fits"
        outpath = outdir / out_name

        try:
            # Using fits.writeto is cleaner and faster than initializing a new PrimaryHDU object manually
            fits.writeto(outpath, sci.data, sci.header, overwrite=True)
            self.logger.info(f"Successfully wrote updated WCS to: {out_name}")
        except Exception as e:
            self.logger.error(f"Failed to write {outpath.name}: {e}")
            return None

        if return_fpath:
            return outpath

    def correct_bdf(self,
                    fpath_fits,
                    outdir,
                    masterdir,
                    ccdmflat,
                    ccdmask=None,
                    return_fpath=True
                    ):
        
        fpath_fits = Path(fpath_fits)
        outdir = Path(outdir)
        masterdir = Path(masterdir)
        outdir.mkdir(parents=True, exist_ok=True)

        try:
            sci = CCDData.read(fpath_fits)
            hdr = sci.header.copy()
            hdr.extend(sci.wcs.to_header(relax=True), update=True)
        except ValueError:
            sci = CCDData.read(fpath_fits, unit='adu')
            hdr = sci.header.copy()
            hdr.extend(sci.wcs.to_header(relax=True), update=True)
        except Exception as e:
            self.logger.error(f"Error reading FITS {fpath_fits.name}: {e}")
            return None

        try:
            mbias = self._select_master(masterdir, 'BIAS', hdr.get('JD', 0))
            mdark = self._select_master(masterdir, 'DARK', hdr.get('JD', 0), hdr.get('EXPTIME'))
        except Exception as e:
            self.logger.error(f"Failed to load master frames for {fpath_fits.name}: {e}")
            return None
        
        # mask (saturated pixels)
        mask_saturated = sci.data >= 3800 # kl4040 saturation level=4096 ADU (12-bit)
    
        # mask (edgeside)
        edge_width = 100  # pixels
        mask_edge = np.zeros(sci.data.shape, dtype=bool)
        mask_edge[:edge_width, :] = True
        mask_edge[-edge_width:, :] = True
        mask_edge[:, :edge_width] = True
        mask_edge[:, -edge_width:] = True    
        
        try:
            # Memory Defense: Overwrite 'sci' variable instead of creating bsci, bdsci, psci
            
            # Bias
            sci = ccdproc.subtract_bias(sci, mbias)
            hdr['BIASCORR'] = (True, "Bias corrected?")
            hdr['BIASNAME'] = (mbias.meta.get('FILENAME', 'Unknown'), "Master bias frame used")
            
            # Dark
            sci = ccdproc.subtract_dark(sci, mdark, exposure_time="EXPTIME", exposure_unit=u.second)
            hdr['DARKCORR'] = (True, "Dark corrected?")
            hdr['DARKNAME'] = (mdark.meta.get('FILENAME', 'Unknown'), "Master dark frame used")
            
            # mask (negative pixels generated by dark subtraction read-noise)
            mask_negative = sci.data < 0
            
            # Flat
            sci = ccdproc.flat_correct(sci, ccdmflat)
            hdr['FLATCORR'] = (True, "Flat corrected?")
            hdr['FLATNAME'] = (ccdmflat.meta.get('FILENAME', 'Unknown'), "Master flat frame used")
            
            # mask (bad pixels)
            mask_badpix = (sci.data <= 0) | np.isnan(sci.data) | np.isinf(sci.data) \
               | (np.isinf(ccdmflat.data)) | (np.isnan(ccdmflat.data) \
               | (ccdmflat.data > 1.5) | (ccdmflat.data <= 0.4))  # flat correction can amplify bad pixels, so mask them too
            
            # mask (nearby very bright & streak-like sources)
            mask_source = self._mask_source(sci.data)
            
            # Combine all masks
            combined_mask = mask_saturated | mask_edge | mask_badpix | mask_source | mask_negative
            if ccdmask is not None:
                combined_mask |= (ccdmask.data.astype(bool))
            if sci.mask is not None:
                combined_mask |= sci.mask.astype(bool)
                
            sci.mask = combined_mask
            # sci.data = np.nan_to_num(np.clip(sci.data, 0, None), nan=0.0) 
            
            # If you MUST fill bad pixels so they don't break simple numpy math later, 
            # fill them with the median of the image, NOT zero.
            # median_sky = np.nanmedian(sci.data)
            # sci.data[sci.data <= 0] = median_sky
            
            hdr['HISTORY'] = f"({datetime.now().isoformat()}) BDF corrected and masked. (solopy.FitsLv1)"
            hdr['NBADPIX'] = (int(np.sum(combined_mask)), "Number of bad pixels masked")
            hdr['MASKNAME'] = (ccdmask.meta.get('FILENAME', 'Unknown') if ccdmask else None, "Master mask frame used")
            self.logger.info(f"Bias, Dark, and Flat processing & Bad pixel mask applied.")
        
        except Exception as e:
            self.logger.error(f"CCD processing failed for {fpath_fits.name}: {e}")
            return None
        
        # Update file name (Defensive Dictionary Checking)
        try:
            ra   = int(round(hdr['RACEN']))
            dec  = int(round(hdr['DECCEN']))
            pm   = 'p' if dec >= 0 else 'n'
            exp  = int(round(hdr['EXPTIME']))
            obst = Time(hdr['DATE-OBS']).strftime("%Y%m%d%H%M%S")
            fname_out = f"kl4040.sci.lv1.{ra:03d}.{pm}{abs(dec):02d}.{exp:03d}.{obst}.fits"
            hdr['FILENAME'] = fname_out
        except KeyError:
            self.logger.warning(f"Missing header keys for naming {fpath_fits.name}. Using default.")
            fname_out = f"{fpath_fits.stem}.lv1.fits"
        
        hdr['DATLEVEL'] = 1
        fpath_out = outdir / fname_out
        
        # Save as Multi-Extension FITS
        sci.data = sci.data.astype(np.float32)  # Ensure data is in a standard format for writing
        primary_hdu = fits.PrimaryHDU(data=sci.data, header=hdr)
        
        hdul_out = fits.HDUList([primary_hdu])
        if sci.mask is not None:
            mask_data = sci.mask.astype(np.uint8)
            mask_hdu = fits.ImageHDU(data=mask_data, name='MASK')
            hdul_out.append(mask_hdu)
        
        try:
            hdul_out.writeto(fpath_out, overwrite=True)
            self.logger.info(f"Preprocessing completed: {fpath_out.name}")
        except Exception as e:
            self.logger.error(f"Failed to write file {fpath_out.name}: {e}")
            return None
            
        if return_fpath:
            return fpath_out

    def _select_master(self, masterdir, imagetyp, jd_target, exptime=None):
        masterdir = Path(masterdir)
        
        # Cache the ImageFileCollection DataFrame to prevent massive I/O bottlenecks
        # Make sure self._master_cache = {} is in __init__
        if str(masterdir) not in getattr(self, '_master_cache', {}):
            coll = ccdproc.ImageFileCollection(masterdir, glob_include="*.fits")
            # Create the cache dictionary if it doesn't exist
            if not hasattr(self, '_master_cache'):
                self._master_cache = {}
            self._master_cache[str(masterdir)] = coll.summary.to_pandas()
            
        # Fetch full dataframe from memory cache
        df_full = self._master_cache[str(masterdir)]
        
        # Filter by image type safely (handles case sensitivity)
        if 'imagetyp' in df_full.columns:
            df = df_full[df_full['imagetyp'].str.upper() == imagetyp.upper()].copy()
        elif 'IMAGETYP' in df_full.columns:
            df = df_full[df_full['IMAGETYP'].str.upper() == imagetyp.upper()].copy()
        else:
            raise KeyError(f"IMAGETYP column missing in {masterdir}")

        if df.empty:
            raise FileNotFoundError(f"No master {imagetyp} frames found in {masterdir}")

        # Find Julian date column safely
        jd_col = 'jd' if 'jd' in df.columns else 'JD' if 'JD' in df.columns else None
        if not jd_col:
            raise KeyError(f"Cannot find 'JD' columns for master {imagetyp} frames.")

        df[jd_col] = df[jd_col].astype(float)
        df['diff'] = (df[jd_col] - jd_target).abs()
        
        if exptime is not None:
            exp_col = 'exptime' if 'exptime' in df.columns else 'EXPTIME' if 'EXPTIME' in df.columns else None
            
            if exp_col:
                df[exp_col] = df[exp_col].astype(float)
                exptime_unique = df[exp_col].unique()
                exptime_closest = min(exptime_unique, key=lambda x: abs(x - float(exptime)))
                self.logger.info(f"Requested exptime={exptime:.1f} sec, found closest master frame exptime={exptime_closest:.1f} sec.")
                df = df[df[exp_col] == exptime_closest].copy()
            else:
                self.logger.warning(f"Cannot find 'exptime' columns for master {imagetyp}.")

        if df.empty:
            raise FileNotFoundError(f"No matching master {imagetyp} frame found.")

        # SAFELY fetch filename using Pandas index label lookup (solves the array mismatch bug)
        idx = df['diff'].idxmin()
        filename = df.loc[idx, 'file']
        selected_file = masterdir / filename
        
        self.logger.info(f"Selected master {imagetyp}: {selected_file.name}")
        
        try:
            if imagetyp == 'MASK':   
                master_frame = CCDData.read(selected_file, unit='bool')
            else:
                master_frame = CCDData.read(selected_file)
        except ValueError:
            master_frame = CCDData.read(selected_file, unit='adu')

        return master_frame
    
    def _mask_source(self, data, minarea=np.pi*12**2, ratio=3, mask_bright_source=True, mask_streak_source=True):
        
        # SEP Byte-Order Defense
        data_sep = data.astype(np.float32)
        if data_sep.dtype.byteorder == '>':
            data_sep = data_sep.byteswap().newbyteorder()
            
        skybkg = sep.Background(data_sep)
        data_bkgsub = data_sep - skybkg.back()
        
        source, segmap = sep.extract(
            data_bkgsub,
            thresh=2.5,
            err=skybkg.globalrms,
            segmentation_map=True
        )
        
        # Initialize masks safely
        mask_bright = np.zeros(segmap.shape, dtype=bool)
        mask_streak = np.zeros(segmap.shape, dtype=bool)

        if mask_bright_source:
            source_bright = ((source['a'] / source['b']) < 2) & (source['npix'] >= minarea)
            idx_bright    = np.where(source_bright)[0]       
            yy, xx        = np.indices(segmap.shape)         

            for idx in idx_bright:
                label = idx + 1
                cx, cy = source['x'][idx], source['y'][idx]
                py, px = np.where(segmap == label)
                
                # Prevent empty array crashes if segmap area is bizarre
                if len(px) > 0 and len(py) > 0:
                    d = np.hypot(px - cx, py - cy)
                    radius = d.max() + 1 
                    circle_bright = (xx - cx)**2 + (yy - cy)**2 <= radius**2
                    mask_bright |= circle_bright
        
        if mask_streak_source:
            source_streak = (source['a'] / source['b']) >= ratio
            idx_streak = np.where(source_streak)[0]    
            for idx in idx_streak:
                label = idx + 1 
                mask_streak |= (segmap == label)
        
        return mask_bright | mask_streak