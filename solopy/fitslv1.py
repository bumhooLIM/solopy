import logging
from pathlib import Path
from astropy.io import fits
from astropy.coordinates import EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u
from datetime import datetime
import ccdproc
from ccdproc import CCDData 
import astrometry # type: ignore
import sep
import numpy as np

class FitsLv1:
    """
    Class for Level-1 processing of RASA lcpy KL4040 science frames.

    Provides methods to solve and update WCS, and to apply bias, dark, and flat corrections.
    Handles .fits and .fits.bz2 file I/O.
    """

    def __init__(self, log_file: str = None):
        """
        Configure logging to console and optional file.
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

    def update_wcs(self,
                   fpath_fits,
                   outdir,
                   cache_directory = "astrometry_cache",
                   return_fpath=True
                   ):
        """
        Solve for WCS and update FITS header.
        Reads .fits and writes to .wcs.fits.
        """
        self.logger.info(f"Find WCS solution: {fpath_fits}")
        
        fpath_fits = Path(fpath_fits)
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        # Read input file
        try:
            sci = CCDData.read(fpath_fits) 
        except ValueError:
            sci = CCDData.read(fpath_fits, unit='adu')
        except FileNotFoundError:
            self.logger.error(f"File not found.")
            return
        except Exception as e:
            self.logger.error(f"Error reading FITS: {e}")
            return

        sci.header["LV0FILE"] = (fpath_fits.name, "Original Level-0 file name")
        
        # Detect sources using SEP
        sci.data = sci.data.astype(np.float32)
        try:
            bkg = sep.Background(sci.data)
            objs = sep.extract(sci.data - bkg.back(), thresh=3.0 * bkg.globalrms, minarea=5)
            bright = objs[np.argsort(objs['flux'])[::-1][:50]]
            coords = np.vstack((bright['x'], bright['y'])).T
        except Exception as e:
            self.logger.error(f"Source detection failed: {e}")
            return

        # Solve WCS
        wcs_solved = False
        try:
            with astrometry.Solver(
                astrometry.series_4100.index_files(cache_directory=cache_directory, scales={8,9,10})
            ) as solver:
                sol = solver.solve(
                    stars=coords,
                    size_hint=astrometry.SizeHint(lower_arcsec_per_pixel=2.90, upper_arcsec_per_pixel=3.00),
                    position_hint=astrometry.PositionHint(
                        ra_deg=sci.header["RA"], dec_deg=sci.header["DEC"], radius_deg=2.0
                    ),
                    solution_parameters=astrometry.SolutionParameters(
                        logodds_callback=lambda l: astrometry.Action.STOP if len(l)>=10 else astrometry.Action.CONTINUE,
                        sip_order=3
                    )
                )
                
                if sol.has_match():
                    wcs_solved = True
                    best = sol.best_match()
                    self.logger.info(f"WCS match: RA={best.center_ra_deg:.2f}, DEC={best.center_dec_deg:.2f}, scale={best.scale_arcsec_per_pixel:.2f}" )
                    sci.wcs = best.astropy_wcs()
                    hdr = best.astropy_wcs().to_header(relax=True)
                    sci.header.extend(hdr, update=True)
                    sci.header['PIXSCALE'] = (best.scale_arcsec_per_pixel, "[arcsec/pixel] Pixel scale")
                    sci.header['HISTORY'] = f"({datetime.now().isoformat()}) WCS updated. (solopy.Lv1.update_wcs)"
                else:
                    self.logger.warning(f"No WCS solution found for {fpath_fits.name}.")
        except Exception as e:
            self.logger.warning(f"Astrometry.net solver failed for {fpath_fits.name}: {e}")

        # Compute center coords
        if wcs_solved:
            try:
                cen = (sci.header['NAXIS1']//2, sci.header['NAXIS2']//2)
                sky = sci.wcs.pixel_to_world(*cen)
                loc = EarthLocation(lat=sci.header['LAT']*u.deg, lon=sci.header['LON']*u.deg, height=sci.header['ELEV']*u.km)
                altaz = sky.transform_to(AltAz(obstime=Time(sci.header['JD'], format='jd'), location=loc))
                sci.header['RACEN'] = (sky.ra.value, "[deg] Center Right Ascension")
                sci.header['DECCEN'] = (sky.dec.value, "[deg] Center Declination")
                sci.header['ALTCEN'] = (altaz.alt.value, "[deg] Center Altitude")
                sci.header['AZCEN'] = (altaz.az.value, "[deg] Center Azimuth")
            except Exception as e:
                self.logger.warning(f"Center coordinate calculation failed. {e}")
        else:
            self.logger.warning(f"Skipping center coord calculation (no WCS).")

        # Save output file
        out_name = f"{fpath_fits.stem}.wcs.fits"
        outpath = outdir / out_name

        new_hdu = fits.PrimaryHDU(data=sci.data, header=sci.header)
        try:
            new_hdu.writeto(outpath, overwrite=True)
            self.logger.info(f"WCS updated: {outpath.name}")
        except Exception as e:
            self.logger.error(f"Failed to write {outpath}: {e}")
            return

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
        """
        Subtract bias, dark, mask bad pixels, and flat-correct.
        Reads .fits and writes Multi-Extension .fits.
        Master frames are assumed to be uncompressed .fits files.
        """
        fpath_fits = Path(fpath_fits)
        outdir = Path(outdir)
        masterdir = Path(masterdir)
        outdir.mkdir(parents=True, exist_ok=True)

        try:
            sci = CCDData.read(fpath_fits)
        except ValueError:
            sci = CCDData.read(fpath_fits, unit='adu')
        except FileNotFoundError:
            self.logger.error(f"File not found: {fpath_fits}")
            return
        except Exception as e:
            self.logger.error(f"Error reading FITS {fpath_fits}: {e}")

        try:
            mbias = self._select_master(masterdir, 'BIAS', sci.header['JD'])
            mdark = self._select_master(masterdir, 'DARK', sci.header['JD'], sci.header['EXPTIME'])
            # mflat = self._select_master(masterdir, 'FLAT', sci.header['JD'])
        except Exception as e:
            self.logger.error(f"Failed to load master frames: {e}")
            return
        
        # try:
        #     mask = self._select_master(masterdir, 'MASK', sci.header['JD'])
        # except Exception as e:
        #     self.logger.warning(f"Failed to load master mask: {e}")
        #     mask = None
        
        # mask (saturated pixels)
        mask_saturated = sci.data >= 4000 # kl4040 saturation level=4096 ADU (12-bit)
    
        # mask (edgeside)
        edge_width = 100  # pixels
        mask_edge = np.zeros(sci.data.shape, dtype=bool)
        mask_edge[:edge_width, :] = True
        mask_edge[-edge_width:, :] = True
        mask_edge[:, :edge_width] = True
        mask_edge[:, -edge_width:] = True    
        
        try:
            # Bias
            bsci = ccdproc.subtract_bias(sci, mbias)
            bsci.meta['BIASCORR'] = (True, "Bias corrected?")
            bsci.meta['HISTORY'] = f"({datetime.now().isoformat()}) Bias subtracted."
            bsci.meta['BIASNAME'] = (mbias.meta.get('FILENAME', None), "Master bias frame used")
            self.logger.info(f"Bias subtracted.")
            
            # dark
            bdsci = ccdproc.subtract_dark(bsci, mdark, exposure_time="EXPTIME", exposure_unit=u.second)
            bdsci.meta['DARKCORR'] = (True, "Dark corrected?")
            bdsci.meta['DARKNAME'] = (mdark.meta.get('FILENAME', None), "Master dark frame used")
            bdsci.meta['HISTORY'] = f"({datetime.now().isoformat()}) Dark subtracted."
            self.logger.info(f"Dark subtracted.")
            
            # mask (negative pixels)
            mask_negative = bdsci.data < 0
            
            # flat
            psci = ccdproc.flat_correct(bdsci, ccdmflat)
            psci.meta['FLATCORR'] = (True, "Flat corrected?")
            psci.meta['HISTORY'] = f"({datetime.now().isoformat()}) Flat corrected."
            psci.meta['FLATNAME'] = (ccdmflat.meta.get('FILENAME', None), "Master flat frame used")
            self.logger.info(f"Flat corrected.")
            
            # mask (bad pixels)
            mask_badpix = (psci.data <= 0) | (ccdmflat.data > 2.0) | np.isnan(psci.data) | np.isinf(psci.data) 
            
            # mask (nearby very bright & streak-like sources)
            mask_source = self._mask_source(psci.data)
            
            # combine masks
            combined_mask = mask_negative | mask_saturated | mask_edge | mask_badpix | mask_source
            if ccdmask is not None:
                combined_mask |= (ccdmask.data.astype(bool))
            if psci.mask is not None:
                combined_mask |= psci.mask.astype(bool)
            psci.mask = combined_mask
            psci.data = np.nan_to_num(np.clip(psci.data, 0, None), nan=0.0) # replace negative & NaN with 0.0
            psci.meta['HISTORY'] = f"({datetime.now().isoformat()}) Bad pixels masked."
            psci.meta['NBADPIX'] = (np.sum(combined_mask), "Number of bad pixels masked")
            psci.meta['MASKNAME'] = (ccdmask.meta.get('FILENAME', None) if ccdmask else None, "Master mask frame used")
            self.logger.info(f"Bad pixel mask applied.")
        
        except Exception as e:
            self.logger.error(f"CCD processing failed: {e}")
            return
        
        # WCS header
        if sci.wcs:
            hdr_wcs = sci.wcs.to_header(relax=True)
            psci.header.extend(hdr_wcs, update=True)
        
        # Update file name
        try:
            ra   = int(round(psci.header['RACEN']))
            dec  = int(round(psci.header['DECCEN']))
            pm   = 'p' if dec >= 0 else 'n'
            exp  = int(round(psci.header['EXPTIME']))
            obst = Time(psci.header['DATE-OBS']).strftime("%Y%m%d%H%M%S")
            fname_out = f"kl4040.sci.lv1.{ra:03d}.{pm}{abs(dec):02d}.{exp:03d}.{obst}.fits"
            psci.meta['FILENAME'] = fname_out
        except KeyError:
            self.logger.warning(f"Failed to find required header keys for naming. Using original name.")
            fname_out = f"{fpath_fits.stem}.lv1.fits"
        
        # Update meta data
        psci.meta['DATLEVEL'] = 1
        
        fpath_out = outdir / fname_out
        
        # Save as Multi-Extension FITS
        primary_hdu = fits.PrimaryHDU(data=psci.data, header=psci.header)
        
        if psci.mask is not None:
            mask_data = psci.mask.astype(np.uint8)
            mask_hdu = fits.ImageHDU(data=mask_data, name='MASK')
            hdul_out = fits.HDUList([primary_hdu, mask_hdu])
        else:
            hdul_out = fits.HDUList([primary_hdu])
        
        try:
            hdul_out.writeto(fpath_out, overwrite=True)
            self.logger.info(f"Preprocessing completed: {fpath_out.name}")
        except Exception as e:
            self.logger.error(f"Failed to write file: {e}")
            return
            
        if return_fpath:
            return fpath_out

    # def zp_calculation(self,

    def _select_master(self, masterdir, imagetyp, jd_target, exptime=None):
        """
        Helper to select the closest master frame by JD (and exposure time if given).
        Assumes master frames are UNCOMPRESSED .fits files.
        """
        coll = ccdproc.ImageFileCollection(masterdir, glob_include="*.fits").filter(imagetyp=imagetyp)
        if not coll.files:
            raise FileNotFoundError(f"No master {imagetyp} frames found in {masterdir}")

        df = coll.summary.to_pandas()
        
        # Find Julian date column
        jd_col = None
        if 'jd' in df.columns:
            jd_col = 'jd'
        elif 'JD' in df.columns:
            jd_col = 'JD'
        else:
            raise KeyError(f"Cannot find 'JD' columns for master {imagetyp} frames.")

        df[jd_col] = df[jd_col].astype(float)
        df['diff'] = (df[jd_col] - jd_target).abs()
        
        if exptime is not None:
            # Find exptime column
            exp_col = None
            if 'exptime' in df.columns:
                exp_col = 'exptime'
            elif 'EXPTIME' in df.columns:
                exp_col = 'EXPTIME'
            
            # Select the master frame with the closest exposure time
            if exp_col:
                df[exp_col] = df[exp_col].astype(float)
                exptime_unique = df[exp_col].unique()
                exptime_closest = min(exptime_unique, key=lambda x: abs(x - float(exptime)))
                self.logger.info(f"Requested exptime={exptime:.1f} sec, found closest master frame exptime={exptime_closest:.1f} sec.")
                df = df[df[exp_col] == exptime_closest].copy()
            else:
                self.logger.warning(f"Cannot find 'exptime' columns for master {imagetyp}, proceeding without exptime filter.")

        if df.empty:
            raise FileNotFoundError(f"No matching master {imagetyp} frame found for JD={jd_target:.1f}, EXPTIME={exptime:.1f} sec.")

        idx = df['diff'].idxmin()
        selected_file = Path(coll.files_filtered(include_path=True)[idx])
        self.logger.info(f"Selected master {imagetyp}: {selected_file.name}")
        
        try:
            if imagetyp == 'MASK':   
                master_frame = CCDData.read(selected_file, unit='bool')
            else:
                master_frame = CCDData.read(selected_file)
        except ValueError:
            master_frame = CCDData.read(selected_file, unit='adu')

        return master_frame
    
    def _mask_source(self, data, minarea=np.pi*12**2, ratio=3, mask_bright_source = True, mask_streak_source = True):
        
        # Source Extraction (sep)
        data = data.astype(np.float32)
        skybkg = sep.Background(data)
        data_bkgsub = data - skybkg.back()
        
        source, segmap = sep.extract(
            data_bkgsub,
            thresh=2.5,
            err=skybkg.globalrms,
            segmentation_map=True
        )
        
        mask_bright = np.zeros_like(segmap, dtype=bool)
        if mask_bright_source:
            # Masking around very bright sources
            source_bright = ((source['a'] / source['b']) < 2) & (source['npix'] >= minarea)
            idx_bright    = np.where(source_bright)[0]       # indices in `source`
            n_bright      = len(idx_bright)                  # number of bright sources 
            yy, xx        = np.indices(segmap.shape)         # pixel grids
            mask_bright   = np.zeros(segmap.shape, bool)     # mask for bright sources

            # Build a circular mask around (cx,cy)
            for idx in idx_bright:
                label = idx + 1
                cx, cy = source['x'][idx], source['y'][idx]
                py, px = np.where(segmap == label)
                d = np.hypot(px - cx, py - cy)
                radius = d.max() + 1 #  # masking r = d + 1
                circle_bright = (xx - cx)**2 + (yy - cy)**2 <= radius**2
                mask_bright |= circle_bright
        
        # Masking streak-like sources (high a/b ratio)
        mask_streak = np.zeros_like(segmap, dtype=bool)
        if mask_streak_source:
            source_streak = (source['a'] / source['b']) >= ratio
            idx_streak = np.where(source_streak)[0]    
            mask_streak = np.zeros_like(segmap, dtype=bool)
            for idx in idx_streak:
                label = idx + 1 
                streak = segmap==label
                mask_streak |= streak
        
        mask_source = mask_bright | mask_streak
        
        return mask_source