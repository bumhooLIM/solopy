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

class Lv1:
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

    def update_wcs(self, fpath_fits, outdir, cache_directory = "astrometry_cache", return_fpath=True):
        """
        Solve for WCS and update FITS header.
        Reads .fits and writes to .wcs.fits.
        """
        fpath_fits = Path(fpath_fits)
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        # Read input file
        try:
            sci = CCDData.read(fpath_fits) 
        except FileNotFoundError:
            self.logger.error(f"File not found: {fpath_fits}")
            return
        except Exception as e:
            self.logger.error(f"Error reading FITS {fpath_fits}: {e}")
            return

        # Detect sources using SEP
        sci.data = sci.data.astype(np.float32)
        try:
            bkg = sep.Background(sci.data)
            objs = sep.extract(sci.data - bkg.back(), thresh=3.0 * bkg.globalrms, minarea=5)
            bright = objs[np.argsort(objs['flux'])[::-1][:50]]
            coords = np.vstack((bright['x'], bright['y'])).T
        except Exception as e:
            self.logger.error(f"Source detection failed for {fpath_fits.name}: {e}")
            return

        # Solve WCS
        wcs_solved = False
        try:
            with astrometry.Solver(
                astrometry.series_4100.index_files(cache_directory=cache_directory, scales={8,9,10}),
                log_level=logging.WARNING
            ) as solver:
                sol = solver.solve(
                    stars=coords,
                    size_hint=astrometry.SizeHint(lower_arcsec_per_pixel=2.90, upper_arcsec_per_pixel=3.00),
                    position_hint=astrometry.PositionHint(
                        ra_deg=sci.header["RA"], dec_deg=sci.header["DEC"], radius_deg=5.0
                    ),
                    solution_parameters=astrometry.SolutionParameters(
                        logodds_callback=lambda l: astrometry.Action.STOP if len(l)>=10 else astrometry.Action.CONTINUE,
                        sip_order=3
                    )
                )
                if sol.has_match():
                    best = sol.best_match()
                    self.logger.info(f"WCS match: RA={best.center_ra_deg:.5f}, DEC={best.center_dec_deg:.5f}, scale={best.scale_arcsec_per_pixel:.3f}" )
                    sci.wcs = best.astropy_wcs()
                    hdr = best.astropy_wcs().to_header(relax=True)
                    sci.header.extend(hdr, update=True)
                    sci.header['PIXSCALE'] = (best.scale_arcsec_per_pixel, "arcsec/pixel")
                    sci.header['HISTORY'] = f"({datetime.now().isoformat()}) WCS updated. (solopy.Lv1.update_wcs)"
                    wcs_solved = True
                else:
                    self.logger.warning(f"No WCS solution found for {fpath_fits.name}.")
        except Exception as e:
             self.logger.warning(f"Astrometry.net solver failed for {fpath_fits.name}: {e}")

        # compute center coords (only if wcs solved)
        if wcs_solved:
            try:
                cen = (sci.header['NAXIS1']//2, sci.header['NAXIS2']//2)
                sky = sci.wcs.pixel_to_world(*cen)
                loc = EarthLocation(lat=sci.header['LAT']*u.deg, lon=sci.header['LON']*u.deg, height=sci.header['ELEV']*u.km)
                altaz = sky.transform_to(AltAz(obstime=Time(sci.header['JD'], format='jd'), location=loc))
                sci.header['RACEN'] = (sky.ra.value, "deg center RA")
                sci.header['DECCEN'] = (sky.dec.value, "deg center DEC")
                sci.header['ALTCEN'] = (altaz.alt.value, "deg center Alt")
                sci.header['AZCEN'] = (altaz.az.value, "deg center Az")
            except Exception as e:
                self.logger.warning(f"Center coordinate calculation failed for {fpath_fits.name}: {e}")
        else:
            self.logger.warning(f"Skipping center coord calculation for {fpath_fits.name} (no WCS).")

        # "image.fits" -> "image.wcs.fits"
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

    def preprocessing(self, fpath_fits, outdir, masterdir, return_fpath=True):
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
            sci = CCDData.read(fpath_fits, unit='adu')
        except FileNotFoundError:
            self.logger.error(f"File not found: {fpath_fits}")
            return
        except Exception as e:
            self.logger.error(f"Error reading FITS {fpath_fits}: {e}")

        try:
            mbias = self._select_master(masterdir, 'BIAS', sci.header['JD'])
            mdark = self._select_master(masterdir, 'DARK', sci.header['JD'], sci.header['EXPTIME'])
            mflat = self._select_master(masterdir, 'FLAT', sci.header['JD'])
        except Exception as e:
            self.logger.error(f"Failed to load master frames for {fpath_fits.name}: {e}")
            return
            
        # Bias
        try:
            bsci = ccdproc.subtract_bias(sci, mbias)
            bsci.meta['BIASCORR'] = (True, "Bias corrected?")
            bsci.meta['HISTORY'] = f"({datetime.now().isoformat()}) Bias subtracted."
            self.logger.info(f"Bias subtracted for {fpath_fits.name}")
            
            # dark
            bdsci = ccdproc.subtract_dark(bsci, mdark, exposure_time="EXPTIME", exposure_unit=u.second)
            bdsci.meta['DARKCORR'] = (True, "Dark corrected?")
            bdsci.meta['HISTORY'] = f"({datetime.now().isoformat()}) Dark subtracted."
            self.logger.info(f"Dark subtracted for {fpath_fits.name}")
            
            # mask (음수 값 마스킹 및 0으로 클리핑)
            bdsci.mask = bdsci.data < 0
            bdsci.data = np.nan_to_num(np.clip(bdsci.data, 0, None), nan=0.0)
            bdsci.meta['HISTORY'] = f"({datetime.now().isoformat()}) Bad pixels masked (negative values)."
            self.logger.info(f"Bad pixels masked for {fpath_fits.name}")
            
            # flat
            psci = ccdproc.flat_correct(bdsci, mflat)
            psci.meta['FLATCORR'] = (True, "Flat corrected?")
            psci.meta['HISTORY'] = f"({datetime.now().isoformat()}) Flat corrected."
            self.logger.info(f"Flat corrected for {fpath_fits.name}")
        
        except Exception as e:
            self.logger.error(f"CCD processing failed for {fpath_fits.name}: {e}")
            return

        # 3. 수정된 파일 쓰기 (멀티-익스텐션 FITS)
        if 'RACEN' not in psci.header or 'DECCEN' not in psci.header:
            self.logger.warning(f"RACEN/DECCEN not in header for {fpath_fits.name}. Using original name.")
            # 'image.wcs.fits' -> 'image.wcs.proc.fits'
            outname = f"{fpath_fits.stem}.proc.fits"
            
        else:
            rac = int(round(psci.header['RACEN']))
            dec = int(round(psci.header['DECCEN']))
            pm = 'p' if dec >= 0 else 'n'
            exp = int(round(psci.header['EXPTIME']))
            obst = Time(psci.header['DATE-OBS']).strftime("%Y%m%d%H%M%S")
            # 파일명 .fits로 수정
            outname = f"kl4040.sci.{rac:03d}.{pm}{abs(dec):02d}.{exp:03d}.{obst}.fits"
        
        outpath = outdir / outname
        
        # 1. Primary HDU (Science Data) 생성
        primary_hdu = fits.PrimaryHDU(data=psci.data, header=psci.header)
        
        # 2. Mask HDU (ImageHDU) 생성
        if psci.mask is not None:
            mask_data = psci.mask.astype(np.uint8)
            mask_hdu = fits.ImageHDU(data=mask_data, name='MASK')
            hdul_out = fits.HDUList([primary_hdu, mask_hdu])
        else:
            hdul_out = fits.HDUList([primary_hdu])
        
        try:
            hdul_out.writeto(outpath, overwrite=True)
            self.logger.info(f"Preprocessing complete (MEF): {outname}")
        except Exception as e:
            self.logger.error(f"Failed to write MEF {outpath}: {e}")
            return
            
        if return_fpath:
            return outpath

    def _select_master(self, masterdir, imagetyp, jd_target, exptime=None):
        """
        Helper to select the closest master frame by JD (and exposure time if given).
        Assumes master frames are UNCOMPRESSED .fits files.
        """
        coll = ccdproc.ImageFileCollection(masterdir, glob_include="*.fits").filter(imagetyp=imagetyp)
        if not coll.files:
            raise FileNotFoundError(f"No master {imagetyp} frames found in {masterdir}")

        df = coll.summary.to_pandas()
        
        # 'jd' 또는 'JD' 키워드 찾기
        jd_col = None
        if 'jd' in df.columns:
            jd_col = 'jd'
        elif 'JD' in df.columns:
            jd_col = 'JD'
        else:
            raise KeyError(f"Cannot find 'jd' or 'JD' column in ImageFileCollection summary for {imagetyp}")

        df[jd_col] = df[jd_col].astype(float)
        df['diff'] = (df[jd_col] - jd_target).abs()
        
        if exptime is not None:
            # 'exptime' 또는 'EXPTIME' 키워드 찾기
            exp_col = None
            if 'exptime' in df.columns:
                exp_col = 'exptime'
            elif 'EXPTIME' in df.columns:
                exp_col = 'EXPTIME'
            
            if exp_col:
                df[exp_col] = df[exp_col].astype(float)
                # 가장 가까운 노출 시간의 다크 프레임 선택
                available_exptimes = df[exp_col].unique()
                closest_exptime = min(available_exptimes, key=lambda x: abs(x - float(exptime)))
                self.logger.info(f"Requested exptime {exptime}, found closest master dark {closest_exptime}")
                df = df[df[exp_col] == closest_exptime].copy() # .copy() to avoid SettingWithCopyWarning
            else:
                self.logger.warning(f"Cannot find 'exptime' or 'EXPTIME' for {imagetyp}, proceeding without exptime filter.")

        if df.empty:
             raise FileNotFoundError(f"No matching master {imagetyp} found for JD={jd_target}, EXPTIME={exptime}")

        idx = df['diff'].idxmin()
        selected_file = Path(coll.files_filtered(include_path=True)[idx])
        self.logger.info(f"Selected master {imagetyp}: {selected_file.name}")
        
        # 마스터 프레임은 .fits이므로 ccdproc.CCDData.read 사용
        return ccdproc.CCDData.read(selected_file, unit='adu')