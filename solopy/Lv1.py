import logging
from pathlib import Path
from astropy.io import fits
from astropy.coordinates import EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u
from datetime import datetime
import ccdproc
from ccdproc import CCDData # ccdproc.CCDData.read 대신 CCDData 클래스 직접 사용
import astrometry # type: ignore
import sep
import numpy as np
from . import _utils # <-- 1. 새로 만든 유틸리티 임포트

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
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        self.logger.addHandler(handler)

        if log_file:
            file_h = logging.FileHandler(log_file)
            file_h.setFormatter(handler.formatter)
            self.logger.addHandler(file_h)
            
    def update_wcs(self, fpath_fits, outdir, return_fpath=True):
        """
        Solve for WCS using astrometry.net and update FITS header.
        Reads .fits/.fits.bz2 and writes to .wcs.fits.bz2.
        """
        fpath_fits = Path(fpath_fits)
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True) # 출력 디렉터리 생성

        # --- 2. 수정된 파일 읽기 ---
        try:
            hdul, was_bz2, fobj = _utils.open_fits_any(fpath_fits)
            # BUNIT이 없는 경우를 대비해 unit='adu'를 기본값으로 가정
            sci = CCDData(hdul[0].data, meta=hdul[0].header, unit='adu') 
        except FileNotFoundError:
            self.logger.error(f"File not found: {fpath_fits}")
            return
        except Exception as e:
            self.logger.error(f"Error reading FITS {fpath_fits}: {e}")
            if 'hdul' in locals() and hdul: hdul.close()
            if 'fobj' in locals() and fobj: fobj.close()
            return
        finally:
            if 'hdul' in locals() and hdul: hdul.close()
            if 'fobj' in locals() and fobj: fobj.close()
        # --- (읽기 완료) ---

        # detect stars
        sci.data = sci.data.astype(np.float32)
        try:
            bkg = sep.Background(sci.data)
            objs = sep.extract(sci.data - bkg.back(), thresh=3.0 * bkg.globalrms, minarea=5)
            bright = objs[np.argsort(objs['flux'])[::-1][:50]]
            coords = np.vstack((bright['x'], bright['y'])).T
        except Exception as e:
            self.logger.error(f"Source detection failed for {fpath_fits.name}: {e}")
            return

        # solve WCS
        wcs_solved = False
        try:
            with astrometry.Solver(
                astrometry.series_4100.index_files(cache_directory="astrometry_cache", scales={8,9,10})
            ) as solver:
                sol = solver.solve(
                    stars=coords,
                    size_hint=astrometry.SizeHint(lower_arcsec_per_pixel=2.95, upper_arcsec_per_pixel=3.00),
                    position_hint=astrometry.PositionHint(
                        ra_deg=sci.header["RA"], dec_deg=sci.header["DEC"], radius_deg=5.0
                    ),
                    solution_parameters=astrometry.SolutionParameters(
                        logodds_callback=lambda l: astrometry.Action.STOP if len(l)>=10 else astrometry.Action.CONTINUE
                    )
                )
                if sol.has_match():
                    best = sol.best_match()
                    self.logger.info(f"WCS match: RA={best.center_ra_deg:.5f}, DEC={best.center_dec_deg:.5f}, scale={best.scale_arcsec_per_pixel:.3f}" )
                    sci.wcs = best.astropy_wcs()
                    hdr = best.astropy_wcs().to_header(relax=False)
                    sci.header.extend(hdr, update=True)
                    sci.header['PIXSCALE'] = (best.scale_arcsec_per_pixel, "arcsec/pixel")
                    sci.header['HISTORY'] = f"({datetime.now().isoformat()}) WCS updated"
                    wcs_solved = True
                else:
                    self.logger.warning(f"No WCS solution found for {fpath_fits.name}.")
        except Exception as e:
             self.logger.warning(f"Astrometry.net solver failed for {fpath_fits.name}: {e}")

        # compute center coords (WCS가 풀린 경우에만)
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


        # --- 3. 수정된 파일 쓰기 ---
        # "image.fits.bz2" -> "image.wcs.fits.bz2"
        true_stem = _utils.get_true_stem(fpath_fits)
        out_name = f"{true_stem}.wcs.fits.bz2" # .fits.bz2로 강제
        outpath = outdir / out_name

        new_hdu = fits.PrimaryHDU(data=sci.data, header=sci.header)
        try:
            _utils.write_fits_any(outpath, new_hdu, as_bz2=True) # as_bz2=True로 압축 저장
            self.logger.info(f"WCS updated: {outpath.name}")
        except Exception as e:
            self.logger.error(f"Failed to write {outpath}: {e}")
            return

        if return_fpath:
            return outpath

    def preprocessing(self, fpath_fits, outdir, masterdir, return_fpath=True):
        """
        Subtract bias, dark, mask bad pixels, and flat-correct a science frame.
        Reads .fits/.fits.bz2 and writes to .fits.bz2.
        Master frames are assumed to be uncompressed .fits files.
        """
        fpath_fits = Path(fpath_fits)
        outdir = Path(outdir)
        masterdir = Path(masterdir)
        outdir.mkdir(parents=True, exist_ok=True)

        # --- 1. 수정된 파일 읽기 (Science Frame) ---
        try:
            hdul, was_bz2, fobj = _utils.open_fits_any(fpath_fits)
            sci = CCDData(hdul[0].data, meta=hdul[0].header, unit='adu')
        except FileNotFoundError:
            self.logger.error(f"File not found: {fpath_fits}")
            return
        except Exception as e:
            self.logger.error(f"Error reading FITS {fpath_fits}: {e}")
            if 'hdul' in locals() and hdul: hdul.close()
            if 'fobj' in locals() and fobj: fobj.close()
            return
        finally:
            if 'hdul' in locals() and hdul: hdul.close()
            if 'fobj' in locals() and fobj: fobj.close()
        # --- (읽기 완료) ---

        try:
            # load masters (Master-frames는 .fits로 가정, _select_master는 ccdproc.read 사용)
            mbias = self._select_master(masterdir, 'BIAS', sci.header['JD'])
            mdark = self._select_master(masterdir, 'DARK', sci.header['JD'], sci.header['EXPTIME'])
            mflat = self._select_master(masterdir, 'FLAT', sci.header['JD']) # 참고: Flat은 필터별로 필요할 수 있음
        except Exception as e:
            self.logger.error(f"Failed to load master frames for {fpath_fits.name}: {e}")
            return
            
        # bias
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
             self.logger.warning(f"RACEN/DECCEN not in header for {fpath_fits.name}. Using original name for output.")
             true_stem = _utils.get_true_stem(fpath_fits)
             outname = f"{true_stem}.proc.fits.bz2"
        else:
            rac = int(round(psci.header['RACEN']))
            dec = int(round(psci.header['DECCEN']))
            pm = 'p' if dec >= 0 else 'n'
            exp = int(round(psci.header['EXPTIME']))
            obst = Time(psci.header['DATE-OBS']).strftime("%Y%m%d%H%M%S")
            outname = f"kl4040.sci.{rac:03d}.{pm}{abs(dec):02d}.{exp:03d}.{obst}.fits.bz2"
            
        # # --- 3. 수정된 파일 쓰기 ---
        # # build output name (WCS가 있어야 RACEN, DECCEN 사용 가능)
        # if 'RACEN' not in psci.header or 'DECCEN' not in psci.header:
        #      self.logger.warning(f"RACEN/DECCEN not in header for {fpath_fits.name}. Using original name for output.")
        #      true_stem = _utils.get_true_stem(fpath_fits)
        #      outname = f"{true_stem}.proc.fits.bz2"
        # else:
        #     rac = int(round(psci.header['RACEN']))
        #     dec = int(round(psci.header['DECCEN']))
        #     pm = 'p' if dec >= 0 else 'n'
        #     exp = int(round(psci.header['EXPTIME']))
        #     obst = Time(psci.header['DATE-OBS']).strftime("%Y%m%d%H%M%S")
        #     # 파일명에 .fits.bz2 강제
        #     outname = f"kl4040.sci.{rac:03d}.{pm}{abs(dec):02d}.{exp:03d}.{obst}.fits.bz2"
        
        outpath = outdir / outname
        
        # # CCDData 객체를 PrimaryHDU로 변환
        # new_hdu = fits.PrimaryHDU(data=psci.data, header=psci.header)
        
        # try:
        #     _utils.write_fits_any(outpath, new_hdu, as_bz2=True) # as_bz2=True로 압축 저장
        #     self.logger.info(f"Preprocessing complete: {outname}")
        # except Exception as e:
        #     self.logger.error(f"Failed to write {outpath}: {e}")
        #     return
            
        # if return_fpath:
        #     return outpath
        
        # 1. Primary HDU (Science Data) 생성
        primary_hdu = fits.PrimaryHDU(data=psci.data, header=psci.header)
        
        # 2. Mask HDU (ImageHDU) 생성
        # FITS 표준은 boolean이 아닌 정수(0, 1)를 권장. uint8이 가장 효율적.
        if psci.mask is not None:
            mask_data = psci.mask.astype(np.uint8)
            mask_hdu = fits.ImageHDU(data=mask_data, name='MASK')
            self.logger.info(f"Mask extension created for {fpath_fits.name}")
            # HDU 리스트로 묶기
            hdul_out = fits.HDUList([primary_hdu, mask_hdu])
        else:
            # 마스크가 없으면 Primary HDU만 사용
            self.logger.warning(f"No mask was generated for {fpath_fits.name}, saving primary HDU only.")
            hdul_out = fits.HDUList([primary_hdu])
        
        try:
            # _utils.write_fits_any는 HDUList 객체를 받아 압축 저장함
            _utils.write_fits_any(outpath, hdul_out, as_bz2=True)
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