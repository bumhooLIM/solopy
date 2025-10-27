import logging
from pathlib import Path
from astropy.io import fits
from astropy.coordinates import EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u
from datetime import datetime
import ccdproc
import astrometry # type: ignore
import sep
import numpy as np

class Lv1:
    """
    Class for Level-1 processing of RASA lcpy KL4040 science frames.

    Provides methods to solve and update WCS, and to apply bias, dark, and flat corrections.
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
        Solve for WCS using astrometry.net and update FITS header accordingly.
        """
        fpath_fits = Path(fpath_fits)
        outdir = Path(outdir)

        try:
            sci = ccdproc.CCDData.read(fpath_fits)
        except FileNotFoundError:
            self.logger.error(f"File not found: {fpath_fits}")
            return
        except ValueError:
            sci = ccdproc.CCDData.read(fpath_fits, unit="adu")
            self.logger.warning(f"BUNIT undefined; defaulting to 'adu': {fpath_fits}")
        except Exception as e:
            self.logger.error(f"Error reading FITS: {e}")
            return

        # detect stars
        sci.data = sci.data.astype(np.float32)
        bkg = sep.Background(sci.data)
        objs = sep.extract(sci.data - bkg.back(), thresh=3.0 * bkg.globalrms, minarea=5)
        bright = objs[np.argsort(objs['flux'])[::-1][:50]]
        coords = np.vstack((bright['x'], bright['y'])).T

        # solve WCS
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
            else:
                self.logger.warning("No WCS solution found.")

        # compute center coords
        cen = (sci.header['NAXIS1']//2, sci.header['NAXIS2']//2)
        sky = sci.wcs.pixel_to_world(*cen)
        loc = EarthLocation(lat=sci.header['LAT']*u.deg, lon=sci.header['LON']*u.deg, height=sci.header['ELEV']*u.km)
        altaz = sky.transform_to(AltAz(obstime=Time(sci.header['JD'], format='jd'), location=loc))
        sci.header['RACEN'] = (sky.ra.value, "deg center RA")
        sci.header['DECCEN'] = (sky.dec.value, "deg center DEC")
        sci.header['ALTCEN'] = (altaz.alt.value, "deg center Alt")
        sci.header['AZCEN'] = (altaz.az.value, "deg center Az")

        outpath = Path(outdir)/ (fpath_fits.stem + ".wcs" + fpath_fits.suffix)
        fits.PrimaryHDU(data=sci.data, header=sci.header).writeto(outpath, overwrite=True)
        self.logger.info(f"WCS updated: {outpath.name}")
        if return_fpath:
            return outpath

    def preprocessing(self, fpath_fits, outdir, masterdir, return_fpath=True):
        """
        Subtract bias, dark, mask bad pixels, and flat-correct a science frame.
        """
        fpath_fits = Path(fpath_fits)
        outdir = Path(outdir)
        masterdir = Path(masterdir)
        outdir.mkdir(parents=True, exist_ok=True)

        try:
            sci = ccdproc.CCDData.read(fpath_fits)
        except FileNotFoundError:
            self.logger.error(f"File not found: {fpath_fits}")
            return
        except ValueError:
            sci = ccdproc.CCDData.read(fpath_fits, unit="adu")
            self.logger.warning(f"BUNIT undefined; defaulting to 'adu': {fpath_fits}")
        except Exception as e:
            self.logger.error(f"Error reading FITS: {e}")
            return

        # load masters
        mbias = self._select_master(masterdir, 'BIAS', sci.header['JD'])
        mdark = self._select_master(masterdir, 'DARK', sci.header['JD'], sci.header['EXPTIME'])
        mflat = self._select_master(masterdir, 'FLAT', sci.header['JD'])

        # bias
        bsci = ccdproc.subtract_bias(sci, mbias)
        bsci.meta['BIASCORR'] = True
        self.logger.info("Bias subtracted.")
        # dark
        bdsci = ccdproc.subtract_dark(bsci, mdark, exposure_time="EXPTIME", exposure_unit=u.second)
        bdsci.meta['DARKCORR'] = True
        self.logger.info("Dark subtracted.")
        # mask
        bdsci.mask = bdsci.data < 0
        bdsci.data = np.nan_to_num(np.clip(bdsci.data, 0, None))
        self.logger.info("Bad pixels masked.")
        # flat
        psci = ccdproc.flat_correct(bdsci, mflat)
        psci.meta['FLATCORR'] = True
        self.logger.info("Flat corrected.")

        # build output name
        rac = int(round(psci.header['RACEN']))
        dec = int(round(psci.header['DECCEN']))
        pm = 'p' if dec >= 0 else 'n'
        exp = int(round(psci.header['EXPTIME']))
        obst = Time(psci.header['DATE-OBS']).strftime("%Y%m%d%H%M%S")
        outname = f"kl4040.sci.{rac:03d}.{pm}{abs(dec):02d}.{exp:03d}.{obst}.fits"
        outpath = outdir / outname
        fits.PrimaryHDU(data=psci.data, header=psci.header).writeto(outpath, overwrite=True)
        self.logger.info(f"Preprocessing complete: {outname}")
        if return_fpath:
            return outpath

    def _select_master(self, masterdir, imagetyp, jd_target, exptime=None):
        """
        Helper to select the closest master frame by JD (and exposure time if given).
        """
        coll = ccdproc.ImageFileCollection(masterdir, glob_include="*.fits").filter(imagetyp=imagetyp)
        df = coll.summary.to_pandas()
        df['jd'] = df.get('jd', df.get('JD')).astype(float)
        df['diff'] = (df['jd'] - jd_target).abs()
        if exptime is not None and 'exptime' in df.columns:
            df = df[df['exptime'].astype(float) == float(exptime)]
        idx = df['diff'].idxmin()
        return ccdproc.CCDData.read(Path(coll.files_filtered(include_path=True)[idx]), unit='adu')
