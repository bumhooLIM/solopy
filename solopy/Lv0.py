import logging
import bz2
import io
from astropy.io import fits
from astropy.coordinates import Angle, SkyCoord, GCRS, GeocentricTrueEcliptic
from astropy.coordinates import get_body, get_sun
from astropy.time import Time
import astropy.units as u
from datetime import datetime
from pathlib import Path

class Lv0:
    """
    Class for calibration Level 0.
    - update FITS headers for Level-0 raw data (update_header).
    """

    def __init__(self, log_file: str = None):

        # set up console + optional file logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        self.logger.addHandler(handler)
        if log_file:
            file_h = logging.FileHandler(log_file)
            file_h.setFormatter(handler.formatter)
            self.logger.addHandler(file_h)

    # --- helpers ---
    def _is_bz2(self, path: Path) -> bool:
        return ".bz2" in "".join(path.suffixes)

    def _open_fits_any(self, path: Path):
        """
        Open .fits or .fits.bz2.

        Returns
        -------
        hdul : fits.HDUList
        was_bz2 : bool
        fobj : file-like or None   # keep this open until after reading data
        """
        if self._is_bz2(path):
            fobj = bz2.BZ2File(path, "rb")
            hdul = fits.open(fobj, memmap=False)
            return hdul, True, fobj
        else:
            hdul = fits.open(path, memmap=False)
            return hdul, False, None

    def _write_fits_any(self, path: Path, hdu: fits.PrimaryHDU, as_bz2: bool):
        if as_bz2:
            buf = io.BytesIO()
            hdu.writeto(buf, overwrite=True)
            with open(path, "wb") as f:
                f.write(bz2.compress(buf.getvalue()))
        else:
            hdu.writeto(path, overwrite=True)
    
    def update_header(self, fpath_fits):
        """
        Validate and update FITS headers for Level-0 images.

        Parameters
        ----------
        fpath_fits : str or pathlib.Path
            Path to the Level-0 FITS file to update. Supports .fits and .fits.bz2
        """
        fpath = Path(fpath_fits)
        if not fpath.exists():
            self.logger.error(f"File not found: {fpath}")
            return

        hdul, was_bz2, fobj = self._open_fits_any(fpath)
        try:  
            hdr = hdul[0].header.copy()
            data = hdul[0].data.copy()
        finally:

            try:
                hdul.close()
            finally:
                if fobj is not None:
                    fobj.close()

        # Remove all HISTORY cards if present
        while 'HISTORY' in hdr:
            hdr.remove('HISTORY')

        # --- Core metadata
        hdr.comments['LT']  = 'Local Time'
        hdr['EXPTIME']      = (float(hdr['EXPTIME']), 'Exposure Time (sec)')
        hdr['CCDTEMP']      = (float(hdr['CCDTEMP']), 'CCD Temperature (C)')
        hdr['PIXSZ']        = (float(hdr['PIXSZ']), 'Pixel Size (um)')
        hdr['JD']           = (float(hdr['JD']), 'Julian Date')
        hdr['DATE-OBS']     = (Time(hdr['JD'], format='jd').isot, 'Observation Datetime')
        hdr['MJD-OBS']      = (Time(hdr['JD'], format='jd').mjd, 'Modified Julian Date')
        hdr['FOCALLEN']     = (float(hdr['FOCALLEN']), 'Focal Length (mm)')
        hdr['APTDIA']       = (float(hdr['APTDIA']), 'Aperture Diameter (mm)')
        hdr['FOCUS']        = (int(hdr['FOCUS']), 'Focal Position')
        hdr['OBSERVER']     = (hdr['OBSERVER'].upper(), 'Observer')
        hdr['IMAGETYP']     = (hdr['IMAGETYP'].upper(), 'Image Type')

        # --- Telescope & Site
        for key, unit, comment in [
            ('RA',  'hourangle', 'Telescope RA (deg)'),
            ('DEC', 'degree',    'Telescope Dec (deg)'),
            ('ALT', 'degree',    'Telescope Alt (deg)'),
            ('AZ',  'degree',    'Telescope Az (deg)'),
        ]:
            if isinstance(hdr.get(key), str):
                hdr[key] = (Angle(hdr[key], unit=unit).degree, comment)
        hdr['LON']   = (-119.4, 'Site Longitude (deg)')
        hdr['LAT']   = ( 37.07, 'Site Latitude (deg)')
        hdr['ELEV']  = (  1.405,'Site Elevation (km)')

        # --- Processing flags
        for flag in ['BIASCORR', 'DARKCORR', 'FLATCORR']:
            hdr[flag] = (False, f'{flag.capitalize()} applied?')
        hdr['DATLEVEL']  = (0, 'Data Process Level')
        hdr['COMBINED']  = (False, 'Combined frames?')

        # --- Coordinates
        obstime = Time(hdr['JD'], format='jd')
        coords  = SkyCoord(ra=hdr['RA']*u.deg, dec=hdr['DEC']*u.deg, frame='icrs')
        gcrs    = coords.transform_to(GCRS(obstime=obstime))
        gal     = coords.galactic
        ecl     = coords.transform_to(GeocentricTrueEcliptic(equinox=obstime))

        sun_gcrs  = get_sun(obstime)
        moon_gcrs = get_body('Moon', obstime)
        hdr['SELONG'] = (gcrs.separation(sun_gcrs).value,  'Solar elongation (deg)')
        hdr['MELONG'] = (gcrs.separation(moon_gcrs).value, 'Lunar elongation (deg)')
        hdr['GXLAT']  = (gal.b.value,  'Galactic latitude (deg)')
        hdr['GXLON']  = (gal.l.value,  'Galactic longitude (deg)')
        hdr['ECLAT']  = (ecl.lat.value,'Ecliptic latitude (deg)')
        hdr['ECLON']  = (ecl.lon.value,'Ecliptic longitude (deg)')

        hdr['HISTORY'] = f"({datetime.now().isoformat()}) LV0 header updated."

        # --- Write back preserving compression
        new_hdu = fits.PrimaryHDU(data=data, header=hdr)
        self._write_fits_any(fpath, new_hdu, as_bz2=was_bz2)
        self.logger.info(f"Updated LV0 header: {fpath}")