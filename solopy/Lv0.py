import logging
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
    - Assumes input files are uncompressed .fits files.
    """

    def __init__(self, log_file: str | None = None):

        # set up console + optional file logging
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
    
    def update_header(self, fpath_fits):
        """
        Validate and update FITS headers for Level-0 images.
        (Operates on .fits files IN-PLACE)
        """
        fpath = Path(fpath_fits)
        if not fpath.exists():
            self.logger.error(f"File not found: {fpath}")
            return

        # Open FITS
        hdul = None
        try:
            hdul = fits.open(fpath)
            hdr  = hdul[0].header.copy()
            data = hdul[0].data.copy()
        except Exception as e:
            self.logger.error(f"Failed to open {fpath}: {e}")
            return
        finally:
            if hdul:
                hdul.close()

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
        hdr['BUNIT']        = ('adu', 'array data unit')

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
        try:
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
        except Exception as e:
            self.logger.warning(f"Coordinate calculation failed for {fpath.name}: {e}")

        hdr['HISTORY'] = f"({datetime.now().isoformat()}) LV0 header updated. (solopy.Lv0.update_header)"
             
        # Write FITS with updated header
        new_hdu = fits.PrimaryHDU(data=data, header=hdr)
        try:
            new_hdu.writeto(fpath, overwrite=True)
            self.logger.info(f"Updated LV0 header: {fpath.name}")
        except Exception as e:
            self.logger.error(f"Failed to write updated header to {fpath}: {e}")