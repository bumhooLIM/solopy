import logging
from astropy.io import fits
from astropy.coordinates import Angle, SkyCoord, GCRS, GeocentricTrueEcliptic
from astropy.coordinates import get_body, get_sun
from astropy.time import Time
import astropy.units as u
from datetime import datetime
from pathlib import Path

class FitsLv0:
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

        hdr.remove('ELAV') # mistaken key

        # --- Core metadata        
        hdr.comments['LT']  = 'Local Solar Time'
        hdr['EXPTIME']      = (float(hdr['EXPTIME']), '[sec] Exposure Time')
        hdr['CCDTEMP']      = (float(hdr['CCDTEMP']), '[C] CCD Temperature')
        hdr['PIXSZ']        = (float(hdr['PIXSZ']), '[micron] Pixel Size')
        hdr['FOCALLEN']     = (float(hdr['FOCALLEN']), '[mm] Telescope Focal Length')
        hdr['APTDIA']       = (float(hdr['APTDIA']), '[mm] Telescope Aperture Diameter')
        hdr['FOCUS']        = (int(hdr['FOCUS']), 'Focuser Focal Position')
        hdr['OBSERVER']     = (hdr['OBSERVER'].upper(), 'Observer')
        hdr['IMAGETYP']     = (hdr['IMAGETYP'].upper(), 'Image Type')
        hdr['BUNIT']        = ('adu', 'array data unit')
        hdr['FILTER']       = ('CLEAR', 'filter used')
        hdr["OBJECT"]       = Path(hdr["OBJECT"]).stem  # Remove file extension if any

        # --- Time information
        if ('UTC-END' in hdr):
            obstime = Time(hdr['UTC-END'].strip()) - 0.5 * hdr['EXPTIME'] * u.s
        else: # At the development stage, some files may have "UTC" as a observation end time
            obstime = Time(hdr['UTC'].strip()) - 0.5 * hdr['EXPTIME'] * u.s 
        hdr['JD']           = (obstime.jd, 'Julian Date')
        hdr['DATE-OBS']     = (obstime.isot, 'Observation Datetime')
        hdr['MJD-OBS']      = (obstime.mjd, 'Modified Julian Date')
        hdr['UTC']          = (obstime.utc.isot, 'Coordinated Universal Time')
        hdr['UTC-STA']      = ((obstime-0.5*hdr['EXPTIME']*u.s).utc.isot, 'Observation Start Time (UTC)')
        hdr['UTC-END']      = ((obstime+0.5*hdr['EXPTIME']*u.s).utc.isot, 'Observation End Time (UTC)')
        hdr['OBSDATE']      = (obstime.to_datetime().strftime('%Y%m%d'), 'YYYYMMDD observation date (UTC)')
        
        # --- Detector properties
        hdr['INSTRUME']     = ('FLI KL4040FI', 'Camera Model')
        hdr['DETECTOR']     = ('GSENSE4040 FI', 'CMOS Detector')
        hdr['EGAIN']        = (25.0, '[e-/ADU] Effective Gain')
        hdr['RDNOISE']      = (3.7, '[e-/pix] Readout Noise')
        
        # --- Telescope & Site
        for key, unit, comment in [
            ('RA',  'hourangle', '[deg] Telescope Right Ascension'),
            ('DEC', 'degree',    '[deg] Telescope Declination'),
            ('ALT', 'degree',    '[deg] Telescope Altitude'),
            ('AZ',  'degree',    '[deg] Telescope Azimuth'),
        ]:
            if isinstance(hdr.get(key), str):
                hdr[key] = (Angle(hdr[key], unit=unit).degree, comment)
        hdr['OBSERVAT']     = ('Sierra Remote Observatory', 'Observatory Name')
        hdr['OBSCODE']      = ("G80", 'MPC Observatory Code')
        hdr['LON']          = (-119.4, '[deg] Site Longitude')
        hdr['LAT']          = ( 37.07, '[deg] Site Latitude')
        hdr['ELEVAT']       = ( 1.405, '[km] Site Elevation')
        hdr['TELESCOP']     = ('Rowe-Ackermann Schmidt Astrograph 11-inch', 'Telescope Tube Model')

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
            hdr['SELONG'] = (gcrs.separation(sun_gcrs).value,  '[deg] Solar elongation')
            hdr['MELONG'] = (gcrs.separation(moon_gcrs).value, '[deg] Lunar elongation')
            hdr['GXLAT']  = (gal.b.value,  '[deg] Galactic latitude')
            hdr['GXLON']  = (gal.l.value,  '[deg] Galactic longitude')
            hdr['ECLAT']  = (ecl.lat.value,'[deg] Ecliptic latitude')
            hdr['ECLON']  = (ecl.lon.value,'[deg] Ecliptic longitude')
        except Exception as e:
            self.logger.warning(f"Coordinate calculation failed for {fpath.name}: {e}")

        hdr['HISTORY'] = f"({datetime.now().isoformat()}) Header updated. (solopy.FitsLv0.update_header)"

        # Write FITS with updated header
        new_hdu = fits.PrimaryHDU(data=data, header=hdr)
        try:
            new_hdu.writeto(fpath, overwrite=True)
            self.logger.info(f"Updated FitsLv0 header: {fpath.name}")
        except Exception as e:
            self.logger.error(f"Failed to write updated header to {fpath}: {e}")