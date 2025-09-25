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

    # def update_header(self, fpath_fits):
    #     """
    #     Update basic headers.
    #     """
    #     fpath = Path(fpath_fits)
    #     if not fpath.exists():
    #         self.logger.error(f"File not found: {fpath}")
    #         return

    #     hdul = fits.open(fpath)
    #     hdr = hdul[0].header.copy()

    #     ### Basic header info.
    #     hdr.comments["LT"]  = "Local Time"
    #     # hdr.comments["UTC"] = "Universal Time Coordinated"
    #     hdr["EXPTIME"]  = (float(hdr["EXPTIME"]), "Exposure Time (sec)")
    #     hdr["CCDTEMP"]  = (float(hdr["CCDTEMP"]), "CCD Temperature (C)")
    #     hdr["PIXSZ"]    = (float(hdr["PIXSZ"]), "Pixel Size (um)")
    #     hdr["JD"]       = (float(hdr["JD"]), "Julian Date")
    #     hdr["DATE-OBS"] = (Time(hdr["JD"], format="jd").isot, "Observation Datetime")
    #     hdr["MJD-OBS"]  = (Time(hdr["JD"], format="jd").mjd, "Modified Julian Date")
    #     hdr["FOCALLEN"] = (float(hdr["FOCALLEN"]), "Focal Length (mm)")
    #     hdr["APTDIA"]   = (float(hdr["APTDIA"]), "Aperture Diameter (mm)")
    #     hdr["FOCUS"]    = (int(hdr["FOCUS"]), "Focal Position")
    #     hdr["OBSERVER"] = (hdr["OBSERVER"].upper(), "Observer")
    #     hdr["IMAGETYP"] = (hdr["IMAGETYP"].upper(), "Image Type (BIAS, DARK, FLAT, LIGHT)")

    #     ### Telescope & Site coordinates
    #     hdr["RA"]       = (Angle(hdr["RA"], unit='hourangle').degree, "Telescope Right Ascension (deg)") if type(hdr["RA"]) is str else hdr["RA"]
    #     hdr["DEC"]      = (Angle(hdr["DEC"], unit='degree').degree, "Telescope Declination (deg)") if type(hdr["DEC"]) is str else hdr["DEC"]
    #     hdr["ALT"]      = (Angle(hdr["ALT"], unit='degree').degree, "Telescope Altitude (deg)") if type(hdr["ALT"]) is str else hdr["ALT"]
    #     hdr["AZ"]       = (Angle(hdr["AZ"], unit='degree').degree, "Telescope Azimuth (deg)") if type(hdr["AZ"]) is str else hdr["AZ"]
    #     hdr["LON"]      = (-119.4, "Site Longitude (deg)")
    #     hdr["LAT"]      = (37.07, "Site Latitude (deg)")
    #     hdr["ELEV"]     = (1.405, "Site Elevation (km)")

    #     ### Additional info.
    #     hdr["BUNIT"]    = ("adu", "array data unit")
    #     hdr["OBSERVAT"] = "Sierra Remote Observatories"
    #     hdr["DETECTOR"] = ("FLI Camera KL4040 FI", "Detector Name")
    #     hdr["FILTER"]   = ("CLEAR", "Filter Name")
    #     hdr["OBSDATE"]  = (Time(hdr["JD"], format="jd").to_datetime().strftime("%Y%m%d"), "YYYYMMD (UT)")

    #     hdr["DATLEVEL"] = (0, "Data Process Level (0-2)")
    #     hdr["COMBINED"] = (False, "Is image combined? (True/False)")
    #     hdr["BIASCORR"] = (False, "Bias Corrected (True/False)")
    #     hdr["DARKCORR"] = (False, "Dark Corrected (True/False)")
    #     hdr["FLATCORR"] = (False, "Flat Corrected (True/False)")
    
    #     ### Coordinates info.
    #     obstime = Time(hdr["JD"], format="jd") # observation time

    #     coords_icrs = SkyCoord(
    #         ra=hdr["RA"]*u.deg,
    #         dec=hdr["DEC"]*u.deg,
    #         frame='icrs'
    #         )
    #     coords_gcrs = coords_icrs.transform_to(GCRS(obstime=obstime))

    #     # Galactic coordinates
    #     gal = coords_icrs.galactic
        
    #     # Ecliptic coordinates
    #     ecl = coords_icrs.transform_to(GeocentricTrueEcliptic(equinox=obstime))
        
    #     # Solar elongation
    #     scoord_gcrs = get_sun(obstime)
    #     selong = coords_gcrs.separation(scoord_gcrs)

    #     # Lunar elongation
    #     mcoord_gcrs = get_body("Moon", obstime)
    #     lelong = coords_gcrs.separation(mcoord_gcrs)
        
    #     hdr["SELONG"] = (selong.value, "Field Solar Elongation (deg)")
    #     hdr["MELONG"] = (lelong.value, "Field Lunar Elongation (deg)")
    #     hdr["ECLAT"]  = (ecl.lat.value, "Field Ecliptic Latitude (deg)")
    #     hdr["ECLON"]  = (ecl.lon.value, "Field Ecliptic Longitude (deg)")
    #     hdr["GXLAT"]  = (gal.l.value, "Field Galactic Latitude (deg)")
    #     hdr["GXLON"]  = (gal.b.value, "Field Galactic Longitude (deg)")
        
    #     hdr["HISTORY"] = f"({datetime.now().isoformat()}) Lv.0 header updated."
        
    #     hdul_updated = fits.PrimaryHDU(data = hdul[0].data, header=hdr)
        
    #     hdul_updated.writeto(fpath_fits, overwrite=True)
    #     print(f"({datetime.now().isoformat()}) Lv.0 header updated: {fpath_fits}")

    #     new_hdu = fits.PrimaryHDU(data=hdul[0].data, header=hdr)
    #     new_hdu.writeto(fpath, overwrite=True)
    #     self.logger.info(f"Updated LV0 header: {fpath}")


    def update_header(self, fpath_fits):
        """
        Validate and update FITS headers.

        Parameters
        ----------
        fpath_fits : str or pathlib.Path
            Path to the Level-0 FITS file to update.
        """
        fpath = Path(fpath_fits)
        if not fpath.exists():
            self.logger.error(f"File not found: {fpath}")
            return

        hdul = fits.open(fpath)
        hdr = hdul[0].header.copy()

        ### Core metadata fields
        hdr.comments['LT']  = 'Local Time'
        # hdr.comments['UTC'] = 'Universal Time Coordinated'
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

        ### Telescope & Site
        for key, unit, comment in [('RA', 'hourangle', 'Telescope RA (deg)'),
                                   ('DEC', 'degree', 'Telescope Dec (deg)'),
                                   ('ALT', 'degree', 'Telescope Alt (deg)'),
                                   ('AZ', 'degree', 'Telescope Az (deg)')]:
            if isinstance(hdr[key], str):
                hdr[key] = (Angle(hdr[key], unit=unit).degree, comment)
        hdr['LON']   = (-119.4, 'Site Longitude (deg)')
        hdr['LAT']   = (37.07, 'Site Latitude (deg)')
        hdr['ELEV']  = (1.405, 'Site Elevation (km)')

        ### Processing flags
        flags = ['BIASCORR', 'DARKCORR', 'FLATCORR']
        for fl in flags:
            hdr[fl] = (False, f'{fl.capitalize()} applied?')
        hdr['DATLEVEL']  = (0, 'Data Process Level')
        hdr['COMBINED']  = (False, 'Combined frames?')

        ### Additional coordinates
        obstime = Time(hdr['JD'], format='jd')
        coords = SkyCoord(ra=hdr['RA']*u.deg, dec=hdr['DEC']*u.deg, frame='icrs')
        gcrs = coords.transform_to(GCRS(obstime=obstime))
        gal  = coords.galactic
        ecl  = coords.transform_to(GeocentricTrueEcliptic(equinox=obstime))

        # Solar & lunar elongation
        sun_gcrs  = get_sun(obstime)
        moon_gcrs = get_body('Moon', obstime)
        hdr['SELONG'] = (gcrs.separation(sun_gcrs).value, 'Solar elongation (deg)')
        hdr['MELONG'] = (gcrs.separation(moon_gcrs).value, 'Lunar elongation (deg)')

        # Galactic & ecliptic coords
        hdr['GXLAT'] = (gal.b.value, 'Galactic latitude (deg)')
        hdr['GXLON'] = (gal.l.value, 'Galactic longitude (deg)')
        hdr['ECLAT'] = (ecl.lat.value, 'Ecliptic latitude (deg)')
        hdr['ECLON'] = (ecl.lon.value, 'Ecliptic longitude (deg)')

        hdr['HISTORY'] = f"({datetime.now().isoformat()}) LV0 header updated."

        # Write back
        new_hdu = fits.PrimaryHDU(data=hdul[0].data, header=hdr)
        new_hdu.writeto(fpath, overwrite=True)
        self.logger.info(f"Updated LV0 header: {fpath}")