import logging
import bz2
import shutil
from astropy.io import fits
from astropy.coordinates import Angle, SkyCoord, GCRS, GeocentricTrueEcliptic
from astropy.coordinates import get_body, get_sun
from astropy.time import Time
import astropy.units as u
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

class FitsLv0:
    """
    Class for calibration Level 0.
    - batch decompress .bz2 files.
    - update FITS headers for Level-0 raw data (in-place).
    """

    def __init__(self, log_file: str | None = None):

        # set up console + optional file logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        
        # Check if handlers already exist before adding new ones
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self.logger.addHandler(handler)
            
            if log_file:
                file_h = logging.FileHandler(log_file)
                file_h.setFormatter(handler.formatter)
                self.logger.addHandler(file_h)

    def batch_decompress(self, in_dir: Path, out_dir: Path, delete_source=False):
        """
        Decompresses all '.fits.bz2' files directly within the specified source directory.
        """
        in_dir = Path(in_dir)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        bz2_files = list(in_dir.glob("*.fits.bz2"))
        if not bz2_files:
            self.logger.warning(f"No .fits.bz2 files found in {in_dir}")
            return
            
        self.logger.info(f"Decompressing {len(bz2_files)} files from {in_dir} to {out_dir}...")
        
        for f_bz2 in tqdm(bz2_files, desc="Decompressing"):
            f_fits = out_dir / f_bz2.stem 
            try:
                # Use multi-context manager for cleaner IO handling
                with bz2.BZ2File(f_bz2, 'rb') as fin, open(f_fits, 'wb') as fout:
                    shutil.copyfileobj(fin, fout)
                if delete_source:
                    f_bz2.unlink()
            except Exception as e:
                self.logger.error(f"Failed to decompress {f_bz2.name}: {e}")
                # Defense: Delete partially written corrupted FITS file
                if f_fits.exists():
                    f_fits.unlink()
    
    def update_header(self, fpath_fits):
        """
        Validate and update FITS headers for Level-0 images.
        (Operates on .fits files IN-PLACE without loading data into memory)
        """
        fpath = Path(fpath_fits)
        if not fpath.exists():
            self.logger.error(f"File not found: {fpath}")
            return

        # Open FITS in 'update' mode. This edits the header directly on the disk.
        try:
            with fits.open(fpath, mode='update') as hdul:
                hdr = hdul[0].header
                
                # --- Cleanup Keys
                while 'HISTORY' in hdr:
                    del hdr['HISTORY']
                
                # Pythonic deletion (avoid if/else None)
                if 'ELAV' in hdr:
                    del hdr['ELAV']

                # --- Core metadata (Defensive parsing using .get)
                exptime = float(hdr.get('EXPTIME', 0.0))
                
                # Update comments safely
                if 'LT' in hdr:
                    hdr.comments['LT'] = 'Local Solar Time'
                    
                hdr['EXPTIME']  = (exptime, '[sec] Exposure Time')
                hdr['CCDTEMP']  = (float(hdr.get('CCDTEMP', 0.0)), '[C] CCD Temperature')
                hdr['PIXSZ']    = (float(hdr.get('PIXSZ', 0.0)), '[micron] Pixel Size')
                hdr['FOCALLEN'] = (float(hdr.get('FOCALLEN', 0.0)), '[mm] Telescope Focal Length')
                hdr['APTDIA']   = (float(hdr.get('APTDIA', 0.0)), '[mm] Telescope Aperture Diameter')
                hdr['FOCUS']    = (int(hdr.get('FOCUS', 0)), 'Focuser Focal Position')
                hdr['OBSERVER'] = (str(hdr.get('OBSERVER', '')).upper(), 'Observer')
                hdr['IMAGETYP'] = (str(hdr.get('IMAGETYP', '')).upper(), 'Image Type')
                hdr['BUNIT']    = ('adu', 'array data unit')
                hdr['FILTER']   = ('CLEAR', 'filter used')
                
                # Safe path stem extraction
                obj_name = str(hdr.get('OBJECT', 'Unknown'))
                hdr["OBJECT"]   = (Path(obj_name).stem, 'Target Object')

                # --- Time information
                time_str = str(hdr.get('UTC-END', hdr.get('UTC', ''))).strip()
                if not time_str:
                    raise ValueError("Could not find UTC or UTC-END in header")

                obstime = Time(time_str) - 0.5 * exptime * u.s
                
                hdr['JD']       = (obstime.jd, 'Julian Date')
                hdr['DATE-OBS'] = (obstime.isot, 'Observation Datetime')
                hdr['MJD-OBS']  = (obstime.mjd, 'Modified Julian Date')
                hdr['UTC']      = (obstime.utc.isot, 'Coordinated Universal Time')
                hdr['UTC-STA']  = ((obstime - 0.5 * exptime * u.s).utc.isot, 'Observation Start Time (UTC)')
                hdr['UTC-END']  = ((obstime + 0.5 * exptime * u.s).utc.isot, 'Observation End Time (UTC)')
                hdr['OBSDATE']  = (obstime.to_datetime().strftime('%Y%m%d'), 'YYYYMMDD observation date (UTC)')
                
                # --- Detector properties
                hdr['INSTRUME'] = ('FLI KL4040FI', 'Camera Model')
                hdr['DETECTOR'] = ('GSENSE4040 FI', 'CMOS Detector')
                hdr['EGAIN']    = (18.69, '[e-/ADU] Effective Gain')
                hdr['RDNOISE']  = (3.7, '[e-/pix] Readout Noise')
                
                # --- Telescope & Site (Handles both strings and pre-parsed floats)
                for key, unit, comment in [
                    ('RA',  'hourangle', '[deg] Telescope Right Ascension'),
                    ('DEC', 'degree',    '[deg] Telescope Declination'),
                    ('ALT', 'degree',    '[deg] Telescope Altitude'),
                    ('AZ',  'degree',    '[deg] Telescope Azimuth'),
                ]:
                    val = hdr.get(key)
                    if isinstance(val, str):
                        hdr[key] = (Angle(val, unit=unit).degree, comment)
                    elif isinstance(val, (int, float)):
                        # If already numeric, just update the comment
                        hdr[key] = (float(val), comment)

                hdr['OBSERVAT'] = ('Sierra Remote Observatory', 'Observatory Name')
                hdr['OBSCODE']  = ("G80", 'MPC Observatory Code')
                hdr['LON']      = (-119.4, '[deg] Site Longitude')
                hdr['LAT']      = ( 37.07, '[deg] Site Latitude')
                hdr['ELEVAT']   = ( 1.405, '[km] Site Elevation')
                hdr['TELESCOP'] = ('Rowe-Ackermann Schmidt Astrograph 11-inch', 'Telescope Tube Model')

                # --- Processing flags
                for flag in ['BIASCORR', 'DARKCORR', 'FLATCORR']:
                    hdr[flag] = (False, f'{flag.capitalize()} applied?')
                hdr['DATLEVEL'] = (0, 'Data Process Level')
                hdr['COMBINED'] = (False, 'Combined frames?')

                # --- Coordinates
                try:
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

                # hdul.flush() is automatically called upon exiting the 'with' block 
                # when opened in 'update' mode.
                # self.logger.info(f"Updated FitsLv0 header: {fpath.name}")

        except Exception as e:
            self.logger.error(f"Failed to process FITS header for {fpath}: {e}")