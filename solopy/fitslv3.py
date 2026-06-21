import logging
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import kete

# Assuming your existing modules are importable
import solopy
import skyloc as sloc

class FitsLv3:
    def __init__(self, orb_path, gaia_path, log_file=None):
        """
        Initialize the Level-3 Science Processor.
        Pre-loads heavy orbital and catalog databases to optimize memory.
        """
        # 1. Setup Logging
        self.logger = logging.getLogger("FitsLv3")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self.logger.addHandler(handler)
            if log_file:
                file_h = logging.FileHandler(log_file)
                file_h.setFormatter(handler.formatter)
                self.logger.addHandler(file_h)
        
        self.logger.info("Initializing Level-3 Processor...")
        
        # 2. Load Orbital Database
        try:
            self.logger.info(f"Loading orbit database from {orb_path}")
            self.orb, _ = sloc.fetch_orb(Path(orb_path), update_output=999)
        except Exception as e:
            self.logger.error(f"Failed to load orbit database: {e}")
            raise
            
        # 3. Load Gaia Catalog
        try:
            self.logger.info(f"Loading Gaia catalog from {gaia_path}")
            self.gaia_all = np.load(Path(gaia_path), mmap_mode='r')
        except Exception as e:
            self.logger.error(f"Failed to load Gaia catalog: {e}")
            raise
            
        # Initialize an instance of Lv2 for its powerful photometry and centroiding tools
        self.lv2 = solopy.FitsLv2(log_file=log_file)
        self.logger.info("Level-3 Processor Ready.")

    def predict_targets(self, science_summary, vmag_upper=16.5):
        """
        Uses kete and skyloc to predict asteroid positions across all provided frames.
        Cross-matches predictions with Gaia to flag potential stellar blends.
        """
        self.logger.info(f"Predicting targets for {len(science_summary)} frames (V < {vmag_upper})...")
        
        fovs = []
        for idx, row in science_summary.iterrows():
            fpath = Path(row['file'])
            try:
                # Fast header read without loading the heavy image array
                hdr = fits.getheader(fpath)
                wcs = WCS(hdr)
                
                jd_mid = hdr["JD"]
                observatory = (hdr['LAT'], hdr['LON'], hdr['ELEVAT'])
                
                observer = kete.spice.earth_pos_to_ecliptic(
                    jd_mid, *observatory, name=fpath.stem
                )
                fovs.append(kete.fov.RectangleFOV.from_wcs(wcs, observer))
            except Exception as e:
                self.logger.warning(f"Failed to create FOV for {fpath.name}: {e}")
                continue

        fovs_col = sloc.FOVCollection(fovs)
        
        # Run N-body Integrator
        self.logger.info("Running N-body integrator (skyloc/kete)... This may take a few minutes.")
        sl1, sl2 = sloc.locator_twice(
            fovs=fovs_col,
            orb=self.orb,
            include_asteroids=(False, True),
            dt_limit=(3, 0.1),
            add_obsid=True,
            drop_obsindex=True,
            add_jds=True
        )

        # Filter by magnitude
        eph_all = sl2.eph[sl2.eph['vmag'] < vmag_upper].reset_index(drop=True)
        
        if eph_all.empty:
            self.logger.warning("No targets found in the specified magnitude range.")
            return eph_all

        # Cross-match with Gaia to identify blends
        self.logger.info("Cross-matching predictions with Gaia to assess blending risks...")
        skycoords_target = SkyCoord(ra=eph_all['ra'].values*u.deg, dec=eph_all['dec'].values*u.deg)
        
        nearest_gaia_sources = solopy.GaiaQuery.query_nearest_gaia(
            target_coords=skycoords_target,
            gaia_data=self.gaia_all,
            gaia_band="g"
        )

        if nearest_gaia_sources:
            source_id, gmag, dist = zip(*nearest_gaia_sources)
            eph_all['nearest_gaia_source_id'] = source_id
            eph_all['nearest_gaia_gmag'] = gmag
            eph_all['nearest_gaia_dist_arcsec'] = dist
        else:
            eph_all['nearest_gaia_source_id'] = np.nan
            eph_all['nearest_gaia_gmag'] = np.nan
            eph_all['nearest_gaia_dist_arcsec'] = np.nan

        self.logger.info(f"Successfully predicted {len(eph_all)} total asteroid appearances.")
        return eph_all

    def extract_sso_photometry(self, science_summary, eph, psf_dir, ap_in_out=(1.5, 3.0, 4.0), base_tile_size=500):
        """
        Executes precision centroiding and spatially varying aperture photometry
        for the predicted asteroids in each frame.
        """
        psf_dir = Path(psf_dir)
        sso_phot_list = []
        
        self.logger.info("Commencing target photometry extraction...")

        for idx, row in science_summary.iterrows():
            fpath_lv1 = Path(row['file'])
            obsid = fpath_lv1.stem
            subdir_name = fpath_lv1.parent.name
            
            # 1. Filter targets for this specific frame
            eph_obsid = eph[eph['obsid'] == obsid].copy()
            
            if eph_obsid.empty:
                continue
                
            # 2. Load Image Data
            try:
                with fits.open(fpath_lv1) as hdul:
                    data = hdul[0].data.astype(np.float32)
                    hdr = hdul[0].header
                    wcs = WCS(hdr)
                    mask = hdul[1].data.astype(bool) if len(hdul) > 1 else np.zeros_like(data, dtype=bool)
                    
                    egain = float(hdr.get("EGAIN", 18.69))
                    rdnoise = float(hdr.get("RDNOISE", 3.9))
                    
                    safe_data = np.maximum(data, 0)
                    err = np.sqrt(safe_data / egain + (rdnoise / egain)**2)
            except Exception as e:
                self.logger.error(f"Failed to process {fpath_lv1.name}: {e}")
                continue

            # 3. Load Calibration Metadata
            fpath_psf = psf_dir / subdir_name / row['psffile']
            try:
                psf_table = pd.read_csv(fpath_psf)
            except Exception as e:
                self.logger.warning(f"Could not load PSF table {fpath_psf.name}, falling back to global FWHM. {e}")
                psf_table = pd.DataFrame()
                
            fwhm_global = float(hdr.get('PSF_FWHM', 2.5))

            # 4. Pixel Mapping
            x_arr, y_arr = wcs.world_to_pixel_values(eph_obsid['ra'].values, eph_obsid['dec'].values)
            eph_obsid['x_init'] = x_arr
            eph_obsid['y_init'] = y_arr

            # Edge exclusion
            edge_margin = 7 * fwhm_global
            naxis1 = int(hdr.get('NAXIS1', data.shape[1]))
            naxis2 = int(hdr.get('NAXIS2', data.shape[0]))
            
            mask_x = (eph_obsid['x_init'] > edge_margin) & (eph_obsid['x_init'] < (naxis1 - edge_margin))
            mask_y = (eph_obsid['y_init'] > edge_margin) & (eph_obsid['y_init'] < (naxis2 - edge_margin))
            eph_obsid = eph_obsid[mask_x & mask_y].copy()
            
            if eph_obsid.empty:
                continue

            # 5. Centroid Refinement
            eph_obsid = self.lv2.find_centroid(
                data=data, 
                sources=eph_obsid,
                fwhm=fwhm_global,
                mask=mask,
                x_col="x_init", y_col="y_init"
            )
            
            if eph_obsid is None or eph_obsid.empty:
                continue

            # 6. Spatially Varying Photometry
            sso_phot_obsid = self.lv2.perform_photometry(
                data=data, 
                sources=eph_obsid,
                exptime=row['exptime'], 
                err=err, 
                mask=mask,
                fwhm=fwhm_global,
                psf_table=psf_table,
                base_tile_size=base_tile_size,
                ap_in_out=ap_in_out,
                x_col='x_winpos', y_col='y_winpos'
            )
            
            if sso_phot_obsid is None or sso_phot_obsid.empty:
                continue
            
            # 7. Metadata Injection
            for col in ['object','exptime', 'filename', 'obsdate', 'altcen', 'azcen', 'zpfile', 'psffile']:
                sso_phot_obsid[col] = row.get(col, np.nan)
                
            sso_phot_obsid['zp_global'] = row.get('zp_g', np.nan)
            sso_phot_obsid['zperr_global'] = row.get('zperr_g', np.nan)
            sso_phot_obsid['fwhm_global'] = fwhm_global

            sso_phot_list.append(sso_phot_obsid)

        # Final Compilation
        if not sso_phot_list:
            self.logger.warning("No photometry was successfully extracted across the dataset.")
            return pd.DataFrame()
            
        sso_phot_summary = pd.concat(sso_phot_list, ignore_index=True)
        self.logger.info(f"Successfully extracted {len(sso_phot_summary)} photometric data points.")
        return sso_phot_summary