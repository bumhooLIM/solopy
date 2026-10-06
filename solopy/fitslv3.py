import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import kete

import skyloc as sloc
from .fitslv2 import FitsLv2
from .gaia import GaiaQuery
from ._timeutil import utc_jd_to_tdb
from ._logutil import get_logger
from .zeropoint import local_zero_points

__all__ = ["FitsLv3"]

class FitsLv3:
    def __init__(self, orb_path, gaia_path, log_file=None):
        """
        Initialize the Level-3 Science Processor.
        Pre-loads heavy orbital and catalog databases to optimize memory.

        `gaia_path` is a path to a Gaia .npy catalog (memory-mapped) or an in-memory catalog.
        Pass the night's subset from `GaiaQuery.build_nightly_subset`: the blend check
        cross-matches against every row, so the full 247.5 M-row catalog costs minutes and GBs.
        """
        # 1. Setup Logging (non-propagating, so lines are not repeated by root handlers)
        self.logger = get_logger("FitsLv3", log_file)
        
        self.logger.info("Initializing Level-3 Processor...")
        
        # 2. Load Orbital Database
        try:
            self.logger.info(f"Loading orbit database from {orb_path}")
            self.orb, _ = sloc.fetch_orb(Path(orb_path), update_output=999)
        except Exception as e:
            self.logger.error(f"Failed to load orbit database: {e}")
            raise
            
        # 3. Gaia Catalog (path -> memory map; arrays such as the nightly subset are used as is)
        try:
            if isinstance(gaia_path, (str, Path)):
                self.logger.info(f"Loading Gaia catalog from {gaia_path}")
                self.gaia_all = np.load(Path(gaia_path), mmap_mode='r')
            else:
                self.gaia_all = gaia_path
            self.logger.info(f"Gaia catalog for blend checks: {len(self.gaia_all):,} sources")
            if len(self.gaia_all) > 20_000_000:
                self.logger.warning("Large Gaia catalog passed to FitsLv3; use GaiaQuery.build_nightly_subset "
                                    "to avoid multi-GB memory use in the blend check.")
        except Exception as e:
            self.logger.error(f"Failed to load Gaia catalog: {e}")
            raise
            
        # Initialize an instance of Lv2 for its powerful photometry and centroiding tools
        self.lv2 = FitsLv2(log_file=log_file)
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
                
                # Header JD is mid-exposure UTC; kete expects TDB.
                jd_mid_tdb = utc_jd_to_tdb(float(hdr["JD"]))
                observatory = (hdr['LAT'], hdr['LON'], hdr['ELEVAT'])

                observer = kete.spice.earth_pos_to_ecliptic(
                    jd_mid_tdb, *observatory, name=fpath.stem
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
        
        nearest_gaia_sources = GaiaQuery.query_nearest_gaia(
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

    def extract_sso_photometry(self, science_summary, eph, psf_dir, ap_in_out=(1.5, 3.0, 4.0), base_tile_size=500,
                               badpix_frac_max=0.05, zp_dir=None, zp_local_radius=500.0, zp_local_min=10,
                               sys_floor_mag=0.01):
        """
        Executes precision centroiding and spatially varying aperture photometry
        for the predicted asteroids in each frame. `badpix_frac_max` is passed to
        `FitsLv2.perform_photometry` (flag `badphot` above this masked fraction of the aperture).

        With `zp_dir` (the Lv2 zero-point tables, one folder per night), each measurement also
        gets a local zero point at solar color from the stars within `zp_local_radius` px
        (robustness review R2/R3), and `mag_err_tot` = sqrt(mag_err^2 + zperr_local^2 +
        sys_floor_mag^2) for weighting (review R5).
        """
        psf_dir = Path(psf_dir)
        zp_dir = Path(zp_dir) if zp_dir is not None else None
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
                    # Lv1 bit mask (solopy.maskbits); photometry needs the saturation bit
                    mask = hdul[1].data if len(hdul) > 1 else np.zeros(data.shape, dtype=np.uint8)
                    
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
                mask=mask != 0,
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
                x_col='x_winpos', y_col='y_winpos',
                badpix_frac_max=badpix_frac_max, gain=egain
            )
            
            if sso_phot_obsid is None or sso_phot_obsid.empty:
                continue
            
            # 7. Metadata Injection
            for col in ['object','exptime', 'filename', 'obsdate', 'altcen', 'azcen', 'zpfile', 'psffile']:
                sso_phot_obsid[col] = row.get(col, np.nan)
                
            sso_phot_obsid['zp_global'] = row.get('zp_g', np.nan)
            sso_phot_obsid['zperr_global'] = row.get('zperr_g', np.nan)
            sso_phot_obsid['fwhm_global'] = fwhm_global

            # 8. Local zero point at solar color (review R2/R3) and total error (review R5)
            zp_sun = float(hdr.get('ZP_SUN', row.get('zp_g', np.nan)))
            sso_phot_obsid['zp_sun'] = zp_sun
            sso_phot_obsid['zp_color'] = float(hdr.get('ZPCOLOR', np.nan))
            if zp_dir is not None:
                local = self._local_zero_points(zp_dir / subdir_name / str(row.get('zpfile', '')), hdr,
                                                sso_phot_obsid['x_winpos'], sso_phot_obsid['y_winpos'],
                                                zp_sun, zp_local_radius, zp_local_min)
                for col in local.columns:
                    sso_phot_obsid[col] = local[col].to_numpy()
                sso_phot_obsid['mag_err_tot'] = np.sqrt(sso_phot_obsid['mag_err']**2
                                                        + sso_phot_obsid['zperr_local']**2 + sys_floor_mag**2)

            sso_phot_list.append(sso_phot_obsid)

        # Final Compilation
        if not sso_phot_list:
            self.logger.warning("No photometry was successfully extracted across the dataset.")
            return pd.DataFrame()
            
        sso_phot_summary = pd.concat(sso_phot_list, ignore_index=True)
        self.logger.info(f"Successfully extracted {len(sso_phot_summary)} photometric data points.")
        return sso_phot_summary

    def _local_zero_points(self, fpath_zp, hdr, x, y, zp_sun, radius, min_stars):
        """Local zero points at (x, y) from a frame's Lv2 table; global fallback when unavailable."""
        n_global = max(int(hdr.get('ZPSOURCE', 1)), 1)
        fallback_err = float(hdr.get('ZPERR_G', np.nan)) / np.sqrt(n_global)
        try:
            table = pd.read_parquet(fpath_zp)
        except Exception as e:
            self.logger.warning(f"No zero-point table {Path(fpath_zp).name} ({e}); using the frame zero point.")
            table = None
        if table is None or 'zp_star_sun' not in table.columns:
            # Tables from before solopy 1.1 have no colors: fall back to the frame value
            return local_zero_points([], [], [], x, y, radius, min_stars, fallback_zp=zp_sun, fallback_err=fallback_err)
        return local_zero_points(table['x'], table['y'], table['zp_star_sun'], x, y, radius, min_stars,
                                 fallback_zp=zp_sun, fallback_err=fallback_err)