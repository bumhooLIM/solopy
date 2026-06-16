import logging
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.stats import sigma_clip, SigmaClip
import sep
from ccdproc import CCDData
from scipy.spatial import cKDTree
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry, ApertureStats
from . import _utils
from astropy.wcs import WCS
from astropy.time import Time
import astropy.units as u
from astroquery.imcce import Skybot
import pandas as pd
import numpy as np

from photutils.centroids import centroid_1dg, centroid_com
from astropy.nddata import Cutout2D

class FitsLv2:
    """
    Class for Level-2 processing (Photometric Zero Point Calculation).
    Assumes all inputs are uncompressed .fits files.
    """

    def __init__(self, log_file: str = None):
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

    def sep_extract_source(self,
                           data,
                           mask=None,
                           fwhm=2.0,
                           thresh=2.5,
                           col_out="all"
                           ):
        """
        Detect sources using SEP.
        """
        try:
            # Ensure data is float32 and handle byte order
            if data.dtype.byteorder == '>':
                data = data.byteswap().view(data.dtype.newbyteorder())
            data = data.astype(np.float32)

            bkg = sep.Background(data, mask=mask)
            data_sub = data - bkg.back()
            
            objects = sep.extract(
                data_sub, 
                thresh=thresh, 
                err=bkg.rms(),
                mask=mask,
                minarea=np.pi*(0.5*fwhm)**2
            )
            
            if len(objects) == 0:
                self.logger.warning("SEP: No sources detected.")
                return pd.DataFrame()
            
            if col_out != "all":
                objects = objects[col_out]
            
            self.logger.info(f"SEP: Detected {len(objects)} sources with SEP.")
            return pd.DataFrame(objects)
        
        except Exception as e:
            self.logger.error(f"SEP: Source detection failed: {e}")
            return None

    def match_catalogs(self, source_cat, ref_cat, tolerance=3.0, 
                       is_sigma_clip = True, sigma_clip_thresh=3.0,
                       source_suffix='_source', ref_suffix='_ref',
                       xcol_source='x', ycol_source='y', xcol_ref='x', ycol_ref='y'
                       ):
        """
        Match source catalog with reference catalog using spatial pixel distance.
        """
        # Ensure coordinates are extracted
        source_coords = np.vstack([source_cat[xcol_source], source_cat[ycol_source]]).T
        ref_coords = np.vstack([ref_cat[xcol_ref], ref_cat[ycol_ref]]).T
        
        # Build tree and query
        tree = cKDTree(ref_coords)
        dist, idx = tree.query(source_coords, distance_upper_bound=tolerance)
        
        # Filter valid matches
        mask = dist != np.inf
        
        # Append suffixes to avoid duplicate column names (Crucial Fix)
        matched_sources = source_cat[mask].copy().add_suffix(source_suffix)
        matched_ref = ref_cat.iloc[idx[mask]].copy().add_suffix(ref_suffix)
        
        # Reset index for safe concatenation
        matched_sources.reset_index(drop=True, inplace=True)
        matched_ref.reset_index(drop=True, inplace=True)
        
        # Combine the catalogs
        matched_df = pd.concat([matched_sources, matched_ref], axis=1)
        
        # Keep the separation distance (since cKDTree already calculated it for free)
        matched_df['separation_pix'] = dist[mask]
        if is_sigma_clip:
            # Sigma clip the separation to remove outliers
            sep_clipped = sigma_clip(matched_df['separation_pix'], sigma=sigma_clip_thresh, maxiters=5)
            matched_df = matched_df[~sep_clipped.mask].reset_index(drop=True)
            self.logger.info(f"Sigma clipped {np.sum(sep_clipped.mask)} sources with separation outliers.")
        
        self.logger.info(f"Matched SEP with Gaia: {len(matched_df)} sources found (maximum separation = {np.max(matched_df['separation_pix'])} pixel)")

        return matched_df

    # def find_centroid(self, data, sources, fwhm=2.0, mask=None,
    #                   x_col='x', y_col='y'):
    #     """
    #     Perform background subtraction and calculate highly accurate windowed 
    #     centroids for a given source catalog.
    #     """
    #     try:
    #         # 1. Prepare the data (NumPy 2.0 compatible byte-swapping)
    #         if data.dtype.byteorder == '>':
    #             data = data.byteswap().view(data.dtype.newbyteorder())
    #         data = data.astype(np.float32)

    #         # 2. Global Background Subtraction
    #         # Mask is passed to prevent bright stars/bad pixels from skewing the background
    #         bkg = sep.Background(data, mask=mask)
    #         data_bkgsub = data - bkg.back()
            
    #         # 3. Calculate Windowed Centroids
    #         x_cen, y_cen, flag = sep.winpos(
    #             data_bkgsub, 
    #             sources[x_col].values, 
    #             sources[y_col].values, 
    #             sig=fwhm/2.355
    #         )

    #         # 4. Apply new coordinates and flags to the catalog
    #         updated_sources = sources.copy()
    #         updated_sources['x_winpos'] = x_cen
    #         updated_sources['y_winpos'] = y_cen
    #         updated_sources['winpos_flag'] = flag

    #         # 5. Filter out bad centroids
    #         good_mask = (flag == 0)
    #         final_sources = updated_sources[good_mask].reset_index(drop=True)

    #         # Log the cleanup
    #         bad_count = len(sources) - len(final_sources)
    #         if bad_count > 0:
    #             self.logger.info(f"SEP: Removed {bad_count} sources with centroiding errors.")

    #         self.logger.info(f"SEP: Calculated windowed centroids for {len(final_sources)} sources.")
    #         return final_sources

    #     except Exception as e:
    #         self.logger.error(f"SEP: Centroiding failed: {e}")
    #         return None

    def find_centroid(self, data, sources, cutout_size=15, mask=None, x_col='x', y_col='y'):
        """
        Perform background subtraction and calculate highly accurate centroids 
        robust against elongated sources (tracking errors).
        """
        try:
            # 1. Prepare the data (NumPy 2.0 compatible byte-swapping)
            if data.dtype.byteorder == '>':
                data = data.byteswap().view(data.dtype.newbyteorder())
            data = data.astype(np.float32)

            # 2. Global Background Subtraction
            bkg = sep.Background(data, mask=mask)
            data_bkgsub = data - bkg.back()
            
            updated_sources = sources.copy()
            new_x, new_y = [], []
            flags = []

            # 3. Calculate Robust Centroids via Cutouts
            # box_size should be large enough to encapsulate the full streak length.
            for idx, row in updated_sources.iterrows():
                init_x, init_y = row[x_col], row[y_col]
                
                try:
                    # Create a 2D cutout around the initial guess
                    cutout = Cutout2D(data_bkgsub, (init_x, init_y), cutout_size)
                    
                    # OPTION 1: 1D Marginal Gaussian Fit (Highly recommended for streaks)
                    # It collapses the streak into 1D profiles, ignoring the asymmetric 2D shape
                    xcen_cutout, ycen_cutout = centroid_1dg(cutout.data)
                    
                    # If 1D Gaussian fails (e.g., too noisy), fallback to Center of Mass
                    if np.isnan(xcen_cutout) or np.isnan(ycen_cutout):
                        xcen_cutout, ycen_cutout = centroid_com(cutout.data)

                    # Convert cutout coordinates back to original image coordinates
                    x_final, y_final = cutout.to_original_position((xcen_cutout, ycen_cutout))
                    
                    new_x.append(x_final)
                    new_y.append(y_final)
                    flags.append(0) # Success flag

                except Exception:
                    # Cutout failed (e.g., star too close to the edge of the sensor)
                    new_x.append(init_x)
                    new_y.append(init_y)
                    flags.append(1) # Bad flag
                    
            # 4. Apply new coordinates and flags to the catalog
            updated_sources['x_winpos'] = new_x # Keeping your column name for pipeline continuity
            updated_sources['y_winpos'] = new_y
            updated_sources['winpos_flag'] = flags

            # 5. Filter out bad centroids
            good_mask = (updated_sources['winpos_flag'] == 0)
            final_sources = updated_sources[good_mask].reset_index(drop=True)

            # Log the cleanup
            bad_count = len(sources) - len(final_sources)
            if bad_count > 0:
                self.logger.info(f"Centroiding: Removed {bad_count} sources due to edge proximity or noise.")

            self.logger.info(f"Centroiding: Calculated robust centroids for {len(final_sources)} sources.")
            return final_sources

        except Exception as e:
            self.logger.error(f"Centroiding failed: {e}")
            return None
    
    def perform_photometry(self,    
                           data,
                           sources,
                           exptime,
                           err=None,
                           mask=None,
                           fwhm=2.0,            # Fallback global FWHM
                           psf_table=None,      # NEW: The DataFrame generated by soloPSF
                           base_tile_size=500,  # NEW: Must match the SOLORegion setup
                           ap_in_out=(2.0, 4.0, 6.0),
                           x_col='x', y_col='y',
                           remove_bad_sources=False):
        """
        Perform fast, science-grade spatially varying aperture photometry.
        Automatically scales aperture radii per-star based on regional PSF variations.
        """
        try:
            # 1. Map Sources to Regional FWHM
            if psf_table is not None and not psf_table.empty:
                # Initialize region boundaries
                regions = soloregion(data.shape, base_tile_size=base_tile_size)
                
                # Vectorized lookup of the (i, j) grid index for every star simultaneously
                region_i = (sources[x_col] // regions.base_tile_size).astype(int).clip(upper=regions.num_tiles_x - 1)
                region_j = (sources[y_col] // regions.base_tile_size).astype(int).clip(upper=regions.num_tiles_y - 1)
                
                # Create a lookup mapping from the PSF DataFrame
                fwhm_map = psf_table.set_index(['region_i', 'region_j'])['fwhm_avg']
                
                # Map the FWHM to the stars based on their (i, j) coordinates
                source_idx = pd.MultiIndex.from_arrays([region_i, region_j])
                fwhm_array = source_idx.map(fwhm_map).values
                
                # Safety Net: If a region failed its PSF fit (NaN), fallback to the global median
                global_median_fwhm = psf_table['fwhm_avg'].median()
                fwhm_array = np.nan_to_num(fwhm_array, nan=global_median_fwhm)
                
                # Prevent absurd values (e.g., > 10 pixels) just in case a bad fit snuck through
                fwhm_array = np.clip(fwhm_array, 1.5, 10.0)
                
                self.logger.debug("Successfully applied spatially varying FWHM to apertures.")
            else:
                # Fallback to static global FWHM if no table is provided
                fwhm_array = np.full(len(sources), float(fwhm))

            # 2. Setup Dynamic Apertures (Photutils seamlessly handles arrays of radii)
            positions = list(zip(sources[x_col], sources[y_col]))
            
            r_ap_array  = ap_in_out[0] * fwhm_array
            r_in_array  = ap_in_out[1] * fwhm_array
            r_out_array = ap_in_out[2] * fwhm_array
            
            aperture = CircularAperture(positions, r=r_ap_array)
            annulus = CircularAnnulus(positions, r_in=r_in_array, r_out=r_out_array)
            
            # 3. Base Photometry
            phot_table = aperture_photometry(data, aperture, error=err, mask=mask)
            
            # 4. Fast Local Background Estimation
            sigclip = SigmaClip(sigma=3.0, maxiters=5)
            sky_stats = ApertureStats(data, annulus, mask=mask, sigma_clip=sigclip)
            
            msky = sky_stats.median
            ssky = sky_stats.std
            nsky = sky_stats.sum_aper_area 
            
            # 5. Fast Bad Pixel Checking
            if mask is not None:
                mask_bool = mask.astype(bool)
                bad_stats = ApertureStats(mask_bool, aperture)
                n_badpixel = bad_stats.sum
                flag_bad = n_badpixel > 0
            else:
                n_badpixel = np.zeros(len(aperture))
                flag_bad = np.zeros(len(aperture), dtype=bool)

            # 6. Math and Columns
            ap_area = aperture.area  # Returns an array of areas for each dynamic aperture!
            phot_table['fwhm_used']      = fwhm_array  # Store the actual FWHM used for traceability
            phot_table['r_ap_pix']       = r_ap_array
            phot_table['aparea']         = ap_area
            phot_table['annulus_median'] = msky
            phot_table['bkg_std']        = ssky
            phot_table['nsky']           = nsky
            
            # Source flux (background subtracted)
            phot_table['source_sum'] = phot_table['aperture_sum'] - (ap_area * msky)
            
            # Error Propagation
            ap_sum_err_sq = phot_table['aperture_sum_err']**2 if 'aperture_sum_err' in phot_table.colnames else 0.0
            
            sky_mean_err_term = np.zeros_like(msky, dtype=float)
            valid_nsky = nsky > 0
            sky_mean_err_term[valid_nsky] = (ap_area[valid_nsky]**2 * ssky[valid_nsky]**2) / nsky[valid_nsky]
            
            phot_table["source_sum_err"] = np.sqrt(ap_sum_err_sq + (ap_area * ssky**2) + sky_mean_err_term) 
            
            # SNR
            phot_table["snr"] = phot_table["source_sum"] / phot_table["source_sum_err"]
            
            # Instrumental Magnitude
            valid_flux = phot_table["source_sum"] > 0
            mag_inst = np.full(len(phot_table), np.nan)
            mag_inst[valid_flux] = -2.5 * np.log10(phot_table["source_sum"][valid_flux] / exptime)
            phot_table["mag_inst"] = mag_inst
            flag_bad |= ~valid_flux  
            
            # Magnitude Error
            phot_table["mag_err"] = (2.5 / np.log(10)) * (1.0 / phot_table["snr"])
            
            # Flags
            phot_table["badphot"] = flag_bad
            phot_table["nbadpix"] = n_badpixel
            
            # Clean up and convert to Pandas
            df_phot = phot_table.to_pandas().drop(columns=["id", "xcenter", "ycenter"])
            
            df_combined = pd.concat([
                sources.reset_index(drop=True), 
                df_phot.reset_index(drop=True)
            ], axis=1)
            
            if remove_bad_sources:
                mask_good = ~df_combined["badphot"]
                df_combined = df_combined[mask_good]
                self.logger.info(f"Removed {np.sum(flag_bad)} sources flagged for bad photometry.")
            
            return df_combined

        except Exception as e:
            self.logger.error(f"Photometry failed: {e}")
            return None

    def calculate_zeropoint(self, 
                            fpath_fits, 
                            gaia_data, 
                            outdir_zp,
                            mag_lower=13.5, 
                            mag_upper=16.0,
                            psf_table=None,       # NEW: Accepts the spatial PSF map
                            base_tile_size=500,   # NEW: Needed for region mapping
                            fallback_fwhm=2.5, 
                            ap_in_out=(2.5, 4.0, 6.0)): # NEW: Multipliers instead of fixed radii
        """
        Calculates the photometric zero-point using Spatially Varying Aperture Photometry.
        """
        fpath_fits = Path(fpath_fits)
        outdir_zp = Path(outdir_zp)
        outdir_zp.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Calculating Zero Point for {fpath_fits.name}...")
        
        # # Determine global FWHM for SEP extraction and centroiding
        # if psf_table is not None and not psf_table.empty:
        #     global_fwhm = float(psf_table['fwhm_avg'].mean())
        # else:
        #     global_fwhm = fallback_fwhm
        
        # 1. Safe File Handling
        try:
            with fits.open(fpath_fits) as hdul:
                data = hdul[0].data.astype(np.float32) 
                mask = hdul[1].data.astype(bool) if len(hdul) > 1 else np.zeros_like(data, dtype=bool)
                hdr = hdul[0].header
                wcs = WCS(hdr)
                
                egain = float(hdr.get("EGAIN", 18.9))
                rdnoise = float(hdr.get("RDNOISE", 3.8))
                exptime = float(hdr.get("EXPTIME", 60.0))
                global_fwhm = float(hdr.get("PSF_FWHM", fallback_fwhm))
                
                safe_data = np.maximum(data, 0)
                err = np.sqrt(safe_data / egain + (rdnoise / egain)**2)
        except Exception as e:
            self.logger.error(f"Failed to read {fpath_fits.name}: {e}")
            return False

        # 2. Source Extraction and Matching (Using global FWHM guess)
        source_gaia = solopy.GaiaQuery.query_gaia(
            wcs=wcs, gaia_data=gaia_data,
            gaia_mag_lower_limit=mag_lower, gaia_mag_upper_limit=mag_upper,
            dist_thresh_pix=15, bright_star_dist_thresh_pix=50
        )
        
        source_sep = self.sep_extract_source(data, mask=mask, thresh=3.0, fwhm=global_fwhm)
        
        if source_gaia is None or source_sep is None or source_sep.empty:
            self.logger.warning(f"Extraction failed or empty for {fpath_fits.name}. Skipping.")
            return False
            
        matched_sources = self.match_catalogs(source_cat=source_sep, ref_cat=source_gaia, tolerance=3.0)
        
        if matched_sources is None or matched_sources.empty:
            self.logger.warning(f"No matched sources found for {fpath_fits.name}. Skipping.")
            return False
        
        # 3. Centroiding
        source_final = self.find_centroid(
            data=data, sources=matched_sources, fwhm=global_fwhm, mask=mask,
            x_col='x_source', y_col='y_source'
        )

        if source_final is None or source_final.empty:
            self.logger.warning(f"Centroiding failed for {fpath_fits.name}. Skipping.")
            return False
        
        keep_cols = ['ra_ref', 'dec_ref', 'x_winpos', 'y_winpos', 'phot_g_mean_mag_ref']
        source_final = source_final[keep_cols].rename(columns={
            'ra_ref': 'ra',
            'dec_ref': 'dec',
            'phot_g_mean_mag_ref': 'phot_g_mean_mag',
            'x_winpos': 'x', 
            'y_winpos': 'y'
            })
        
        # 4. Spatially Varying Aperture Photometry
        phot = self.perform_photometry(
            data, source_final, 
            exptime=exptime, err=err, mask=mask, 
            fwhm=global_fwhm,               # Fallback 
            psf_table=psf_table,            # NEW: Pass spatial table
            base_tile_size=base_tile_size,  # NEW: Pass tile size
            ap_in_out=ap_in_out,            # NEW: Dynamic Multipliers
            x_col='x', y_col='y', remove_bad_sources=True
        )
        
        if phot is None or phot.empty:
            self.logger.warning(f"Photometry returned empty for {fpath_fits.name}. Skipping.")
            return False
            
        # 5. Calculate Field Zero Point
        phot['mag_diff_g_inst'] = phot['phot_g_mean_mag'] - phot['mag_inst']
        
        sigma_clipped = sigma_clip(phot['mag_diff_g_inst'], sigma=3, maxiters=5)
        outlier_mask = np.ma.getmaskarray(sigma_clipped)
        phot_clipped = phot[~outlier_mask]
        
        if phot_clipped.empty:
            self.logger.warning(f"All sources clipped out during ZP calculation for {fpath_fits.name}.")
            return False
            
        zp = phot_clipped['mag_diff_g_inst'].median()
        zp_err = phot_clipped['mag_diff_g_inst'].std()
        num_sources = len(phot_clipped)
        
        # 6. Save to Parquet
        fpath_out_pq = outdir_zp / f"zp.{fpath_fits.stem}.parquet"
        try:
            phot.to_parquet(fpath_out_pq, index=False)
        except Exception as e:
            self.logger.error(f"Failed to save Parquet file for {fpath_fits.name}: {e}")
            return False
        
        # 7. Update FITS Header
        try:
            fits.setval(fpath_fits, 'ZP_G', value=float(zp), comment='Photometric Zeropoint (Gaia G-band)')
            fits.setval(fpath_fits, 'ZPERR_G', value=float(zp_err), comment='Estimated error of the zeropoint')
            fits.setval(fpath_fits, 'ZPSOURCE', value=int(num_sources), comment='Number of sources used for ZP')
            fits.setval(fpath_fits, 'ZPFILE', value=fpath_out_pq.name, comment='Zero-point catalog file name')
            
            self.logger.info(f"Updated header ZP={zp:.3f}$\\pm${zp_err:.3f} (N={num_sources})")
        except Exception as e:
            self.logger.error(f"Failed to write ZP headers to {fpath_fits.name}: {e}")
            return False
            
        return True

    def find_asteroids_in_fov(self, wcs, hdr, mag_upper_limit=18.0):
        """
        Query SkyBoT (IMCCE) for all known minor planets within the FITS Field of View.
        """
        try:
            # 1. Parse Time from Header
            # Assumes hdr['jd'] is a float Julian Date
            epoch = Time(hdr['jd'], format='jd')

            # 2. Determine FOV Center and Radius
            nx = hdr['NAXIS1']
            ny = hdr['NAXIS2']

            # Find the center of the image
            center_x, center_y = nx / 2, ny / 2
            center_world = wcs.pixel_to_world(center_x, center_y)

            # Find max radius to the corners to ensure the cone covers the whole rectangle
            corners_x = [0, nx, nx, 0]
            corners_y = [0, 0, ny, ny]
            corners_world = wcs.pixel_to_world(corners_x, corners_y)
            
            # The radius of our search cone is the distance from center to the furthest corner
            search_radius = np.max(center_world.separation(corners_world))

            self.logger.info(f"Querying SkyBoT at {epoch.isot} with radius {search_radius.to(u.deg):.2f}...")

            # 3. Query SkyBoT
            # Skybot natively returns a rich Astropy QTable
            results = Skybot.cone_search(center_world, search_radius, epoch)
            
            if results is None or len(results) == 0:
                self.logger.info("No asteroids found in this FOV.")
                return pd.DataFrame()

            # Convert to Pandas for easier manipulation downstream
            df_ast = results.to_pandas()

            # 4. Filter by Magnitude
            # Skybot returns the predicted V-band magnitude in the 'V' column
            df_ast = df_ast[df_ast['V'] <= mag_upper_limit].copy()

            if df_ast.empty:
                self.logger.info(f"No asteroids brighter than {mag_upper_limit} mag found.")
                return df_ast

            # 5. Calculate Pixel Coordinates
            coords = SkyCoord(df_ast['RA']*u.deg, df_ast['DEC']*u.deg)
            x, y = wcs.world_to_pixel(coords)
            df_ast['x_pixel'] = x
            df_ast['y_pixel'] = y

            # 6. Strict Rectangular Masking
            # Skybot queries a circle. We must trim off the corners that fall outside the CCD.
            in_fov_mask = (
                (df_ast['x_pixel'] >= 0) & (df_ast['x_pixel'] <= nx) &
                (df_ast['y_pixel'] >= 0) & (df_ast['y_pixel'] <= ny)
            )
            df_ast = df_ast[in_fov_mask].reset_index(drop=True)

            if df_ast.empty:
                self.logger.info("Asteroids found, but they fell outside the CCD rectangle.")
                return df_ast


            # 8. Clean up and rename columns for sanity
            rename_map = {
                'Name': 'name',
                'Number': 'number',
                'RA': 'ra',
                'DEC': 'dec',
                'V': 'mag_v_pred',    # Predicted V magnitude
                'r': 'r_hel_au',      # Heliocentric distance
                'delta': 'delta_geo_au', # Geocentric distance
                'alpha': 'phase_angle',  
                'elong': 'solar_elong'
            }
            df_ast.rename(columns=rename_map, inplace=True)

            # Select and order the columns of interest
            cols_to_keep = [
                'name', 'number', 'ra', 'dec', 'x_pixel', 'y_pixel', 
                'mag_v_pred', 'r_hel_au', 'delta_geo_au', 'phase_angle', 
                'solar_elong',
            ]
            
            # Ensure we only try to keep columns that actually exist (failsafe)
            final_cols = [c for c in cols_to_keep if c in df_ast.columns]
            
            self.logger.info(f"Successfully tracked {len(df_ast)} asteroids in FOV.")
            return df_ast[final_cols]

        except Exception as e:
            self.logger.error(f"Asteroid FoV search failed: {e}")
            return pd.DataFrame()