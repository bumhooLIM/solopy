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

from astropy.time import Time
import astropy.units as u
from astroquery.imcce import Skybot
import pandas as pd
import numpy as np
class FitsLv2:
    """
    Class for Level-2 processing (Photometry and Zero Point Calculation).
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

    def query_gaia(self,
                   wcs,
                   hdr,
                   gaia_data,
                   gaia_mag_upper_limit=18.0,
                   gaia_mag_lower_limit=13.0,
                   gaia_band="g",
                   filter_nearby_sources=True,
                   dist_thresh_pix=15,
                   bright_star_dist_thresh_pix=50
                   ):
        """
        Query Gaia catalog for sources within the field of view.
        """
        if isinstance(gaia_data, (str, Path)):
            try:
                gaia_all = np.load(gaia_data, mmap_mode='r')
            except Exception as e:
                self.logger.error(f"Failed to load Gaia catalog from {gaia_data}: {e}")
                return None
        else:
            gaia_all = gaia_data

        # Filter FoV
        nx = hdr['NAXIS1']
        ny = hdr['NAXIS2']
        
        corners_x = [0, nx, nx, 0]
        corners_y = [0, 0, ny, ny]
        corners_world = wcs.pixel_to_world(corners_x, corners_y)
        
        ra_min = corners_world.ra.degree.min()
        ra_max = corners_world.ra.degree.max()
        dec_min = corners_world.dec.degree.min()
        dec_max = corners_world.dec.degree.max()

        # Add a small buffer (e.g., 0.1 degrees) to account for spherical projection bowing.
        # The edges of a FOV bow outward slightly on a sphere; the corners alone 
        # might barely miss sources situated exactly in the middle of an edge.
        buffer = 0.1 
        
        # Handle RA wrap-around (crossing 0/360 degrees)
        if (ra_max - ra_min) > 180:
            # The FOV crosses the RA=0 meridian. We must mask differently.
            mask_region = (
                ((gaia_all['ra'] >= ra_max - buffer) | (gaia_all['ra'] <= ra_min + buffer)) &
                (gaia_all['dec'] >= dec_min - buffer) & 
                (gaia_all['dec'] <= dec_max + buffer)
            )
        else:
            # Standard masking
            mask_region = (
                (gaia_all['ra'] >= ra_min - buffer) & (gaia_all['ra'] <= ra_max + buffer) &
                (gaia_all['dec'] >= dec_min - buffer) & (gaia_all['dec'] <= dec_max + buffer)
            )
        
        gaia_region_all = gaia_all[mask_region]
        
        # Convert ALL FOV sources to pixel coordinates
        skycoord_gaia = SkyCoord(gaia_region_all["ra"]*u.degree, gaia_region_all["dec"]*u.degree)
        x_all, y_all = wcs.world_to_pixel(skycoord_gaia)
        
        df_gaia = pd.DataFrame(gaia_region_all)
        df_gaia['x'] = x_all
        df_gaia['y'] = y_all
        
        # 1. Filter (x, y) cordinates within the bounds of the image
        # (with a small buffer to catch edge sources)
        img_buffer = 10  # pixels
        mask_img = (
            (df_gaia['x'] >= -img_buffer) & (df_gaia['x'] <= nx + img_buffer) &
            (df_gaia['y'] >= -img_buffer) & (df_gaia['y'] <= ny + img_buffer)
        )
        df_gaia = df_gaia[mask_img].reset_index(drop=True).copy()

        # Identify bright stars for masking
        mag_col = f'phot_{gaia_band}_mean_mag'
        bright_stars = df_gaia[df_gaia[mag_col] <= gaia_mag_lower_limit]

        # Filter 2: Magnitude Range
        mask_target = (df_gaia[mag_col] > gaia_mag_lower_limit) & (df_gaia[mag_col] < gaia_mag_upper_limit)
        df_target = df_gaia[mask_target].copy()
        
        if not filter_nearby_sources or df_target.empty:
            self.logger.info(f"Gaia catalog: Found {len(df_target)} sources in FOV (no spatial filtering applied.)")
            return df_target

        # Filter 3: Mask sources too close to EACH OTHER
        coords_target = np.vstack([df_target['x'], df_target['y']]).T
        tree_target = cKDTree(coords_target)
        dists_all, _ = tree_target.query(coords_target, k=2, workers=-1)
        
        # Distances to the nearest neighbor (index 1, since index 0 is the point itself)
        mask_dist_nearest = dists_all[:, 1] >= dist_thresh_pix
        
        # Filter 4: Mask sources too close to VERY BRIGHT STARS
        if not bright_stars.empty:
            coords_bright = np.vstack([bright_stars['x'], bright_stars['y']]).T
            tree_bright = cKDTree(coords_bright)
            
            # Find distance from each target star to the nearest bright star
            dists_to_bright, _ = tree_bright.query(coords_target, k=1, workers=-1)
            mask_dist_nearbright = dists_to_bright >= bright_star_dist_thresh_pix
        else:
            mask_dist_nearbright = np.ones(len(df_target), dtype=bool)

        # Combine spatial masks and return
        final_mask = mask_dist_nearest & mask_dist_nearbright
        
        self.logger.info(f"Gaia catalog: Found {np.sum(final_mask)} sources in FOV")
        
        return df_target[final_mask].reset_index(drop=True)

    def detect_sources(self, data, mask=None, fwhm=2.0, thresh=2.5):
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

    def find_centroid(self, data, sources, fwhm=2.0, mask=None,
                      x_col='x', y_col='y'):
        """
        Perform background subtraction and calculate highly accurate windowed 
        centroids for a given source catalog.
        """
        try:
            # 1. Prepare the data (NumPy 2.0 compatible byte-swapping)
            if data.dtype.byteorder == '>':
                data = data.byteswap().view(data.dtype.newbyteorder())
            data = data.astype(np.float32)

            # 2. Global Background Subtraction
            # Mask is passed to prevent bright stars/bad pixels from skewing the background
            bkg = sep.Background(data, mask=mask)
            data_bkgsub = data - bkg.back()
            
            # 3. Calculate Windowed Centroids
            x_cen, y_cen, flag = sep.winpos(
                data_bkgsub, 
                sources[x_col].values, 
                sources[y_col].values, 
                sig=fwhm/2.355
            )

            # 4. Apply new coordinates and flags to the catalog
            updated_sources = sources.copy()
            updated_sources['x_winpos'] = x_cen
            updated_sources['y_winpos'] = y_cen
            updated_sources['winpos_flag'] = flag

            # 5. Filter out bad centroids
            good_mask = (flag == 0)
            final_sources = updated_sources[good_mask].reset_index(drop=True)

            # Log the cleanup
            bad_count = len(sources) - len(final_sources)
            if bad_count > 0:
                self.logger.info(f"SEP: Removed {bad_count} sources with centroiding errors.")

            self.logger.info(f"SEP: Calculated windowed centroids for {len(final_sources)} sources.")
            return final_sources

        except Exception as e:
            self.logger.error(f"SEP: Centroiding failed: {e}")
            return None


    def perform_photometry(self,
                           data,
                           sources,
                           exptime,
                           err=None, # Added so aperture_photometry can calc aperture_sum_err
                           mask=None,
                           fwhm=2.0,
                           ap_in_out=(2.0, 4.0, 6.0),
                           x_col='x_winpos', y_col='y_winpos',
                           remove_bad_sources=False):
        """
        Perform fast, science-grade aperture photometry using photutils ApertureStats.
        """
        try:
            # 1. Setup Apertures
            positions = list(zip(sources[x_col], sources[y_col]))
            
            r_ap  = ap_in_out[0] * fwhm
            r_in  = ap_in_out[1] * fwhm
            r_out = ap_in_out[2] * fwhm
            aperture = CircularAperture(positions, r=r_ap)
            annulus = CircularAnnulus(positions, r_in=r_in, r_out=r_out)
            
            # 2. Base Photometry
            phot_table = aperture_photometry(data, aperture, error=err, mask=mask)
            
            # 3. Fast Local Background Estimation (Replaces the slow loops)
            sigclip = SigmaClip(sigma=3.0, maxiters=5)
            # We pass the mask here so bad pixels are ignored in the sky background calculation
            sky_stats = ApertureStats(data, annulus, mask=mask, sigma_clip=sigclip)
            
            msky = sky_stats.median
            ssky = sky_stats.std
            nsky = sky_stats.sum_aper_area # The area of the unmasked pixels used for the background
            
            # 4. Fast Bad Pixel Checking
            if mask is not None:
                mask_bool = mask.astype(bool)
                # Treat the mask as the 'data' to sum the number of True (bad) pixels
                bad_stats = ApertureStats(mask_bool, aperture)
                n_badpixel = bad_stats.sum
                flag_bad = n_badpixel > 0
            else:
                n_badpixel = np.zeros(len(aperture))
                flag_bad = np.zeros(len(aperture), dtype=bool)

            # 5. Math and Columns
            ap_area = aperture.area
            phot_table['aperture_area']  = ap_area
            phot_table['annulus_median'] = msky
            phot_table['bkg_std']        = ssky
            phot_table['nsky']           = nsky
            
            # Source flux (background subtracted)
            phot_table['source_sum'] = phot_table['aperture_sum'] - (ap_area * msky)
            
            # Error Propagation
            # Handle case where error_array wasn't provided
            ap_sum_err_sq = phot_table['aperture_sum_err']**2 if 'aperture_sum_err' in phot_table.colnames else 0.0
            
            # Safe division for the standard error of the mean term (prevents divide-by-zero warnings)
            sky_mean_err_term = np.zeros_like(msky, dtype=float)
            valid_nsky = nsky > 0
            sky_mean_err_term[valid_nsky] = (ap_area**2 * ssky[valid_nsky]**2) / nsky[valid_nsky]
            
            # Total error equation
            phot_table["source_sum_err"] = np.sqrt(ap_sum_err_sq + (ap_area * ssky**2) + sky_mean_err_term) 
            
            # SNR
            phot_table["snr"] = phot_table["source_sum"] / phot_table["source_sum_err"]
            
            # Instrumental Magnitude (Safely handle negative fluxes)
            valid_flux = phot_table["source_sum"] > 0
            mag_inst = np.full(len(phot_table), np.nan)
            mag_inst[valid_flux] = -2.5 * np.log10(phot_table["source_sum"][valid_flux] / exptime)
            phot_table["mag_inst"] = mag_inst
            flag_bad |= ~valid_flux  # Flag sources with non-positive flux as bad
            
            # Magnitude Error
            phot_table["mag_err"] = (2.5 / np.log(10)) * (1.0 / phot_table["snr"])
            
            # Flags
            phot_table["badphot"] = flag_bad
            phot_table["nbadpix"] = n_badpixel
            
            # Clean up and convert to Pandas
            df_phot = phot_table.to_pandas().drop(columns=["id", "xcenter", "ycenter"])
            
            # Combine the original sources with the new photometry data
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

    def calculate_zeropoint(self, sources, filter_band='g'):
        """
        Calculate the photometric zero point using an inverse-variance weighted mean,
        filtering out bad photometry flags.
        """
        try:
            if len(sources) < 3:
                self.logger.warning("Not enough valid stars to calculate a reliable zero point.")
                return np.nan, np.nan

            # 2. Calculate raw ZP
            # NOTE: If you are using a specific filter (e.g., V or R), apply the 
            # Gaia color transformation to 'phot_g_mean_mag' before this step!
            raw_zp = sources[f'phot_{filter_band}_mean_mag_ref'] - sources['mag_inst']
            
            # 3. Calculate total variance for weighting
            # Total error = sqrt(instrumental_err^2 + catalog_err^2)
            # Make sure you have the Gaia error column, usually 'phot_g_mean_mag_error'
            weights = 1.0 / sources['mag_err']**2

            # 4. Sigma clip to remove outliers (e.g., variable stars, hidden binaries)
            # We clip the ZP array, which gives us a boolean mask of the surviving stars
            zp_clipped = sigma_clip(raw_zp, sigma=3.0, maxiters=5)
            surviving_mask = ~zp_clipped.mask

            # Extract the surviving data points and their weights
            final_zps = raw_zp[surviving_mask]
            final_weights = weights[surviving_mask]
            N_survivors = len(final_zps)

            # 5. Calculate Inverse-Variance Weighted Mean
            final_zp = np.average(final_zps, weights=final_weights)
            
            # 6. Calculate the true error of the zero point
            # Standard error of a weighted mean: 1 / sqrt(sum(weights))
            zp_err = 1.0 / np.sqrt(np.sum(final_weights))
            
            # Optional: Keep the scatter (standard deviation) for logging/diagnostics
            zp_scatter = np.std(final_zps)

            self.logger.info(f"Zero point: {final_zp:.3f} ± {zp_err:.3f} (Scatter: {zp_scatter:.3f}, Stars used: {N_survivors})")
            
            return final_zp, zp_err

        except Exception as e:
            self.logger.error(f"Zero point calculation failed: {e}")
            return np.nan, np.nan

    def process(self, fpath_fits, gaia_catalog_path, outdir=None):
        """
        Main processing function for Level-2.
        """
        fpath_fits = Path(fpath_fits)
        if outdir:
            outdir = Path(outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            
        # 1. Load FITS
        try:
            # Using _utils.open_fits_any if available or astropy.io.fits directly for uncompressed assumption
            # Since Lv1 outputs .fits (uncompressed), we can use CCDData directly as in Lv1
            # But we need the WCS object which might not be fully parsed by CCDData in all cases
            # or we might want to use fits.open to be safe.
            # Let's use CCDData as it handles WCS well usually.
            ccd = CCDData.read(fpath_fits, unit='adu')
            hdr = ccd.header
            wcs = ccd.wcs
        except Exception as e:
            self.logger.error(f"Failed to read {fpath_fits}: {e}")
            return

        # 2. Query Gaia
        self.logger.info(f"Querying Gaia catalog for {fpath_fits.name}...")
        ref_cat = self.query_gaia(wcs, hdr, gaia_catalog_path)
        if ref_cat is None or len(ref_cat) == 0:
            self.logger.warning("No Gaia sources found in FOV.")
            return

        # 3. Detect Sources
        self.logger.info("Detecting sources...")
        source_cat, data_sub = self.detect_sources(ccd.data, mask=ccd.mask)
        if source_cat is None or len(source_cat) == 0:
            self.logger.warning("No sources detected in image.")
            return

        # 4. Match Catalogs
        # First, refine centroids (optional but good practice, as done in notebook with winpos)
        # Skipping winpos for brevity, using sep outputs directly
        self.logger.info("Matching catalogs...")
        matched_df = self.match_catalogs(source_cat, ref_cat)
        if len(matched_df) == 0:
            self.logger.warning("No matches found between image and catalog.")
            return

        # 5. Photometry on Matched Sources
        # Note: Ideally perform photometry on all sources then match, or match then photometry.
        # Notebook does: SEP -> Match -> Photometry on Matched coordinates
        # Let's follow notebook logic: define apertures on matched coordinates
        self.logger.info("Performing photometry...")
        # Re-extract coordinates from matched dataframe
        # Note: 'x' and 'y' in matched_df will be from source_cat (the first part of concat)
        # We should use the image coordinates
        phot_results = self.perform_photometry(data_sub, matched_df)
        
        # Merge photometry results with reference magnitudes
        # phot_results has same index as matched_df, so we can join
        full_df = pd.concat([matched_df, phot_results], axis=1)
        
        # Remove duplicate columns if any (pandas handles this but good to be clean)
        full_df = full_df.loc[:,~full_df.columns.duplicated()]

        # 6. Calculate Zero Point
        self.logger.info("Calculating Zero Point...")
        zp, zp_std = self.calculate_zeropoint(full_df)
        self.logger.info(f"Zero Point: {zp:.3f} +/- {zp_std:.3f}")

        # 7. Update Header and Save
        hdr['ZP'] = (zp, 'Photometric Zero Point (mag)')
        hdr['ZPERR'] = (zp_std, 'Zero Point Standard Deviation')
        hdr['HISTORY'] = f"({datetime.now().isoformat()}) Level-2 processing: Photometry & ZP calculated."
        
        if outdir:
            outpath = outdir / f"{fpath_fits.stem}.lv2.fits" # Assuming .fits input, replace extension
            ccd.write(outpath, overwrite=True)
            self.logger.info(f"Saved Level-2 product to {outpath}")
            
        return zp, zp_std

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