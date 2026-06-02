
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
from scipy.spatial import cKDTree

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

class GaiaQuery:

    def __init__(self):
        # Initialize any necessary attributes or configurations here
        pass

    def query_gaia(self,
                wcs_input,
                gaia_data,
                gaia_mag_upper_limit=18.0,
                gaia_mag_lower_limit=13.0,
                gaia_band="g",
                filter_nearby_sources=True,
                dist_thresh_pix=15,
                bright_star_dist_thresh_pix=50
                ):
        """
        Query Gaia catalog for sources within the field of view(s).
        Accepts a single Astropy WCS object, a list of WCSs, or a dictionary of WCSs.
        """
        # 1. Load Gaia Data
        if isinstance(gaia_data, (str, Path)):
            try:
                gaia_all = np.load(gaia_data, mmap_mode='r')
            except Exception as e:
                raise ValueError(f"Failed to load Gaia catalog from {gaia_data}: {e}")
        else:
            gaia_all = gaia_data

        # 2. Standardize WCS Input
        if isinstance(wcs_input, WCS):
            wcs_list = [wcs_input]
        elif isinstance(wcs_input, dict):
            wcs_list = list(wcs_input.values())
        elif isinstance(wcs_input, (list, tuple)):
            wcs_list = list(wcs_input)
        else:
            raise ValueError("wcs_input must be a WCS object, a list of WCS, or a dict of WCS.")

        all_valid_targets = []
        buffer = 0.1  # degrees buffer for spherical projection bowing
        img_buffer = 10  # pixel buffer for edges

        # 3. Iterate over all provided fields of view
        for i, wcs in enumerate(wcs_list):
            try:
                # Use wcs.pixel_shape instead of FITS header NAXIS
                nx, ny = wcs.pixel_shape
            except AttributeError:
                warnings.warn(f"WCS object at index {i} lacks pixel_shape. Skipping.", stacklevel=2)
                continue

            # Calculate Corners & Rough RA/Dec Mask
            corners_x = [0, nx, nx, 0]
            corners_y = [0, 0, ny, ny]
            corners_world = wcs.pixel_to_world(corners_x, corners_y)
            
            ra_min = corners_world.ra.degree.min()
            ra_max = corners_world.ra.degree.max()
            dec_min = corners_world.dec.degree.min()
            dec_max = corners_world.dec.degree.max()

            # Handle RA wrap-around (crossing 0/360 degrees)
            if (ra_max - ra_min) > 180:
                mask_region = (
                    ((gaia_all['ra'] >= ra_max - buffer) | (gaia_all['ra'] <= ra_min + buffer)) &
                    (gaia_all['dec'] >= dec_min - buffer) & 
                    (gaia_all['dec'] <= dec_max + buffer)
                )
            else:
                mask_region = (
                    (gaia_all['ra'] >= ra_min - buffer) & (gaia_all['ra'] <= ra_max + buffer) &
                    (gaia_all['dec'] >= dec_min - buffer) & (gaia_all['dec'] <= dec_max + buffer)
                )
            
            gaia_region_all = gaia_all[mask_region]
            
            if len(gaia_region_all) == 0:
                continue

            # Convert to pixel coordinates for THIS specific WCS
            skycoord_gaia = SkyCoord(gaia_region_all["ra"]*u.degree, gaia_region_all["dec"]*u.degree)
            x_all, y_all = wcs.world_to_pixel(skycoord_gaia)
            
            df_gaia = pd.DataFrame(gaia_region_all)
            df_gaia['x'] = x_all
            df_gaia['y'] = y_all
            
            # Filter 1: Bounds of the image
            mask_img = (
                (df_gaia['x'] >= -img_buffer) & (df_gaia['x'] <= nx + img_buffer) &
                (df_gaia['y'] >= -img_buffer) & (df_gaia['y'] <= ny + img_buffer)
            )
            df_gaia = df_gaia[mask_img].reset_index(drop=True).copy()

            if df_gaia.empty:
                continue

            # Filter 2: Magnitude Range
            mag_col = f'phot_{gaia_band}_mean_mag'
            bright_stars = df_gaia[df_gaia[mag_col] <= gaia_mag_lower_limit]
            
            mask_target = (df_gaia[mag_col] > gaia_mag_lower_limit) & (df_gaia[mag_col] < gaia_mag_upper_limit)
            df_target = df_gaia[mask_target].copy()
            
            if not filter_nearby_sources or df_target.empty:
                all_valid_targets.append(df_target)
                continue

            # Filter 3: Mask sources too close to EACH OTHER
            coords_target = np.vstack([df_target['x'], df_target['y']]).T
            tree_target = cKDTree(coords_target)
            dists_all, _ = tree_target.query(coords_target, k=2, workers=-1)
            
            mask_dist_nearest = dists_all[:, 1] >= dist_thresh_pix
            
            # Filter 4: Mask sources too close to VERY BRIGHT STARS
            if not bright_stars.empty:
                coords_bright = np.vstack([bright_stars['x'], bright_stars['y']]).T
                tree_bright = cKDTree(coords_bright)
                
                dists_to_bright, _ = tree_bright.query(coords_target, k=1, workers=-1)
                mask_dist_nearbright = dists_to_bright >= bright_star_dist_thresh_pix
            else:
                mask_dist_nearbright = np.ones(len(df_target), dtype=bool)

            # Combine spatial masks for this WCS
            final_mask = mask_dist_nearest & mask_dist_nearbright
            valid_targets = df_target[final_mask].reset_index(drop=True)
            
            all_valid_targets.append(valid_targets)
            
        # 4. Combine Results and Drop Duplicates
        if not all_valid_targets:
            warnings.warn("Gaia catalog: Found 0 sources across all provided FOVs.", stacklevel=2)
            return pd.DataFrame()

        df_combined = pd.concat(all_valid_targets, ignore_index=True)
        
        # If FOVs overlap, the same star will appear multiple times. We drop duplicates based on 
        # source_id if it exists in your array, otherwise fallback to RA/Dec coordinates.
        dup_subset = 'source_id' if 'source_id' in df_combined.columns else ['ra', 'dec']
        df_combined = df_combined.drop_duplicates(subset=dup_subset).reset_index(drop=True)
        
        print(f"Gaia catalog: Found {len(df_combined)} unique sources across all FOVs.")
        return df_combined


    def query_nearest_gaia(self,
                    target_coords, 
                    gaia_data, 
                    gaia_band="g"
                    ):
        """
        Find the nearest Gaia sources to the provided target coordinates.
        
        Parameters:
        -----------
        target_coords : astropy.coordinates.SkyCoord or list
            A single SkyCoord object or a list of SkyCoord objects.
        gaia_data : numpy.ndarray or pandas.DataFrame
            The Gaia catalog data containing at least 'ra', 'dec', 'source_id', 
            and the relevant magnitude column.
        gaia_band : str, optional
            The magnitude band to extract (default is "g").
            
        Returns:
        --------
        tuple or list of tuples
            If a single SkyCoord is provided, returns a single tuple: 
            (source_id, magnitude, angular_distance_arcsec).
            If a list/array of SkyCoords is provided, returns a list of such tuples.
        """
        # 1. Handle empty catalog
        if gaia_data is None or len(gaia_data) == 0:
            return None

        # 2. Standardize Gaia data into a pandas DataFrame for uniform column access
        if not isinstance(gaia_data, pd.DataFrame):
            df_gaia = pd.DataFrame(gaia_data)
        else:
            df_gaia = gaia_data

        # Ensure required columns exist
        mag_col = f'phot_{gaia_band}_mean_mag'
        required_cols = ['ra', 'dec', 'source_id']
        for col in required_cols:
            if col not in df_gaia.columns:
                raise ValueError(f"Missing required column '{col}' in gaia_data.")

        # 3. Build the SkyCoord catalog for the Gaia dataset
        catalog_coords = SkyCoord(ra=df_gaia['ra'].values * u.degree, 
                                dec=df_gaia['dec'].values * u.degree)

        # 4. Standardize target_coords input (differentiate between scalar and array/list)
        is_scalar = False
        if isinstance(target_coords, SkyCoord):
            if target_coords.isscalar:
                is_scalar = True
                # Convert scalar to a 1D SkyCoord array for uniform processing
                targets = SkyCoord([target_coords]) 
            else:
                targets = target_coords
        elif isinstance(target_coords, (list, tuple)):
            # Astropy can parse a list of SkyCoord objects natively
            targets = SkyCoord(target_coords)
        else:
            raise TypeError("target_coords must be a SkyCoord object or a list of SkyCoord objects.")

        # 5. Perform the spatial cross-match
        # idx: indices of the closest catalog matches
        # d2d: 2D angular distances to the matches
        idx, d2d, _ = targets.match_to_catalog_sky(catalog_coords)

        # 6. Extract the matched data
        matched_source_ids = df_gaia['source_id'].values[idx]
        matched_dists_arcsec = d2d.arcsec
        
        # Handle the magnitude column safely in case a specific band is missing
        if mag_col in df_gaia.columns:
            matched_mags = df_gaia[mag_col].values[idx]
        else:
            matched_mags = np.full(len(idx), np.nan)

        # 7. Format the output
        results = [
            (sid, mag, dist) 
            for sid, mag, dist in zip(matched_source_ids, matched_mags, matched_dists_arcsec)
        ]

        # Return a single tuple if input was a scalar, otherwise return the list of tuples
        if is_scalar:
            return results[0]
        
        return results