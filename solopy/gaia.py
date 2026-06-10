
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
from scipy.spatial import cKDTree

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
class GaiaQuery:

    @staticmethod
    def _boxes_overlap(b1, b2):
        """Check if two [ra_min, ra_max, dec_min, dec_max] bounding boxes overlap."""
        # Returns False if one box is completely to the left, right, top, or bottom of the other.
        return not (b1[1] < b2[0] or b1[0] > b2[1] or b1[3] < b2[2] or b1[2] > b2[3])

    @staticmethod
    def _merge_bounding_boxes(boxes):
        """Merge a list of bounding boxes until no more overlaps exist."""
        merged_state = True
        while merged_state:
            merged_state = False
            new_boxes = []
            while boxes:
                current = boxes.pop(0)
                overlap_idx = -1
                
                # Check if 'current' overlaps with any remaining boxes
                for i, other in enumerate(boxes):
                    if GaiaQuery._boxes_overlap(current, other):
                        overlap_idx = i
                        break
                
                if overlap_idx >= 0:
                    other = boxes.pop(overlap_idx)
                    # Merge the two overlapping boxes into one larger box
                    new_box = [
                        min(current[0], other[0]),  # RA min
                        max(current[1], other[1]),  # RA max
                        min(current[2], other[2]),  # Dec min
                        max(current[3], other[3])   # Dec max
                    ]
                    boxes.append(new_box)
                    merged_state = True  # We did a merge, so we must run the check again
                else:
                    new_boxes.append(current)
            
            boxes = new_boxes
            
        return boxes

    @staticmethod
    def query_gaia_subset(wcs_input, 
                          gaia_data, 
                          gaia_mag_upper_limit=18.0, 
                          gaia_mag_lower_limit=13.0, 
                          gaia_band="g"
                          ):
        """
        Extract a subset of the Gaia catalog that covers all provided WCS fields.
        Optimized for large catalogs and many WCS fields by merging overlapping FOVs
        before masking the array.
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

        # 3. Extract and standardise bounding boxes for all WCS
        buffer = 0.1
        boxes = []
        
        for i, wcs in enumerate(wcs_list):
            try:
                nx, ny = wcs.pixel_shape
            except AttributeError:
                warnings.warn(f"WCS object at index {i} lacks pixel_shape. Skipping.", stacklevel=2)
                continue
            
            corners_x = [0, nx, nx, 0]
            corners_y = [0, 0, ny, ny]
            corners_world = wcs.pixel_to_world(corners_x, corners_y)
            
            ra_min = corners_world.ra.degree.min()
            ra_max = corners_world.ra.degree.max()
            dec_min = corners_world.dec.degree.min()
            dec_max = corners_world.dec.degree.max()
            
            # Handle RA Wrap-around (crossing 0/360 degrees) mathematically
            # We split the FOV into two separate non-wrapping boxes before merging
            if (ra_max - ra_min) > 180:
                boxes.append([ra_max - buffer, 360.0, dec_min - buffer, dec_max + buffer])
                boxes.append([0.0, ra_min + buffer, dec_min - buffer, dec_max + buffer])
            else:
                boxes.append([ra_min - buffer, ra_max + buffer, dec_min - buffer, dec_max + buffer])

        if not boxes:
            return pd.DataFrame()

        # 4. Merge overlapping bounding boxes into Master Regions
        merged_boxes = GaiaQuery._merge_bounding_boxes(boxes)
        
        # 5. Create the Master Boolean Mask
        master_mask = np.zeros(len(gaia_all), dtype=bool)
        
        for box in merged_boxes:
            r_min, r_max, d_min, d_max = box
            
            # Apply bounds logic
            box_mask = (
                (gaia_all['ra'] >= r_min) & (gaia_all['ra'] <= r_max) &
                (gaia_all['dec'] >= d_min) & (gaia_all['dec'] <= d_max)
            )
            # Logically OR the mask into the master mask
            master_mask |= box_mask
            
        # 6. Extract the Master Data (Happens only ONCE)
        gaia_subset = gaia_all[master_mask]
        
        if len(gaia_subset) == 0:
            print("Gaia subset: Found 0 sources across all provided FOVs.")
            return pd.DataFrame()
            
        # 7. Apply Magnitude Filtering to shrink the output further
        df_subset = pd.DataFrame(gaia_subset)
        mag_col = f'phot_{gaia_band}_mean_mag'
        
        if mag_col in df_subset.columns:
            mask_mag = (df_subset[mag_col] > gaia_mag_lower_limit) & (df_subset[mag_col] < gaia_mag_upper_limit)
            df_subset = df_subset[mask_mag].copy()

        print(f"Gaia subset: Extracted {len(df_subset)} unique sources from {len(merged_boxes)} master regions.")
        
        # No drop_duplicates required! The master mask naturally extracted every star exactly once.
        return df_subset.reset_index(drop=True)


    @staticmethod
    def query_gaia(wcs,
                   gaia_data,
                   gaia_mag_upper_limit=18.0,
                   gaia_mag_lower_limit=13.0,
                   gaia_band="g",
                   filter_nearby_sources=True,
                   dist_thresh_pix=15,
                   bright_star_dist_thresh_pix=50
                   ):
        """
        Query Gaia catalog for sources within a SINGLE field of view.
        Calculates X/Y pixel coordinates and applies spatial/magnitude filtering.
        """
        # 1. Load Gaia Data
        if isinstance(gaia_data, (str, Path)):
            try:
                gaia_all = np.load(gaia_data, mmap_mode='r')
            except Exception as e:
                raise ValueError(f"Failed to load Gaia catalog from {gaia_data}: {e}")
        else:
            gaia_all = gaia_data

        if len(gaia_all) == 0:
            return pd.DataFrame()

        # 2. Extract Image Dimensions
        try:
            nx, ny = wcs.pixel_shape
        except AttributeError:
            raise ValueError("Provided WCS object lacks 'pixel_shape' attribute.")

        # 3. Calculate Corners & Fast RA/Dec Mask
        # (This prevents massive memory spikes if a non-subsetted catalog is accidentally passed in)
        buffer = 0.1  # degrees
        corners_x = [0, nx, nx, 0]
        corners_y = [0, 0, ny, ny]
        corners_world = wcs.pixel_to_world(corners_x, corners_y)
        
        ra_min = corners_world.ra.degree.min()
        ra_max = corners_world.ra.degree.max()
        dec_min = corners_world.dec.degree.min()
        dec_max = corners_world.dec.degree.max()

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
            return pd.DataFrame()

        # 4. Convert to pixel coordinates
        skycoord_gaia = SkyCoord(gaia_region_all["ra"]*u.degree, gaia_region_all["dec"]*u.degree)
        x_all, y_all = wcs.world_to_pixel(skycoord_gaia)
        
        df_gaia = pd.DataFrame(gaia_region_all)
        df_gaia['x'] = x_all
        df_gaia['y'] = y_all
        
        # 5. Filter 1: Bounds of the image (with pixel buffer)
        img_buffer = 10  # pixels
        mask_img = (
            (df_gaia['x'] >= -img_buffer) & (df_gaia['x'] <= nx + img_buffer) &
            (df_gaia['y'] >= -img_buffer) & (df_gaia['y'] <= ny + img_buffer)
        )
        df_gaia = df_gaia[mask_img].reset_index(drop=True).copy()

        if df_gaia.empty:
            return pd.DataFrame()

        # 6. Filter 2: Magnitude Range
        mag_col = f'phot_{gaia_band}_mean_mag'
        
        if mag_col in df_gaia.columns:
            bright_stars = df_gaia[df_gaia[mag_col] <= gaia_mag_lower_limit]
            mask_target = (df_gaia[mag_col] > gaia_mag_lower_limit) & (df_gaia[mag_col] < gaia_mag_upper_limit)
            df_target = df_gaia[mask_target].copy()
        else:
            warnings.warn(f"Magnitude column '{mag_col}' not found. Skipping magnitude filters.", stacklevel=2)
            df_target = df_gaia.copy()
            bright_stars = pd.DataFrame()
        
        if not filter_nearby_sources or df_target.empty:
            return df_target.reset_index(drop=True)

        # 7. Filter 3: Mask sources too close to EACH OTHER
        coords_target = np.vstack([df_target['x'], df_target['y']]).T
        tree_target = cKDTree(coords_target)
        dists_all, _ = tree_target.query(coords_target, k=2, workers=-1)
        
        mask_dist_nearest = dists_all[:, 1] >= dist_thresh_pix
        
        # 8. Filter 4: Mask sources too close to VERY BRIGHT STARS
        if not bright_stars.empty:
            coords_bright = np.vstack([bright_stars['x'], bright_stars['y']]).T
            tree_bright = cKDTree(coords_bright)
            
            dists_to_bright, _ = tree_bright.query(coords_target, k=1, workers=-1)
            mask_dist_nearbright = dists_to_bright >= bright_star_dist_thresh_pix
        else:
            mask_dist_nearbright = np.ones(len(df_target), dtype=bool)

        # 9. Combine spatial masks and return
        final_mask = mask_dist_nearest & mask_dist_nearbright
        
        return df_target[final_mask].reset_index(drop=True)

    @staticmethod
    def query_nearest_gaia(target_coords, 
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