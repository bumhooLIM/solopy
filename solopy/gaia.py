
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
from scipy.spatial import cKDTree

import json
import warnings
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from pathlib import Path

__all__ = ["GaiaQuery", "NIGHTLY_SUBSET_RADIUS_DEG"]


# Radius of the nightly Gaia subset around each telescope pointing [deg]:
# FoV half-diagonal (4096 px * sqrt(2)/2 * 2.98"/px = 2.40 deg) + astrometry.net search radius
# around the header pointing (2.0 deg, FitsLv1.update_wcs) + 0.1 deg buffer.
NIGHTLY_SUBSET_RADIUS_DEG = 4.5


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

    # ------------------------------------------------------------------
    # Nightly subset: one pass over the full catalog per night
    # ------------------------------------------------------------------
    @staticmethod
    def _as_catalog(gaia_data):
        """Path -> memory-mapped .npy; DataFrame -> record array; arrays are returned as is."""
        if isinstance(gaia_data, (str, Path)):
            try:
                return np.load(gaia_data, mmap_mode='r')
            except Exception as e:
                raise ValueError(f"Failed to load Gaia catalog from {gaia_data}: {e}")
        if isinstance(gaia_data, pd.DataFrame):
            return gaia_data.to_records(index=False)
        return gaia_data

    @staticmethod
    def _split_ra(ra_min, ra_max, dec_min, dec_max):
        """Split an RA interval that crosses 0/360 deg into boxes inside [0, 360]."""
        if ra_max - ra_min >= 360.0:
            return [[0.0, 360.0, dec_min, dec_max]]
        if ra_min < 0.0:
            return [[ra_min + 360.0, 360.0, dec_min, dec_max], [0.0, ra_max, dec_min, dec_max]]
        if ra_max > 360.0:
            return [[ra_min, 360.0, dec_min, dec_max], [0.0, ra_max - 360.0, dec_min, dec_max]]
        return [[ra_min, ra_max, dec_min, dec_max]]

    @staticmethod
    def cap_boxes(ra_deg, dec_deg, radius_deg):
        """
        RA/Dec boxes [ra_min, ra_max, dec_min, dec_max] that contain the spherical cap of
        `radius_deg` around each (ra, dec). The RA half-width is the exact extent of the cap,
        arcsin(sin r / cos dec); caps reaching a pole span all RA, and boxes crossing
        RA = 0/360 are split in two.
        """
        ra = np.atleast_1d(np.asarray(ra_deg, dtype=float)) % 360.0
        dec = np.atleast_1d(np.asarray(dec_deg, dtype=float))
        sin_r = np.sin(np.radians(radius_deg))
        boxes = []
        for a, d in zip(ra, dec):
            dec_min, dec_max = max(d - radius_deg, -90.0), min(d + radius_deg, 90.0)
            cos_d = np.cos(np.radians(d))
            if dec_max >= 90.0 or dec_min <= -90.0 or sin_r >= cos_d:
                boxes.append([0.0, 360.0, dec_min, dec_max])
                continue
            half = np.degrees(np.arcsin(sin_r / cos_d))
            boxes.extend(GaiaQuery._split_ra(a - half, a + half, dec_min, dec_max))
        return boxes

    @staticmethod
    def _footprint_grid(wcs, margin_pix, n_sample):
        """RA/Dec of an n_sample x n_sample grid over the image expanded by margin_pix."""
        nx, ny = wcs.pixel_shape
        xs = np.linspace(-0.5 - margin_pix, nx - 0.5 + margin_pix, n_sample)
        ys = np.linspace(-0.5 - margin_pix, ny - 0.5 + margin_pix, n_sample)
        xx, yy = np.meshgrid(xs, ys)
        ra, dec = wcs.pixel_to_world_values(xx.ravel(), yy.ravel())
        return np.asarray(ra, dtype=float) % 360.0, np.asarray(dec, dtype=float)

    @staticmethod
    def wcs_boxes(wcs, margin_pix=60, n_sample=9):
        """
        RA/Dec boxes enclosing an image footprint expanded by `margin_pix`. The default 60 px
        covers the +-10 px edge buffer and the 50 px bright-star radius used by `query_gaia`.
        """
        ra, dec = GaiaQuery._footprint_grid(wcs, margin_pix, n_sample)
        dec_min, dec_max = float(dec.min()), float(dec.max())
        nx, ny = wcs.pixel_shape
        for pole_dec in (90.0, -90.0):
            try:
                px, py = wcs.world_to_pixel_values(0.0, pole_dec)
            except Exception:  # e.g. SIP inversion does not converge far from the field
                continue
            if np.isfinite(px) and np.isfinite(py) and \
               -margin_pix <= px <= nx + margin_pix and -margin_pix <= py <= ny + margin_pix:
                return [[0.0, 360.0, min(dec_min, pole_dec), max(dec_max, pole_dec)]]
        if ra.max() - ra.min() > 180.0:  # footprint straddles RA = 0
            ra = np.where(ra > 180.0, ra - 360.0, ra)
        return GaiaQuery._split_ra(float(ra.min()), float(ra.max()), dec_min, dec_max)

    @staticmethod
    def in_boxes(boxes, ra_deg, dec_deg):
        """Boolean array: which (ra, dec) lie inside any box. RA must be in [0, 360)."""
        ra = np.asarray(ra_deg, dtype=float)
        dec = np.asarray(dec_deg, dtype=float)
        inside = np.zeros(ra.shape, dtype=bool)
        for ra_min, ra_max, dec_min, dec_max in boxes:
            inside |= (ra >= ra_min) & (ra <= ra_max) & (dec >= dec_min) & (dec <= dec_max)
        return inside

    @staticmethod
    def footprint_covered(boxes, wcs, margin_pix=60, n_sample=9):
        """True if the image footprint (expanded by margin_pix) lies inside the union of boxes."""
        ra, dec = GaiaQuery._footprint_grid(wcs, margin_pix, n_sample)
        return bool(GaiaQuery.in_boxes(boxes, ra, dec).all())

    @staticmethod
    def isolation_flags(ra_deg, dec_deg, radius_arcsec=20.0):
        """
        True where the nearest *other* source lies at least `radius_arcsec` away.
        Same criterion (3-D unit-vector KD-tree) that built gaiadr3_20arcsec.npy.
        """
        ra = np.radians(np.asarray(ra_deg, dtype=float))
        dec = np.radians(np.asarray(dec_deg, dtype=float))
        if ra.size < 2:
            return np.ones(ra.size, dtype=bool)
        xyz = np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
        dist, _ = cKDTree(xyz).query(xyz, k=2, workers=-1)
        chord = 2.0 * np.sin(np.radians(radius_arcsec / 3600.0) / 2.0)
        return dist[:, 1] >= chord

    @staticmethod
    def build_subset(gaia_data, boxes, isolation_arcsec=20.0, chunk_rows=4_000_000):
        """
        Extract every catalog row inside `boxes` in ONE sequential pass over a (memory-mapped)
        catalog, and add a boolean field `iso` (no other catalog source within `isolation_arcsec`).

        Isolation is evaluated inside the subset. That equals the all-sky criterion for every
        source farther than `isolation_arcsec` from the subset boundary, which holds for all
        frames because the boxes include a margin of degrees around every field.
        """
        gaia_all = GaiaQuery._as_catalog(gaia_data)
        merged = GaiaQuery._merge_bounding_boxes([[float(v) for v in box] for box in boxes])
        parts = []
        for start in range(0, len(gaia_all), chunk_rows):
            chunk = gaia_all[start:start + chunk_rows]
            keep = GaiaQuery.in_boxes(merged, chunk['ra'], chunk['dec'])
            if keep.any():
                parts.append(np.asarray(chunk[keep]))
        subset = np.concatenate(parts) if parts else np.asarray(gaia_all[:0])

        names = [name for name in subset.dtype.names if name != 'iso']
        out = np.empty(len(subset), dtype=[(name, subset.dtype[name]) for name in names] + [('iso', '?')])
        for name in names:
            out[name] = subset[name]
        out['iso'] = GaiaQuery.isolation_flags(subset['ra'], subset['dec'], isolation_arcsec)
        return out

    @staticmethod
    def build_nightly_subset(gaia_data, ra_deg, dec_deg,
                             radius_deg=NIGHTLY_SUBSET_RADIUS_DEG, isolation_arcsec=20.0):
        """
        Gaia subset that safely encompasses every field of one night, built before calibration.

        `ra_deg`/`dec_deg` are the telescope pointings (Lv0 header RA/DEC). Every plate-solved
        footprint lies within `radius_deg` of its pointing (see NIGHTLY_SUBSET_RADIUS_DEG); confirm
        after Lv1 with `footprint_covered` and extend with `wcs_boxes` if a frame falls outside.

        Returns (subset, boxes). `subset[subset['iso']]` replaces gaiadr3_20arcsec.npy (zero
        points); the full subset replaces gaiadr3.npy (Lv3 blend check).
        """
        points = np.column_stack([np.asarray(ra_deg, dtype=float) % 360.0, np.asarray(dec_deg, dtype=float)])
        points = points[np.all(np.isfinite(points), axis=1)]
        if len(points) == 0:
            raise ValueError("No valid pointings to build a Gaia subset from.")
        points = np.unique(np.round(points, 2), axis=0)  # 0.01 deg rounding << 0.1 deg buffer
        boxes = GaiaQuery.cap_boxes(points[:, 0], points[:, 1], radius_deg)
        return GaiaQuery.build_subset(gaia_data, boxes, isolation_arcsec), boxes

    @staticmethod
    def save_subset(npy_path, subset, boxes, **meta):
        """Save a subset as .npy plus a .json sidecar holding its boxes and provenance."""
        npy_path = Path(npy_path)
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, subset)
        sidecar = {
            "boxes": [[float(v) for v in box] for box in boxes],
            "n_sources": int(len(subset)),
            "n_isolated": int(np.count_nonzero(subset['iso'])),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec='seconds'),
            **meta,
        }
        npy_path.with_suffix('.json').write_text(json.dumps(sidecar, indent=2, default=str))

    @staticmethod
    def load_subset(npy_path):
        """Load a subset saved by `save_subset`. Returns (subset, boxes, metadata)."""
        npy_path = Path(npy_path)
        meta = json.loads(npy_path.with_suffix('.json').read_text())
        return np.load(npy_path), meta["boxes"], meta

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
        # corners_world = wcs.pixel_to_world(corners_x, corners_y)
        ra_corners, dec_corners = wcs.pixel_to_world_values(corners_x, corners_y)
        
        ra_min = ra_corners.min()
        ra_max = ra_corners.max()
        dec_min = dec_corners.min()
        dec_max = dec_corners.max()

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
        # skycoord_gaia = SkyCoord(gaia_region_all["ra"]*u.degree, gaia_region_all["dec"]*u.degree)
        # x_all, y_all = wcs.world_to_pixel(skycoord_gaia)
        x_all, y_all = wcs.world_to_pixel_values(gaia_region_all['ra'], gaia_region_all['dec'])
        
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