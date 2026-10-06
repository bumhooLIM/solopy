from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clip, SigmaClip
from astropy.wcs import WCS
import sep
from scipy.spatial import cKDTree
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry, ApertureStats
from .region import SOLORegion
from .gaia import GaiaQuery
from ._logutil import get_logger
from . import maskbits


__all__ = ["FitsLv2"]


def _as_float_array(values):
    """Plain 1-D float array from photutils outputs, which may be Quantity (e.g. areas in pix2)."""
    return np.atleast_1d(np.asarray(getattr(values, "value", values), dtype=float))


class FitsLv2:
    """
    Class for Level-2 processing (Photometric Zero Point Calculation).
    Assumes all inputs are uncompressed .fits files.
    """

    def __init__(self, log_file: str = None):
        # FitsLv3 creates its own FitsLv2, so this must not stack handlers.
        self.logger = get_logger(self.__class__.__name__, log_file)

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
            # FIX: Safely extract boolean mask to prevent Pandas KeyError on 0 outliers
            # matched_df = matched_df[~sep_clipped.mask].reset_index(drop=True)
            outlier_mask = np.ma.getmaskarray(sep_clipped)
            matched_df = matched_df[~outlier_mask].reset_index(drop=True)
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
                           err=None,
                           mask=None,
                           fwhm=2.5,            
                           psf_table=None,      
                           base_tile_size=500,  
                           ap_in_out=(2.5, 4.0, 6.0),
                           x_col='x', y_col='y',
                           remove_bad_sources=False,
                           badpix_frac_max=0.05):
        """
        Perform fast, science-grade spatially varying aperture photometry.
        Automatically scales aperture radii per-region using GroupBy optimizations.

        A source is flagged `badphot` when masked pixels cover more than `badpix_frac_max`
        of its aperture area (default 5%; 0 reproduces the old "any masked pixel" rule),
        when any saturated pixel touches its aperture, or when its background-subtracted
        flux is not positive. Masked pixels are excluded from both the aperture sum and
        `aperture_area`.

        `mask` may be boolean or an Lv1 bit mask (`solopy.maskbits`); saturation can only
        be recognized from a bit mask.
        """
        try:
            # 1. Map Sources to Regional FWHM
            if sources.empty:
                    self.logger.warning("No sources provided for photometry.")
                    return pd.DataFrame()

            # Integer masks are Lv1 bit masks; any non-zero pixel is excluded from the photometry.
            sat_map = None
            if mask is not None:
                mask = np.asarray(mask)
                if np.issubdtype(mask.dtype, np.integer):
                    sat_map = (mask & maskbits.SATURATED) != 0
                mask = mask != 0

            if psf_table is not None and not psf_table.empty:
                regions = SOLORegion(data.shape, base_tile_size=base_tile_size)
                
                region_i = (sources[x_col] // regions.base_tile_size).astype(int).clip(upper=regions.num_tiles_x - 1)
                region_j = (sources[y_col] // regions.base_tile_size).astype(int).clip(upper=regions.num_tiles_y - 1)
                
                fwhm_map = psf_table.set_index(['region_i', 'region_j'])['fwhm_avg']
                source_idx = pd.MultiIndex.from_arrays([region_i, region_j])
                fwhm_array = source_idx.map(fwhm_map).values
                
                global_median_fwhm = psf_table['fwhm_avg'].median()
                fwhm_array = np.nan_to_num(fwhm_array, nan=global_median_fwhm)
                fwhm_array = np.clip(fwhm_array, 1.5, 10.0)
                
                # Attach the mapped fwhm directly to the sources dataframe so we can group by it
                sources = sources.copy()
                sources['mapped_fwhm'] = fwhm_array
                
            else:
                sources = sources.copy()
                sources['mapped_fwhm'] = float(fwhm)

            # 2. GroupBy Execution (The Secret to Fast Spatial Photometry)
            photometry_results = []
            
            # Group stars that share the exact same FWHM size
            for local_fwhm, group in sources.groupby('mapped_fwhm'):
                
                positions = list(zip(group[x_col], group[y_col]))
                
                # Calculate the exact scalar radii for this specific group of stars
                r_ap  = ap_in_out[0] * local_fwhm
                r_in  = ap_in_out[1] * local_fwhm
                r_out = ap_in_out[2] * local_fwhm
                
                # Because r_ap is a scalar float, photutils is happy!
                aperture = CircularAperture(positions, r=r_ap)
                annulus = CircularAnnulus(positions, r_in=r_in, r_out=r_out)
                
                # Base Photometry
                phot_table = aperture_photometry(data, aperture, error=err, mask=mask)
                
                # Background Estimation
                sigclip = SigmaClip(sigma=3.0, maxiters=5)
                sky_stats = ApertureStats(data, annulus, mask=mask, sigma_clip=sigclip)
                
                msky = _as_float_array(sky_stats.median)
                ssky = _as_float_array(sky_stats.std)
                nsky = _as_float_array(sky_stats.sum_aper_area)  # Quantity [pix2] -> float
                
                # Bad Pixel Checking: fraction of the aperture area covered by masked pixels
                if mask is not None:
                    n_badpixel = np.atleast_1d(np.asarray(ApertureStats(mask, aperture).sum, dtype=float))
                else:
                    n_badpixel = np.zeros(len(aperture))
                badpix_frac = n_badpixel / aperture.area

                # Saturated pixels always flag: the lost core flux cannot be recovered (review R1)
                if sat_map is not None:
                    n_satpix = np.atleast_1d(np.asarray(ApertureStats(sat_map, aperture).sum, dtype=float))
                else:
                    n_satpix = np.zeros(len(aperture))
                saturated = n_satpix > 0
                flag_bad = (badpix_frac > badpix_frac_max) | saturated

                # Math and Columns
                # Unmasked aperture area, matching the masked aperture sum. photutils returns a
                # Quantity [pix2]; subtracting it from the unitless aperture_sum raised and made
                # every call return None (regression in 82036f8).
                ap_stats = ApertureStats(data, aperture, mask=mask)
                ap_area = _as_float_array(ap_stats.sum_aper_area)
                phot_table['fwhm_used']      = local_fwhm  
                phot_table['r_ap_pixel']     = r_ap
                phot_table['aperture_area']  = ap_area
                phot_table['annulus_median'] = msky
                phot_table['bkg_std']        = ssky
                phot_table['nsky']           = nsky
                
                phot_table['source_sum'] = phot_table['aperture_sum'] - (ap_area * msky)
                
                ap_sum_err_sq = phot_table['aperture_sum_err']**2 if 'aperture_sum_err' in phot_table.colnames else 0.0
                
                sky_mean_err_term = np.zeros_like(msky, dtype=float)
                valid_nsky = nsky > 0
                sky_mean_err_term[valid_nsky] = (ap_area[valid_nsky]**2 * ssky[valid_nsky]**2) / nsky[valid_nsky]
                
                phot_table["source_sum_err"] = np.sqrt(ap_sum_err_sq + (ap_area * ssky**2) + sky_mean_err_term) 
                
                phot_table["snr"] = phot_table["source_sum"] / phot_table["source_sum_err"]
                
                valid_flux = phot_table["source_sum"] > 0
                mag_inst = np.full(len(phot_table), np.nan)
                mag_inst[valid_flux] = -2.5 * np.log10(phot_table["source_sum"][valid_flux] / exptime)
                phot_table["mag_inst"] = mag_inst
                flag_bad |= ~valid_flux  
                
                phot_table["mag_err"] = (2.5 / np.log(10)) * (1.0 / phot_table["snr"])
                
                phot_table["badphot"] = flag_bad
                phot_table["nbadpix"] = n_badpixel
                phot_table["badpix_frac"] = badpix_frac
                phot_table["saturated"] = saturated
                phot_table["nsatpix"] = n_satpix
                
                # Convert this group's results to pandas
                df_phot = phot_table.to_pandas().drop(columns=["id", "xcenter", "ycenter"])
                
                # Combine original group data with its photometry
                df_combined = pd.concat([
                    group.reset_index(drop=True), 
                    df_phot.reset_index(drop=True)
                ], axis=1)
                
                photometry_results.append(df_combined)

            # 3. Finalize
            # Concatenate all the spatial groups back together
            final_df = pd.concat(photometry_results, ignore_index=True)
            
            if remove_bad_sources:
                mask_good = ~final_df["badphot"]
                final_df = final_df[mask_good]
                self.logger.info(f"Removed {np.sum(~mask_good)} sources flagged for bad photometry.")
            
            return final_df

        except Exception as e:
            self.logger.error(f"Photometry failed: {e}")
            return None
          
    def calculate_zeropoint(self, 
                            fpath_fits, 
                            gaia_data, 
                            outdir_zp,
                            mag_lower=13.0, 
                            mag_upper=18.0,
                            psf_table=None,       # NEW: Accepts the spatial PSF map
                            base_tile_size=500,   # NEW: Needed for region mapping
                            fallback_fwhm=2.5, 
                            ap_in_out=(2.5, 4.0, 6.0), # NEW: Multipliers instead of fixed radii
                            badpix_frac_max=0.05):     # max masked fraction of an aperture
        """
        Calculates the photometric zero-point using Spatially Varying Aperture Photometry.
        """
        fpath_fits = Path(fpath_fits)
        outdir_zp = Path(outdir_zp)
        outdir_zp.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Calculating Zero Point for {fpath_fits.name}...")
        
        # 1. Safe File Handling
        try:
            with fits.open(fpath_fits) as hdul:
                data = hdul[0].data.astype(np.float32) 
                # Keep the Lv1 bit mask (saturation is needed by perform_photometry)
                mask = hdul[1].data if len(hdul) > 1 else np.zeros(data.shape, dtype=np.uint8)
                if len(hdul) > 1 and not maskbits.has_bits(hdul[1].header):
                    self.logger.warning(f"{fpath_fits.name}: MASK has no saturation bits (Lv1 made before solopy 1.1); "
                                        "saturated stars are rejected only through the masked-fraction rule.")
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
        source_gaia = GaiaQuery.query_gaia(
            wcs=wcs, gaia_data=gaia_data,
            gaia_mag_lower_limit=12.0, gaia_mag_upper_limit=18.0,
            dist_thresh_pix=15, bright_star_dist_thresh_pix=50
        )
        
        source_sep = self.sep_extract_source(data, mask=mask != 0, thresh=3.0, fwhm=global_fwhm)
        
        if source_gaia is None or source_sep is None or source_sep.empty:
            self.logger.warning(f"Extraction failed or empty for {fpath_fits.name}. Skipping.")
            return False
            
        matched_sources = self.match_catalogs(source_cat=source_sep, ref_cat=source_gaia, tolerance=3.0)
        
        if matched_sources is None or matched_sources.empty:
            self.logger.warning(f"No matched sources found for {fpath_fits.name}. Skipping.")
            return False
        
        # Skip centroiding and use SEP positions. (temporary)
        keep_cols = ['ra_ref', 'dec_ref', 'x_source', 'y_source', 'phot_g_mean_mag_ref']
        source_final = matched_sources[keep_cols].rename(columns={
            'ra_ref': 'ra',
            'dec_ref': 'dec',
            'phot_g_mean_mag_ref': 'phot_g_mean_mag',
            'x_source': 'x', 
            'y_source': 'y'
            })
        mask_gaia_mag = (source_final['phot_g_mean_mag'] >= mag_lower) & (source_final['phot_g_mean_mag'] <= mag_upper)
        source_final = source_final[mask_gaia_mag].reset_index(drop=True)
        
        # 4. Spatially Varying Aperture Photometry
        phot = self.perform_photometry(
            data, source_final, 
            exptime=exptime, err=err, mask=mask, 
            fwhm=global_fwhm,               # Fallback 
            psf_table=psf_table,            # NEW: Pass spatial table
            base_tile_size=base_tile_size,  # NEW: Pass tile size
            ap_in_out=ap_in_out,            # NEW: Dynamic Multipliers
            x_col='x', y_col='y', remove_bad_sources=True,
            badpix_frac_max=badpix_frac_max
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
            # Keep the comment short: the long file name leaves ~13 characters on the 80-char card.
            fits.setval(fpath_fits, 'ZPFILE', value=fpath_out_pq.name, comment='ZP table')
            
            self.logger.info(f"Updated header ZP={zp:.3f}$\\pm${zp_err:.3f} (N={num_sources})")
        except Exception as e:
            self.logger.error(f"Failed to write ZP headers to {fpath_fits.name}: {e}")
            return False
            
        return True
