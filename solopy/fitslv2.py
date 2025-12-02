import logging
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.stats import sigma_clip
import sep
from ccdproc import CCDData
from scipy.spatial import cKDTree
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from . import _utils

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
                   gaia_catalog_path,
                   gaia_mag_upper_limit=18.0,
                   gaia_mag_lower_limit=13.0,
                   gaia_band = "G",
                   filter_nearby_sources=True,
                   dist_thresh_pix=10,
                   ):
        """
        Query Gaia catalog for sources within the field of view.
        """
        # Load Gaia catalog
        # This assumes the Gaia catalog is a single file for simplicity, but might need adjustment for large catalogs or online queries
        try:
            gaia_all = np.load(gaia_catalog_path, mmap_mode='r')
        except Exception as e:
            self.logger.error(f"Failed to load Gaia catalog: {e}")
            return None

        # Determine rectangle FOV
        edge_1 = wcs.pixel_to_world(0, 0)
        edge_2 = wcs.pixel_to_world(hdr['NAXIS1'], hdr['NAXIS2'])
        ra_min = min(edge_1.ra.degree, edge_2.ra.degree)
        ra_max = max(edge_1.ra.degree, edge_2.ra.degree)
        dec_min = min(edge_1.dec.degree, edge_2.dec.degree)
        dec_max = max(edge_1.dec.degree, edge_2.dec.degree)

        # Mask sources within FOV
        mask_region = ((gaia_all['ra'] >= ra_min) & (gaia_all['ra'] <= ra_max) &
                       (gaia_all['dec'] >= dec_min) & (gaia_all['dec'] <= dec_max))
        
        # Additional masking (brightness, etc.) can be added here as per the notebook
        mask_bright = gaia_all[f'phot_{gaia_band}_mean_mag'] <= gaia_mag_lower_limit
        mask_faint = gaia_all[f'phot_{gaia_band}_mean_mag'] >= gaia_mag_upper_limit
        mask_source = mask_region & (~mask_bright) & (~mask_faint)

        gaia_region = gaia_all[mask_source]
        
        # Convert to pixel coordinates for masking nearby sources
        skycoord_gaia = SkyCoord(gaia_region["ra"]*u.degree, gaia_region["dec"]*u.degree)
        x, y = wcs.world_to_pixel(skycoord_gaia)
        
        # Add x, y to the structured array (requires creating a new array or using pandas)
        # Using pandas for easier manipulation
        df_gaia = pd.DataFrame(gaia_region)
        df_gaia['x'] = x
        df_gaia['y'] = y
        
        # Mask nearby sources using cKDTree
        if not filter_nearby_sources:
            return df_gaia
        
        coords = np.vstack([x, y]).T
        tree_all = cKDTree(coords)
        dists_all, _ = tree_all.query(coords, k=2, workers=-1)
        dists_nearest = dists_all[:, 1]
        
        thresh_dist = dist_thresh_pix # pixels
        mask_dist = dists_nearest >= thresh_dist
        
        return df_gaia[mask_dist]

    def detect_sources(self, data, mask=None, fwhm=2.0, thresh=2.5):
        """
        Detect sources using SEP.
        """
        try:
            # Ensure data is float32 and handle byte order
            if data.dtype.byteorder == '>':
                data = data.byteswap().newbyteorder()
            data = data.astype(np.float32)

            bkg = sep.Background(data)
            data_sub = data - bkg.back()
            
            objects = sep.extract(
                data_sub, 
                thresh=thresh, 
                err=bkg.globalrms, 
                mask=mask,
                minarea=np.pi*(0.5*fwhm)**2
            )
            return pd.DataFrame(objects)
        
        except Exception as e:
            self.logger.error(f"Source detection failed: {e}")
            return None

    def perform_photometry(self,
                           data,
                           sources,
                           exptime,
                           mask=None,
                           fwhm=2.0,
                           ap_in_out = (2.0, 4.0, 6.0),
                           ):
        """
        Perform aperture photometry.
        """
        positions = list(zip(sources['x'], sources['y']))
        r_ap = ap_in_out[0] * fwhm
        r_in = ap_in_out[1] * fwhm
        r_out = ap_in_out[2] * fwhm
        
        aperture = CircularAperture(positions, r=r_ap)
        annulus = CircularAnnulus(positions, r_in=r_in, r_out=r_out)
        
        phot_table = aperture_photometry(data, aperture)
        
        # Estimate local background
        annulus_masks = annulus.to_mask(method='center')
        msky = []
        nsky = []
        ssky = []
        # for mask in annulus_masks:
        #     annulus_data = mask.multiply(data)
        #     annulus_data_1d = annulus_data[mask.data > 0]
        #     _, median_sigclip, _ = sigma_clip(annulus_data_1d, sigma=3.0, maxiters=5, masked=False, return_bounds=True) # Using sigma_clip
        #     msky.append(np.median(median_sigclip)) # Median of clipped data
    
        for mask in annulus_masks:
            ann_data = mask.multiply(data)
            if ann_data is None:
                msky.append(np.nan)
                ssky.append(np.nan)
                nsky.append(0)
            else:
                sky_data = ann_data[mask.data==1]
                sky_data_clipped = sigma_clip(sky_data, sigma=3, maxiters=5)
                med = np.ma.median(sky_data_clipped)
                stddev = np.ma.std(sky_data_clipped)
                msky.append(med)
                ssky.append(stddev)
                nsky.append(sky_data_clipped.count())

        # Check if any bad pixel inside aperture
        flag_bad = []
        n_badpixel = []
        if mask:
            mask = mask.astype(bool)

            for ap in aperture:
                ap_mask = ap.to_mask(method="center")
                cutout_mask = ap_mask.multiply(mask)
                contain_bad = np.any(cutout_mask > 0)
                flag_bad.append(contain_bad)
                n_badpixel.append(int(cutout_mask.sum()))
        else:
            flag_bad = [None] * len(aperture)
            n_badpixel = [None] * len(aperture)
        
        phot_table['aperture_area']  = aperture.area    
        phot_table['annulus_median'] = msky
        phot_table['bkg_std']        = ssky
        phot_table['source_sum']     = phot_table['aperture_sum'] - aperture.area * msky
        phot_table["source_sum_err"] = np.sqrt(phot_table["aperture_sum_err"]**2 + aperture.area * ssky**2) 
        phot_table["snr"]            = phot_table["source_sum"] / phot_table["source_sum_err"]
        phot_table["mag_inst"]       = -2.5 * np.log10(phot_table["source_sum"] / exptime)
        phot_table["mag_err"]        = 2.5/np.log(10)*(1/phot_table["snr"])
        phot_table["badphot"]        = flag_bad
        phot_table["nbadpix"]        = n_badpixel
        
        phot_table = phot_table.to_pandas().drop(columns=["id", "xcenter", "ycenter"])
        
        return phot_table

    def match_catalogs(self, source_cat, ref_cat, wcs, tolerance=3.0):
        """
        Match source catalog with reference catalog.
        """
        # Use pixel coordinates for matching
        source_coords = np.vstack([source_cat['x'], source_cat['y']]).T
        ref_coords = np.vstack([ref_cat['x'], ref_cat['y']]).T
        
        tree = cKDTree(ref_coords)
        dist, idx = tree.query(source_coords, distance_upper_bound=tolerance)
        
        # Filter matches
        mask = dist != np.inf
        matched_sources = source_cat[mask].reset_index(drop=True)
        matched_ref = ref_cat.iloc[idx[mask]].reset_index(drop=True)
        
        return pd.concat([matched_sources, matched_ref], axis=1)

    def calculate_zeropoint(self, matched_df, exptime):
        """
        Calculate photometric zero point.
        """
        # Calculate instrumental magnitude
        # inst_mag = -2.5 * log10(counts / exptime)
        # Filter out negative or zero flux
        valid_flux = matched_df['aper_sum_bkgsub'] > 0
        matched_df = matched_df[valid_flux].copy()
        
        matched_df['inst_mag'] = -2.5 * np.log10(matched_df['aper_sum_bkgsub'] / exptime)
        
        # Calculate zero point
        # ZP = m_ref - m_inst
        # Here we assume 'phot_g_mean_mag' is the reference magnitude
        matched_df['zp'] = matched_df['phot_g_mean_mag'] - matched_df['inst_mag']
        
        # Sigma clip to remove outliers
        zp_clipped = sigma_clip(matched_df['zp'], sigma=3.0, maxiters=5)
        
        final_zp = np.ma.median(zp_clipped)
        zp_std = np.ma.std(zp_clipped)
        
        return final_zp, zp_std

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
        matched_df = self.match_catalogs(source_cat, ref_cat, wcs)
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
        zp, zp_std = self.calculate_zeropoint(full_df, hdr['EXPTIME'])
        self.logger.info(f"Zero Point: {zp:.3f} +/- {zp_std:.3f}")

        # 7. Update Header and Save
        hdr['ZP'] = (zp, 'Photometric Zero Point (mag)')
        hdr['ZPSTD'] = (zp_std, 'Zero Point Standard Deviation')
        hdr['HISTORY'] = f"({datetime.now().isoformat()}) Level-2 processing: Photometry & ZP calculated."
        
        if outdir:
            outpath = outdir / f"{fpath_fits.stem}.lv2.fits" # Assuming .fits input, replace extension
            ccd.write(outpath, overwrite=True)
            self.logger.info(f"Saved Level-2 product to {outpath}")
            
        return zp, zp_std