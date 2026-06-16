import numpy as np
import pandas as pd
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.table import Table
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import Gaussian2D
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAnnulus, ApertureStats
import copy
import sep
from .region import SOLORegion as soloregion

import warnings
from astropy.utils.exceptions import AstropyUserWarning
# Add this at the top of your master script, right after your imports:
warnings.simplefilter('ignore', category=AstropyUserWarning)
warnings.simplefilter('ignore', category=RuntimeWarning)

class soloPSF:
    
    def __init__(self, init_fwhm=2.5, n_star=20, peakmin=300, peakmax=3000,
                 max_ab_ratio=2.0, max_deviation=3.0, base_tile_size=500):
        self.init_fwhm = init_fwhm
        self.n_star = n_star
        self.peakmin = peakmin
        self.peakmax = peakmax         # Upper limit to ignore saturated blooming stars
        # self.minarea = minarea         # Minimum pixels required to be a source
        self.max_ab_ratio = max_ab_ratio # Streak filter: max allowed a/b ratio (1.0 is a perfect circle)
        self.max_deviation = max_deviation
        self.base_tile_size = base_tile_size
        self.psf_cutout_size = int(5 * self.init_fwhm)
        self.fitter = LevMarLSQFitter()

    def _extract_psfs(self, ccd_cutout):
        """Extract PSFs using SEP with morphological streak filtering."""
        
        # 1. SEP Strict Data Formatting
        # SEP requires C-contiguous, native byte-order float arrays.
        data = np.ma.getdata(ccd_cutout.data)
        if data.dtype.byteorder == '>':
                data = data.byteswap().view(data.dtype.newbyteorder())
        data = data.astype(np.float32)
        
        # 2. SEP Background Estimation
        try:
            bkg = sep.Background(data)
            data_sub = data - bkg
        except Exception as e:
            print(f"Error occurred while estimating background: {e}")
            return [], [] # Failsafe if the tile is completely empty or corrupted
            
        # 3. Source Extraction
        try:
            objects = sep.extract(data_sub, thresh=3.0, err=bkg.globalrms, minarea=np.pi*(0.5*self.init_fwhm)**2)
        except Exception as e:
            print(f"Error occurred while extracting sources: {e}")
            return [], []
            
        
        if objects is None or len(objects) == 0:
            return [], []
            
        # 4. Strict Morphological Filtering (Streak & Saturation Defense)
        # Calculate ellipticity: a / b (1.0 = perfect circle, > 2.0 = streak)
        ab_ratio = objects['a'] / objects['b']
        
        mask_valid = (
            (objects['peak'] > self.peakmin) & 
            (objects['peak'] < self.peakmax) &
            (ab_ratio <= self.max_ab_ratio) # Filter streaks, cosmic rays, and satellites
        )
        
        objects = objects[mask_valid]
        
        if len(objects) == 0:
            return [], []
            
        # 5. Sort by brightness (flux) and select the top 'n_star' candidates
        objects = np.sort(objects, order='flux')[::-1][:self.n_star]
        
        # 6. Edge exclusion logic
        center_pos = np.transpose((objects['x'], objects['y']))
        image_h, image_w = data.shape
        x_pos, y_pos = center_pos[:, 0], center_pos[:, 1]
        
        dist_to_border = np.minimum.reduce([
            x_pos, image_w - x_pos, 
            y_pos, image_h - y_pos
        ])
        
        mask_dist = dist_to_border >= (self.psf_cutout_size * 0.5)
        objects = objects[mask_dist]
        
        if len(objects) == 0:
            return [], []
            
        # Update valid coordinates
        center_pos = np.transpose((objects['x'], objects['y']))
        
        # 7. High-Precision Local Background (Photutils Annulus)
        # SEP evaluates background globally across the tile. We use an annulus here 
        # to guarantee the exact local pedestal is removed for perfect PSF normalization.
        aps_sky = CircularAnnulus(center_pos, r_in=4.*self.init_fwhm, r_out=6.*self.init_fwhm)
        sky_stats = ApertureStats(data, aps_sky, sigma_clip=SigmaClip())
        list_background = sky_stats.median 

        list_psf = []
        peak_values = []
        
        # 8. Extract Cutouts
        for i, (x_center, y_center) in enumerate(zip(objects['x'], objects['y'])):
            cutout = Cutout2D(data, position=(x_center, y_center), size=self.psf_cutout_size)
            psf_data = cutout.data.astype('float64')
            
            # STRICT SHAPE CHECK
            if psf_data.shape != (self.psf_cutout_size, self.psf_cutout_size):
                continue
            
            if np.isnan(psf_data).all():
                continue
                
            peak_val = np.nanmax(psf_data)
            
            # Subtraction and Normalization
            psf_data -= list_background[i]
            
            sum_flux = np.nansum(psf_data)
            if sum_flux > 0 and not np.isinf(sum_flux):
                psf_data /= sum_flux
            else:
                continue 
                
            peak_values.append(peak_val)
            list_psf.append(psf_data)
            
        return list_psf, peak_values

    def _fit_gaussians(self, psf_list):
        """Internal method to fit 2D Gaussians with bounds and safety checks."""
        fwhms, thetas, flags = [], [], []
        
        for psf_data in psf_list:
            y, x = np.indices(psf_data.shape)
            y_center, x_center = psf_data.shape[0] / 2, psf_data.shape[1] / 2
            
            init_guess = Gaussian2D(
                amplitude=np.max(psf_data),
                x_mean=x_center, y_mean=y_center,
                x_stddev=self.init_fwhm/2.355, y_stddev=self.init_fwhm/2.355,
                theta=0
            )
            # Fixed: Force positive bounds to prevent negative square root warnings
            init_guess.x_stddev.bounds = (1e-5, None)
            init_guess.y_stddev.bounds = (1e-5, None)
            init_guess.amplitude.bounds = (0, None)
            
            model = self.fitter(init_guess, x, y, psf_data, filter_non_finite=True)
            
            # Fixed: Check if fitter successfully converged
            if self.fitter.fit_info['ierr'] not in [1, 2, 3, 4]:
                flags.append(False)
                fwhms.append(np.nan)
                thetas.append(np.nan)
                continue
                
            x_stddev, y_stddev = model.x_stddev.value, model.y_stddev.value
            fwhm = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(x_stddev * y_stddev)
            
            deviation = np.sqrt((model.x_mean.value - x_center)**2 + (model.y_mean.value - y_center)**2)
            flag = deviation <= self.max_deviation
            
            fwhms.append(fwhm)
            thetas.append(model.theta.value)
            flags.append(flag)
            
        return fwhms, thetas, flags

    def process_ccd(self, ccd):
        """
        Process the entire CCD, splitting it into regions and evaluating PSF variations.
        
        Returns:
        - pd.DataFrame containing the FWHM statistics for each spatial tile.
        """
        regions = soloregion(ccd.shape, self.base_tile_size)
        result_table = []
        
        for i in range(regions.num_tiles_x):
            for j in range(regions.num_tiles_y):
                
                tile_info = regions.get_tile_info(i, j)
                
                # Perform the image cutout for this region dynamically
                position = (tile_info['x_center'], tile_info['y_center'])
                size = (tile_info['size_y'], tile_info['size_x'])
                ccd_cutout = Cutout2D(ccd, position=position, size=size)
                
                # Extract and Fit
                psf_list, peak_values = self._extract_psfs(ccd_cutout)
                
                if not psf_list:
                    continue  # Skip if no valid stars found
                    
                fwhms, thetas, flags = self._fit_gaussians(psf_list)
                
                # --- NEW: Strict Filtering and Median Stacking ---
                valid_fwhms = []
                valid_thetas = []
                valid_psfs_data = [] # Store only the clean 2D arrays
                
                for psf_arr, fwhm, theta, flag in zip(psf_list, fwhms, thetas, flags):
                    # Keep if fit succeeded AND FWHM is physically reasonable (e.g., 1.5 to 10 pixels)
                    if flag and not np.isnan(fwhm) and (1.5 < fwhm < 8.0):
                        valid_fwhms.append(fwhm)
                        valid_thetas.append(theta)
                        valid_psfs_data.append(psf_arr)
                
                if not valid_fwhms:
                    continue
                
                # Use np.median instead of np.mean to completely erase streaks, cosmic rays, and background anomalies!
                avg_psf = np.median(valid_psfs_data, axis=0)
                # ------------------------------------------------
                
                result_table.append({
                    "region_i": i,
                    "region_j": j,
                    "x_center": tile_info['x_center'],
                    "y_center": tile_info['y_center'],
                    "tile_size_x": tile_info['size_x'],
                    "tile_size_y": tile_info['size_y'],
                    "fwhm_avg": np.mean(valid_fwhms),
                    "fwhm_median": np.median(valid_fwhms),
                    "fwhm_stddev": np.std(valid_fwhms),
                    "theta_avg": np.mean(valid_thetas),
                    "peak_min": np.min(peak_values),  
                    "peak_max": np.max(peak_values),
                    "n_star": len(valid_fwhms),
                    "avg_psf_data": avg_psf # Store the average PSF data for this region (can be used for later analysis or visualization)
                })
                
        return pd.DataFrame(result_table)