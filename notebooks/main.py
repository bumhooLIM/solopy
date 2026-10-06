# Main script for the SOLO (Solar system Object Light-curve Observatory) data reduction pipeline.
# Processes one night: Lv0 (decompress + headers) -> master bias/dark -> Lv1 (WCS + BDF)
# -> Lv2 (spatial PSF + zero point) -> Lv3 (asteroid photometry).
#
# Run it from a folder that also holds `directory.py` (the path configuration); in production that is
# ~/Desktop/data/solo/notebooks/. This file is versioned in the solopy repository as notebooks/main.py.
#
#   python main.py -s 2026_0630                  # all levels
#   python main.py -s 2026_0630 --levels 2,3     # re-run zero points and asteroid photometry only
#
# Last update: 2026-10-06
# version 1.1.0

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from ccdproc import ImageFileCollection, CCDData

import solopy
import directory  # type: ignore

ALL_LEVELS = (0, 1, 2, 3)


def parse_args(argv=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SOLO Data Reduction Pipeline (one night)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Require the subdirectory (observation date)
    parser.add_argument(
        "-s", "--subdir",
        type=str,
        required=True,
        help="Observation date in YYYY_MMDD format (used as the subdirectory name)."
    )

    # Optional detector name, defaults to kl4040
    parser.add_argument(
        "-d", "--detector",
        type=str,
        default="kl4040",
        help="Detector name used for master calibration file naming."
    )

    parser.add_argument(
        "--levels",
        type=str,
        default="0,1,2,3",
        help="Comma-separated levels to run: 0 = decompress, headers, master bias/dark; "
             "1 = WCS + bias/dark/flat; 2 = spatial PSF + zero point; 3 = asteroid photometry."
    )

    parser.add_argument(
        "--badpix-frac-max",
        type=float,
        default=0.05,
        help="Flag photometry (badphot) when masked pixels cover more than this fraction of the aperture."
    )

    parser.add_argument(
        "--rebuild-gaia",
        action="store_true",
        help="Rebuild the nightly Gaia subset even if it already exists."
    )

    args = parser.parse_args(argv)
    args.levels = sorted({int(level) for level in args.levels.split(",") if level.strip()})
    if not set(args.levels) <= set(ALL_LEVELS):
        parser.error(f"--levels must be a subset of {ALL_LEVELS}")
    return args


def read_summary(location, glob_include="*.fits"):
    """
    FITS header summary of a folder as a DataFrame whose 'file' column holds absolute paths.
    ccdproc lists bare file names (and returns None for an empty folder), so build full paths here.
    """
    summary = ImageFileCollection(location, glob_include=glob_include, glob_exclude="._*").summary
    if summary is None or len(summary) == 0:
        return pd.DataFrame(columns=["file"])
    df = summary.to_pandas()
    df["file"] = [str(Path(location) / name) for name in df["file"]]
    return df


def science_frames(summary, keywords_object, name_col):
    """LIGHT frames whose `name_col` (OBJECT for Lv0, LV0FILE for Lv1) starts with dawn/dusk."""
    if summary.empty or not {"imagetyp", name_col} <= set(summary.columns):
        return summary.iloc[0:0]
    is_science = (summary["imagetyp"] == "LIGHT") & \
        summary[name_col].astype(str).str.startswith(tuple(keywords_object))
    return summary[is_science].reset_index(drop=True)


def load_or_build_gaia_subset(fpath_night, fpath_gaia_all, ra_deg, dec_deg, subdir_name, rebuild, logger):
    """Nightly Gaia subset covering every field of the night (built once, then reused)."""
    if fpath_night.exists() and not rebuild:
        subset, boxes, meta = solopy.GaiaQuery.load_subset(fpath_night)
        logger.info(f"Loaded nightly Gaia subset {fpath_night.name}: {len(subset):,} sources "
                    f"({meta['n_isolated']:,} isolated)")
        return subset, boxes

    logger.info(f"Building nightly Gaia subset from {len(ra_deg)} pointings "
                f"(radius {solopy.NIGHTLY_SUBSET_RADIUS_DEG} deg around each)...")
    subset, boxes = solopy.GaiaQuery.build_nightly_subset(fpath_gaia_all, ra_deg, dec_deg)
    solopy.GaiaQuery.save_subset(
        fpath_night, subset, boxes,
        night=subdir_name, source_catalog=str(fpath_gaia_all),
        radius_deg=solopy.NIGHTLY_SUBSET_RADIUS_DEG, isolation_arcsec=20.0,
    )
    logger.info(f"Saved {fpath_night.name}: {len(subset):,} sources ({int(subset['iso'].sum()):,} isolated)")
    return subset, boxes


def ensure_gaia_coverage(subset, boxes, lv1_files, fpath_night, fpath_gaia_all, subdir_name, logger):
    """Extend the nightly subset if a plate-solved footprint is not fully covered by it."""
    extra_boxes = []
    for fpath in lv1_files:
        try:
            hdr = fits.getheader(fpath)
            wcs = WCS(hdr)
            wcs.pixel_shape = (hdr["NAXIS1"], hdr["NAXIS2"])
        except Exception as e:
            logger.warning(f"Cannot read WCS of {Path(fpath).name} for the Gaia coverage check: {e}")
            continue
        if not solopy.GaiaQuery.footprint_covered(boxes + extra_boxes, wcs):
            logger.warning(f"{Path(fpath).name} lies partly outside the nightly Gaia subset; extending it.")
            extra_boxes.extend(solopy.GaiaQuery.wcs_boxes(wcs))

    if not extra_boxes:
        logger.info(f"Nightly Gaia subset covers all {len(lv1_files)} Lv1 footprints.")
        return subset, boxes

    boxes = boxes + extra_boxes
    subset = solopy.GaiaQuery.build_subset(fpath_gaia_all, boxes)
    solopy.GaiaQuery.save_subset(
        fpath_night, subset, boxes,
        night=subdir_name, source_catalog=str(fpath_gaia_all),
        radius_deg=solopy.NIGHTLY_SUBSET_RADIUS_DEG, isolation_arcsec=20.0, extended_with_wcs=True,
    )
    logger.info(f"Extended nightly Gaia subset: {len(subset):,} sources")
    return subset, boxes


def main(argv=None):
    args = parse_args(argv)

    subdir_name = args.subdir
    detector_name = args.detector
    levels = args.levels

    # ===========================================================================================
    # Directory setup
    # ===========================================================================================

    WORK_DIR = directory.WORK_DIR
    MASTER_DIR = directory.MASTER_DIR

    # Master flat and bad pixel mask paths
    fpath_mflat = MASTER_DIR / f"{detector_name}.flat.clear.comb.20260526.fits"
    fpath_bpm = MASTER_DIR / f"{detector_name}.bpm.20260616.fits"

    LV0_DIR = directory.LV0_DIR
    LV0_SUBDIR = LV0_DIR / subdir_name

    LV1_DIR = directory.LV1_DIR
    LV1_SUBDIR = LV1_DIR / subdir_name

    RESULT_DIR = directory.RESULT_DIR

    ZP_DIR = directory.ZP_DIR
    ZP_SUBDIR = ZP_DIR / subdir_name
    ZP_SUBDIR.mkdir(parents=True, exist_ok=True)

    PSF_DIR = directory.PSF_DIR
    PSF_SUBDIR = PSF_DIR / subdir_name
    PSF_SUBDIR.mkdir(parents=True, exist_ok=True)

    LOG_DIR = directory.LOG_DIR
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fpath_log = LOG_DIR / f'solopy_{subdir_name}.log'
    if fpath_log.exists():
        fpath_log.unlink()

    # Optional entries in directory.py (defaults keep the previous layout)
    ASTROMETRY_CACHE_DIR = getattr(directory, "ASTROMETRY_CACHE_DIR", WORK_DIR / "astrometry_cache")
    GAIA_NIGHTLY_DIR = getattr(directory, "GAIA_NIGHTLY_DIR", directory.GAIA_DIR / "nightly")

    # Gaia catalog paths: the full catalog is only read once per night to build the nightly subset
    fpath_gaia_all = directory.GAIA_DIR / "gaiadr3.npy"
    fpath_gaia_night = GAIA_NIGHTLY_DIR / f"gaiadr3.{subdir_name}.npy"

    # Orbital elements file path for SSO ephemeris prediction
    fpath_orb = directory.SLOC_DIR / "orb_sbdb.parq"

    # keywords for science frames summary
    keywords_object = ['dawn', 'dusk']
    keywords_header = [
        'file', 'filename', 'zpfile', 'psffile', 'lv0file', 'jd', 'obsdate', 'date-obs', 'exptime', 'filter', 'object',
        'egain', 'rdnoise', 'racen', 'deccen', 'altcen', 'azcen', 'pixscale', 'psf_fwhm', 'zp_g', 'zperr_g', 'zpsource'
        ]
    keywords_to_numeric = [
        'jd', 'exptime', 'racen', 'deccen', 'altcen', 'azcen',
        'psf_fwhm', 'pixscale', 'egain', 'rdnoise', 'zp_g', 'zperr_g', 'zpsource'
        ]

    # Setup root logger for the main script (solopy class loggers do not propagate here)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(str(fpath_log))]
    )
    logger = logging.getLogger("MAIN")
    logger.info(f"solopy {solopy.__version__}: night {subdir_name}, levels {levels}")

    # ===========================================================================================
    # Fits Lv0 Processing (decompress .bz2 + update headers)
    # ===========================================================================================
    if not LV0_SUBDIR.exists():
        logger.error(f"Target directory does not exist: {LV0_SUBDIR}")
        return

    if 0 in levels:
        # Initialize the Level-0 Processor
        logger.info(f"Starting Lv0 Pipeline for {subdir_name}...")
        lv0 = solopy.FitsLv0(log_file=str(fpath_log))

        # 1. Decompress .fits.bz2 files in-place
        lv0.batch_decompress(in_dir=LV0_SUBDIR, out_dir=LV0_SUBDIR, delete_source=True)

    all_lv0fits = ImageFileCollection(LV0_SUBDIR, glob_include="*.fits", glob_exclude="._*")

    if 0 in levels:
        # 2. Update FITS headers for all Lv0 files
        fits_files = all_lv0fits.files_filtered(include_path=True)
        logger.info(f"Updating headers for {len(fits_files)} files...")
        for fpath_fits in fits_files:
            lv0.update_header(fpath_fits)

        all_lv0fits.refresh()  # Refresh the file collection to reflect updated headers

    lv0_science = science_frames(read_summary(LV0_SUBDIR), keywords_object, "object")
    lv0_frame = lv0_science["file"].to_list()
    if not lv0_frame:
        logger.error(f"No dawn/dusk LIGHT frames found in {LV0_SUBDIR}")
        return

    # ===========================================================================================
    # Nightly Gaia subset, built before calibration from the telescope pointings
    # ===========================================================================================
    gaia_night, gaia_boxes = None, None
    if {2, 3} & set(levels):
        pointing_ra = pd.to_numeric(lv0_science["ra"], errors="coerce").to_numpy()
        pointing_dec = pd.to_numeric(lv0_science["dec"], errors="coerce").to_numpy()
        gaia_night, gaia_boxes = load_or_build_gaia_subset(
            fpath_gaia_night, fpath_gaia_all, pointing_ra, pointing_dec, subdir_name, args.rebuild_gaia, logger
        )

    # ===========================================================================================
    # Master calibration frames
    # ===========================================================================================
    if 0 in levels:
        bias_frame = all_lv0fits.files_filtered(imagetyp="BIAS", include_path=True)
        dark_frame = all_lv0fits.files_filtered(imagetyp="DARK", include_path=True)

        logger.info(f"Found {len(bias_frame)} BIAS frames and {len(dark_frame)} DARK frames.")

        comb = solopy.CombMaster(log_file=str(fpath_log))

        if len(bias_frame) > 0:
            comb.comb_master_bias(bias_frame, MASTER_DIR, outname=detector_name)
        if len(dark_frame) > 0:
            comb.comb_master_dark(dark_frame, MASTER_DIR, outname=detector_name)

    # ===========================================================================================
    # Fits Lv1 Processing (WCS update + BDF correction)
    # ===========================================================================================
    if 1 in levels:
        # Load Master flat
        try:
            mflat = CCDData.read(fpath_mflat)
            # Prevent Division-by-Zero errors (inf)
            mflat.data = np.nan_to_num(np.clip(mflat.data, 1e-5, None), nan=1.0)
        except Exception as e:
            logger.error(f"Failed to load Master Flat: {e}")
            return

        # Load Master bad pixel mask
        try:
            ccdmask = CCDData.read(fpath_bpm)
        except ValueError:
            # Safely load the unitless mask file if BUNIT is missing in the header
            ccdmask = CCDData.read(fpath_bpm, unit='adu')
        except Exception as e:
            logger.warning(f"Master BPM not found or failed to load. Proceeding without external mask. Error: {e}")
            ccdmask = None

        lv1 = solopy.FitsLv1(log_file=str(fpath_log))
        logger.info(f"Starting Lv1 Processing for {len(lv0_frame)} LIGHT frames...")

        for fpath_fits in lv0_frame:
            fpath_wcs = lv1.update_wcs(
                fpath_fits,
                outdir=LV1_SUBDIR,
                cache_directory=ASTROMETRY_CACHE_DIR,
                return_fpath=True
            )

            # update_wcs returns None when the frame has no astrometric solution
            if not fpath_wcs:
                logger.warning(f"Skipping BDF correction for {Path(fpath_fits).name} due to WCS failure.")
                continue

            fpath_bdf = lv1.correct_bdf(
                fpath_wcs,
                outdir=LV1_SUBDIR,
                masterdir=MASTER_DIR,
                ccdmflat=mflat,
                ccdmask=ccdmask,
                return_fpath=True
            )

            # Clean up intermediate WCS file
            if fpath_wcs.exists():
                fpath_wcs.unlink()

        logger.info(f"Level-1 processing completed.")

    if not ({2, 3} & set(levels)):
        return

    # Lv1 science frames of the night
    lv1_science_files = science_frames(read_summary(LV1_SUBDIR, "*lv1*.fits"), keywords_object, "lv0file")["file"].to_list()
    if not lv1_science_files:
        logger.error(f"No Lv1 science frames found in {LV1_SUBDIR}")
        return

    # The plate-solved footprints must lie inside the nightly Gaia subset
    gaia_night, gaia_boxes = ensure_gaia_coverage(
        gaia_night, gaia_boxes, lv1_science_files, fpath_gaia_night, fpath_gaia_all, subdir_name, logger
    )

    # ===========================================================================================
    # Fits Lv2 Processing (Zero-Point Calculation)
    # ===========================================================================================
    if 2 in levels:
        lv2 = solopy.FitsLv2(log_file=str(fpath_log))
        gaia_isolated = gaia_night[gaia_night["iso"]]  # replaces gaiadr3_20arcsec.npy

        # Initialize the PSF Processor
        psf_processor = solopy.soloPSF(init_fwhm=2.5,     # initial guess for PSF FWHM in pixels
                                       n_star=20,         # maximum number of sources in each tile
                                       peakmin=300,       # minimum peak value for a source to be considered
                                       peakmax=3000,      # maximum peak value for a source to be considered
                                       max_ab_ratio=2.0,  # maximum allowed axis ratio for a source to be considered (to filter out elongated sources)
                                       max_deviation=3.0, # maximum allowed deviation from the center peak for a source to be considered (to filter out off-center sources)
                                       base_tile_size=500
                                       )

        logger.info(f"Starting Lv2 Processing for {len(lv1_science_files)} science frames...")

        for fpath_fits in map(Path, lv1_science_files):

            # ---------------------------------------------------------
            # STEP 1: Spatial PSF Calculation
            # ---------------------------------------------------------
            logger.info(f"Calculating spatial PSF for {fpath_fits.name}...")

            try:
                data = fits.getdata(fpath_fits).astype(np.float32)
            except Exception as e:
                logger.error(f"Failed to load image data for {fpath_fits.name}: {e}")
                continue

            psf_df = psf_processor.process_ccd(data)

            if psf_df is None or psf_df.empty:
                logger.warning(f"PSF evaluation failed for {fpath_fits.name}. Falling back to default FWHM=2.5")
                overall_fwhm = 2.5
                psf_table_pass = None
            else:
                # We use median instead of mean here to guard against bad regions skewing the header
                overall_fwhm = float(psf_df['fwhm_avg'].median())
                psf_table_pass = psf_df.copy()

                # Save PSF table to CSV (dropping the 2D array)
                fpath_out_psf = PSF_SUBDIR / f"psf.{fpath_fits.stem}.csv"
                psf_df_save = psf_df.drop(columns=['avg_psf_data'])
                psf_df_save.to_csv(fpath_out_psf, index=False)
                logger.info(f"Saved spatial PSF catalog to {fpath_out_psf.name}")

                try:
                    with fits.open(fpath_fits, mode='update') as hdul:
                        hdul[0].header['PSFFILE'] = (fpath_out_psf.name, 'PSF table')  # short: fits on the card
                        hdul[0].header['PSF_FWHM'] = (overall_fwhm, '[pixels] Median field PSF FWHM')
                        hdul.flush()
                except Exception as e:
                    logger.error(f"Failed to write PSF_FWHM header to {fpath_fits.name}: {e}")

            # ---------------------------------------------------------
            # STEP 2: Zero Point Calculation (Spatially Varying)
            # ---------------------------------------------------------
            success = lv2.calculate_zeropoint(
                fpath_fits=fpath_fits,
                gaia_data=gaia_isolated,
                outdir_zp=ZP_SUBDIR,
                mag_lower=13.0,
                mag_upper=15.0,
                psf_table=psf_table_pass,       # Pass the DataFrame to the photometer
                base_tile_size=500,             # Keep synchronized with soloPSF
                fallback_fwhm=overall_fwhm,     # Used if a specific region failed to fit
                ap_in_out=(1.5, 3.0, 4.0),      # Standard dynamic FWHM multipliers
                badpix_frac_max=args.badpix_frac_max
            )

            if not success:
                logger.warning(f"Failed to generate Level-2 Zero Point for {fpath_fits.name}")

        logger.info("Lv2 Zero-Point processing complete.")

    # ===========================================================================================
    # Fits Lv3 Processing (SSO Photometry)
    # ===========================================================================================
    if 3 in levels:
        lv3 = solopy.FitsLv3(orb_path=fpath_orb, gaia_path=gaia_night, log_file=str(fpath_log))

        # Re-read headers: Lv2 has just written PSF_FWHM and ZP_G (test frames are filtered out)
        science_summary = science_frames(read_summary(LV1_SUBDIR, "*lv1*.fits"), keywords_object, "lv0file")
        missing = [key for key in keywords_header if key not in science_summary.columns]
        if missing:
            logger.error(f"Lv1 headers lack {missing}; run level 2 before level 3.")
            return
        science_summary = science_summary[keywords_header]
        for col in keywords_to_numeric:
            science_summary[col] = pd.to_numeric(science_summary[col], errors='coerce')
        science_summary = science_summary.dropna().reset_index(drop=True)

        # Predict Targets
        eph_all = lv3.predict_targets(science_summary, vmag_upper=16.5)
        if eph_all.empty:
            print("No targets found. Exiting.")
            return

        # Extract SSO Photometry
        sso_phot_summary = lv3.extract_sso_photometry(science_summary, eph=eph_all, psf_dir=PSF_DIR,
                                                      ap_in_out=(1.5, 3.0, 4.0),
                                                      badpix_frac_max=args.badpix_frac_max,
                                                      zp_dir=ZP_DIR)
        if sso_phot_summary.empty:
            print("Photometry extraction failed. Exiting.")
            return

        # 6. Apply Absolute Calibrations: local zero point at solar color (falls back to the frame
        #    value at solar color when fewer than 10 stars lie within 500 px)
        sso_phot_summary.dropna(subset=['mag_inst', 'zp_local'], inplace=True)

        sso_phot_summary['gmag'] = sso_phot_summary['mag_inst'] + sso_phot_summary['zp_local']
        sso_phot_summary['gmag_distcorr'] = sso_phot_summary['gmag'] - 5 * np.log10(sso_phot_summary['r_hel'] * sso_phot_summary['r_obs'])

        # 7. Save Results
        colnames = [
            'desig', 'jd_tdb', 'jd_utc', 'r_hel', 'r_obs', 'vmag', 'alpha',
            'ra', 'dec', 'altcen', 'azcen',
            'filename', 'obsid', 'obsdate', 'exptime', 'object',
            'x_winpos', 'y_winpos', 'mapped_fwhm', 'r_ap_pixel', 'aperture_area',
            'aperture_sum', 'aperture_sum_err', 'annulus_median',
            'bkg_std', 'nsky', 'source_sum', 'source_sum_err', 'snr',
            'mag_inst', 'mag_err', 'mag_err_tot', 'badphot', 'nbadpix', 'badpix_frac',
            'saturated', 'psf_lost_frac',
            'zp_global', 'zperr_global', 'zp_sun', 'zp_color',
            'zp_local', 'zperr_local', 'zp_local_spread', 'zp_local_n', 'zp_local_fallback',
            'gmag', 'gmag_distcorr',
            'nearest_gaia_source_id', 'nearest_gaia_gmag', 'nearest_gaia_dist_arcsec'
        ]

        final_df = sso_phot_summary[colnames].copy()
        final_df.rename(columns={'x_winpos': 'x', 'y_winpos': 'y', 'mapped_fwhm': 'psf_fwhm'}, inplace=True)

        for obsdate, gp in final_df.groupby('obsdate'):
            out_path = RESULT_DIR / f"solo.summary.{obsdate}.csv"
            gp.to_csv(out_path, index=False)
            print(f"Saved {len(gp)} targets to {out_path.name}")


if __name__ == "__main__":
    main()
