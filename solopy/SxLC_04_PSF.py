import ccdproc as ccdp
from astropy.nddata import CCDData
from astropy.wcs import FITSFixedWarning
from astropy.visualization import ZScaleInterval
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy import units as u
from astropy.time import Time
from astropy.nddata import Cutout2D
from photutils.detection import DAOStarFinder
from photutils.aperture import (CircularAperture,  CircularAnnulus, ApertureStats)
import astroalign as aa

from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt
from matplotlib import rcParams

from pathlib import Path
import warnings
import numpy as np
import sys
from datetime import datetime, timedelta

SCRIPTDIR = sys.path.append('/home/bumhoo7/__MYresearch__/__script__/')
import ccdutil

warnings.filterwarnings("ignore", category=FITSFixedWarning)

# rcParams
rcParams['font.family']         = 'DejaVu Sans'
rcParams['font.weight']         = 'bold'  
rcParams['font.size']           = 15 
rcParams['figure.figsize']      = (10, 9)
rcParams['figure.titlesize']    = 20
rcParams['axes.titlesize']      = 20
rcParams['axes.labelsize']      = 15
rcParams['axes.titleweight']    = 'bold' 
rcParams['axes.labelweight']    = 'bold'
rcParams['xtick.labelsize']     = 15  
rcParams['ytick.labelsize']     = 15  
rcParams['xtick.direction']     = 'in'
rcParams['ytick.direction']     = 'in'
rcParams['xtick.top']           = True
rcParams['ytick.right']         = True
rcParams['xtick.minor.visible'] = True
rcParams['ytick.minor.visible'] = True
rcParams['savefig.bbox']        = 'tight'
rcParams['savefig.dpi']         = 200

FIGDIR    = Path('__fig__')
WORKDIR   = Path('./__data__/20241112_Seoul/')
INDIR     = WORKDIR/'03_merged'
OUTDIR    = WORKDIR/'04_psf'
fig_name  = WORKDIR.name

for FOCUS_POS in [13000, 13500, 14000, 14500, 15000, 15500, 16000, 16500, 17000]:
    fpath_obj = INDIR/f'Focus_{FOCUS_POS}-001-010_r_merged.fits'
    fpath_out = OUTDIR/(fpath_obj.stem + '.psf' + fpath_obj.suffix)

    try:
        OUTDIR.mkdir(exist_ok=False)
    except FileExistsError: # directory already exists.     
        pass

    fwhm        = 2.0 + (np.abs(FOCUS_POS-15000))/1e+3  # expected fwhm of sources [pixel]
    ccd         = CCDData.read(fpath_obj)
    ny, nx      = ccd.shape
    ccd.header['FOC-POS'] = (FOCUS_POS, 'Focal position')

    #
    # Cutout
    #

    cutout_size     = 500
    cutout_position = (nx//2, ny//2)
    cutout_ccd      = ccdutil.CreateCutoutCCD(ccd, position=cutout_position, size=cutout_size)

    #
    # source finding with DAOStarFinder
    #

    avg, med, std = sigma_clipped_stats(cutout_ccd.data) 

    finder        = DAOStarFinder(
        threshold=3.*std,
        fwhm=fwhm,
        exclude_border=True, 
        peakmax=5e+4, 
        brightest=20 # only brightest 20 sources
        )

    sources       = finder(cutout_ccd.data - med)

    # for col in sources.colnames:
    #     sources[col].info.format = "%d" if col in ('id', 'npix') else '%.2f'
    # sources.pprint(max_width=100)

    #
    # Sky subtraction & normzalization
    #

    center_pos    = np.transpose((sources['xcentroid'], sources['ycentroid']))
    aps           = CircularAperture(center_pos, r=1.5*fwhm)
    aps_sky       = CircularAnnulus(center_pos, r_in=2.*fwhm, r_out=4.*fwhm)

    sky_stats = ApertureStats(
        cutout_ccd.data,
        aps_sky,
        sigma_clip=SigmaClip()
        )

    sky_value = (sky_stats.sum / sky_stats.sum_aper_area).to_value('1/pix2')

    fig = plt.figure(figsize= (16, 20))
    gs  = GridSpec(len(sources)//4, 4)

    PSF = list()
    for i, (xcentroid, ycentroid) in enumerate(zip(sources['xcentroid'], sources['ycentroid'])):
        psf = ccdutil.CreateCutoutCCD(cutout_ccd, position=(xcentroid, ycentroid), size=25)
        if psf.data.shape != (25, 25):  # Exclude cutouts that are not the expected size
            continue
        psf.data -= sky_value[i]
        psf.header.add_history(f'Background subtracted (value={sky_value[i]}).')
        psf = ccdutil.Normalize_CCDData(psf)
        PSF.append(psf)
        ax = fig.add_subplot(gs[i])
        ax.imshow(psf.data, cmap='jet', origin='lower', aspect='auto', vmin=0, vmax=1)

    plt.savefig(OUTDIR/f'FOCUS{FOCUS_POS}_01')
    plt.close()

    #
    # PSF combined
    #
    psf_combined = ccdp.combine(
        PSF,
        method='median',
        mem_limit=350e6,
        dtype=np.float32
        )
    
    psf_combined = ccdutil.Normalize_CCDData(psf_combined)

    fig = plt.figure()
    ax  = fig.add_subplot()
    ax.imshow(psf_combined.data, cmap='jet', origin='lower', vmin=0, vmax=1)
    ax.set_title(f'{fpath_out.name}')
    plt.savefig(OUTDIR/f'FOCUS{FOCUS_POS}_02')

    psf_combined.meta['combined'] = True
    psf_combined.write(fpath_out, overwrite=True)
    print(f'{fpath_out} has been saved. ({datetime.utcnow().isoformat()})')