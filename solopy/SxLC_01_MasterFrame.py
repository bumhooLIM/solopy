from astropy.wcs import FITSFixedWarning
from astropy.nddata import CCDData
from astropy.stats import mad_std
import astropy.units as u
import ccdproc as ccdp

import random
import numpy as np
import warnings
from datetime import datetime
from pathlib import Path
import _fileutil as fileutil

warnings.filterwarnings("ignore", category=FITSFixedWarning)

is_skyflat  = False # use flat frames for the master flat. If False, use median-combined science frames as alternatives. 
n_flatframe = 99    # if no flat images, use randomly chosen n frames from the science data.
seed        = 1234  # random seed

# file path
Filtername  = 'r'                                                                   # filter name
RAWDIR      = Path.home()/'__MYresearch__/__rawdata__/SPHERExLC/20241218_Seoul_r/'  # file path to raw data
WORKDIR     = Path('./20241218_Seoul/')                                             # working directory
MASTERDIR   = WORKDIR/'lv00_masterframe'                                            # file path to save preprocessing data
TMPDIR      = WORKDIR/'tmp'                                                         # temporary directory (removed after end of process)
fpath_mbias = MASTERDIR/'Mbias.fit'                                                   # file path to save master bias
fpath_mflat = MASTERDIR/f'Mflat_{Filtername}.fit'                                     # file path to save master flat

for fpath in [WORKDIR, MASTERDIR, TMPDIR]:

    try:
        fpath.mkdir(exist_ok=False)
    except FileExistsError: # directory already exists. 
        continue

allfits         = ccdp.ImageFileCollection(RAWDIR)                                                       # all fits file in raw data

list_fpath_bias       = [Path(fpath) for fpath in allfits.files_filtered(imagetyp='Bias Frame' , include_path=True)] # bias
list_fpath_dark       = [Path(fpath) for fpath in allfits.files_filtered(imagetyp='Dark Frame' , include_path=True)] # dark
list_fpath_obj        = [Path(fpath) for fpath in allfits.files_filtered(imagetyp='Light Frame', include_path=True)] # science targets

frame, n_frame = np.unique(allfits.summary['imagetyp'], return_counts=True)
print(dict(zip(frame, n_frame)))

# collect flat frames
if is_skyflat:
    list_fpath_flat   = [Path(fpath) for fpath in allfits.files_filtered(imagetyp='Flat Field' , include_path=True)]
    
else:
    random.seed(seed)
    if len(list_fpath_obj) < n_flatframe:
        n_flatframe = len(list_fpath_obj)
        # raise ValueError("The number of available science frames is less than the required n_flatframe.")
    list_fpath_flat = random.sample(list_fpath_obj, n_flatframe) 
    print(f'is_flat == False. {len(list_fpath_flat)} Science frame will be alterantively used.')

#
# master bias
#

print(f'Combining master bias ({len(list_fpath_bias)} frames)...')
mbias = ccdp.combine(list_fpath_bias,
                     method='median',
                     unit='adu',
                     sigma_clip=True,
                     sigma_clip_low_thresh=5,
                     sigma_clip_high_thresh=5,
                     sigma_clip_func=np.ma.median,
                     sigma_clip_dev_func=mad_std,
                     mem_limit=350e6,
                     dtype=np.float32
                     )

mbias.meta['combined'] = True
mbias.write(fpath_mbias, overwrite=True)

print(f"{fpath_mbias} has been saved ({datetime.now()}).")

#
# master dark
#

# clear out the temporary directory
fileutil.clear_dir(TMPDIR) 

# load master bias
fileutil.check_file_exsist(fpath_mbias)
mbias = CCDData.read(fpath_mbias)

# subtract master bias from dark frame
print(f'Subtracting master bias from {len(list_fpath_dark)} dark frames...')
for fpath_dark in list_fpath_dark:
    
    fpath_bdark = TMPDIR / (fpath_dark.stem+'_b'+fpath_dark.suffix)
    
    dark       = CCDData.read(fpath_dark, unit='adu')
    bdark      = ccdp.subtract_bias(dark, mbias)
    bdark.write(fpath_bdark, overwrite=True)
    print(f"{fpath_bdark} has been saved ({datetime.now()}).")
    
allbdark   = ccdp.ImageFileCollection(TMPDIR)
allexptime = set(allbdark.summary['exptime']) # expoure times of dark frames

# combine master dark (for each exposure time)
for exptime in allexptime:
    
    list_bdark = [Path(bdark) for bdark in allbdark.files_filtered(imagetyp='Dark Frame', exptime=exptime, include_path=True)]
    fpath_mdark = MASTERDIR / f'Mdark_{exptime:3.1f}s.fit'
    
    print(f'Combining master dark (exptime={exptime}s, {len(list_bdark)} frames)...')
    mdark = ccdp.combine(list_bdark,
                         method='median',
                         sigma_clip=True,
                         sigma_clip_low_thresh=5,
                         sigma_clip_high_thresh=5,
                         sigma_clip_func=np.ma.median,
                         sigma_clip_dev_func=mad_std,
                         mem_limit=350e6,
                         dtype=np.float32
                         )    
    
    mdark.meta['combined'] = True
    mdark.write(fpath_mdark, overwrite=True)
    print(f"{fpath_mdark} has been saved ({datetime.now()}).")
            
#
# master flat
#

# clear out the temporary directory
fileutil.clear_dir(TMPDIR)

# load master bias
fileutil.check_file_exsist(fpath_mbias)
mbias = CCDData.read(fpath_mbias)

# subtract master bias from flat frames
print(f'Subtracting master bias from {len(list_fpath_flat)} flat frames...')
for fpath_flat in list_fpath_flat:
    
    fpath_bflat = TMPDIR / (fpath_flat.stem+'_b'+fpath_flat.suffix) # file path to save bias-subtracted dark
    
    flat       = CCDData.read(fpath_flat, unit='adu')
    bflat      = ccdp.subtract_bias(flat, mbias)
    bflat.write(fpath_bflat, overwrite=True)
    print(f"{fpath_bflat} has been saved ({datetime.now()}).")
    
allbflat   = ccdp.ImageFileCollection(TMPDIR)
allexptime = set(allbflat.summary['exptime']) # expoure times of dark frames

# subtract master dark from flat frames
list_bdflat = []
for exptime in allexptime:
    fpath_mdark = MASTERDIR/f'Mdark_{exptime:3.1f}s.fit' 
    fileutil.check_file_exsist(fpath_mdark)
    
    try:
        mdark       = CCDData.read(fpath_mdark)
        
    # if corresponding master dark does not exsist, find alternatives with the cloesest expsoure time.   
    except FileNotFoundError: 
        allmdark        = ccdp.ImageFileCollection(MASTERDIR).filter(imagetyp='Dark Frame')
        exptime_closest = min(set(allmdark.summary['exptime']),key=lambda x:abs(x-exptime))
        fpath_mdark      = MASTERDIR/f'Mdark_{exptime_closest:3.1f}s.fit'
        mdark           = CCDData.read(fpath_mdark)   
        print(f'{fpath_mdark} was alternatively used.')
    
    # list_bflat = [Path(bflat) for bflat in allbflat.files_filtered(imagetyp='Flat Field', exptime=exptime, include_path=True)]
    list_bflat = [Path(bflat) for bflat in allbflat.files_filtered(exptime=exptime, include_path=True)]
    
    print(f'Subtracting master dark from {len(list_bflat)} flat frames (exptime={exptime:3.1f}s)...')
    for fpath_bflat in list_bflat:
    
        fpath_bdflat = TMPDIR / (fpath_bflat.stem+'d'+fpath_bflat.suffix)
        
        bflat       = CCDData.read(fpath_bflat)
        bdflat      = ccdp.subtract_dark(bflat, mdark, exposure_time='exptime', exposure_unit=u.second)
        
        bdflat.write(fpath_bdflat)
        print(f"{fpath_bdflat} has been saved ({datetime.now()}).")
        fpath_bflat.unlink() # to save the disk volume
        list_bdflat.append(fpath_bdflat)

# combine master flat
print(f'Combining master flat ({len(list_bdflat)} frames)...')

# # average-combine (recommneded for fast-calculation)
# mflat = ccdp.combine(list_bdflat,
#                      method='average',
#                      scale=funcs.inv_median,
#                     #  sigma_clip=True,
#                     #  sigma_clip_low_thresh=5,
#                     #  sigma_clip_high_thresh=5,
#                     #  sigma_clip_func=np.ma.median,
#                     #  sigma_clip_dev_func=mad_std,
#                      mem_limit=350e6,
#                      dtype=np.float32
#                      )    

# median combine (recommended for science frames)
mflat = ccdp.combine(list_bdflat,
                     method='median',
                     scale=fileutil.inv_median,
                     sigma_clip=True,
                     sigma_clip_low_thresh=5,
                     sigma_clip_high_thresh=5,
                     sigma_clip_func=np.ma.median,
                     sigma_clip_dev_func=mad_std,
                     mem_limit=350e6,
                     dtype=np.float32
                     )

mflat.meta['combined'] = True
mflat.write(fpath_mflat, overwrite=True)
print(f"{fpath_mflat} has been saved ({datetime.now()}).")

# delete temporary directory
fileutil.clear_dir(TMPDIR)
TMPDIR.rmdir()
print(f"{TMPDIR} has been deleted ({datetime.now()}).")