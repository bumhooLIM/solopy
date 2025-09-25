from astropy.io import fits
from astropy.wcs import FITSFixedWarning
from astropy.nddata import CCDData
import astropy.units as u
import ccdproc as ccdp

import numpy as np
import warnings
from datetime import datetime
from pathlib import Path
import _fileutil as fileutil

warnings.filterwarnings("ignore", category=FITSFixedWarning)

filter_name    = 'r'                                                                  # filter name
RAWDIR         = Path.home()/'__MYresearch__/__rawdata__/SPHERExLC/20241218_Seoul_r/' # file path to raw data
WORKDIR        = Path('./20241218_Seoul/')                                            # working directory
MASTERDIR      = WORKDIR/'lv00_masterframe'                                           # file path for Master bias, dark, and flat
CALDIR         = WORKDIR/'lv01_preprocess'                                            # file path to save calibrated data                                               # temporary directory (removed after end of process)
fpath_mbias    = MASTERDIR/'Mbias.fit'                                               # file path to save master bias
fpath_mflat    = MASTERDIR/f'Mflat_{filter_name}.fit'                                # file path to save master flat

try:
    CALDIR.mkdir(exist_ok=False)
except FileExistsError: # directory already exists. 
    pass

ALLFITS        = ccdp.ImageFileCollection(RAWDIR)                                                       # all fits file in raw data
list_obj       = [Path(fpath) for fpath in ALLFITS.files_filtered(imagetyp='Light Frame', include_path=True)] # science target

# load master bias
fileutil.check_file_exsist(fpath_mbias)
mbias = CCDData.read(fpath_mbias)

# load master flat
fileutil.check_file_exsist(fpath_mflat)
mflat = CCDData.read(fpath_mflat)

# list_obj_calibrated = [] 
# subtract master bias and dark, correct flat
print(f'Bias, dark, flat correction for {len(list_obj)} science frames...')
for path_obj in list_obj:
    
    # file path to save preprocessed data
    fpath_pobj = CALDIR / (path_obj.stem+'_p'+path_obj.suffix)
    
    obj         = CCDData.read(path_obj, unit='adu')
    
    # load master dark (corresponding to exposure time of science frame)
    exptime     = fits.getheader(path_obj)['exptime']    
    fpath_mdark  = MASTERDIR/f'Mdark_{exptime:3.1f}s.fit'
    # funcs.check_file_exsist(fpath_mdark)  
    try:
        mdark       = CCDData.read(fpath_mdark)
    # if corresponding master dark does not exsist, find alternatives with the cloesest expsoure time.   
    except FileNotFoundError: 
        allmdark        = ccdp.ImageFileCollection(MASTERDIR).filter(imagetyp='Dark Frame')
        exptime_closest = min(set(allmdark.summary['exptime']),key=lambda x:abs(x-exptime))
        fpath_mdark     = MASTERDIR/f'Mdark_{exptime_closest:3.1f}s.fit'
        mdark           = CCDData.read(fpath_mdark)   
        print(f'Warning: Master dark frame (exptime={exptime:3.1f}s) does not exist. {fpath_mdark} was alternatively used.')
            
    # subtract bias
    bobj        = ccdp.subtract_bias(obj , mbias)    
    
    # subtract dark                                               
    bdobj       = ccdp.subtract_dark(bobj, mdark, exposure_time='exptime', exposure_unit=u.second)
    
    # flat correection
    pobj      = ccdp.flat_correct(bdobj, mflat)
    
    # correct data dtype (to save the disk space)
    pobj.data = pobj.data.astype(np.uint16)
    
    pobj.write(fpath_pobj, overwrite=True)
    print(f"{fpath_pobj} has been saved ({datetime.now()}).")
    # list_obj_calibrated.append(fpath_pobj)