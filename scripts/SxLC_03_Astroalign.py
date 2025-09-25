from astropy.nddata import CCDData, Cutout2D
from astropy.wcs import FITSFixedWarning
from astropy.stats import mad_std
import astroalign as aa
import ccdproc as ccdp
import _ccdutil as ccdutil

import numpy as np
import re
import _fileutil as fileutil
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", category=FITSFixedWarning)

target_name    = '755Quintilla'                                              # target name
filter_name    = 'r'                                                         # filter name
start_fnum     = 1                                                           # frame number (start)
ref_fnum       = 1                                                          # frame number (reference alignment)
end_fnum       = 75                                                         # frame number (end)
fname_pattern  = re.compile(target_name+r'-(\d{3})_'+filter_name+'_p.fit') # filename patterns to extract

WORKDIR        = Path('./20241218_Seoul/')                          # working directory
CALDIR         = WORKDIR/'lv01_preprocess'                                   # file path for calibrated data
OUTDIR         = WORKDIR/'lv02_astroalign'                                   # file path to save merged data

for fpath in [OUTDIR]:
    try:
        fpath.mkdir(exist_ok=False)
    except FileExistsError: # directory already exists. 
        pass

# extract data (start_fnum - end_fnum)
list_obj, fpath_refobj = fileutil.FileCollection(CALDIR, fname_pattern, start_fnum=start_fnum, end_fnum=end_fnum, ref_fnum=ref_fnum)
list_obj.sort()

# reference data for alignment
ref = CCDData.read(fpath_refobj, format='fits')
if ref.data.dtype.byteorder == '>':  # Check if data is big-endian and convert to little-endian
    ref.data = ref.data.astype(np.float32)

#
# data alignment
#

# list_obj_aligned = []
print(f"""Align {len(list_obj)} science frames...
Reference frame: {fpath_refobj.name}""")
for fpath_obj in list_obj:

    obj = CCDData.read(fpath_obj)
    if obj.data.dtype.byteorder == '>': 
        obj.data = obj.data.astype(np.float32) 
        
    data_aligned, footprint = aa.register(source=obj, target=ref)
    obj.data = data_aligned

    obj_cutout = ccdutil.CreateCutoutCCD(obj, position=(2545, 1355), size=500)

    fpath_obj_aligned = OUTDIR / (fpath_obj.stem+'aa'+fpath_obj.suffix)
    obj_cutout.write(fpath_obj_aligned, overwrite=True)
    
    # list_obj_aligned.append(fpath_obj_aligned)
    
    print(f"{fpath_obj_aligned} has been saved ({datetime.now()}).")
