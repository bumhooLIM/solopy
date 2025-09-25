from astropy.nddata import CCDData
from astropy.wcs import FITSFixedWarning
from astropy.stats import mad_std
import astroalign as aa
import ccdproc as ccdp

import numpy as np
import re
import funcs
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", category=FITSFixedWarning)

target_name    = 'Focus_17000'                                                 # target name
filter_name    = 'r'                                                         # filter name
start_fnum     = 1                                                           # frame number (start)
ref_fnum       = 5                                                          # frame number (reference alignment)
end_fnum       = 10                                                          # frame number (end)
fname_pattern  = re.compile(target_name+r'-(\d{3})_'+filter_name+'_bdf.fit') # filename patterns to extract

path_workdir   = Path('./__data__/20241112_Seoul/')                          # working directory
path_caldata   = path_workdir/'02_calibration'                               # file path for calibrated data
path_mergedata = path_workdir/'03_merged'                                    # file path to save merged data
path_tmpdata   = path_workdir/'tmp'                                          # temporary directory

for fpath in [path_mergedata, path_tmpdata]:
    try:
        fpath.mkdir(exist_ok=False)
    except FileExistsError: # directory already exists. 
        pass

funcs.clear_dir(path_tmpdata)

# extract data (start_fnum - end_fnum)
list_obj, path_refobj = funcs.FileCollection(path_caldata, fname_pattern, start_fnum=start_fnum, end_fnum=end_fnum, ref_fnum=ref_fnum)
list_obj.sort()

# reference data for alignment
ref = CCDData.read(path_refobj, format='fits')
if ref.data.dtype.byteorder == '>':  # Check if data is big-endian and convert to little-endian
    ref.data = ref.data.astype(np.float32)

#
# data alignment
#

list_obj_aligned = []
print(f"""Align {len(list_obj)} science frames...
Reference frame: {path_refobj.name}""")
for path_obj in list_obj:

    obj = CCDData.read(path_obj)
    if obj.data.dtype.byteorder == '>': 
        obj.data = obj.data.astype(np.float32) 
        
    data_aligned, footprint = aa.register(source=obj, target=ref)
    
    obj.data = data_aligned

    path_obj_aligned = path_tmpdata / (path_obj.stem+'a'+path_obj.suffix)
    obj.write(path_obj_aligned, overwrite=True)
    
    list_obj_aligned.append(path_obj_aligned)
    
    print(f"{path_obj_aligned} has been saved ({datetime.now()}).")

#
# data merge
#

list_obj_aligned.sort()

print(f'Merging {len(list_obj_aligned)} science frames ({list_obj_aligned[0].name} - {list_obj_aligned[-1].name})...')

obj_merged = ccdp.combine(list_obj_aligned,
                          method='median',
                          sigma_clip=True,
                          sigma_clip_low_thresh=5,
                          sigma_clip_high_thresh=3,
                          sigma_clip_func=np.ma.median,
                          sigma_clip_dev_func=mad_std,
                          mem_limit=350e6,
                          dtype=np.float32
                          )

# path_obj_merged = re.sub(r'(\d{3})', f'{start_fnum:03d}-{end_fnum:03d}', path_refobj.stem+'_merged'+path_refobj.suffix)
path_obj_merged = f"{target_name}-{start_fnum:03d}-{end_fnum:03d}_{filter_name}_merged.fits"
obj_merged.meta['combined'] = True
obj_merged.write(path_mergedata/path_obj_merged, overwrite=True)
print(f'{path_obj_merged} has been saved ({datetime.now()})')

funcs.clear_dir(path_tmpdata)
path_tmpdata.rmdir()
print(f"{path_tmpdata} has been deleted ({datetime.now()}).")
