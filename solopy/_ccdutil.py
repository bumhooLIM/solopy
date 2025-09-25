import ccdproc as ccdp
from ccdproc import CCDData
from astropy.nddata import Cutout2D
from datetime import datetime
import numpy as np

def CreateCutoutCCD(ccd, position, size):
    """
    Creates a cutout from the provided CCDData object, including updated WCS information.
    
    Parameters:
        ccd (CCDData): CCDData object containing image data, header, and optional WCS.
        position (tuple): Center (x, y) of the cutout.
        size (tuple): Size (height, width) of the cutout.
    
    Returns:
        CCDData: A new CCDData object containing the cutout, updated header, and WCS.
    """
    ccd = ccd.copy()
    
    # Create the cutout with WCS adjustment
    cutout = Cutout2D(ccd.data, position=position, size=size, wcs=ccd.wcs if ccd.wcs is not None else None)
    
    # Handle mask cutout
    if ccd.mask is not None:
        mask_cutout = Cutout2D(ccd.mask, position=position, size=size).data
    else:
        mask_cutout = None
    
    # Modify header
    header = ccd.header.copy()
    
    try:
        header['NAXIS1'], header['NAXIS2'] = size
    except TypeError:
        header['NAXIS1'] = size
        header['NAXIS2'] = size
    
    header.add_history(f"Cutout from original size={ccd.data.shape} ({datetime.utcnow().isoformat()}).")
    header.add_history(f"Cutout center position (x, y): {position}")
    header.add_history(f"Cutout size (height, width): {size}")
    
    # Create new CCDData object
    cutout_ccd = CCDData(cutout.data, header=header, mask=mask_cutout, unit=ccd.unit, wcs=cutout.wcs)
    
    return cutout_ccd

def CCDBadPixel(list_fpath_ccd):
    
    list_fpath_ccd.sort()
    ccd_01 = CCDData.read(list_fpath_ccd[0])
    ccd_02 = CCDData.read(list_fpath_ccd[-1])
    ratio  = ccd_02.divide(ccd_01)
    mask_badpixel = ccdp.ccdmask(ratio)
    
    return mask_badpixel

def CCDTranspose(ccd):
    '''
    transpose x and y axis of ccd (astropy.nddata.ccddata.CCDData).
    '''
    
    ccd_transpose = ccd.copy()
    
    ccd_transpose.data  = np.transpose(ccd.data)
    
    if ccd.mask is not None:
        ccd_transpose.mask  = np.transpose(ccd.mask)    
        
    if ccd.flags is not None:
        ccd_transpose.flags = np.transpose(ccd.flags)
    
    ccd_transpose.header.add_history(f'x and y axes transposed ({datetime.utcnow().isoformat()}).')
    
    ccd_transpose.header['naxis1'] = ccd_transpose.data.shape[1] # x-axis
    ccd_transpose.header['naxis2'] = ccd_transpose.data.shape[0] # y-axis

    return ccd_transpose

def CCDFlipXY(ccd):
    '''
    Flip CCDData (astropy.nddata.ccddata.CCDData) along both x and y axes (180-degree rotation).
    
    Parameters:
        ccd (CCDData): CCDData object to be flipped.
    
    Returns:
        CCDData: A new CCDData object with the data flipped.
    '''
    
    ccd_flipped = ccd.copy()

    ccd_flipped.data = np.flip(ccd.data, axis=(0, 1)) # Flip the data along both axes
    
    # Flip the mask, if present
    if ccd.mask is not None:
        ccd_flipped.mask = np.flip(ccd.mask, axis=(0, 1))
        
    # Flip the flags, if present
    if ccd.flags is not None:
        ccd_flipped.flags = np.flip(ccd.flags, axis=(0, 1))
    
    # Update the header with a history entry
    ccd_flipped.header.add_history(f'x and y axes flipped (180-degree rotation) ({datetime.utcnow().isoformat()}).')
    
    # Update the header with the flipped dimensions
    ccd_flipped.header['NAXIS1'] = ccd_flipped.data.shape[1]  # x-axis size
    ccd_flipped.header['NAXIS2'] = ccd_flipped.data.shape[0]  # y-axis size
    
    return ccd_flipped

def Normalize_CCDData(ccd, scale_func=(lambda x: 1/np.nanmax(x))):
    
    ccd.data   = ccd.data * scale_func(ccd.data)
    ccd.header.add_history(f'Normalized by {scale_func}. ({datetime.utcnow().isoformat()})')
    
    return ccd