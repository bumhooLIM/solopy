from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple
import numpy as np
import re

def check_file_exsist(fpath):
    '''
    Check if the fpath exists.
    '''
    
    if not fpath.exists():
        print(f"Warning: {fpath} does not exsist.")
        
def clear_dir(fpath):
    '''
    Clear all the files inside the fpath. `fpath` should be directory.
    '''
    
    directory = Path(fpath)

    if directory.exists() and directory.is_dir():
        for item in directory.rglob('*'):
            item.unlink() if item.is_file() else item.rmdir()
            
        print(f"{directory} has been cleared out. ({datetime.now()})")
    
    else:
        print(f"{directory} does not exist.")


def inv_median(a):
    return 1 / np.median(a)

# def extract_files(dir: Path, pattern: re.Pattern, start_fnum: int, end_fnum: int, ref_fnum: int):
#     """
#     Extract files from a directory that match a specified filename pattern and 
#     have a numeric part within a specified range.

#     Parameters:
#     -----------
#     dir : Path
#         The directory path where the files are located.
        
#     pattern : re.Pattern
#         A compiled regular expression pattern used to match file names. The 
#         pattern should contain a capturing group for the numeric part of the filename.
        
#     start_fnum : int
#         The start of the range for the numeric part of the filename (inclusive).
        
#     end_fnum : int
#         The end of the range for the numeric part of the filename (inclusive).

#     ref_fnum : int
#         The reference frame number for the numeric part of the filename. Should be 
#         located between (start_fnum, end_fnum).

#     Returns:
#     --------
#     List[Path], Path
#         A list of `Path` objects representing the matching files with filenames
#         that contain a numeric part within the specified range, and the path to
#         the reference file (for alignment) if found.
#     """
    
#     matching_files = []
#     fpath_refobj   = None 
    
#     for fpath in dir.glob('*'):
#         match = pattern.match(fpath.name)
#         if match:
#             number = int(match.group(1))  # Extract the number part from the filename
#             # Check if the number is between start_fnum and end_fnum
#             if start_fnum <= number <= end_fnum:
#                 matching_files.append(fpath)
#             if number == ref_fnum:
#                 fpath_refobj = fpath
    
#     # Check if reference object was found
#     if fpath_refobj is None:
#         raise ValueError(f"Reference file with number {ref_fnum} not found.")
    
#     return matching_files, fpath_refobj


def FileCollection(
    dir: Path, 
    pattern: re.Pattern, 
    start_fnum: int, 
    end_fnum: int, 
    ref_fnum: Optional[int] = None
) -> Tuple[List[Path], Optional[Path]]:
    """
    Extract files from a directory that match a specified filename pattern and 
    have a numeric part within a specified range. Optionally, find a reference file.

    Parameters:
    -----------
    dir : Path
        The directory path where the files are located.
        
    pattern : re.Pattern
        A compiled regular expression pattern used to match file names. The 
        pattern should contain a capturing group for the numeric part of the filename.
        
    start_fnum : int
        The start of the range for the numeric part of the filename (inclusive).
        
    end_fnum : int
        The end of the range for the numeric part of the filename (inclusive).

    ref_fnum : Optional[int]
        The reference frame number for the numeric part of the filename. If `None`, 
        no reference file is required, and only matching files are collected.

    Returns:
    --------
    Tuple[List[Path], Optional[Path]]
        A tuple containing:
        - A list of `Path` objects representing the matching files with filenames
          that contain a numeric part within the specified range.
        - The path to the reference file (if specified and found), or `None` if
          `ref_fnum` is `None`.
    """
    matching_files = []
    fpath_refobj = None 

    for fpath in dir.glob('*'):
        match = pattern.match(fpath.name)
        if match:
            number = int(match.group(1))  # Extract the number part from the filename
            # Check if the number is between start_fnum and end_fnum
            if start_fnum <= number <= end_fnum:
                matching_files.append(fpath)
            # Check if this is the reference file
            if ref_fnum is not None and number == ref_fnum:
                fpath_refobj = fpath
    
    # Raise an error if a reference number was specified but not found
    if ref_fnum is not None and fpath_refobj is None:
        raise ValueError(f"Reference file with number {ref_fnum} not found.")
    
    return matching_files, fpath_refobj

