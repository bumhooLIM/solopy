import solopy
from ccdproc import ImageFileCollection
from pathlib import Path

# Observation date (=subdirectory name)
subdir = "2025_0717" 
detector_name = "kl4040"

# ============= 
# Directory setup 
# =============

# Working directory
WORKDIR = Path.home() / "Desktop" / "solo-data"

# Lv0 data directory
LV0DIR = WORKDIR/"Lv0"
LV0_subdir = LV0DIR / subdir

# Lv1 data directory
LV1DIR = WORKDIR/"Lv1"
LV1_subdir = LV1DIR / subdir

# Master calibration files directory
MASTERDIR = WORKDIR / "calibration_files"

# Log directory
LOGDIR = WORKDIR/"log"
LOGDIR.mkdir(parents=True, exist_ok=True)
fpath_log = LOGDIR/f'solopy_{subdir}.log' # general log file path
if fpath_log.exists():
    fpath_log.unlink()

# =============
# Fits Lv0 header update
# =============

allfits = ImageFileCollection(LV0_subdir, glob_include="*.fits")

lv0 = solopy.FitsLv0(log_file=fpath_log)

for fpath_fits in allfits.files_filtered(include_path=True):
    lv0.update_header(fpath_fits)

allfits.refresh()  # Refresh the file collection to reflect updated headers

# =============
# Master calibration frames
# =============

bias_frame = allfits.files_filtered(imagetyp="BIAS", include_path=True)
dark_frame = allfits.files_filtered(imagetyp="DARK", include_path=True)

print(f"Number of bias frames: {len(bias_frame)}")
print(f"Number of dark frames: {len(dark_frame)}")

comb = solopy.CombMaster(log_file=fpath_log)

# Master bias
comb.comb_master_bias(bias_frame, MASTERDIR, outname=detector_name)

# Master dark
comb.comb_master_dark(dark_frame, MASTERDIR, outname=detector_name)

# =============
# Lv1 WCS solution
# =============