#!/bin/bash
# Batch driver: runs main.py for every night listed below, one after another.
#
#   ./run_solopy.sh                       # all levels (0,1,2,3)
#   LEVELS=2,3 ./run_solopy.sh            # re-run zero points and asteroid photometry only
#
# - Uses the `solopy` conda env explicitly (a bare `python` in a non-activated shell is the
#   miniconda base env, which lacks kete/skyloc). Override with SOLOPY_PYTHON=/path/to/python.
# - Each night's stdout/stderr goes to ../log/run_<night>.out, so tracebacks are kept.
# - A failing night does not stop the batch.

PYTHON="${SOLOPY_PYTHON:-$HOME/miniconda3/envs/solopy/bin/python}"
LEVELS="${LEVELS:-0,1,2,3}"
cd "$(dirname "$0")" || exit 1
mkdir -p ../log

# Array of observation dates to process
SUBDIRS=(
    "2026_0522" "2026_0523" "2026_0524" "2026_0525" "2026_0526"
    "2026_0531" "2026_0601" "2026_0602" "2026_0603" "2026_0604"
    "2026_0605" "2026_0606" "2026_0607" "2026_0610" "2026_0611"
    "2026_0613" "2026_0614" "2026_0615" "2026_0616" "2026_0617"
    "2026_0618" "2026_0619" "2026_0620" "2026_0621" "2026_0622"
    "2026_0623" "2026_0624" "2026_0625" "2026_0626" "2026_0627"
    "2026_0628" "2026_0629" "2026_0630"
)

echo "Starting SOLO batch pipeline (levels ${LEVELS}) with ${PYTHON}"
for dir in "${SUBDIRS[@]}"; do
    echo "=== ${dir}  $(date '+%Y-%m-%d %H:%M:%S')"
    "$PYTHON" main.py -s "$dir" --levels "$LEVELS" > "../log/run_${dir}.out" 2>&1
    echo "    exit $?  $(date '+%H:%M:%S')"
done
echo "Batch processing complete!"
