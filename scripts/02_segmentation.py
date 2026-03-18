"""
02_segmentation.py
==================
Whole-cell segmentation of the G4X protein OME-TIFF image using the Mesmer
deep-learning model.

The image (19200x15232 px) is too large for a single GPU pass, so inference
is performed on overlapping tiles (tile_size=2000 px, overlap=200 px) that
are stitched into a single label mask.

Inputs
------
- data/raw/ometiff/G04.ome.tiff          : Multi-channel protein OME-TIFF

Outputs
-------
- data/processed/<SEGMENTATION_TAG>/segmentation_mask.tiff
    Integer label mask (each cell has a unique positive integer; background=0)
- data/processed/<SEGMENTATION_TAG>/dataScaleSize.csv
    Per-cell mean marker intensity normalized by cell area
- data/processed/<SEGMENTATION_TAG>/segmentation_markers.ome.tiff
    Preprocessed internal/boundary channels used for segmentation
- data/processed/<SEGMENTATION_TAG>/parameter_segmentation.json
    Serialized segmentation parameters
- data/processed/<SEGMENTATION_TAG>/segmentation.log
    Segmentation log file
"""

import sys
import warnings
from pathlib import Path

from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

# Allow scripts to find the utils/ package
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from io_setup import setup_gpu
from segmentation import (
    run_segmentation_mesmer_cell,
    run_segmentation_mesmer_compartments,
)
from segmentation_mask import (
    create_rgb_segmentation_mask,
    find_segmentation_boundaries,
)

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_ROOT = Path(__file__).parent.parent / "data"

# Segmentation tag — used as the output sub-directory name
SEGMENTATION_TAG = "260316_g4xtonsil_cell"

# Directory containing the input OME-TIFF file
unit_dir = DATA_ROOT / "raw" / "ometiff"

# Output goes here: data/processed/<SEGMENTATION_TAG>/
# (created automatically by run_segmentation_mesmer_cell)

# Mesmer parameters
internal_markers = ["nuclear"]
boundary_markers = ["CD20", "PanCK", "CD45RA", "CD3"]
thresh_q_min = 0
thresh_q_max = 0.99
thresh_otsu = False
scale = True
pixel_size_um = 0.3125
maxima_threshold = 0.075
interior_threshold = 0.20
# ──────────────────────────────────────────────────────────────────────────────

# Configure GPU (set to "0" to use the first GPU, or "" for CPU)
setup_gpu("2")

# Run whole-cell segmentation; results are written to unit_dir / SEGMENTATION_TAG
run_segmentation_mesmer_cell(
    unit_dir=unit_dir,
    internal_markers=internal_markers,
    boundary_markers=boundary_markers,
    thresh_q_min=thresh_q_min,
    thresh_q_max=thresh_q_max,
    thresh_otsu=thresh_otsu,
    scale=scale,
    pixel_size_um=pixel_size_um,
    maxima_threshold=maxima_threshold,
    interior_threshold=interior_threshold,
    tag=SEGMENTATION_TAG,
)
