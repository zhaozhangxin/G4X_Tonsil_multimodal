"""
03_convert_to_geojson.py
========================
Convert the integer segmentation mask to a GeoJSON FeatureCollection for
interactive visualization and quality control in QuPath.

Each cell becomes a Polygon feature carrying its label, classification ("Cell"),
and contour coordinates extracted from the bounding-box-local region of the mask.

Algorithm
---------
1. A single O(W*H) pass with skimage.measure.regionprops obtains every cell's
   bounding box (no repeated full-image scans).
2. Per-cell contour extraction (find_contours) operates only within the
   bounding box, keeping memory usage low and avoiding redundant computation.

This single-threaded approach is faster and more memory-efficient than
multiprocessing alternatives (e.g. mask_to_geojson_joblib) for large masks.

Inputs
------
- data/processed/<SEGMENTATION_TAG>/segmentation_mask.tiff

Outputs
-------
- data/processed/<SEGMENTATION_TAG>/segmentation.geojson
"""

import json
from pathlib import Path

import numpy as np
import tifffile
from skimage import measure

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_ROOT = Path(__file__).parent.parent / "data"

SEGMENTATION_TAG = "260316_g4xtonsil_cell"

MASK_PATH = (
    DATA_ROOT / "processed" / SEGMENTATION_TAG / "segmentation_mask.tiff"
)
OUTPUT_PATH = (
    DATA_ROOT / "processed" / SEGMENTATION_TAG / "segmentation.geojson"
)
# ──────────────────────────────────────────────────────────────────────────────

print("Loading mask...")
mask = tifffile.imread(MASK_PATH)
print(f"Shape: {mask.shape}, dtype: {mask.dtype}")

print("Extracting contours via regionprops (single-pass)...")
props = measure.regionprops(mask)
print(f"Total cells: {len(props)}")

features = []
for prop in props:
    label = prop.label
    # Work only within the bounding box of each cell
    min_row, min_col, max_row, max_col = prop.bbox
    # Pad by 1 pixel to ensure closed contours at image borders
    r0 = max(min_row - 1, 0)
    c0 = max(min_col - 1, 0)
    r1 = min(max_row + 1, mask.shape[0])
    c1 = min(max_col + 1, mask.shape[1])

    local_mask = (mask[r0:r1, c0:c1] == label).astype(np.uint8)
    contours = measure.find_contours(local_mask, 0.5)
    if not contours:
        continue

    # Use the longest contour
    contour = max(contours, key=len)
    # Convert local coordinates back to global coordinates.
    # find_contours returns (row, col); QuPath expects (x=col, y=row).
    coords = [(c0 + c, r0 + r) for r, c in contour]
    if len(coords) < 3:
        continue
    # Close the polygon
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    features.append({
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [coords],
        },
        "properties": {
            "objectType": "detection",
            "name": str(label),
            "classification": {
                "name": "Cell",
                "color": [0, 255, 0],
            },
        },
    })

print(f"Writing {len(features)} cells to GeoJSON...")
geojson = {"type": "FeatureCollection", "features": features}
with open(OUTPUT_PATH, "w") as f:
    json.dump(geojson, f)

print(f"Done: {OUTPUT_PATH}")
