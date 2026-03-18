"""
01_visualize_transcripts_on_he.py
==================================
Overlay RNA transcript coordinates on the H&E image to visually verify
that the RNA signal is spatially aligned with tissue morphology.

Inputs
------
- data/raw/h_and_e/h_and_e.jp2          : Whole-tissue H&E image (JP2 format)
- data/raw/rna/transcript_table.csv.gz  : Transcript table with pixel coordinates

Outputs
-------
- results/transcript_on_he.png          : Side-by-side comparison figure
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_ROOT   = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

HE_PATH         = DATA_ROOT / "raw" / "h_and_e" / "h_and_e.jp2"
TRANSCRIPT_PATH = DATA_ROOT / "raw" / "rna" / "transcript_table.csv.gz"
OUTPUT_PATH     = RESULTS_DIR / "transcript_on_he.png"

THUMBNAIL_SCALE = 1 / 10   # Downsample ratio for visualization
SCATTER_SIZE    = 0.05      # Scatter point size
SCATTER_ALPHA   = 0.15      # Scatter point transparency
# ──────────────────────────────────────────────────────────────────────────────

Image.MAX_IMAGE_PIXELS = None   # Allow reading very large images

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load transcript coordinates (only x/y pixel columns needed)
print("Loading transcript table...")
df = pd.read_csv(TRANSCRIPT_PATH, usecols=["x_pixel_coordinate", "y_pixel_coordinate"])
print(f"  Total transcripts: {len(df):,}")

# Load H&E image and create thumbnail for visualization
print("Loading H&E image...")
img = Image.open(HE_PATH)
full_w, full_h = img.size
print(f"  Full image size: {full_w} x {full_h} px")

thumb_w = full_w // int(1 / THUMBNAIL_SCALE)
thumb_h = full_h // int(1 / THUMBNAIL_SCALE)
thumb = img.resize((thumb_w, thumb_h), Image.LANCZOS)

# Plot: left = H&E only, right = H&E + transcripts
fig, axes = plt.subplots(1, 2, figsize=(28, 14))

axes[0].imshow(np.array(thumb))
axes[0].set_title("H&E", fontsize=14)
axes[0].axis("off")

axes[1].imshow(np.array(thumb))
axes[1].scatter(
    df["x_pixel_coordinate"].values * THUMBNAIL_SCALE,
    df["y_pixel_coordinate"].values * THUMBNAIL_SCALE,
    s=SCATTER_SIZE,
    alpha=SCATTER_ALPHA,
    c="red",
    linewidths=0,
)
axes[1].set_title("Transcripts on H&E", fontsize=14)
axes[1].axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved: {OUTPUT_PATH}")
