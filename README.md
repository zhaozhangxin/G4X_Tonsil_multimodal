# G4X Spatial Transcriptomics Preprocessing Pipeline

A preprocessing pipeline for **Singular Genomics G4X** platform data,
demonstrated on a human tonsil dataset (Rep 1). The pipeline takes raw H&E,
RNA transcript, and multiplexed protein images as inputs and produces a
ready-to-analyze multimodal single-cell object (MuData) combining 348 RNA
genes and 16 protein markers across 230,446 segmented cells.

---

## Pipeline Overview

```
Raw data
   │
   ├── H&E image (h_and_e.jp2)
   ├── RNA transcripts (transcript_table.csv.gz)
   └── Protein OME-TIFF (G04.ome.tiff)
   │
   v
Step 01 — Visualize Transcripts on H&E
   Overlay transcript pixel coordinates on the H&E thumbnail to verify
   spatial alignment of RNA signal with tissue morphology.
   Output: results/transcript_on_he.png
   │
   v
Step 02 — Whole-Cell Segmentation (Mesmer)
   Tiled Mesmer inference (tile=2000 px, overlap=200 px) on the 19200x15232 px
   protein image. Produces integer label mask + per-cell protein feature table.
   Output: data/processed/260316_g4xtonsil_cell/segmentation_mask.tiff
           data/processed/260316_g4xtonsil_cell/dataScaleSize.csv
   │
   v
Step 03 — Convert Mask to GeoJSON
   Efficient single-pass regionprops + bbox-local contour extraction.
   Each of 230,446 cells becomes a GeoJSON Polygon importable into QuPath.
   Output: data/processed/260316_g4xtonsil_cell/segmentation.geojson
   │
   v
Step 04 — Build Cell-Gene Matrix
   Assign each transcript to a cell via mask lookup. Build sparse RNA count
   matrix (zero-pad cells with no transcripts). Merge with protein AnnData
   into a MuData object.
   Output: results/tonsil_rep1_rna.h5ad
           results/tonsil_rep1_protein.h5ad
           results/tonsil_rep1.h5mu
```

---

## Repository Structure

```
G4X_prepro/
├── scripts/
│   ├── 01_visualize_transcripts_on_he.py
│   ├── 02_segmentation.py
│   ├── 03_convert_to_geojson.py
│   └── 04_build_cell_gene_matrix.py
├── utils/
│   ├── __init__.py
│   ├── io_setup.py            # GPU setup, logging, metadata I/O
│   ├── segmentation.py        # Mesmer segmentation utilities
│   └── segmentation_mask.py   # Mask visualization utilities
├── data/
│   └── README.md              # Data download and setup instructions
├── results/                   # Pipeline outputs (gitignored for large files)
├── requirements.txt
├── environment.yml
├── .gitignore
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/G4X_prepro.git
cd G4X_prepro
```

### 2. Create and activate the conda environment

```bash
conda env create -f environment.yml
conda activate g4x_prepro
```

> **Note:** `deepcell` requires TensorFlow. The `environment.yml` installs
> TensorFlow first so that deepcell can resolve a compatible version.
> GPU support requires a CUDA-compatible driver; see the
> [TensorFlow GPU guide](https://www.tensorflow.org/install/pip) for details.

---

## Data

See [`data/README.md`](data/README.md) for full instructions on:
- Where to download the Singular Genomics G4X tonsil dataset
- Required directory layout under `data/`
- File names, sizes, and descriptions

---

## Usage

Run the scripts in order from the repository root. Each script reads its inputs
from `data/` and writes outputs to `data/processed/` or `results/`.

### Step 1 — Visualize transcripts on H&E

```bash
python scripts/01_visualize_transcripts_on_he.py
```

### Step 2 — Whole-cell segmentation

```bash
python scripts/02_segmentation.py
```

Edit the GPU ID at the top of the script (`setup_gpu("0")`) to select the
appropriate device.

### Step 3 — Convert segmentation mask to GeoJSON

```bash
python scripts/03_convert_to_geojson.py
```

### Step 4 — Build cell-gene matrix

```bash
python scripts/04_build_cell_gene_matrix.py
```

---

## Outputs

| File | Description |
|---|---|
| `results/transcript_on_he.png` | Side-by-side H&E / H&E + transcripts figure |
| `data/processed/.../segmentation_mask.tiff` | Integer label mask (background=0, cells=1..N) |
| `data/processed/.../dataScaleSize.csv` | Per-cell protein intensities normalized by cell area |
| `data/processed/.../segmentation.geojson` | GeoJSON polygons for QuPath |
| `results/tonsil_rep1_rna.h5ad` | RNA AnnData (230,446 cells x 348 genes) |
| `results/tonsil_rep1_protein.h5ad` | Protein AnnData (230,446 cells x 16 markers) |
| `results/tonsil_rep1.h5mu` | MuData combining RNA and protein modalities |

---

## Dependencies

Core dependencies and their roles:

| Package | Role |
|---|---|
| `tensorflow` + `deepcell` | Mesmer deep-learning cell segmentation |
| `tifffile` | Read/write TIFF and OME-TIFF images |
| `scikit-image` | `regionprops`, `find_contours`, `find_boundaries` |
| `anndata` / `mudata` | Single-cell data containers |
| `scipy` | Sparse matrix construction |
| `pandas` / `numpy` | Data wrangling |
| `pyqupath` | OME-TIFF pyramid I/O and GeoJSON utilities |
| `pycodex` | CODEX/multiplex imaging utilities (logging, setup) |
| `opencv-python` | Otsu thresholding in marker preprocessing |
| `Pillow` | JPEG 2000 H&E image loading |
| `matplotlib` | Visualization |

---

## Key Technical Highlights

### Tiled Mesmer Inference for Large Images

The raw protein image is 19,200 x 15,232 pixels — far too large for a single
GPU pass. `_run_mesmer_tiled` in `utils/segmentation.py` splits the image into
2,000 px tiles with 200 px overlap:

- Only the inner (non-overlapping) portion of each tile is used, eliminating
  boundary artifacts.
- Non-zero labels in each tile are globally renumbered via `label_offset`,
  guaranteeing unique cell IDs across all tiles.
- The stitched mask is written as a single TIFF.

This design enables segmentation of arbitrarily large images without exceeding
GPU memory.

### Efficient GeoJSON Export

`scripts/03_convert_to_geojson.py` uses a two-step strategy instead of
multiprocessing:

1. A single O(W*H) pass with `skimage.measure.regionprops` collects every
   cell's bounding box.
2. `find_contours` runs only within each cell's bounding box, avoiding repeated
   full-image scans.

This single-threaded approach avoids joblib process-pool overhead and is faster
and more memory-efficient on large masks.

### MuData Integration of RNA and Protein Modalities

`scripts/04_build_cell_gene_matrix.py` builds a fully aligned multimodal object:

- RNA: sparse count matrix built by mask-lookup of transcript pixel coordinates.
  Cells present in the mask but with zero transcripts are zero-padded so that
  the RNA cell count always matches the segmentation.
- Protein: per-cell mean intensities from `dataScaleSize.csv`, including spatial
  metadata (`Y_cent`, `X_cent`, `cellSize`).
- Both modalities are intersected on cell ID and saved as a `MuData` object.

---

## Dataset Statistics (Tonsil Rep 1)

| Statistic | Value |
|---|---|
| Segmented cells | 230,446 |
| RNA genes | 348 |
| Total transcripts | 22,148,004 |
| Transcripts assigned to cells | 18,262,215 |
| Protein markers | 16 |
| Image size | 19,200 x 15,232 px |
| Pixel size | 0.3125 um/px |
