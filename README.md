# G4X Spatial Transcriptomics Preprocessing Pipeline

A preprocessing pipeline for **Singular Genomics G4X** platform data,
demonstrated on a human tonsil dataset (Rep 1). The pipeline takes raw H&E,
RNA transcript, and multiplexed protein images as inputs and produces
fully preprocessed, analysis-ready AnnData objects for both modalities:
348 RNA genes and 16 protein markers across 230,446 segmented cells,
with QC filtering, normalization, dimensionality reduction, and clustering
applied to each modality.

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
   │
   v
Step 05 — Preprocess RNA and Protein
   RNA: QC (min 50 genes/cell, min 3 cells/gene) → Scrublet doublet removal →
   normalization (1e4) → log1p → top-300 HVGs → PCA → UMAP → Leiden clustering.
   Protein: cell-size outlier filtering (3 MAD) → zero-nuclear removal →
   nucleus-signal normalization → arcsinh (cofactor=0.01) → quantile normalization.
   Output: results/tonsil_rep1_rna_processed.h5ad
           results/tonsil_rep1_protein_processed.h5ad
           results/plots/
   │
   v
Step 07 — Extract Per-Cell H&E Patches  [env: g4x_prepro]
   For every cell in the preprocessed RNA AnnData, crop a square patch centred
   on the cell from the H&E JPEG-2000 image at six sizes (64, 128, 256, 448, 512 px,
   plus original bounding-box size). A single O(H×W) bounding-box scan and a
   multiprocessing pool (fork context) make this step fast even for 200k+ cells.
   Checkpoint-resume: already-written patches are skipped automatically.
   Output: results/cell_patches_tiff/{fixed_64,fixed_128,...,original_size}/<cell_id>.tiff
   │
   v
Step 08 — CONCH Feature Extraction  [env: conch]
   Pass each cell patch through the CONCH ViT-B/16 image encoder to obtain a
   512-dimensional morphological embedding. Requires the CONCH model weights
   (see Installation → CONCH model below).
   Output: results/conch_features/conch_features_tonsil_{size_folder}.csv
           results/conch_features/conch_metadata_tonsil_{size_folder}.csv
   │
   v
Step 09 — CONCH Features → AnnData  [env: conch]
   Load each feature CSV, apply PCA (512 → 500 dims), and save as an AnnData
   (.h5ad) ready for multimodal integration.
   Output: results/conch_features/tonsil_CONCH_he_conch_{size_folder}_prepro.h5ad
   │
   v
Step 10 — Build Multimodal MuData  [env: g4x_prepro]
   Intersect cells across RNA, protein, and H&E modalities by cell ID and
   combine into a MuData object with modalities 'rna', 'protein', and 'he'.
   Spatial coordinates (X_cent, Y_cent) are stored in mdata.obsm['spatial'].
   Output: results/conch_features/tonsil_multimodal_{size_folder}_matched.h5mu
```

---

## Repository Structure

```
G4X_prepro/
├── scripts/
│   ├── 01_visualize_transcripts_on_he.py   # env: g4x_prepro
│   ├── 02_segmentation.py                  # env: g4x_prepro
│   ├── 03_convert_to_geojson.py            # env: g4x_prepro
│   ├── 04_build_cell_gene_matrix.py        # env: g4x_prepro
│   ├── 05_preprocess_rna_protein.py        # env: g4x_prepro
│   ├── 07_extract_cell_patches.py          # env: g4x_prepro
│   ├── 08_conch_extract_features.py        # env: conch
│   ├── 09_conch_to_anndata.py              # env: conch
│   └── 10_build_multimodal_mudata.py       # env: g4x_prepro
├── utils/
│   ├── __init__.py
│   ├── io_setup.py            # GPU setup, logging, metadata I/O
│   ├── segmentation.py        # Mesmer segmentation utilities
│   └── segmentation_mask.py   # Mask visualization utilities
├── data/
│   └── README.md              # Data download and setup instructions
├── results/                   # Pipeline outputs (gitignored for large files)
├── requirements.txt
├── environment.yml            # Main conda environment (Steps 01–05, 07, 10)
├── environment_conch.yml      # CONCH conda environment (Steps 08–09)
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

### 2. Create the main conda environment (Steps 01–05, 07, 10)

```bash
conda env create -f environment.yml
conda activate g4x_prepro
```

> **Note:** `deepcell` requires TensorFlow. The `environment.yml` installs
> TensorFlow first so that deepcell can resolve a compatible version.
> GPU support requires a CUDA-compatible driver; see the
> [TensorFlow GPU guide](https://www.tensorflow.org/install/pip) for details.

### 3. Create the CONCH conda environment (Steps 08–09)

Steps 08 and 09 run in a separate environment that pins `anndata==0.11.4`
(the version compatible with the CONCH package) and bundles PyTorch.

```bash
conda env create -f environment_conch.yml
conda activate conch
pip install git+https://github.com/mahmoodlab/CONCH.git
```

### 4. Download the CONCH model weights

Request access and download `pytorch_model.bin` from the official release:

> https://huggingface.co/MahmoodLab/CONCH

Place the file at:

```
data/CONCH/checkpoints/conch/pytorch_model.bin
```

> **Reference:** Lu, M. Y. et al. (2024). A visual-language foundation model
> for computational pathology. *Nature Medicine*, 30, 863–874.
> https://doi.org/10.1038/s41591-024-02856-4

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

### Step 5 — Preprocess RNA and protein

```bash
python scripts/05_preprocess_rna_protein.py
```

### Step 7 — Extract per-cell H&E patches  `[env: g4x_prepro]`

```bash
conda activate g4x_prepro
python scripts/07_extract_cell_patches.py
```

Adjust `NUM_WORKERS` at the top of the script to match available CPU cores.

### Step 8 — Extract CONCH features  `[env: conch]`

```bash
conda activate conch
python scripts/08_conch_extract_features.py
```

Set `GPU_ID` at the top of the script to select the target GPU.

### Step 9 — CONCH features → AnnData  `[env: conch]`

```bash
conda activate conch
python scripts/09_conch_to_anndata.py
```

### Step 10 — Build multimodal MuData  `[env: g4x_prepro]`

```bash
conda activate g4x_prepro
python scripts/10_build_multimodal_mudata.py
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
| `results/tonsil_rep1_rna_processed.h5ad` | Preprocessed RNA (QC + normalization + UMAP + Leiden) |
| `results/tonsil_rep1_protein_processed.h5ad` | Preprocessed protein (filtered + arcsinh + quantile normalized) |
| `results/plots/` | QC, HVG, PCA variance, and UMAP figures |
| `results/cell_patches_tiff/fixed_{64,128,256,448,512}/` | Per-cell H&E patches at fixed sizes |
| `results/cell_patches_tiff/original_size/` | Per-cell H&E patches at original bounding-box size |
| `results/conch_features/conch_features_tonsil_*.csv` | Raw CONCH 512-dim features per size folder |
| `results/conch_features/tonsil_CONCH_he_conch_*_prepro.h5ad` | CONCH AnnData with PCA (n_cells × 512 / 500) |
| `results/conch_features/tonsil_multimodal_*_matched.h5mu` | MuData combining RNA + protein + H&E |

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
| `scanpy` | RNA QC, normalization, PCA, UMAP, Leiden clustering |
| `seaborn` | KDE plots for protein normalization QC |
| `igraph` | Leiden community detection backend |
| `codex_preprocessing` | Protein nucleus-signal normalization, arcsinh, quantile normalization |
| `torch` + `conch` | CONCH ViT-B/16 H&E image encoder (Steps 08–09, `conch` env) |
| `mudata` ≥ 0.3.3 | Multimodal data container combining RNA, protein, H&E |

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

### RNA Preprocessing

`scripts/05_preprocess_rna_protein.py` applies a standard scanpy single-cell
workflow to the RNA modality:

- **QC**: cells with fewer than 50 detected genes and genes detected in fewer
  than 3 cells are removed.
- **Doublet removal**: Scrublet is run on the single-sample data (no batch key)
  to flag and remove predicted doublets.
- **Normalization**: total-count normalization to 10,000 counts per cell,
  followed by log1p transformation.
- **Dimensionality reduction**: top-300 highly variable genes selected for PCA;
  UMAP computed from the PCA embedding (no Harmony needed for a single sample).
- **Clustering**: Leiden algorithm (`flavor='igraph'`, `resolution=1`) applied
  to the neighbor graph.

### Protein Preprocessing

The protein modality uses `codex_preprocessing` to handle multiplex imaging
signal characteristics:

- **Filtering**: cells outside 3 MAD of the median cell size are removed;
  cells with zero nuclear signal are discarded.
- **Nucleus-signal normalization**: each marker intensity is normalized by the
  nuclear signal within each sample, correcting for cell-to-cell variation in
  staining depth.
- **Arcsinh transformation**: applied with cofactor 0.01, compressing the
  heavy-tailed distribution typical of protein imaging data.
- **Quantile normalization**: per-marker 1st–99.9th percentile rescaling to
  bring all markers onto a comparable range.

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
| RNA cells after QC + doublet removal | 27,145 |
| RNA genes after QC | 321 |
| Protein cells after size + nuclear filtering | ~208,177 |
