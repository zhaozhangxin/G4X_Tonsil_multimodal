"""
04_build_cell_gene_matrix.py
============================
Build a multimodal single-cell data object (MuData) combining RNA transcript
counts and protein marker intensities, both indexed by cell ID from the
segmentation mask.

Steps
-----
1.  Load segmentation mask; extract all non-zero cell labels (230,446 cells).
2.  Load transcript table (22,148,004 transcripts); look up each transcript's
    rounded pixel coordinate in the mask to assign a cell ID.
3.  Filter out background transcripts (cell_id == 0); retain transcripts
    inside cells (18,262,215 transcripts covering 227,081 cells).
4.  Group by (cell_id, gene_name) to count; build sparse RNA count matrix.
    Zero-pad the 3,365 cells that exist in the mask but received no transcripts
    so that RNA cell count matches the total segmented cell count.
5.  Load dataScaleSize.csv; build Protein AnnData with cellSize, Y_cent,
    X_cent as cell metadata and 16 protein marker columns as features.
6.  Intersect RNA and Protein obs indices to align cells across modalities.
7.  Save each modality as .h5ad, then merge into a MuData object (.h5mu).

Inputs
------
- data/processed/<SEGMENTATION_TAG>/segmentation_mask.tiff
- data/raw/rna/transcript_table.csv.gz
- data/processed/<SEGMENTATION_TAG>/dataScaleSize.csv   (only if
  RECOMPUTE_PROTEIN_FROM_OMETIFF == False)
- data/raw/ometiff/G04.ome.tiff                          (only if
  RECOMPUTE_PROTEIN_FROM_OMETIFF == True)

Outputs
-------
- results/tonsil_rep1_rna.h5ad         : RNA AnnData  (230,446 cells x 348 genes)
- results/tonsil_rep1_protein.h5ad     : Protein AnnData (230,446 cells x 16 markers)
- results/tonsil_rep1.h5mu             : MuData (RNA + Protein)

Custom segmentation
-------------------
Every modality this script writes (RNA + Protein) is grounded in MASK_PATH:
- RNA counts are always re-binned by mask cell ID, so they always match.
- Protein quantification comes from PROTEIN_PATH, which is produced by
  02_segmentation.py with the *Mesmer* mask. If you swap MASK_PATH for your
  own segmentation, set RECOMPUTE_PROTEIN_FROM_OMETIFF = True so the protein
  table is re-derived from OMETIFF_PATH using YOUR mask.
"""

import sys
from pathlib import Path

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import tifffile
from scipy.sparse import csr_matrix

# Allow imports from the utils/ package next to scripts/
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_ROOT   = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

SEGMENTATION_TAG = "260316_g4xtonsil_cell"

# Set True when MASK_PATH points at a custom (non-Mesmer) segmentation, so
# protein per-cell values are re-computed from the OME-TIFF using YOUR mask.
RECOMPUTE_PROTEIN_FROM_OMETIFF = False

MASK_PATH       = DATA_ROOT / "processed" / SEGMENTATION_TAG / "segmentation_mask.tiff"
TRANSCRIPT_PATH = DATA_ROOT / "raw" / "rna" / "transcript_table.csv.gz"
PROTEIN_PATH    = DATA_ROOT / "processed" / SEGMENTATION_TAG / "dataScaleSize.csv"
OMETIFF_PATH    = DATA_ROOT / "raw" / "ometiff" / "G04.ome.tiff"

OUTPUT_PATH         = RESULTS_DIR / "tonsil_rep1.h5mu"
RNA_OUTPUT_PATH     = RESULTS_DIR / "tonsil_rep1_rna.h5ad"
PROTEIN_OUTPUT_PATH = RESULTS_DIR / "tonsil_rep1_protein.h5ad"
# ──────────────────────────────────────────────────────────────────────────────

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Step 1: Load segmentation mask ──────────────────────────────────────────
print("Loading segmentation mask...")
mask = tifffile.imread(MASK_PATH)
print(f"  Mask shape: {mask.shape}, dtype: {mask.dtype}")
mask_h, mask_w = mask.shape

# Collect all non-zero cell labels in the mask (used later to zero-pad RNA)
all_mask_cell_ids = np.unique(mask)
all_mask_cell_ids = all_mask_cell_ids[all_mask_cell_ids > 0]
all_mask_cell_ids_str = all_mask_cell_ids.astype(str)
print(f"  Total segmented cells in mask: {len(all_mask_cell_ids)}")

# ── Step 2: Load transcript table ───────────────────────────────────────────
print("Loading transcript table...")
df = pd.read_csv(
    TRANSCRIPT_PATH,
    usecols=["y_pixel_coordinate", "x_pixel_coordinate", "gene_name"],
)
print(f"  Total transcripts: {len(df)}")

# ── Step 3: Round coordinates and clip to mask bounds ───────────────────────
df["y"] = df["y_pixel_coordinate"].round().astype(int).clip(0, mask_h - 1)
df["x"] = df["x_pixel_coordinate"].round().astype(int).clip(0, mask_w - 1)

# ── Step 4: Look up cell_id from mask ───────────────────────────────────────
print("Looking up cell IDs from mask...")
df["cell_id"] = mask[df["y"].values, df["x"].values]

# ── Step 5: Drop background (cell_id == 0) ──────────────────────────────────
df_cells = df[df["cell_id"] > 0].copy()
print(f"  Transcripts assigned to cells: {len(df_cells)} / {len(df)}")
print(f"  Unique cells: {df_cells['cell_id'].nunique()}")
print(f"  Unique genes: {df_cells['gene_name'].nunique()}")

# ── Step 6: Count gene occurrences per cell ──────────────────────────────────
print("Counting gene expression per cell...")
counts = (
    df_cells.groupby(["cell_id", "gene_name"])
    .size()
    .reset_index(name="count")
)

# ── Step 7: Build RNA AnnData (zero-pad cells with no transcripts) ───────────
print("Building RNA AnnData...")
matrix = (
    counts.pivot(index="cell_id", columns="gene_name", values="count")
    .fillna(0)
    .astype(np.float32)
)
matrix.columns.name = None
matrix.index = matrix.index.astype(str)

# Reindex to all mask cells; cells with no transcripts get 0 across all genes
matrix = matrix.reindex(all_mask_cell_ids_str, fill_value=0)
n_zero_padded = len(all_mask_cell_ids_str) - (matrix.sum(axis=1) > 0).sum()
print(f"  Zero-padded cells (no transcripts): {n_zero_padded}")

rna_obs = pd.DataFrame(index=matrix.index)
rna_var = pd.DataFrame(index=matrix.columns)
adata_rna = ad.AnnData(
    X=csr_matrix(matrix.values),
    obs=rna_obs,
    var=rna_var,
)
print(f"  RNA AnnData: {adata_rna.shape}  (cells x genes)")

# ── Step 8: Build Protein AnnData ────────────────────────────────────────────
if RECOMPUTE_PROTEIN_FROM_OMETIFF:
    # Load OME-TIFF channels and quantify per cell using OUR mask, so cellLabel
    # IDs come from MASK_PATH rather than 02_segmentation.py's Mesmer mask.
    print(f"Recomputing per-cell protein from OME-TIFF: {OMETIFF_PATH}")
    from pyqupath.tiff import TiffZarrReader
    from segmentation import extract_cell_features

    marker_dict = TiffZarrReader.from_ometiff(OMETIFF_PATH).zimg_dict
    print(f"  Markers in OME-TIFF: {list(marker_dict.keys())}")

    _, prot_df = extract_cell_features(marker_dict, mask)
    prot_df = prot_df.set_index("cellLabel")
    prot_df.index = prot_df.index.astype(str)
else:
    print(f"Loading protein data from {PROTEIN_PATH} ...")
    prot_df = pd.read_csv(PROTEIN_PATH, index_col=0)

    # cellLabel is the cell ID; use it as obs index
    prot_df = prot_df.set_index("cellLabel")
    prot_df.index = prot_df.index.astype(str)

# Metadata columns to keep in obs
meta_cols = ["cellSize", "Y_cent", "X_cent"]
prot_obs = prot_df[meta_cols].copy()

# Protein marker columns (everything after meta columns)
marker_cols = [c for c in prot_df.columns if c not in meta_cols]
prot_matrix = prot_df[marker_cols].astype(np.float32)

adata_prot = ad.AnnData(
    X=csr_matrix(prot_matrix.values),
    obs=prot_obs,
    var=pd.DataFrame(index=marker_cols),
)
print(f"  Protein AnnData: {adata_prot.shape}  (cells x proteins)")

# ── Step 9: Align cells present in both modalities ───────────────────────────
common_cells = adata_rna.obs_names.intersection(adata_prot.obs_names)
print(f"  Common cells (RNA n Protein): {len(common_cells)}")
adata_rna  = adata_rna[common_cells].copy()
adata_prot = adata_prot[common_cells].copy()

# ── Step 10: Print AnnData attributes ────────────────────────────────────────
print("\nRNA AnnData attributes:")
print(f"  shape:       {adata_rna.shape}  (cells x genes)")
print(f"  obs columns: {list(adata_rna.obs.columns)}")
print(f"  var columns: {list(adata_rna.var.columns)}")
print(f"  obsm keys:   {list(adata_rna.obsm.keys())}")
print(f"  obsp keys:   {list(adata_rna.obsp.keys())}")
print(f"  uns keys:    {list(adata_rna.uns.keys())}")
print(f"  X dtype:     {adata_rna.X.dtype}, nnz: {adata_rna.X.nnz}")

print("\nProtein AnnData attributes:")
print(f"  shape:       {adata_prot.shape}  (cells x proteins)")
print(f"  obs columns: {list(adata_prot.obs.columns)}")
print(f"  var columns: {list(adata_prot.var.columns)}")
print(f"  obsm keys:   {list(adata_prot.obsm.keys())}")
print(f"  obsp keys:   {list(adata_prot.obsp.keys())}")
print(f"  uns keys:    {list(adata_prot.uns.keys())}")
print(f"  X dtype:     {adata_prot.X.dtype}, nnz: {adata_prot.X.nnz}")

# ── Step 11: Save individual AnnData files ────────────────────────────────────
print(f"\nSaving RNA AnnData to {RNA_OUTPUT_PATH} ...")
adata_rna.write_h5ad(RNA_OUTPUT_PATH)

print(f"Saving Protein AnnData to {PROTEIN_OUTPUT_PATH} ...")
adata_prot.write_h5ad(PROTEIN_OUTPUT_PATH)

# ── Step 12: Build MuData and save ───────────────────────────────────────────
print("\nBuilding MuData...")
mdata = md.MuData({"rna": adata_rna, "protein": adata_prot})
print(f"  MuData: {mdata}")

print(f"Saving to {OUTPUT_PATH} ...")
mdata.write(OUTPUT_PATH)
print("Done.")
