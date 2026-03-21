# env: g4x_prepro
"""
Preprocess tonsil_rep1 RNA and Protein AnnData objects.

RNA pipeline (scanpy):
  QC → doublet removal → normalization → HVG → PCA → UMAP → Leiden clustering

Protein pipeline (codex_preprocessing):
  Cell-size / nuclear filtering → nucleus-signal normalization →
  arcsinh transformation → quantile normalization

Inputs  (from Step 04):
  results/tonsil_rep1_rna.h5ad
  results/tonsil_rep1_protein.h5ad

Outputs:
  results/tonsil_rep1_rna_processed.h5ad
  results/tonsil_rep1_protein_processed.h5ad
  results/plots/  (QC and embedding figures)
"""

import os
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from codex_preprocessing.preprocessing import (
    ExtremeCutoff,
    arcsinh_transformation,
    nucleus_signal_normalization,
    quantile_normalization,
    print_processing_history,
    _downsample_cells,
)

# ─────────────────────────────────────────────
# Paths  (run from repository root)
# ─────────────────────────────────────────────
RESULTS_DIR = "results"
PLOT_DIR    = os.path.join(RESULTS_DIR, "plots")
RNA_IN      = os.path.join(RESULTS_DIR, "tonsil_rep1_rna.h5ad")
PROT_IN     = os.path.join(RESULTS_DIR, "tonsil_rep1_protein.h5ad")
RNA_OUT     = os.path.join(RESULTS_DIR, "tonsil_rep1_rna_processed.h5ad")
PROT_OUT    = os.path.join(RESULTS_DIR, "tonsil_rep1_protein_processed.h5ad")

os.makedirs(PLOT_DIR, exist_ok=True)

sc.settings.figdir = PLOT_DIR
sc.settings.verbosity = 1


# ═════════════════════════════════════════════
# Part 1 – RNA preprocessing
# ═════════════════════════════════════════════
print("\n" + "="*60)
print("Part 1: RNA preprocessing")
print("="*60)

adata_rna = ad.read_h5ad(RNA_IN)
print(f"Loaded RNA: {adata_rna}")
print(f"X dtype: {adata_rna.X.dtype}")

adata_rna.layers["raw"] = adata_rna.X.copy()

# ── Step 1: Quality control ──────────────────
sc.pp.filter_cells(adata_rna, min_genes=50)
sc.pp.filter_genes(adata_rna, min_cells=3)
print(f"✅ QC: {adata_rna.n_obs} cells, {adata_rna.n_vars} genes")

# ── Step 2: Doublet detection ────────────────
# Single sample → no batch_key
sc.pp.scrublet(adata_rna)
n_before = adata_rna.n_obs
adata_rna = adata_rna[~adata_rna.obs["predicted_doublet"]].copy()
print(f"✅ Doublet removal: {n_before} → {adata_rna.n_obs} cells "
      f"({n_before - adata_rna.n_obs} doublets removed)")

# ── Step 3: Normalization ────────────────────
sc.pp.normalize_total(adata_rna, target_sum=1e4)
sc.pp.log1p(adata_rna)
print("✅ Normalization done")

# ── Step 4: Highly variable genes ───────────
sc.pp.highly_variable_genes(adata_rna, n_top_genes=300)
sc.pl.highly_variable_genes(adata_rna, show=False, save="_tonsil_rep1_rna_hvg.png")
print("✅ HVG selection done")

# ── Step 5: PCA ──────────────────────────────
sc.tl.pca(adata_rna, mask_var="highly_variable")
sc.pl.pca_variance_ratio(adata_rna, n_pcs=50, show=False, save="_tonsil_rep1_rna_pca_variance.png")
print("✅ PCA done")

# ── Step 6: Neighbors + UMAP ─────────────────
# Single sample: use X_pca directly (no Harmony needed)
sc.pp.neighbors(adata_rna, use_rep="X_pca", n_neighbors=15, n_pcs=30)
sc.tl.umap(adata_rna, min_dist=0.5, spread=1.0, random_state=0)
print("✅ Neighbors + UMAP done")

# ── Step 7: Leiden clustering ────────────────
sc.tl.leiden(adata_rna, resolution=1, flavor="igraph")
sc.pl.umap(adata_rna, color=["leiden"], save="_tonsil_rep1_rna_leiden.png", show=False)
print("✅ Leiden clustering done")

# ── Save RNA ─────────────────────────────────
adata_rna.write_h5ad(RNA_OUT)
print(f"\n✅ RNA saved to: {RNA_OUT}")


# ═════════════════════════════════════════════
# Part 2 – Protein preprocessing
# ═════════════════════════════════════════════
print("\n" + "="*60)
print("Part 2: Protein preprocessing")
print("="*60)

adata_prot = ad.read_h5ad(PROT_IN)
print(f"Loaded Protein: {adata_prot}")
print(f"Markers: {list(adata_prot.var_names)}")

# nucleus_signal_normalization requires a sample-id column
adata_prot.obs["sample_id"] = "tonsil_rep1"

# ── Step 1: Cell-size filtering ──────────────
extreme_cutoff = ExtremeCutoff(values=adata_prot.obs["cellSize"])
print(extreme_cutoff)
mask_size = extreme_cutoff.filter_values(method="median", n_sigma=3)
print(f"Cell-size filter: removing {np.sum(~mask_size):,} cells")

# ── Step 2: Zero-nuclear filter ──────────────
mask_nuclear = (np.asarray(adata_prot[:, "nuclear"].X.todense()) > 0).flatten()
print(f"Zero-nuclear filter: removing {np.sum(~mask_nuclear):,} cells")

adata_prot = adata_prot[mask_size & mask_nuclear].copy()
# codex_preprocessing requires a dense matrix
if sp.issparse(adata_prot.X):
    adata_prot.X = adata_prot.X.toarray()
adata_prot.layers["scale_size"] = adata_prot.X.copy()
print(f"✅ Filtering done: {adata_prot.n_obs} cells remaining")

# ── Step 3: Nucleus-signal normalization ─────
nucleus_signal_normalization(
    adata_prot,
    col_data_id="sample_id",
    marker_nucleus="nuclear",
    inplace=True,
)
print_processing_history(adata_prot)

# Plot nuclear signal before / after normalization
adata_sm = _downsample_cells(adata_prot, sample_size=min(100000, adata_prot.n_obs))
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.kdeplot(
    x=np.asarray(adata_sm[:, "nuclear"].layers["scale_size"]).flatten(),
    hue=adata_sm.obs["sample_id"],
    log_scale=(True, False),
    legend=True,
    ax=axes[0],
)
axes[0].set_title("nuclear Before Normalization")
sns.kdeplot(
    x=np.asarray(adata_sm[:, "nuclear"].X).flatten(),
    hue=adata_sm.obs["sample_id"],
    log_scale=(True, False),
    legend=True,
    ax=axes[1],
)
axes[1].set_title("nuclear After Normalization")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "protein_nuclear_normalization.png"), dpi=150)
plt.close()
print("✅ Nucleus-signal normalization done")

# ── Step 4: Arcsinh transformation ───────────
arcsinh_transformation(adata_prot, cofactor=0.01, inplace=True)
print_processing_history(adata_prot)
print("✅ Arcsinh transformation done")

# ── Step 5: Quantile normalization ───────────
quantile_normalization(adata_prot, min_quantile=0.01, max_quantile=0.999, inplace=True)
print_processing_history(adata_prot)
print("✅ Quantile normalization done")

# ── Save Protein ──────────────────────────────
adata_prot.uns.pop("processing_history", None)
adata_prot.write_h5ad(PROT_OUT)
print(f"\n✅ Protein saved to: {PROT_OUT}")

print("\n" + "="*60)
print("All done.")
print(f"Plots saved to: {PLOT_DIR}")
print("="*60)
