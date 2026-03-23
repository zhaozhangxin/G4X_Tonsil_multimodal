# env: g4x_prepro
"""
Preprocess tonsil_rep1 RNA and Protein AnnData objects.

RNA pipeline  (scanpy, CPU):
  QC → doublet removal → normalization → HVG → PCA → UMAP → Leiden clustering

Protein pipeline (codex_preprocessing):
  Cell-size / nuclear filtering → nucleus-signal normalization →
  arcsinh transformation → quantile normalization
"""

import os
import logging
import datetime
import anndata as ad
import scanpy as sc
import rapids_singlecell as rsc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from codex_preprocessing.preprocessing import (
    ExtremeCutoff,
    arcsinh_transformation,
    nucleus_signal_normalization,
    plot_arcsinh_transformation,
    plot_quantile_normalization,
    quantile_normalization,
    print_processing_history,
    _downsample_cells,
)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR   = "/mnt/md0/home/zhangxinzhao/G4X_prepro/public_datasets/G4X_prepro"
PLOT_DIR   = os.path.join(BASE_DIR, "G4X_tonsil_prepro_plot")
RNA_IN     = os.path.join(BASE_DIR, "tonsil_rep1_rna.h5ad")
PROT_IN    = os.path.join(BASE_DIR, "tonsil_rep1_protein.h5ad")
RNA_OUT    = os.path.join(BASE_DIR, "tonsil_rep1_rna_processed.h5ad")
PROT_OUT   = os.path.join(BASE_DIR, "tonsil_rep1_protein_processed.h5ad")

LOG_OUT    = os.path.join(BASE_DIR, "tonsil_rep1_preprocess_cell_counts.log")

os.makedirs(PLOT_DIR, exist_ok=True)

# ── Logging setup ─────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(LOG_OUT, mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger()
log.info(f"Run time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log.info(f"Log file: {LOG_OUT}")

# scanpy saves figures to figdir; we point it to our plot folder
sc.settings.figdir = PLOT_DIR
sc.settings.verbosity = 1


# ═════════════════════════════════════════════
# Part 1 – RNA preprocessing
# ═════════════════════════════════════════════
log.info("\n" + "="*60)
log.info("Part 1: RNA preprocessing")
log.info("="*60)

adata_rna = ad.read_h5ad(RNA_IN)
log.info(f"Loaded RNA: {adata_rna}")
log.info(f"X dtype: {adata_rna.X.dtype}")

adata_rna.layers["raw"] = adata_rna.X.copy()

# ── Step 1: Quality control ──────────────────
n0_rna = adata_rna.n_obs
g0_rna = adata_rna.n_vars

# min/max UMI threshold: median(log(total_counts)) ± 2 * MAD(log(total_counts))
import scipy.sparse as sp
_counts = np.array(adata_rna.X.sum(axis=1)).flatten()
_log_counts = np.log(_counts + 1)
_median = np.median(_log_counts)
_mad = np.median(np.abs(_log_counts - _median))
_min_log = _median - 3 * _mad
mask_umi = (_log_counts >= _min_log) 
adata_rna = adata_rna[mask_umi].copy()

n1_rna = adata_rna.n_obs
sc.pp.filter_genes(adata_rna, min_cells=3)
g1_rna = adata_rna.n_vars
log.info(f"\n[RNA] Start: {n0_rna:,} cells, {g0_rna:,} genes")
log.info(f"[RNA] Step1 filter_cells (median±2·MAD on log(UMI)): {n0_rna:,} → {n1_rna:,} cells  (removed {n0_rna - n1_rna:,})")
log.info(f"[RNA]   threshold: log(UMI) ∈ [{_min_log:.3f}, Inf]  "
         f"(UMI ∈ [{np.expm1(_min_log):.1f}, Inf])")
log.info(f"[RNA] Step1 filter_genes (min_cells=3):  {g0_rna:,} → {g1_rna:,} genes  (removed {g0_rna - g1_rna:,})")

# ── Step 2: Doublet detection (GPU) ──────────
# rsc.pp.scrublet 需要原始 counts，在归一化前运行
rsc.get.anndata_to_GPU(adata_rna)
rsc.pp.scrublet(adata_rna)
rsc.get.anndata_to_CPU(adata_rna)
n_before = adata_rna.n_obs
adata_rna.obs["predicted_doublet"] = adata_rna.obs["predicted_doublet"].fillna(False).astype(bool)
adata_rna = adata_rna[~adata_rna.obs["predicted_doublet"]].copy()
log.info(f"[RNA] Step2 doublet removal:             {n_before:,} → {adata_rna.n_obs:,} cells  (removed {n_before - adata_rna.n_obs:,})")
log.info(f"[RNA] After all filters: {adata_rna.n_obs:,} cells remaining")

# ── Step 3: Normalization (GPU) ───────────────
rsc.get.anndata_to_GPU(adata_rna)
rsc.pp.normalize_total(adata_rna, target_sum=1e4)
rsc.pp.log1p(adata_rna)
print("✅ Normalization done")

# ── Step 4: Highly variable genes (GPU) ──────
# Single sample → no batch_key
rsc.pp.highly_variable_genes(adata_rna, n_top_genes=300)
rsc.get.anndata_to_CPU(adata_rna)
sc.pl.highly_variable_genes(
    adata_rna, show=False,
    save="_tonsil_rep1_rna_hvg.png",
)
rsc.get.anndata_to_GPU(adata_rna)
print("✅ HVG selection done")

# ── Step 5: PCA (GPU) ─────────────────────────
rsc.tl.pca(adata_rna, use_highly_variable=True)
rsc.get.anndata_to_CPU(adata_rna)
sc.pl.pca_variance_ratio(
    adata_rna, n_pcs=50, show=False,
    save="_tonsil_rep1_rna_pca_variance.png",
)
rsc.get.anndata_to_GPU(adata_rna)
print("✅ PCA done")

# ── Step 6: Neighbors + UMAP (GPU) ───────────
# Single sample: use X_pca directly (no Harmony needed)
rsc.pp.neighbors(adata_rna, use_rep="X_pca", n_neighbors=15, n_pcs=30)
rsc.tl.umap(adata_rna, min_dist=0.5, spread=1.0, random_state=0)
rsc.get.anndata_to_CPU(adata_rna)
print("✅ Neighbors + UMAP done")

# ── Step 7: Leiden clustering (CPU, cugraph 不可用) ──
sc.tl.leiden(adata_rna, resolution=1, flavor='igraph')
sc.pl.umap(
    adata_rna,
    color=["leiden"],
    save="_tonsil_rep1_rna_leiden.png",
    show=False,
)
print("✅ Leiden clustering done")

# ── Save RNA ─────────────────────────────────
adata_rna.write_h5ad(RNA_OUT)
log.info(f"\n✅ RNA saved to: {RNA_OUT}")


# ═════════════════════════════════════════════
# Part 2 – Protein preprocessing
# ═════════════════════════════════════════════
log.info("\n" + "="*60)
log.info("Part 2: Protein preprocessing")
log.info("="*60)

adata_prot = ad.read_h5ad(PROT_IN)
log.info(f"Loaded Protein: {adata_prot}")
log.info(f"Markers: {list(adata_prot.var_names)}")

n0_prot = adata_prot.n_obs
log.info(f"\n[Protein] Start: {n0_prot:,} cells")

# nucleus_signal_normalization requires a sample-id column
adata_prot.obs["sample_id"] = "tonsil_rep1"

# ── Step 1: Cell-size filtering ──────────────
extreme_cutoff = ExtremeCutoff(values=adata_prot.obs["cellSize"])
print(extreme_cutoff)
mask_size = extreme_cutoff.filter_values(method="median", n_sigma=3)
log.info(f"[Protein] Step1 cell-size filter (median±3σ): removes {np.sum(~mask_size):,} cells")

# ── Step 2: Zero-nuclear filter ──────────────
mask_nuclear = (adata_prot[:, "nuclear"].X.toarray() > 0).flatten()
log.info(f"[Protein] Step2 zero-nuclear filter:          removes {np.sum(~mask_nuclear):,} cells")

only_size_fail   = np.sum(~mask_size & mask_nuclear)
only_nuc_fail    = np.sum(mask_size & ~mask_nuclear)
both_fail        = np.sum(~mask_size & ~mask_nuclear)
log.info(f"[Protein]   (fail size only: {only_size_fail:,} | fail nuclear only: {only_nuc_fail:,} | fail both: {both_fail:,})")

adata_prot = adata_prot[mask_size & mask_nuclear].copy()
# codex_preprocessing 不支持稀疏矩阵，转为 dense
import scipy.sparse as sp
if sp.issparse(adata_prot.X):
    adata_prot.X = adata_prot.X.toarray()
adata_prot.layers["scale_size"] = adata_prot.X.copy()
log.info(f"[Protein] After all filters: {n0_prot:,} → {adata_prot.n_obs:,} cells  (removed {n0_prot - adata_prot.n_obs:,})")

# ── Step 3: Nucleus-signal normalization ─────
nucleus_signal_normalization(
    adata_prot,
    col_data_id="sample_id",
    marker_nucleus="nuclear",
    inplace=True,
)
print_processing_history(adata_prot)

# Plot DAPI (nuclear) before / after normalization
adata_sm = _downsample_cells(adata_prot, sample_size=min(100000, adata_prot.n_obs))
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.kdeplot(
    x=adata_sm[:, "nuclear"].layers["scale_size"].toarray().flatten(),
    hue=adata_sm.obs["sample_id"],
    log_scale=(True, False),
    legend=True,
    ax=axes[0],
)
axes[0].set_title("nuclear Before Normalization")
sns.kdeplot(
    x=adata_sm[:, "nuclear"].X.toarray().flatten(),
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
quantile_normalization(
    adata_prot,
    min_quantile=0.01,
    max_quantile=0.999,
    inplace=True,
)
print_processing_history(adata_prot)
print("✅ Quantile normalization done")

# ── Save Protein ──────────────────────────────
adata_prot.uns.pop("processing_history", None)
adata_prot.write_h5ad(PROT_OUT)
log.info(f"\n✅ Protein saved to: {PROT_OUT}")

log.info("\n" + "="*60)
log.info("All done.")
log.info(f"Plots saved to: {PLOT_DIR}")
log.info(f"Cell-count log saved to: {LOG_OUT}")
log.info("="*60)
