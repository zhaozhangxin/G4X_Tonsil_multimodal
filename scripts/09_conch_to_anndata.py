# env: conch
"""
Convert CONCH H&E feature CSVs to AnnData format.

For each size folder processed in Step 08, this script:
  1. Loads the feature matrix CSV (n_cells × 512 raw CONCH features).
  2. Applies PCA (512 → 500 dimensions).
  3. Builds an AnnData with:
       .X              — raw CONCH features (n_cells × 512)
       .obs['cell_ID'] — cell identifier (string)
       .obsm['X_pca']  — PCA-reduced features (n_cells × 500)
       .uns['pca']     — explained variance info
  4. Saves the AnnData as an .h5ad file.

The output .h5ad files are written with anndata 0.11.x and can be read by
any anndata ≥ 0.10. Step 10 (multimodal matching) requires anndata ≥ 0.12
and must run in the `g4x_prepro` environment, NOT in this `conch` environment.

Inputs  (from Step 08)
------
  results/conch_features/conch_features_tonsil_{size_folder}.csv

Outputs
-------
  results/conch_features/tonsil_CONCH_he_conch_{size_folder}_prepro.h5ad
"""

import os
import anndata as ad
import pandas as pd
from sklearn.decomposition import PCA

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
SIZE_FOLDERS     = ["fixed_64", "fixed_128", "fixed_256", "fixed_448",
                    "fixed_512", "original_size"]
BASE_DIR         = os.path.join("results", "conch_features")
INPUT_TEMPLATE   = "conch_features_tonsil_{size_folder}.csv"
OUTPUT_TEMPLATE  = "tonsil_CONCH_he_conch_{size_folder}_prepro.h5ad"
N_PCA_COMPONENTS = 500
PCA_RANDOM_STATE = 42


# ─────────────────────────────────────────────
# Processing function
# ─────────────────────────────────────────────
def process_conch_features(size_folder, input_dir=BASE_DIR,
                           output_dir=BASE_DIR):
    print(f"\n{'=' * 80}")
    print(f"Processing: {size_folder}")
    print(f"{'=' * 80}")

    input_path  = os.path.join(input_dir,
                               INPUT_TEMPLATE.format(size_folder=size_folder))
    output_path = os.path.join(output_dir,
                               OUTPUT_TEMPLATE.format(size_folder=size_folder))

    if not os.path.exists(input_path):
        print(f"  Input file not found — skipping: {input_path}")
        return None

    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")

    features_df  = pd.read_csv(input_path)
    feature_cols = [c for c in features_df.columns if c.startswith("feature_")]
    X_value      = features_df[feature_cols].values
    print(f"  X shape: {X_value.shape}")

    n_comp = min(N_PCA_COMPONENTS, X_value.shape[1], X_value.shape[0])
    print(f"  PCA: {X_value.shape[1]} → {n_comp} dimensions")
    pca   = PCA(n_components=n_comp, random_state=PCA_RANDOM_STATE)
    X_pca = pca.fit_transform(X_value)
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    obs = pd.DataFrame({"cell_ID": features_df["Cell_id"].astype(str)})
    adata = ad.AnnData(X=X_value, obs=obs)
    adata.obsm["X_pca"] = X_pca
    adata.uns["pca"] = {
        "variance_ratio": pca.explained_variance_ratio_,
        "variance":       pca.explained_variance_,
        "n_components":   n_comp,
    }
    adata.uns["size_folder"] = size_folder
    adata.write(output_path)

    print(f"  Saved AnnData {adata.shape} → {output_path}")
    return adata


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 80)
    print("CONCH Features → AnnData")
    print("=" * 80)
    print(f"Base directory : {BASE_DIR}")
    print(f"PCA components : {N_PCA_COMPONENTS}")
    print(f"Size folders   : {SIZE_FOLDERS}")

    results = {}
    for sf in SIZE_FOLDERS:
        adata = process_conch_features(sf)
        if adata is not None:
            results[sf] = adata

    print(f"\n{'=' * 80}")
    print(f"Summary: {len(results)}/{len(SIZE_FOLDERS)} size folders processed")
    for sf, a in results.items():
        print(f"  {sf}: {a.shape[0]} cells")
    failed = set(SIZE_FOLDERS) - set(results)
    for sf in sorted(failed):
        print(f"  {sf}: FAILED")
    print("=" * 80)
