# env: g4x_prepro
# NOTE: requires anndata >= 0.12 and mudata >= 0.3.3 (see environment.yml).
#       Do NOT run in the `conch` environment — it ships anndata 0.11.x,
#       which cannot read .h5ad files written by anndata 0.12.x.
"""
Combine RNA, protein, and H&E CONCH features into a MuData object.

For each size folder, this script:
  1. Loads preprocessed RNA and protein AnnData objects (once, shared).
  2. Loads the CONCH H&E AnnData for the current size folder.
  3. Finds cells present in all three modalities (intersection on cell ID).
  4. Builds a MuData with modalities 'rna', 'protein', and 'he'.
  5. Promotes spatial coordinates and key protein metadata to mdata.obs /
     mdata.obsm so that downstream tools can access them easily.
  6. Saves the MuData and unmatched-cell CSVs.

Cell-ID normalisation
---------------------
  RNA     — obs index is named 'cell_id', values like '117'
  Protein — obs index is named 'cellLabel', values like '1' (cast to str)
  H&E     — obs column 'cell_ID' (set as index)
  All three are cast to str before the intersection.

Inputs  (from Steps 05 and 09)
------
  results/tonsil_rep1_rna_processed.h5ad
  results/tonsil_rep1_protein_processed.h5ad
  results/conch_features/tonsil_CONCH_he_conch_{size_folder}_prepro.h5ad

Outputs
-------
  results/conch_features/tonsil_multimodal_{size_folder}_matched.h5mu
  results/conch_features/tonsil_unmatched_rna_{size_folder}_cells.csv
  results/conch_features/tonsil_unmatched_protein_{size_folder}_cells.csv
  results/conch_features/tonsil_unmatched_he_{size_folder}_cells.csv
"""

import os
import anndata as ad
import mudata as md
import pandas as pd

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
SIZE_FOLDERS = ["fixed_64", "fixed_128", "fixed_256", "fixed_448",
                "fixed_512", "original_size"]

RNA_PATH     = os.path.join("results", "tonsil_rep1_rna_processed.h5ad")
PROTEIN_PATH = os.path.join("results", "tonsil_rep1_protein_processed.h5ad")
HE_TEMPLATE  = os.path.join("results", "conch_features",
                             "tonsil_CONCH_he_conch_{size_folder}_prepro.h5ad")
OUTPUT_DIR   = os.path.join("results", "conch_features")

MUDATA_TEMPLATE            = "tonsil_multimodal_{size_folder}_matched.h5mu"
UNMATCHED_RNA_TEMPLATE     = "tonsil_unmatched_rna_{size_folder}_cells.csv"
UNMATCHED_PROTEIN_TEMPLATE = "tonsil_unmatched_protein_{size_folder}_cells.csv"
UNMATCHED_HE_TEMPLATE      = "tonsil_unmatched_he_{size_folder}_cells.csv"


# ─────────────────────────────────────────────
# Matching function
# ─────────────────────────────────────────────
def match_multimodal(adata_rna, adata_protein, adata_he, size_folder):
    """Intersect RNA / protein / H&E on cell ID and return a MuData."""
    print(f"\n{'=' * 80}")
    print(f"Processing: {size_folder}")
    print(f"{'=' * 80}")

    rna     = adata_rna.copy()
    protein = adata_protein.copy()
    he      = adata_he.copy()

    print(f"RNA     shape: {rna.shape}")
    print(f"Protein shape: {protein.shape}")
    print(f"H&E     shape: {he.shape}")

    # Normalise indices to str
    rna.obs.index     = rna.obs.index.astype(str)
    rna.obs.index.name = "cell_id"
    protein.obs.index = protein.obs.index.astype(str)
    protein.obs.index.name = "cell_id"
    he.obs.index      = he.obs["cell_ID"].astype(str)
    he.obs.index.name = "cell_id"

    print(f"\nRNA     index sample: {rna.obs.index[:5].tolist()}")
    print(f"Protein index sample: {protein.obs.index[:5].tolist()}")
    print(f"H&E     index sample: {he.obs.index[:5].tolist()}")

    common = (set(rna.obs.index)
              .intersection(protein.obs.index)
              .intersection(he.obs.index))
    print(f"\nCommon cells (all 3 modalities): {len(common)}")
    print(f"  RNA only / missing:     {len(set(rna.obs.index) - common)}")
    print(f"  Protein only / missing: {len(set(protein.obs.index) - common)}")
    print(f"  H&E only / missing:     {len(set(he.obs.index) - common)}")

    sorted_ids = sorted(common, key=lambda x: int(x))

    rna_m     = rna[sorted_ids].copy()
    protein_m = protein[sorted_ids].copy()
    he_m      = he[sorted_ids].copy()

    # Deduplicate if needed
    for name, obj in [("RNA", rna_m), ("Protein", protein_m), ("H&E", he_m)]:
        dups = obj.obs.index.duplicated().sum()
        if dups:
            print(f"  {name}: {dups} duplicate cell IDs — keeping first")
            obj = obj[~obj.obs.index.duplicated(keep="first")].copy()

    assert (list(rna_m.obs.index) == list(protein_m.obs.index)
            == list(he_m.obs.index)), \
        "Cell indices are not aligned across modalities!"
    print(f"  Alignment check passed: {rna_m.n_obs} cells")

    # Unmatched cells
    unmatched_rna  = rna.obs.loc[~rna.obs.index.isin(common), []].reset_index()
    unmatched_prot = protein.obs.loc[
        ~protein.obs.index.isin(common), []].reset_index()
    unmatched_he   = he.obs.loc[
        ~he.obs.index.isin(common), ["cell_ID"]].reset_index()

    # Build MuData
    mdata = md.MuData({"rna": rna_m, "protein": protein_m, "he": he_m})
    mdata.uns["he_size_folder"] = size_folder

    # Promote selected protein metadata to mdata.obs
    PROTEIN_META = ["cellSize", "sample_id", "nucleus_norm_factor"]
    for col in PROTEIN_META:
        mdata.obs[f"protein:{col}"] = protein_m.obs[col].values
    print(f"  Promoted to mdata.obs: {['protein:' + c for c in PROTEIN_META]}")

    # Store spatial coordinates
    mdata.obsm["spatial"] = protein_m.obs[["X_cent", "Y_cent"]].values
    print(f"  mdata.obsm['spatial'] shape: {mdata.obsm['spatial'].shape}")

    print(f"\nMuData summary:")
    print(f"  n_obs:     {mdata.n_obs}")
    print(f"  rna:       {mdata['rna'].shape}")
    print(f"  protein:   {mdata['protein'].shape}")
    print(f"  he:        {mdata['he'].shape}")
    if "X_pca" in mdata["he"].obsm:
        print(f"  he X_pca:  {mdata['he'].obsm['X_pca'].shape}")

    return mdata, unmatched_rna, unmatched_prot, unmatched_he


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 80)
    print("Tonsil Multimodal (RNA + Protein + H&E) Matching")
    print("=" * 80)

    for p in [RNA_PATH, PROTEIN_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    adata_rna     = ad.read_h5ad(RNA_PATH)
    adata_protein = ad.read_h5ad(PROTEIN_PATH)
    print(f"Loaded RNA:     {adata_rna.shape}")
    print(f"Loaded protein: {adata_protein.shape}")

    results = {}
    for size_folder in SIZE_FOLDERS:
        try:
            he_path = HE_TEMPLATE.format(size_folder=size_folder)
            if not os.path.exists(he_path):
                print(f"\nH&E file not found — skipping: {he_path}")
                continue

            adata_he = ad.read_h5ad(he_path)
            print(f"\nLoaded H&E ({size_folder}): {adata_he.shape}")

            mdata, unm_rna, unm_prot, unm_he = match_multimodal(
                adata_rna, adata_protein, adata_he, size_folder)

            mu_path   = os.path.join(OUTPUT_DIR,
                         MUDATA_TEMPLATE.format(size_folder=size_folder))
            rna_path  = os.path.join(OUTPUT_DIR,
                         UNMATCHED_RNA_TEMPLATE.format(size_folder=size_folder))
            prot_path = os.path.join(OUTPUT_DIR,
                         UNMATCHED_PROTEIN_TEMPLATE.format(
                             size_folder=size_folder))
            he_path2  = os.path.join(OUTPUT_DIR,
                         UNMATCHED_HE_TEMPLATE.format(size_folder=size_folder))

            mdata.write(mu_path)
            print(f"  Saved MuData    → {mu_path}")
            unm_rna.to_csv(rna_path,   index=False)
            unm_prot.to_csv(prot_path, index=False)
            unm_he.to_csv(he_path2,    index=False)

            results[size_folder] = {
                "matched":          mdata.n_obs,
                "unmatched_rna":    len(unm_rna),
                "unmatched_protein": len(unm_prot),
                "unmatched_he":     len(unm_he),
            }
        except Exception as exc:
            import traceback
            print(f"\nERROR processing {size_folder}: {exc}")
            traceback.print_exc()

    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")
    if results:
        fmt = "{:<16} {:>10} {:>14} {:>14} {:>12}"
        print(fmt.format("Size Folder", "Matched",
                         "Unmatch RNA", "Unmatch Prot", "Unmatch HE"))
        print("-" * 68)
        for sf in SIZE_FOLDERS:
            if sf in results:
                s = results[sf]
                print(fmt.format(sf, s["matched"], s["unmatched_rna"],
                                 s["unmatched_protein"], s["unmatched_he"]))
    failed = set(SIZE_FOLDERS) - set(results)
    if failed:
        print(f"\nFailed: {sorted(failed)}")
    print("=" * 80)
