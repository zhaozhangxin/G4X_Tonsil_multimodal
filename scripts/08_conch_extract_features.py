# env: conch
"""
Extract CONCH visual features from per-cell H&E patches.

For each size subfolder produced by Step 07, every TIFF patch is passed
through the CONCH ViT-B/16 image encoder (without contrast projection /
normalisation) and the raw 512-dimensional embedding is saved to a CSV.

CONCH preprocessing applied to each patch:
  1. Resize so the shorter side is 448 px (BICUBIC interpolation).
  2. Centre-crop to 448 × 448.
  3. Convert to RGB.
  4. Normalise with ImageNet mean/std.

Model checkpoint
----------------
Download the CONCH weights from the official release:
  https://huggingface.co/MahmoodLab/CONCH

Place (or symlink) the file at:
  data/CONCH/checkpoints/conch/pytorch_model.bin

Install the CONCH package (in the `conch` conda environment):
  pip install git+https://github.com/mahmoodlab/CONCH.git

Inputs  (from Step 07)
------
  results/cell_patches_tiff/{fixed_64,fixed_128,...,original_size}/*.tiff

Outputs
-------
  results/conch_features/conch_features_tonsil_{size_folder}.csv
  results/conch_features/conch_metadata_tonsil_{size_folder}.csv

References
----------
Lu, M. Y. et al. (2024). A visual-language foundation model for
computational pathology. *Nature Medicine*, 30, 863–874.
https://doi.org/10.1038/s41591-024-02856-4
"""

import os
import torch
from pathlib import Path
from PIL import Image
import pandas as pd
from tqdm import tqdm
from conch.open_clip_custom import create_model_from_pretrained

# ─────────────────────────────────────────────
# Paths  (run from repository root)
# ─────────────────────────────────────────────
BASE_FOLDER      = os.path.join("results", "cell_patches_tiff")
CHECKPOINT_PATH  = os.path.join("data", "CONCH", "checkpoints", "conch",
                                 "pytorch_model.bin")
OUTPUT_DIR       = os.path.join("results", "conch_features")
GPU_ID           = 0   # change to select a different GPU

# Size subfolders to process
SIZE_FOLDERS = ["fixed_64", "fixed_128", "fixed_256", "fixed_448",
                "fixed_512", "original_size"]

# ─────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading CONCH model...")
device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
model, preprocess = create_model_from_pretrained(
    "conch_ViT-B-16", checkpoint_path=CHECKPOINT_PATH)
model = model.to(device)
model.eval()
print(f"Model loaded on {device}")

# ─────────────────────────────────────────────
# Feature extraction loop
# ─────────────────────────────────────────────
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

for size_folder in SIZE_FOLDERS:
    print(f"\n{'#' * 80}")
    print(f"# Processing: {size_folder}")
    print(f"{'#' * 80}")

    image_folder = os.path.join(BASE_FOLDER, size_folder)
    if not os.path.isdir(image_folder):
        print(f"  Skipping — folder not found: {image_folder}")
        continue

    out_csv_features = os.path.join(
        OUTPUT_DIR, f"conch_features_tonsil_{size_folder}.csv")
    out_csv_metadata = os.path.join(
        OUTPUT_DIR, f"conch_metadata_tonsil_{size_folder}.csv")

    image_files = sorted(
        f for f in os.listdir(image_folder)
        if f.lower().endswith(IMAGE_EXTS))

    if not image_files:
        print(f"  No images found in {image_folder}")
        continue
    print(f"  Found {len(image_files)} images")

    batch_features = []
    batch_metadata = []
    total_images = 0

    for img_file in tqdm(image_files,
                         desc=f"  Extracting features [{size_folder}]"):
        img_path = os.path.join(image_folder, img_file)
        cell_id  = Path(img_file).stem
        try:
            image = Image.open(img_path).convert("RGB")
            orig_size = image.size  # (width, height)
            img_tensor = preprocess(image).unsqueeze(0).to(device)
            with torch.inference_mode():
                emb = model.encode_image(
                    img_tensor, proj_contrast=False, normalize=False)
            feat = emb.cpu().numpy().squeeze()  # shape (512,)
            if total_images == 0:
                print(f"\n  Feature dimension: {feat.shape}")
            batch_features.append({
                "Cell_id": cell_id,
                "filename": img_file,
                **{f"feature_{i}": feat[i] for i in range(len(feat))},
            })
            batch_metadata.append({
                "Cell_id": cell_id,
                "filename": img_file,
                "original_width":  orig_size[0],
                "original_height": orig_size[1],
            })
            total_images += 1
        except Exception as exc:
            print(f"\n  Error processing {img_file}: {exc}")

    pd.DataFrame(batch_features).to_csv(out_csv_features, index=False)
    pd.DataFrame(batch_metadata).to_csv(out_csv_metadata, index=False)

    print(f"\n  Saved {total_images} features → {out_csv_features}")
    print(f"  Saved metadata           → {out_csv_metadata}")

print(f"\n{'#' * 80}")
print("All size folders completed!")
print(f"{'#' * 80}")
