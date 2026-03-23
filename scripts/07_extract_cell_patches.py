# env: g4x_prepro
"""
Extract per-cell H&E image patches from the whole-slide H&E image and
segmentation mask.

For each cell present in the preprocessed RNA AnnData, a square patch
centred on the cell is cropped from the H&E JPEG-2000 image at multiple
fixed sizes (64, 128, 256, 448, 512 px) plus the original bounding-box
size. All patches are saved as deflate-compressed TIFF files.

Algorithm
---------
1. Compute every cell's bounding box in a single O(H×W) scan with
   ``scipy.ndimage.find_objects`` — avoids repeated full-image lookups.
2. Spawn a multiprocessing pool (fork context) so workers share the
   read-only numpy arrays without copying.
3. Each worker crops the patch for one cell at all requested sizes and
   writes only the files that do not already exist (checkpoint-resume).

Inputs  (from Steps 02 and 05)
------
  data/processed/segmentation_mask.tiff   — integer label mask (0 = background)
  data/h_and_e.jp2                        — whole-slide H&E image (JPEG 2000)
  results/tonsil_rep1_rna_processed.h5ad  — preprocessed RNA AnnData (for cell list)

Outputs
-------
  results/cell_patches_tiff/original_size/<cell_id>.tiff
  results/cell_patches_tiff/fixed_64/<cell_id>.tiff
  results/cell_patches_tiff/fixed_128/<cell_id>.tiff
  results/cell_patches_tiff/fixed_256/<cell_id>.tiff
  results/cell_patches_tiff/fixed_448/<cell_id>.tiff
  results/cell_patches_tiff/fixed_512/<cell_id>.tiff
"""

import os
import numpy as np
from PIL import Image
import tifffile
import anndata as ad
import multiprocessing
import scipy.ndimage
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

# ─────────────────────────────────────────────
# Paths  (run from repository root)
# ─────────────────────────────────────────────
MASK_PATH    = os.path.join("data", "processed", "segmentation_mask.tiff")
HE_PATH      = os.path.join("data", "h_and_e.jp2")
RNA_PATH     = os.path.join("results", "tonsil_rep1_rna_processed.h5ad")
OUTPUT_DIR   = os.path.join("results", "cell_patches_tiff")

PADDING      = 20
TARGET_SIZES = [64, 128, 256, 448, 512]
SAVE_ORIGINAL = True
NUM_WORKERS  = 8   # adjust to available CPU cores

# ─────────────────────────────────────────────
# Worker globals (shared via fork, zero-copy)
# ─────────────────────────────────────────────
_nuclei      = None
_he_array    = None
_bboxes      = None
_output_dirs = None
_target_sizes = None
_save_original = None
_padding     = None


def _init_worker(nuclei, he_array, bboxes, output_dirs, target_sizes,
                 save_original, padding):
    global _nuclei, _he_array, _bboxes, _output_dirs
    global _target_sizes, _save_original, _padding
    _nuclei       = nuclei
    _he_array     = he_array
    _bboxes       = bboxes
    _output_dirs  = output_dirs
    _target_sizes = target_sizes
    _save_original = save_original
    _padding      = padding


def _resize_and_center(patch, target_size=128, bg=255):
    """Crop centre or pad to *target_size × target_size*."""
    if isinstance(target_size, int):
        th, tw = target_size, target_size
    else:
        th, tw = target_size
    h, w = patch.shape[:2]
    if h == th and w == tw:
        return patch
    if h > th or w > tw:
        cy, cx = h // 2, w // 2
        y0 = cy - th // 2
        y1 = y0 + th
        x0 = cx - tw // 2
        x1 = x0 + tw
        return patch[y0:y1, x0:x1]
    result = np.full((th, tw, patch.shape[2]), bg, dtype=patch.dtype)
    oy = (th - h) // 2
    ox = (tw - w) // 2
    result[oy:oy + h, ox:ox + w] = patch
    return result


def _process_cell(cell_id):
    """Crop and save patches for one cell at every requested size."""
    bb = _bboxes.get(cell_id)
    if bb is None:
        return
    bb_y0, bb_y1, bb_x0, bb_x1 = bb
    H, W = _nuclei.shape
    cy = (bb_y0 + bb_y1) // 2
    cx = (bb_x0 + bb_x1) // 2

    if _save_original:
        out = os.path.join(_output_dirs["original"], f"{cell_id}.tiff")
        if not os.path.exists(out):
            y0 = max(0, bb_y0 - _padding)
            y1 = min(H, bb_y1 + _padding + 1)
            x0 = max(0, bb_x0 - _padding)
            x1 = min(W, bb_x1 + _padding + 1)
            Image.fromarray(_he_array[y0:y1, x0:x1]).save(
                out, compression="tiff_deflate")

    for size in _target_sizes:
        out = os.path.join(_output_dirs[size], f"{cell_id}.tiff")
        if os.path.exists(out):
            continue
        half = size // 2
        y0 = max(0, cy - half)
        y1 = min(H, cy + half + (size % 2))
        x0 = max(0, cx - half)
        x1 = min(W, cx + half + (size % 2))
        patch = _he_array[y0:y1, x0:x1]
        if patch.shape[0] != size or patch.shape[1] != size:
            patch = _resize_and_center(patch, target_size=size)
        Image.fromarray(patch).save(out, compression="tiff_deflate")


def extract_and_save_cells(nuclei, he_array, adata, output_base_dir,
                           padding=20, target_sizes=None,
                           save_original=True, num_workers=8):
    if target_sizes is None:
        target_sizes = [64, 128, 256, 448, 512]

    # Create output directories
    dirs = {}
    if save_original:
        d = os.path.join(output_base_dir, "original_size")
        os.makedirs(d, exist_ok=True)
        dirs["original"] = d
    for sz in target_sizes:
        d = os.path.join(output_base_dir, f"fixed_{sz}")
        os.makedirs(d, exist_ok=True)
        dirs[sz] = d

    # Cell IDs from AnnData obs index (integer strings like "117")
    print("\n=== Parsing cell IDs from AnnData ===")
    cell_ids_in_adata = set()
    for cell_str in adata.obs.index.astype(str):
        try:
            cell_ids_in_adata.add(int(cell_str))
        except ValueError as e:
            print(f"Warning: cannot parse cell_id '{cell_str}': {e}")
    print(f"  {len(cell_ids_in_adata)} cell IDs loaded from AnnData")

    # Single-pass bounding-box computation
    print("\nPre-computing bounding boxes (single mask scan)...")
    slices = scipy.ndimage.find_objects(nuclei)
    bboxes = {}
    for idx, sl in enumerate(slices):
        if sl is None:
            continue
        cid = idx + 1
        if cid not in cell_ids_in_adata:
            continue
        y_sl, x_sl = sl
        bboxes[cid] = (y_sl.start, y_sl.stop, x_sl.start, x_sl.stop)

    cell_ids = list(bboxes.keys())
    print(f"  Matched {len(cell_ids)} cells")
    if not cell_ids:
        print("No matching cells found — check that cell IDs in the mask "
              "align with those in the AnnData index.")
        return

    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(nuclei, he_array, bboxes, dirs, target_sizes,
                  save_original, padding),
    ) as pool:
        list(tqdm(
            pool.imap_unordered(_process_cell, cell_ids, chunksize=20),
            total=len(cell_ids),
            desc="Extracting cell patches",
        ))

    print(f"\nDone — {len(cell_ids)} cells saved to:")
    for name, path in dirs.items():
        print(f"  {name}: {path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading RNA AnnData...")
    adata = ad.read_h5ad(RNA_PATH)
    print(f"  {adata.n_obs} cells")

    print("Loading segmentation mask...")
    nuclei = tifffile.imread(MASK_PATH)

    print("Loading H&E image...")
    he_array = np.array(Image.open(HE_PATH))

    extract_and_save_cells(
        nuclei,
        he_array,
        adata,
        OUTPUT_DIR,
        padding=PADDING,
        target_sizes=TARGET_SIZES,
        save_original=SAVE_ORIGINAL,
        num_workers=NUM_WORKERS,
    )
