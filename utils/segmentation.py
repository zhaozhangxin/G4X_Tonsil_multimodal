"""
segmentation.py
===============
Mesmer-based whole-cell segmentation utilities for the G4X preprocessing
pipeline.

Key functions
-------------
preprocess_marker          : Quantile clip, optional Otsu threshold, and
                             intensity rescale a single marker image.
construct_channel          : Combine one or more markers into a single channel
                             by averaging preprocessed images.
segmentation_mesmer        : Run Mesmer on an image stack, with optional tiled
                             inference for images that exceed GPU memory.
_run_mesmer_tiled          : Internal helper — splits a large image into
                             overlapping tiles, runs Mesmer on each, and
                             stitches the results.
extract_cell_features      : Extract per-cell area, centroid, and marker
                             intensities from a segmentation mask.
run_segmentation_mesmer_cell         : End-to-end whole-cell segmentation
                                       pipeline with logging and output saving.
run_segmentation_mesmer_compartments : End-to-end compartment segmentation
                                       (cell, nuclear, membrane) pipeline.
generate_segmentation_mask_geojson_cell        : Convert whole-cell mask to GeoJSON.
generate_segmentation_mask_geojson_compartments: Convert compartment masks to GeoJSON.
"""

# %%
import json
import logging
import re
import time
from pathlib import Path
from typing import Union

import cv2 as cv
import numpy as np
import pandas as pd
import skimage.measure
import tifffile
from pyqupath.geojson import mask_to_geojson_joblib
from pyqupath.tiff import PyramidWriter, TiffZarrReader
from skimage.exposure import rescale_intensity

from io_setup import setup_logging

###############################################################################
# segmentation
###############################################################################


def preprocess_marker(
    image: np.ndarray,
    thresh_q_min: float = 0,
    thresh_q_max: float = 1,
    thresh_otsu: bool = False,
    scale: bool = True,
) -> np.ndarray:
    """
    Helper function to preprocess a single marker image.

    Parameters
    ----------
    image : np.ndarray
        Image to be preprocessed.
    thresh_q_min : float, optional
        Lower quantile to cut at. Values below this quantile will be set to 0.
        Defaults to 0.
    thresh_q_max : float, optional
        Upper quantile to cut at. Values above this quantile will be set to
        the quantile value. Defaults to 1.
    thresh_otsu: bool, optional
        Whether to perform OTSU thresholding to the image or not. Defaults
        to False.
    scale : bool, optional
        Whether to scale the image or not. Defaults to True.

    Returns
    -------
    np.ndarray
        Preprocessed image.
    """

    if thresh_q_min != 0 or thresh_q_max != 1:
        value_q_min = np.quantile(image, thresh_q_min)
        value_q_max = np.quantile(image, thresh_q_max)
        image = np.where(image < value_q_min, 0, image)
        image = np.where(image > value_q_max, value_q_max, image)
    if thresh_otsu:
        min_type = np.min_scalar_type(np.max(image).astype(np.int_))
        _, mask_otsu = cv.threshold(
            image.astype(min_type), 0, 1, cv.THRESH_BINARY + cv.THRESH_OTSU
        )
    if scale:
        image = rescale_intensity(image, out_range=(0, 1))
    return image if not thresh_otsu else image * mask_otsu


def construct_channel(
    marker_list: list[str],
    marker_dict: dict[str : np.ndarray],
    thresh_q_min: float = 0,
    thresh_q_max: float = 1,
    thresh_otsu: bool = False,
    scale: bool = True,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Construct a single channel by combining single or multiple channels.

    Parameters
    ----------
    marker_list : list[str]
        List of marker names to be scaled and summed.
    marker_dict : dict
        Dictionary containing marker names as keys and corresponding images as
        values.
    thresh_q_min : float, optional
        Lower quantile to cut at. Values below this quantile will be set to 0.
        Defaults to 0.
    thresh_q_max : float, optional
        Upper quantile to cut at. Values above this quantile will be set to
        the quantile value. Defaults to 1.
    thresh_otsu: bool, optional
        Whether to perform OTSU thresholding to the image or not. Defaults
        to False.
    scale : bool, optional
        Whether to scale the image or not. Defaults to True.

    Returns
    -------
    tuple[np.ndarray, dict[str : np.ndarray]]
        - Single channel image after preprocessing.
        - Dictionary of preprocessed images for each channel in `marker_list`.
    """

    # Check if any marker has constant value
    constant = [
        np.min(marker_dict[marker]) == np.max(marker_dict[marker])
        for marker in marker_list
    ]
    if any(constant):
        logging.warning(
            f"Marker with constant value: {np.array(marker_list)[constant].tolist()}"
        )

    # Construct channel of the specified markers
    image_dict = {
        marker: preprocess_marker(
            marker_dict[marker],
            thresh_q_min=thresh_q_min,
            thresh_q_max=thresh_q_max,
            thresh_otsu=thresh_otsu,
            scale=scale,
        )
        for marker in marker_list
    }
    image_channel = np.mean([image for image in image_dict.values()], axis=0)
    return image_channel, image_dict


def _run_mesmer_tiled(
    image_stack: np.ndarray,
    pixel_size_um: float,
    maxima_threshold: float,
    interior_threshold: float,
    compartment: str,
    tile_size: int = 2000,
    overlap: int = 200,
) -> np.ndarray:
    """
    Run Mesmer on a large image by splitting it into overlapping tiles.

    Each tile is processed independently. The overlap region is discarded —
    only the inner (non-overlapping) portion of each tile contributes to the
    output. Cell labels are renumbered across tiles to avoid conflicts.

    Parameters
    ----------
    image_stack : np.ndarray
        Input array of shape (1, H, W, 2).
    pixel_size_um : float
        Pixel size in micrometers.
    maxima_threshold : float
        Mesmer maxima threshold.
    interior_threshold : float
        Mesmer interior threshold.
    compartment : str
        "whole-cell", "nuclear", or "both".
    tile_size : int
        Tile width/height in pixels (before Mesmer's internal rescaling).
    overlap : int
        Overlap between adjacent tiles in pixels. Cells within the overlap
        border are assigned to the neighbouring tile, so overlap should be
        larger than the expected maximum cell diameter.

    Returns
    -------
    np.ndarray
        Stitched segmentation mask of shape (1, H, W, n_outputs).
    """
    from deepcell.applications import Mesmer

    _, H, W, _ = image_stack.shape
    n_outputs = 1 if compartment != "both" else 2
    segmentation_mask = np.zeros((1, H, W, n_outputs), dtype=np.int32)
    label_offset = 0

    mesmer = Mesmer()
    step = tile_size - 2 * overlap
    y_starts = list(range(0, H, step))
    x_starts = list(range(0, W, step))
    n_tiles = len(y_starts) * len(x_starts)
    logging.info(
        f"Tiled segmentation: {len(y_starts)}x{len(x_starts)} = {n_tiles} tiles "
        f"(tile_size={tile_size}, overlap={overlap})"
    )

    for ti, y0 in enumerate(y_starts):
        for tj, x0 in enumerate(x_starts):
            y1 = min(y0 + tile_size, H)
            x1 = min(x0 + tile_size, W)
            tile = image_stack[:, y0:y1, x0:x1, :]

            tile_mask = mesmer.predict(
                tile,
                image_mpp=pixel_size_um,
                batch_size=1,
                postprocess_kwargs_whole_cell={
                    "maxima_threshold": maxima_threshold,
                    "interior_threshold": interior_threshold,
                },
                compartment=compartment,
            )

            # Inner region: exclude overlap borders (keeps cells fully inside)
            iy0 = overlap if y0 > 0 else 0
            ix0 = overlap if x0 > 0 else 0
            iy1 = (y1 - y0) - (overlap if y1 < H else 0)
            ix1 = (x1 - x0) - (overlap if x1 < W else 0)

            inner = tile_mask[:, iy0:iy1, ix0:ix1, :].copy()
            inner[inner > 0] += label_offset
            label_offset = int(inner.max())

            out_y0, out_y1 = y0 + iy0, y0 + iy1
            out_x0, out_x1 = x0 + ix0, x0 + ix1
            segmentation_mask[:, out_y0:out_y1, out_x0:out_x1, :] = inner

            logging.info(
                f"  Tile ({ti},{tj}) done -- y={y0}:{y1}, x={x0}:{x1}, "
                f"labels so far: {label_offset}"
            )

    return segmentation_mask


def segmentation_mesmer(
    marker_dict: dict[str : np.ndarray],
    internal_markers: list[str],
    boundary_markers: list[str],
    thresh_q_min: float,
    thresh_q_max: float,
    thresh_otsu: bool,
    scale: bool,
    pixel_size_um: float,
    maxima_threshold: float = 0.075,
    interior_threshold: float = 0.20,
    compartment="whole-cell",
    tile_size: int = None,
    overlap: int = 300,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]
]:
    """
    Perform segmentation (Mesmer) on a given image.

    Parameters
    ----------
    marker_dict : dict
        Dictionary containing marker names as keys and corresponding images as
        values.
    boundary_markers : list
        List of boundary marker names.
    internal_markers : list
        List of internal marker names.
    thresh_q_min : float, optional
        Lower quantile to cut at for each marker in `internal_markers` and
        `boundary_markers`. Values below this quantile will be set to 0.
    thresh_q_max : float, optional
        Upper quantile to cut at for each marker in `internal_markers` and
        `boundary_markers`. Values above this quantile will be set to the
        quantile value.
    thresh_otsu: bool, optional
        Whether to perform OTSU thresholding for each marker in `internal_markers`
        and `boundary_markers`. Values below the OTSU threshold will be set to 0.
    scale : bool, optional
        Whether to scale each marker in `internal_markers` and `boundary_markers`
        before summing.
    pixel_size_um : float
        Pixel size in micrometers for marker images.
        Note:
        - Fusion: 0.5068164319979996
        - Keyence: 0.3775202
    maxima_threshold : float, optional
        Maxima threshold for Mesmer. Lower values will result in more separate
        cells being predicted, whereas higher values will result in fewer cells.
        Defaults to 0.075.
    interior_threshold : float, optional
        Interior threshold for Mesmer. Lower values will result in larger cells,
        whereas higher values will result in smaller cells. Defaults to 0.20.
    compartment : str, optional
        Specify type of segmentation to predict. Must be one of "whole-cell",
        "nuclear", "both". Defaults to "whole-cell".

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]
        - Segmentation mask.
        - Single internal channel image.
        - Single boundary channel image.
        -
    """
    # Data for Mesmer
    internal_channel, internal_dict = construct_channel(
        marker_list=internal_markers,
        marker_dict=marker_dict,
        thresh_q_min=thresh_q_min,
        thresh_q_max=thresh_q_max,
        thresh_otsu=thresh_otsu,
        scale=scale,
    )
    internal_channel = rescale_intensity(internal_channel, out_range=(0, 1))
    boundary_channel, boundary_dict = construct_channel(
        marker_list=boundary_markers,
        marker_dict=marker_dict,
        thresh_q_min=thresh_q_min,
        thresh_q_max=thresh_q_max,
        thresh_otsu=thresh_otsu,
        scale=scale,
    )
    boundary_channel = rescale_intensity(boundary_channel, out_range=(0, 1))
    image_stack = np.stack((internal_channel, boundary_channel), axis=-1).astype(np.float32)
    image_stack = np.expand_dims(image_stack, 0)

    # Do segmentation
    _, H, W, _ = image_stack.shape
    if tile_size is not None and (H > tile_size or W > tile_size):
        segmentation_mask = _run_mesmer_tiled(
            image_stack=image_stack,
            pixel_size_um=pixel_size_um,
            maxima_threshold=maxima_threshold,
            interior_threshold=interior_threshold,
            compartment=compartment,
            tile_size=tile_size,
            overlap=overlap,
        )
    else:
        from deepcell.applications import Mesmer
        mesmer = Mesmer()
        segmentation_mask = mesmer.predict(
            image_stack,
            image_mpp=pixel_size_um,
            batch_size=1,
            postprocess_kwargs_whole_cell={
                "maxima_threshold": maxima_threshold,
                "interior_threshold": interior_threshold,
            },
            compartment=compartment,
        )
    return (
        segmentation_mask,
        internal_channel,
        boundary_channel,
        internal_dict,
        boundary_dict,
    )


def extract_cell_features(
    marker_dict: dict[str, np.ndarray],
    segmentation_mask: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract single cell features from segmentation mask.

    Parameters
    ----------
    marker_dict : dict
        Dictionary containing marker names as keys and corresponding images as
        values.
    segmentation_mask : np.ndarray
        A 2D segmentation mask with the same shape as the marker images, in
        which each cell is labeled with a unique integer.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - Dataframe containing single-cell features.
        - Dataframe containing single-cell features with marker intensities
          scaled by cell size.
    """
    marker_name = [marker for marker in marker_dict.keys()]
    marker_array = np.stack([marker_dict[marker] for marker in marker_name], axis=2)

    # extract properties
    props = skimage.measure.regionprops_table(
        segmentation_mask,
        properties=["label", "area", "centroid"],
    )
    props_df = pd.DataFrame(props)
    props_df.columns = ["cellLabel", "cellSize", "Y_cent", "X_cent"]

    # extract marker intensity
    stats = skimage.measure.regionprops(segmentation_mask)
    n_cell = len(stats)
    n_marker = len(marker_name)
    sums = np.zeros((n_cell, n_marker))
    avgs = np.zeros((n_cell, n_marker))
    for i, region in enumerate(stats):
        # Extract the pixel values for the current region from the marker_array
        label_counts = [marker_array[coord[0], coord[1], :] for coord in region.coords]
        sums[i] = np.sum(label_counts, axis=0)  # Sum of marker intensities
        avgs[i] = sums[i] / region.area  # Average intensity per unit area

    sums_df = pd.DataFrame(sums, columns=marker_name)
    avgs_df = pd.DataFrame(avgs, columns=marker_name)
    data = pd.concat([props_df, sums_df], axis=1)
    data_scale_size = pd.concat([props_df, avgs_df], axis=1)
    return data, data_scale_size


def _save_marker_dict_for_segmentation(
    segmentation_dir: Union[Path, str],
    internal_channel: np.ndarray,
    boundary_channel: np.ndarray,
    internal_dict: dict[str, np.ndarray],
    boundary_dict: dict[str, np.ndarray],
    scale: bool,
    dtype: np.dtype,
    num_threads: int,
):
    """
    Create a dictionary containing preprocessed marker images for segmentation.

    Parameters
    ----------
    segmentation_dir : Union[Path, str]
        Directory to save the segmentation markers.
    internal_channel : np.ndarray
        Summed/combined internal marker image after preprocessing
    boundary_channel : np.ndarray
        Summed/combined boundary marker image after preprocessing
    internal_dict : dict[str, np.ndarray]
        Individual preprocessed internal marker images
    boundary_dict : dict[str, np.ndarray]
        Individual preprocessed boundary marker images
    scale : bool
        Whether the images were scaled during preprocessing
    dtype : np.dtype
        Output data type for marker images
    num_threads : int
        The number of threads to use for writing the OME-TIFF file.
    """
    segmentation_markers_dict = {}
    segmentation_markers_dict.update(internal_dict)
    segmentation_markers_dict.update(boundary_dict)
    if scale:
        segmentation_markers_dict["internal_sum"] = internal_channel
        segmentation_markers_dict["boundary_sum"] = boundary_channel
        segmentation_markers_dict = {
            marker: (image * np.iinfo(dtype).max).astype(dtype)
            for marker, image in segmentation_markers_dict.items()
        }
    else:
        segmentation_markers_dict["internal_sum"] = (
            internal_channel * np.iinfo(dtype).max
        )
        segmentation_markers_dict["boundary_sum"] = (
            boundary_channel * np.iinfo(dtype).max
        )
        segmentation_markers_dict = {
            marker: image.astype(dtype)
            for marker, image in segmentation_markers_dict.items()
        }

    segmentation_markers_f = segmentation_dir / "segmentation_markers.ome.tiff"
    tiff_writer = PyramidWriter.from_dict(segmentation_markers_dict)
    tiff_writer.export_ometiff_pyramid(
        segmentation_markers_f, overwrite=True, num_threads=num_threads
    )


def _get_marker_dict(
    unit_dir: Union[Path, str],
    ometiff_path: Union[Path, str, None] = None,
    marker_dict: dict[str, np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """
    Get marker dictionary from OME-TIFF file in the specified directory.
    If `ometiff_path` is not provided, it will search for the only OME-TIFF file
    in the `unit_dir`.

    Parameters
    ----------
    unit_dir : Union[Path, str]
        Directory to search for the OME-TIFF file.
    ometiff_path : Union[Path, str, None], optional
        Path to the OME-TIFF file. If None, it will search for the only
        OME-TIFF file in `unit_dir`. Defaults to None.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing marker names as keys and corresponding images as
        values.
    """
    if marker_dict is not None:
        return marker_dict

    if ometiff_path is None:
        pattern = re.compile(r".*\.ome\.tif[f]?", re.IGNORECASE)
        ometiff_paths = [f for f in unit_dir.glob("*") if pattern.match(f.name)]
        if len(ometiff_paths) == 0:
            logging.error("No OME-TIFF file found in the directory.")
            raise FileNotFoundError("No OME-TIFF file found in the directory.")
        elif len(ometiff_paths) > 1:
            logging.error("Multiple OME-TIFF files found in the directory.")
            raise ValueError("Multiple OME-TIFF files found in the directory.")
        else:
            ometiff_path = ometiff_paths[0]
    else:
        ometiff_path = Path(ometiff_path)
    tiff_reader = TiffZarrReader.from_ometiff(ometiff_path)
    marker_dict = tiff_reader.zimg_dict

    return marker_dict


def run_segmentation_mesmer_cell(
    unit_dir: str,
    internal_markers: list[str],
    boundary_markers: list[str],
    thresh_q_min: float,
    thresh_q_max: float,
    thresh_otsu: bool,
    scale: bool,
    pixel_size_um: float,
    maxima_threshold: float = 0.075,
    interior_threshold: float = 0.20,
    tag: str = None,
    ometiff_path: str = None,
    marker_dict: dict[str, np.ndarray] = None,
    num_threads: int = 8,
    tile_size: int = 2000,
    overlap: int = 200,
):
    """
    Run whole-cell segmentation using Mesmer.

    Parameters
    ----------
    unit_dir : str
        Directory to load and save data for segmentation.
    internal_markers : list
        List of internal marker names.
    boundary_markers : list
        List of boundary marker names.
    thresh_q_min : float, optional
        Lower quantile to cut at for each marker in `internal_markers` and
        `boundary_markers`. Values below this quantile will be set to 0.
    thresh_q_max : float, optional
        Upper quantile to cut at for each marker in `internal_markers` and
        `boundary_markers`. Values above this quantile will be set to the
        quantile value.
    thresh_otsu: bool, optional
        Whether to perform OTSU thresholding for each marker in `internal_markers`
        and `boundary_markers`. Values below the OTSU threshold will be set to 0.
    scale : bool, optional
        Whether to scale each marker in `internal_markers` and `boundary_markers`
        before constructing into a single channel.
    pixel_size_um : float
        Pixel size in micrometers for marker images.
        Note:
        - Fusion: 0.5068164319979996
        - Keyence: 0.3775202
    maxima_threshold : float, optional
        Maxima threshold for Mesmer. Lower values will result in more separate
        cells being predicted, whereas higher values will result in fewer cells.
        Defaults to 0.075.
    interior_threshold : float, optional
        Interior threshold for Mesmer. Lower values will result in larger cells,
        whereas higher values will result in smaller cells. Defaults to 0.20.
    tag : str, optional
        Tag for the segmentation directory. Defaults to None, using time as tag
        (YYYYMMDD_HHMMSS).
    ometiff_path : str, optional
        Path to the OME-TIFF file containing the marker images for segmentation.
        Defaults to None, using the only OME-TIFF file in `unit_dir`.
    marker_dict : dict[str, np.ndarray], optional
        Dictionary containing marker names as keys and corresponding images as
        values. If provided, it will be used directly instead of loading from
        `ometiff_path` or searching in `unit_dir`. Defaults to None.
    num_threads : int, optional
        The number of threads to use for writing the OME-TIFF file. Defaults to 8.
    """
    # Set up directories
    unit_dir = Path(unit_dir)
    if tag is None:
        tag = time.strftime("%Y%m%d_%H%M%S")
    segmentation_dir = unit_dir / tag
    segmentation_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    setup_logging(segmentation_dir / "segmentation.log")

    # Load OME-TIFF file
    marker_dict = _get_marker_dict(
        unit_dir=unit_dir, ometiff_path=ometiff_path, marker_dict=marker_dict
    )
    logging.info(f"OME-TIFF file loaded: {ometiff_path}.")

    # Check whether selected markers are present in the OME-TIFF file
    markers = list(marker_dict.keys())
    missing_markers = [
        marker
        for marker in boundary_markers + internal_markers
        if marker not in markers
    ]
    if len(missing_markers) > 0:
        logging.error(f"Missing markers: {missing_markers}")
        raise ValueError(f"Missing markers: {missing_markers}")

    # Write parameters
    params = {
        "internal_markers": internal_markers,
        "boundary_markers": boundary_markers,
        "thresh_q_min": thresh_q_min,
        "thresh_q_max": thresh_q_max,
        "thresh_otsu": thresh_otsu,
        "scale": scale,
        "pixel_size_um": pixel_size_um,
        "maxima_threshold": maxima_threshold,
        "interior_threshold": interior_threshold,
        "compartment": "whole-cell",
    }
    with open(
        f"{segmentation_dir}/parameter_segmentation.json", "w", encoding="utf-8"
    ) as file:
        json.dump(params, file, indent=4, ensure_ascii=False)

    # Segmentation
    (
        segmentation_mask,
        internal_channel,
        boundary_channel,
        internal_dict,
        boundary_dict,
    ) = segmentation_mesmer(
        marker_dict=marker_dict, **params, tile_size=tile_size, overlap=overlap
    )
    segmentation_mask = segmentation_mask[0, :, :, 0]
    segmentation_mask_f = segmentation_dir / "segmentation_mask.tiff"
    tifffile.imwrite(str(segmentation_mask_f), segmentation_mask)
    logging.info("Segmentation completed.")

    # Save markers for segmentation, boundary and internal channels
    dtype = next(iter(marker_dict.values())).dtype
    _save_marker_dict_for_segmentation(
        segmentation_dir=segmentation_dir,
        internal_channel=internal_channel,
        boundary_channel=boundary_channel,
        internal_dict=internal_dict,
        boundary_dict=boundary_dict,
        scale=scale,
        dtype=dtype,
        num_threads=num_threads,
    )
    logging.info("Markers used for segmentation saved as OME-TIFF.")

    # Extract single-cell features
    data, data_scale = extract_cell_features(marker_dict, segmentation_mask)
    data.to_csv(segmentation_dir / "data.csv")
    data_scale.to_csv(segmentation_dir / "dataScaleSize.csv")
    logging.info("Single-cell features extracted.")


def generate_segmentation_mask_geojson_cell(
    unit_dir: str,
    tag: str,
    n_jobs: int = 10,
    batch_size: int = 10,
):
    """
    Generate GeoJSON file for segmentation mask.

    Parameters
    ----------
    unit_dir : str
        Directory to load and save data for segmentation.
    tag : str
        Tag for the segmentation directory, where the segmentation mask will
        be used to generate the GeoJSON file.
    n_jobs : int, optional
        The number of parallel workers (CPU cores or threads) are spawned to
        process the tasks. Default is 10.
    batch_size : int, optional
        The number of labels to process in each batch. Default is 10.
    """
    unit_dir = Path(unit_dir)
    segmentation_dir = unit_dir / tag

    segmentation_mask_f = segmentation_dir / "segmentation_mask.tiff"
    segmentation_mask = tifffile.imread(segmentation_mask_f)
    mask_to_geojson_joblib(
        segmentation_mask,
        segmentation_dir / "segmentation_mask.geojson",
        n_jobs=n_jobs,
        batch_size=batch_size,
    )


def run_segmentation_mesmer_compartments(
    unit_dir: str,
    internal_markers: list[str],
    boundary_markers: list[str],
    thresh_q_min: float,
    thresh_q_max: float,
    thresh_otsu: bool,
    scale: bool,
    pixel_size_um: float,
    maxima_threshold: float = 0.075,
    interior_threshold: float = 0.20,
    tag: str = None,
    ometiff_path: str = None,
    marker_dict: dict[str, np.ndarray] = None,
    num_threads: int = 8,
    tile_size: int = 2000,
    overlap: int = 200,
):
    """
    Run segmentation using Mesmer to separate cell compartments (cell, nuclear,
    and membrane).

    Parameters
    ----------
    unit_dir : str
        Directory to load and save data for segmentation.
    internal_markers : list
        List of internal marker names.
    boundary_markers : list
        List of boundary marker names.
    thresh_q_min : float, optional
        Lower quantile to cut at for each marker in `internal_markers` and
        `boundary_markers`. Values below this quantile will be set to 0.
    thresh_q_max : float, optional
        Upper quantile to cut at for each marker in `internal_markers` and
        `boundary_markers`. Values above this quantile will be set to the
        quantile value.
    thresh_otsu: bool, optional
        Whether to perform OTSU thresholding for each marker in `internal_markers`
        and `boundary_markers`. Values below the OTSU threshold will be set to 0.
    scale : bool, optional
        Whether to scale each marker in `internal_markers` and `boundary_markers`
        before constructing into a single channel.
    pixel_size_um : float
        Pixel size in micrometers for marker images.
        Note:
        - Fusion: 0.5068164319979996
        - Keyence: 0.3775202
    maxima_threshold : float, optional
        Maxima threshold for Mesmer. Lower values will result in more separate
        cells being predicted, whereas higher values will result in fewer cells.
        Defaults to 0.075.
    interior_threshold : float, optional
        Interior threshold for Mesmer. Lower values will result in larger cells,
        whereas higher values will result in smaller cells. Defaults to 0.20.
    tag : str, optional
        Tag for the segmentation directory. Defaults to None, using time as tag
        (YYYYMMDD_HHMMSS).
    ometiff_path : str, optional
        Path to the OME-TIFF file containing the marker images for segmentation.
        Defaults to None, using the only OME-TIFF file in `unit_dir`.
    marker_dict : dict[str, np.ndarray], optional
        Dictionary containing marker names as keys and corresponding images as
        values. If provided, it will be used directly instead of loading from
        `ometiff_path` or searching in `unit_dir`. Defaults to None.
    num_threads : int, optional
        The number of threads to use for writing the OME-TIFF file. Defaults to 8.
    """
    # Set up directories
    unit_dir = Path(unit_dir)
    if tag is None:
        tag = time.strftime("%Y%m%d_%H%M%S")
    segmentation_dir = unit_dir / tag
    segmentation_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    setup_logging(segmentation_dir / "segmentation.log")

    # Load OME-TIFF file
    marker_dict = _get_marker_dict(
        unit_dir=unit_dir, ometiff_path=ometiff_path, marker_dict=marker_dict
    )
    logging.info(f"OME-TIFF file loaded: {ometiff_path}.")

    # Check whether selected markers are present in the OME-TIFF file
    markers = list(marker_dict.keys())
    missing_markers = [
        marker
        for marker in boundary_markers + internal_markers
        if marker not in markers
    ]
    if len(missing_markers) > 0:
        logging.error(f"Missing markers: {missing_markers}")
        raise ValueError(f"Missing markers: {missing_markers}")

    # Write parameters
    params = {
        "internal_markers": internal_markers,
        "boundary_markers": boundary_markers,
        "thresh_q_min": thresh_q_min,
        "thresh_q_max": thresh_q_max,
        "thresh_otsu": thresh_otsu,
        "scale": scale,
        "pixel_size_um": pixel_size_um,
        "maxima_threshold": maxima_threshold,
        "interior_threshold": interior_threshold,
        "compartment": "both",
    }
    with open(
        f"{segmentation_dir}/parameter_segmentation.json", "w", encoding="utf-8"
    ) as file:
        json.dump(params, file, indent=4, ensure_ascii=False)

    # Segmentation
    (
        segmentation_mask,
        internal_channel,
        boundary_channel,
        internal_dict,
        boundary_dict,
    ) = segmentation_mesmer(
        marker_dict=marker_dict, **params, tile_size=tile_size, overlap=overlap
    )
    segmentation_mask_cell = segmentation_mask[0, :, :, 0]
    segmentation_mask_nuclear = segmentation_mask[0, :, :, 1]
    segmentation_mask_nuclear = (
        segmentation_mask_nuclear.astype(bool) * segmentation_mask_cell
    )
    segmentation_mask_membrane = segmentation_mask_cell - segmentation_mask_nuclear
    tifffile.imwrite(
        str(segmentation_dir / "segmentation_mask_cell.tiff"), segmentation_mask_cell
    )
    tifffile.imwrite(
        str(segmentation_dir / "segmentation_mask_nuclear.tiff"),
        segmentation_mask_nuclear,
    )
    tifffile.imwrite(
        str(segmentation_dir / "segmentation_mask_membrane.tiff"),
        segmentation_mask_membrane,
    )
    logging.info("Segmentation completed.")

    # Save markers for segmentation, boundary and internal channels
    dtype = next(iter(marker_dict.values())).dtype
    _save_marker_dict_for_segmentation(
        segmentation_dir=segmentation_dir,
        internal_channel=internal_channel,
        boundary_channel=boundary_channel,
        internal_dict=internal_dict,
        boundary_dict=boundary_dict,
        scale=scale,
        dtype=dtype,
        num_threads=num_threads,
    )
    logging.info("Markers used for segmentation saved as OME-TIFF.")

    # Extract single-cell features
    data_cell, data_cell_scale = extract_cell_features(
        marker_dict, segmentation_mask_cell
    )
    data_nuclear, data_nuclear_scale = extract_cell_features(
        marker_dict, segmentation_mask_nuclear
    )
    data_membrane, data_membrane_scale = extract_cell_features(
        marker_dict, segmentation_mask_membrane
    )
    data_cell.to_csv(segmentation_dir / "cell_data.csv")
    data_cell_scale.to_csv(segmentation_dir / "cell_dataScaleSize.csv")
    data_nuclear.to_csv(segmentation_dir / "nuclear_data.csv")
    data_nuclear_scale.to_csv(segmentation_dir / "nuclear_dataScaleSize.csv")
    data_membrane.to_csv(segmentation_dir / "membrane_data.csv")
    data_membrane_scale.to_csv(segmentation_dir / "membrane_dataScaleSize.csv")
    logging.info("Single-cell features extracted.")


def generate_segmentation_mask_geojson_compartments(
    unit_dir: str,
    tag: str,
    n_jobs: int = 10,
    batch_size: int = 10,
):
    """
    Generate GeoJSON file for segmentation mask.

    Parameters
    ----------
    unit_dir : str
        Directory to load and save data for segmentation.
    tag : str
        Tag for the segmentation directory, where the segmentation mask will
        be used to generate the GeoJSON file.
    n_jobs : int, optional
        The number of parallel workers (CPU cores or threads) are spawned to
        process the tasks. Default is 10.
    batch_size : int, optional
        The number of labels to process in each batch. Default is 10.
    """
    unit_dir = Path(unit_dir)
    segmentation_dir = unit_dir / tag

    for component in ["cell", "membrane", "nuclear"]:
        segmentation_mask_f = segmentation_dir / f"segmentation_mask_{component}.tiff"
        segmentation_mask = tifffile.imread(segmentation_mask_f)
        mask_to_geojson_joblib(
            segmentation_mask,
            segmentation_dir / f"segmentation_mask_{component}.geojson",
            n_jobs=n_jobs,
            batch_size=batch_size,
        )
