"""
io_setup.py
===========
I/O and environment setup utilities for the G4X preprocessing pipeline.

Functions
---------
setup_logging  : Configure root logger to write to file and console.
setup_gpu      : Select GPU(s) and enable TensorFlow memory growth.
rename_invalid_marker   : Replace "/" and ":" in marker names with "_".
rename_duplicate_markers: Append numeric suffixes to deduplicate marker lists.
parse_marker_fusion     : Parse metadata from a Fusion-format tiff filename.
parse_marker_keyence    : Parse metadata from a Keyence-format tiff filename.
parse_region            : Load metadata for all marker files in a region dir.
organize_metadata       : Collect metadata across regions into a dict.
organize_metadata_fusion   : Convenience wrapper for Fusion data.
organize_metadata_keyence  : Convenience wrapper for Keyence data.
summary_dir             : Print a summary of folder/file contents.
summary_metadata        : Print and return region/marker summary statistics.
write_ometiff           : Save a list of images as a multi-channel OME-TIFF.
"""

import logging
import os
import re

import numpy as np
import pandas as pd
import tifffile

################################################################################
# setup
################################################################################


def setup_logging(
    log_file=os.path.join(os.getcwd(), "output.log"),
    log_format="%(asctime)s - %(levelname)s - %(message)s",
    log_mode="w",
    logger_level=logging.INFO,
    file_handler_level=logging.INFO,
    stream_handler_level=logging.WARNING,
):
    """
    Configures logging to output messages to both a file and the console.

    Parameters
    ----------
    log_file : str, optional
        Path to the log file. Default is 'output.log' in the current working directory.
    log_format : str, optional
        Format for log messages. Default includes timestamp, log level, and message.
    log_mode : str, optional
        File mode for the log file. Default is 'w' (write mode, overwrites file).
    logger_level : int, optional
        Logging level for the root logger. Default is logging.WARNING.
    file_handler_level : int, optional
        Logging level for the FileHandler. Default is logging.INFO.
    stream_handler_level : int, optional
        Logging level for the StreamHandler (console output). Default is logging.WARNING.

    Returns
    -------
    None
        This function configures the logger and does not return any value.

    Notes
    -----
    - The logger level controls the global filtering of messages.
    - The FileHandler writes log messages to a file.
    - The StreamHandler displays log messages in the console.
    """
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logger_level)  # Set the logger's global logging level

    # Create a FileHandler for writing logs to a file
    file_handler = logging.FileHandler(log_file, mode=log_mode)
    file_handler.setLevel(file_handler_level)  # Set the level for the file handler
    file_handler.setFormatter(logging.Formatter(log_format))  # Apply the log format

    # Create a StreamHandler for console output
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(
        stream_handler_level
    )  # Set the level for the stream handler
    stream_handler.setFormatter(
        logging.Formatter(log_format)
    )  # Apply the same log format

    # Add both handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Optional: Print confirmation to the console
    print(
        f"\nLogging is configured. Logs are saved to: {log_file} and displayed in the console."
    )


def setup_gpu(gpu_ids="0"):
    """
    Configures TensorFlow to use specified GPU(s) with memory growth enabled.

    TensorFlow is imported lazily inside this function so that modules which
    import io_setup but do not call setup_gpu are not forced to load TensorFlow.

    Parameters
    ----------
    gpu_ids : str
        Comma-separated GPU IDs to be made visible (e.g., "0,1").

    Returns
    -------
    None
    """
    import tensorflow as tf  # lazy import — avoids loading TF at module import time

    # Specify which GPU(s) to use
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    try:
        # Get available GPU devices
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            # Enable memory growth for each GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f"Using GPU(s): {[gpu.name for gpu in gpus]}")
            print(f"Using GPU(s): {[gpu.name for gpu in gpus]}")
        else:
            logging.warning("No GPU detected, using CPU.")
    except RuntimeError as e:
        # Handle any runtime errors (e.g., GPUs already initialized)
        logging.error(f"GPU setup failed: {e}")


################################################################################
# rename marker
################################################################################


def rename_invalid_marker(marker_name: str) -> str:
    """
    Rename the invalid marker name (containing "/" and ":").

    Parameters
    ----------
    marker_name: str
        Original marker name.

    Returns
    -------
    str
        Renamed marker name with invalid characters replaced by "_".
    """
    marker_name = re.sub(r"[/:]", "_", marker_name)
    return marker_name


def rename_duplicate_markers(marker_list: list[str]) -> list[str]:
    """
    Renames duplicate markers by appending a numeric suffix to ensure uniqueness.

    Parameters
    ----------
    marker_list: list)
        A list of strings representing marker names.

    Returns
    -------
    list
        A list of marker names where duplicates are renamed with a suffix (e.g.,
        '_2', '_3').
    """
    # Count the occurrences of each marker
    marker_list_count = pd.Series(marker_list).value_counts()
    marker_duplicated = marker_list_count.index[marker_list_count > 1]
    logging.warning(f"Duplicated markers: {marker_duplicated.tolist()}")

    # Rename duplicated markers: add a numeric suffix for duplicates
    marker_seen = {}
    renamed_list = []
    for marker in marker_list:
        if marker in marker_duplicated:
            idx = marker_seen.get(marker, 1)
            renamed_list.append(f"{marker}_{idx}")
            marker_seen[marker] = idx + 1
        else:
            renamed_list.append(marker)

    return renamed_list


################################################################################
# organize metadata
################################################################################


def parse_marker_fusion(marker_path: str) -> pd.DataFrame:
    """
    Fusion: Parse marker information from the file name.

    Parameters
    ----------
    marker_path : str
        Path to the marker image file (*.tif).

    Returns
    -------
    pd.DataFrame
        DataFrame of metadata.
    """
    marker_basename = os.path.basename(marker_path)
    pattern = r"(.+)\.tiff?$"
    match = re.match(pattern, marker_basename)
    (marker,) = match.groups()
    metadata = pd.DataFrame(
        [[marker_path, rename_invalid_marker(marker)]],
        columns=["path", "marker"],
    )
    return metadata


def parse_marker_keyence(marker_path: str) -> pd.DataFrame:
    """
    Keyence: Parse marker information from the file name.

    Parameters
    ----------
    marker_path : str
        Path to the marker image file (*.tif).

    Returns
    -------
    pd.DataFrame
        DataFrame of metadata.
    """
    marker_basename = os.path.basename(marker_path)
    pattern = r".*(reg\d+)_(cyc\d+)_(ch\d+)_([^.]+)(?:\.ome)?\.tiff?$"
    match = re.match(pattern, marker_basename)
    region, cycle, channel, marker = match.groups()
    metadata = pd.DataFrame(
        [[marker_path, region, cycle, channel, rename_invalid_marker(marker)]],
        columns=["path", "_region", "cycle", "channel", "marker"],
    )
    return metadata


def parse_region(
    region_dir: str, parse_marker_func, region: str, extensions: list[str]
) -> pd.DataFrame:
    """
    Load and process metadata for a specific region.

    Parameters
    ----------
    region_dir : str
        Directory containing marker images for a specific region.
    parse_marker_func : function
        Function to parse individual marker image files into DataFrames.
    region : str
        Name of the region to extract markers for.
    extensions : list
        List of allowed file extensions.

    Returns
    -------
    pd.DataFrame
        DataFrame containing metadata for the specified region.
    """
    metadata_dfs = [
        parse_marker_func(os.path.join(region_dir, marker))
        for marker in os.listdir(region_dir)
        if re.search("([^.]+)(.*)", marker).groups()[1] in extensions
    ]
    metadata_df = pd.concat(metadata_dfs, axis=0).reset_index(drop=True)
    metadata_df["region"] = region
    return metadata_df


def organize_metadata(
    marker_dir: str,
    parse_marker_func,
    subfolders: bool = True,
    extensions: list[str] = [".tiff", ".tif"],
) -> dict[str, pd.DataFrame]:
    """
    Organize metadata from marker files.

    Parameters
    ----------
    marker_dir : str
        Directory containing marker images or subdirectories of marker images.
    parse_marker_func : function
        Function to parse individual marker image files into DataFrames.
    subfolders : bool, optional
        If True, marker files are in subfolders of `marker_dir` (e.g., one
        subfolder per region). If False, marker files are directly in `marker_dir`.
        Default is True.
    extensions : list, optional
        List of allowed file extensions. Default is [".tiff", ".tif"].

    Returns
    -------
    dict
        Dictionary with region names as keys and metadata DataFrames as values.
    """
    metadata_dict = {}
    if subfolders:
        for region in os.listdir(marker_dir):
            region_dir = os.path.join(marker_dir, region)
            metadata_dict[region] = parse_region(
                region_dir, parse_marker_func, region, extensions
            )
    else:
        metadata_dict["region"] = parse_region(
            marker_dir, parse_marker_func, "region", extensions
        )
    return metadata_dict


def organize_metadata_fusion(
    marker_dir: str,
    subfolders: bool = True,
    extensions: list[str] = [".tiff", ".tif"],
) -> dict[str, pd.DataFrame]:
    """
    Organize metadata from marker files for Fusion output.
    """
    return organize_metadata(marker_dir, parse_marker_fusion, subfolders, extensions)


def organize_metadata_keyence(
    marker_dir: str,
    subfolders: bool = True,
    extensions: list[str] = [".tiff", ".tif"],
) -> dict[str, pd.DataFrame]:
    """
    Organize metadata from marker files for Keyence output.
    """
    return organize_metadata(marker_dir, parse_marker_keyence, subfolders, extensions)


################################################################################
# Summary
################################################################################


def summary_dir(dir: str, indent="    "):
    """
    Summarize the contents of the marker directory.

    Parameters
    ----------
    dir : str
        Directory path to summarize.
    indent : str, optional
        Indentation string for nested levels. Default is four spaces.

    Returns
    -------
    None
    """
    items = os.listdir(dir)

    # Folders
    folders = [item for item in items if os.path.isdir(os.path.join(dir, item))]
    if folders:
        print(f"Folders: {folders}")

    # Files with different extensions
    files = [item for item in items if os.path.isfile(os.path.join(dir, item))]
    if files:
        files_dict = {}
        for file in files:
            ext = os.path.splitext(file)[1] or "no_ext"
            files_dict.setdefault(ext, []).append(file)
        print("Files by extension:")
        for ext, file_list in files_dict.items():
            print(f"{indent}- {ext}: {file_list}")


def summary_metadata(
    metadata_dict: dict[str, pd.DataFrame], indent="    "
) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Summarize marker information across regions.

    Parameters
    ----------
    metadata_dict : dict
        Dictionary containing region names as keys and metadata DataFrames as values.
    indent : str, optional
        Indentation string for nested levels. Default is four spaces.

    Returns
    -------
    tuple
        Tuple containing lists of regions, unique markers, blank markers, and missing
        markers.
    """
    # Summary of regions
    regions = list(metadata_dict.keys())

    # Summary of markers
    ## Get All unique markers
    combined_metadata_df = pd.concat(metadata_dict.values(), ignore_index=True)
    all_markers = combined_metadata_df["marker"].unique()

    ## Identify and filter out blank markers
    blank_markers = [
        marker for marker in all_markers if re.match(r"blank", marker, re.IGNORECASE)
    ]
    metadata_df = combined_metadata_df.loc[
        ~combined_metadata_df["marker"].isin(blank_markers)
    ]
    count_pivot = metadata_df.pivot_table(
        index="marker", columns="region", aggfunc="size", fill_value=0
    )

    ## Identify markers that are missing in some regions
    missing_markers_n = (count_pivot == 0).sum(axis=1)
    missing_markers_df = count_pivot.loc[missing_markers_n > 0]
    missing_markers = list(missing_markers_df.index)
    missing_long_df = pd.melt(
        missing_markers_df.reset_index(),
        id_vars=["marker"],
        var_name="region",
        value_name="count",
    )
    missing_long_df = missing_long_df[missing_long_df["count"] == 0]
    missing_markers_dict = (
        missing_long_df.groupby("region", group_keys=False)["marker"]
        .apply(list)
        .to_dict()
    )

    ## Identify unique markers (not blank, not duplicated, and not missing in any region)
    unique_markers = [
        marker
        for marker in all_markers
        if marker not in (blank_markers + missing_markers)
    ]
    unique_markers = sorted(unique_markers)

    # Display summary information
    print(f"Summary of Regions:\n{indent}- Total regions: {len(regions)} {regions}")
    print(
        f"Summary of Markers:\n"
        f"{indent}- Total unique markers: {len(all_markers)}\n"
        f"{indent}- Unique markers: {len(unique_markers)} {unique_markers}\n"
        f"{indent}- Blank markers: {len(blank_markers)} {blank_markers}\n"
        f"{indent}- Missing markers: {len(missing_markers)}"
    )
    if missing_markers:
        for region, markers in missing_markers_dict.items():
            print(f"{indent * 2}- {region}: {markers}")

    return regions, unique_markers, blank_markers, missing_markers


################################################################################
# export OME-TIFF
################################################################################


def write_ometiff(path_ometiff, names, images, dtype=np.uint16):
    """
    Save a list of images as an OME-TIFF file with channel metadata.

    Parameters
    ----------
    path_ometiff : str
        Path to the output OME-TIFF file.
    names : list of str
        List of channel names corresponding to each image.
    images : list of numpy.ndarray
        List of 2D arrays representing individual images for each channel. All
        images must have the same shape.
    dtype : numpy.dtype, optional
        Data type to cast the images to before saving. Default is `np.uint16`.

    Returns
    -------
    None
    """
    images_stack = np.stack(images, axis=0).astype(dtype)
    tifffile.imwrite(
        path_ometiff,
        images_stack,
        metadata={"axes": "CYX", "Channel": {"Name": names}},
        ome=True,
    )
