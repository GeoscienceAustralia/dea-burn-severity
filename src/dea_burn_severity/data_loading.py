"""
Data loading helpers for burn severity processing.
"""

from __future__ import annotations

from typing import Iterable

import datacube
import numpy as np
import xarray as xr
from datacube.utils.geometry import Geometry

from dea_tools.datahandling import load_ard

from .configuration import RuntimeConfig


LOAD_DASK_CHUNKS: dict[str, int] = {"x": 2048, "y": 2048}


def _select_reference_band(dataset: xr.Dataset) -> str:
    preferred = ("nbart_green", "nbart_red", "nbart_nir_1", "nbart_blue")
    for candidate in preferred:
        if candidate in dataset.data_vars:
            return candidate
    for name, data in dataset.data_vars.items():
        if "time" in data.dims:
            return name
    raise ValueError("Dataset lacks a data variable with a 'time' dimension.")


def find_latest_valid_pixel(dataset: xr.Dataset) -> xr.Dataset:
    """
    For a multi-temporal xarray Dataset (masked for clouds/contiguity),
    return the latest clear value for each pixel across all bands.
    Output: Dataset with same bands, no time dimension.
    """
    # Ensure 'time' is a dimension
    if 'time' not in dataset.dims:
        raise ValueError("Dataset must have a 'time' dimension")

    # Convert time to numeric for comparison
    time_numeric = xr.DataArray(dataset['time'].astype('datetime64[ns]').astype(np.int64), dims='time')

    # Create mask for valid values (non-NaN)
    valid_mask = ~np.isnan(dataset['nbart_green'])

    # Multiply mask by time to get numeric time where valid, else 0
    valid_times = valid_mask * time_numeric

    # Find latest valid time for each pixel
    latest_valid_time = valid_times.max(dim='time')

    # Create mask for latest time
    latest_mask = valid_times == latest_valid_time

    # Select latest valid values for each band
    latest_values = dataset.where(latest_mask).max(dim='time')

    return latest_values


def load_ard_with_fallback(
    dc: datacube.Datacube,
    gpgon: Geometry,
    time: tuple[str, str],
    config: RuntimeConfig,
    min_gooddata_thresholds: Iterable[float] = (0.99, 0.90),
    **kwargs,
) -> xr.Dataset:
    """
    Loads ARD data, trying a list of 'min_gooddata' thresholds in order.
    """
    base_params = {
        "dc": dc,
        "products": config.s2_products,
        "geopolygon": gpgon,
        "time": time,
        "measurements": list(config.s2_measurements),
        "output_crs": config.output_crs,
        "resolution": config.resolution,
        "group_by": "solar_day",
        "cloud_mask": "s2cloudless",
        "dask_chunks": LOAD_DASK_CHUNKS,
        **kwargs,
    }

    data = xr.Dataset()
    for threshold in min_gooddata_thresholds:
        print(f"Attempting load_ard with min_gooddata={threshold} ...")
        base_params["min_gooddata"] = threshold
        data = load_ard(**base_params)
        time_size = int(getattr(getattr(data, "time", None), "size", 0))
        if time_size > 0:
            print(f"Success: Loaded {time_size} time slices.")
            return data

    print(
        f"Warning: No data found for time range {time} "
        f"even with min_gooddata={list(min_gooddata_thresholds)[-1]}"
    )
    return data


def load_baseline_stack(
    dc: datacube.Datacube, gpgon: Geometry, time: tuple[str, str], config: RuntimeConfig
) -> tuple[xr.Dataset, xr.Dataset | None]:
    """
    Load baseline ARD with strict cloud limits, then fall back to a relaxed
    load that is composited to the latest valid pixel if required.
    """
    base_params = {
        "dc": dc,
        "products": config.s2_products,
        "geopolygon": gpgon,
        "time": time,
        "measurements": list(config.s2_measurements),
        "output_crs": config.output_crs,
        "resolution": config.resolution,
        "group_by": "solar_day",
        "cloud_mask": "s2cloudless",
        "dask_chunks": LOAD_DASK_CHUNKS,
        "min_gooddata": 0.99,
    }

    baseline = load_ard(**base_params)
    baseline_time_size = int(getattr(getattr(baseline, "time", None), "size", 0))
    if baseline_time_size > 0:
        return baseline, baseline.isel(time=-1)

    print("Baseline load empty; retrying with mask dilation and relaxed clouds.")
    relaxed_params = base_params | {
        "mask_filters": [("dilation", 15)],
        "min_gooddata": 0.20,
    }
    baseline = load_ard(**relaxed_params)
    if baseline.time.size == 0:
        return baseline, None

    composite = find_latest_valid_pixel(baseline)
    return baseline, composite


__all__ = [
    "LOAD_DASK_CHUNKS",
    "find_latest_valid_pixel",
    "load_ard_with_fallback",
    "load_baseline_stack",
]
