"""
Burn severity masking and classification helpers.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from dea_tools.bandindices import calculate_indices


def calculate_severity(
    delta_nbr: xr.DataArray, landcover: xr.Dataset, grass_classes: tuple[int, ...] | list[int]
) -> xr.DataArray:
    """
    Calculate burn severity classifications using woody vs. grass thresholds.
    """
    print("Calculating severity based on landcover...")

    nbr = delta_nbr["NBR"] if isinstance(delta_nbr, xr.Dataset) else delta_nbr

    # Mirror the previous behaviour where grass pixels carried a value of 2.
    grass_mask = landcover.level4.isin(grass_classes).astype(np.uint8) * 2
    woody_mask = grass_mask == 0

    low = (nbr >= 0.1).astype(np.uint8) * 2
    medium = (nbr >= 0.27).astype(np.uint8)
    high = (nbr >= 0.44).astype(np.uint8)
    very_high = (nbr >= 0.66).astype(np.uint8)

    severity_woody = (low + medium + high + very_high).where(woody_mask, 0)
    severity_grass = grass_mask.where(nbr >= 0.1, 0)

    severity = severity_woody + severity_grass

    severity.name = "severity"
    return severity


def create_debug_mask(pre_fire_scene: xr.Dataset, post_fire_stack: xr.Dataset) -> xr.DataArray:
    """
    Build a composite mask describing why pixels are excluded from severity output.
    """
    print("Creating debug/masking layer...")

    debug_layer_blank = xr.ones_like(pre_fire_scene.nbart_red, dtype=np.uint16)

    post_mndwi = calculate_indices(
        post_fire_stack, index="MNDWI", collection="ga_s2_3", drop=True
    )
    max_mndwi = post_mndwi.max("time")
    post_water = debug_layer_blank.where(max_mndwi.MNDWI > 0, 0)
    new_debug = post_water

    pre_cloud = (debug_layer_blank.where(pre_fire_scene.oa_s2cloudless_mask == 2, 0)) * 10
    new_debug = new_debug + pre_cloud

    post_cloud = post_fire_stack.oa_s2cloudless_mask.where(
        post_fire_stack.oa_s2cloudless_mask >= 1, 1
    )
    persistent_cloud = post_cloud.min("time")
    post_cloud_mask = (debug_layer_blank.where(persistent_cloud == 2, 0)) * 100
    new_debug = new_debug + post_cloud_mask

    pre_contiguity = (
        debug_layer_blank.where(pre_fire_scene.oa_nbart_contiguity != 1, 0)
    ) * 1000
    new_debug = new_debug + pre_contiguity

    post_contiguity = post_fire_stack.oa_nbart_contiguity.where(
        post_fire_stack.oa_nbart_contiguity == 1, 0
    )
    persistent_cont = post_contiguity.max("time")
    post_contiguity_mask = (debug_layer_blank.where(persistent_cont != 1, 0)) * 10000
    new_debug = new_debug + post_contiguity_mask

    new_debug.name = "debug_mask"
    return new_debug


__all__ = ["calculate_severity", "create_debug_mask"]
