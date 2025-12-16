"""
Command-line interface for the DEA burn severity classification workflow.

The original monolithic script has been reorganised so configuration,
severity analytics and CLI entry points live in dedicated modules.
"""

from __future__ import annotations

import os
import re
import traceback
from datetime import datetime, timedelta
from typing import Any, Iterable

import click
import datacube
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from datacube.utils.geometry import CRS, Geometry

from dea_tools.bandindices import calculate_indices
from datacube.utils.cog import write_cog
from dea_tools.spatial import xr_vectorize

from .configuration import (
    DEFAULT_CONFIG_DICT,
    RuntimeConfig,
    build_runtime_config,
    get_default_runtime_config,
)
from .data_loading import load_ard_with_fallback, load_baseline_stack
from .database import load_and_prepare_polygons
from .logging_utils import append_log
from .result_io import (
    is_valid_geojson,
    parse_s3_uri,
    s3_key_exists_and_nonempty,
    upload_dir_to_s3_and_cleanup,
)
from .severity import calculate_severity, create_debug_mask

ATTRIBUTE_COPY_RULES: tuple[dict[str, Any], ...] = (
    {"target": "fire_id", "sources": ("fire_id",), "date": False},
    {"target": "fire_name", "sources": ("fire_name",), "date": False},
    {"target": "fire_type", "sources": ("fire_type",), "date": False},
    {
        "target": "ignition_date",
        "sources": ("ignition_date", "ignition_d"),
        "date": True,
    },
    {
        "target": "capt_date",
        "sources": ("capt_date", "capture_date", "capture_da"),
        "date": True,
    },
    {"target": "capt_method", "sources": ("capt_method", "capt_metho"), "date": False},
    {"target": "area_ha", "sources": ("area_ha",), "date": False},
    {"target": "perim_km", "sources": ("perim_km",), "date": False},
    {"target": "state", "sources": ("state",), "date": False},
    {"target": "agency", "sources": ("agency",), "date": False},
    {
        "target": "date_retrieved",
        "sources": ("date_retrieved", "date_retri"),
        "date": True,
    },
    {
        "target": "date_processed",
        "sources": ("date_processed", "date_proce"),
        "date": True,
    },
    {
        "target": "extinguish_date",
        "sources": ("extinguish_date",),
        "date": True,
    },
)

FIRE_NAME_FIELDS: tuple[str, ...] = ("fire_name",)
FIRE_ID_FIELDS: tuple[str, ...] = ("fire_id",)
IGNITION_DATE_FIELDS: tuple[str, ...] = ("ignition_date", "ignition_d")
EXTINGUISH_DATE_FIELDS: tuple[str, ...] = ("extinguish_date", "date_retrieved", "date_retri")


# =============================
# ======= CORE HELPERS ========
# =============================


def _first_valid_value(series: pd.Series, keys: Iterable[str]) -> Any | None:
    for key in keys:
        if key not in series:
            continue
        value = series[key]
        if pd.isna(value):
            continue
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped or stripped.lower() in {"none", "nan", "null"}:
                continue
            value = stripped
        return value
    return None

def is_date_iso(value: str) -> bool:
    """ checks to see if date string has desiered ios UTC format by trying to convert it to datetime object
    returns TRUE if string is in correct format
    returns FLASE if it is no :(
    
    """
    try:
        datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ")
        return True
    except (ValueError, TypeError):
        return False


def is_date_DEA_format(value: str) -> bool:
    # test for YYYY-MM-DD format date:
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except (ValueError, TypeError):
        return False


def process_date(value: Any) -> str | None:
    """check the format for a date is as desiered 'yyyy-MM-ddTHH:mm:ss.sZ ', If it is not then format assuming it's 
    a valid date either as YYYY-MM-DD, YYYYMMDD or YYYYMMDDhhmmss and format as above (assume the time is noon if no time is given)

    otherwise return None. """
    if pd.isna(date):  # Handles NaT and None
        return None
    
    elif type(date) == pd._libs.tslibs.timestamps.Timestamp:
        date = date.to_pydatetime()
        return date.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    elif isinstance(date, datetime):
        return date.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    elif is_date_iso(date) == True:
        return date
        
    elif is_date_DEA_format(date) == True:
        return f'{date}T12:00:00.0Z'
        
    elif bool(re.match(r'^\d+$', date)) == True:
        if len(date) == 14:
            return f'{date[0:4]}-{date[4:6]}-{date[6:8]}T{date[8:10]}:{date[10:12]}:{date[12:14]}.0Z'
        else:
            return f'{date[0:4]}-{date[4:6]}-{date[6:8]}T12:00:00.0Z'
    else:
        try:
            as_date = datetime.strptime(date, "%Y%m%d%H%M%S.%f")
            return as_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            
        except ValueError:
            return

def clean_fire_slug(name: str | None) -> str:
    if not name:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9\s]", "", str(name))
    cleaned = re.sub(r"\s+", "_", cleaned).strip("_")
    return cleaned


def _build_vector_filename(
    fire_series: pd.Series, attributes: dict[str, Any]) -> tuple[str, str]:
        """Return file-friendly identifier/date tuple for DEA outputs."""
    identifier_src = attributes.get("fire_id")
    if identifier_src in (None, ""):
        identifier_src = _first_valid_value(fire_series, FIRE_ID_FIELDS)
    processed_date_src =  str(datetime.now()).strip()
    fire_id_for_save = f"{identifier_src}_{processed_date_src}"
    return fire_id_for_save, f"DEA_burn_severity_{fire_id_for_save}.geojson"


def _extract_attribute_values(series: pd.Series) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    for rule in ATTRIBUTE_COPY_RULES:
        raw_value = _first_valid_value(series, rule["sources"])
        if raw_value is None:
            continue
        if rule["date"]:
            formatted = process_date(raw_value)
            if formatted is None:
                continue
            attributes[rule["target"]] = formatted
        else:
            attributes[rule["target"]] = raw_value
    return attributes

def process_single_fire(
    fire_series: pd.Series,
    poly_crs: CRS,
    dc: datacube.Datacube,
    unique_fire_name: str,
    config: RuntimeConfig,
    log_path: str | None = None,
    out_dir: str | None = None,
) -> gpd.GeoDataFrame | None:
    """
    Full burn mapping workflow for a single fire polygon.
    Returns:
        GeoDataFrame dissolved by 'severity' (with 'severity' column),
  .
    """

    os.environ["AWS_NO_SIGN_REQUEST"] = "Yes"

    gpgon = Geometry(fire_series.geometry, crs=poly_crs)

    poly = gpd.GeoDataFrame([fire_series], crs=poly_crs).copy()


    attributes = _extract_attribute_values(fire_series)

    fire_id = attributes.get("fire_id")
    fire_name_value = attributes.get("fire_name")


    fire_date = process_date(attributes.get("ignition_date"))
    if not fire_date:
        fallback_raw = _first_valid_value(fire_series, ("capt_date", "capture_date", "capture_da"))
        fire_date = process_date(fallback_raw)
    if not fire_date:
        raise ValueError(
            f"Could not determine ignition date for fire '{fire_display_name}'."
        )

    processed_date_value = process_date(attributes.get("date_processed"))
    if processed_date_value:
            extinguish_date = datetime.strftime(
            ((datetime.strptime(assumed_extinguish_date, "%Y-%m-%dT%H:%M:%S.%fZ")) - timedelta(days=config.nli_holding_days)),
            "%Y-%m-%dT%H:%M:%S.%fZ")
    else:
        extinguish_date = ( datetime.strptime(fire_date, "%Y-%m-%dT%H:%M:%S.%fZ") + 
                           timedelta(days=config.post_fire_start_days)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    start_date_pre = (
        datetime.strptime(fire_date, "%Y-%m-%dT%H:%M:%S.%fZ") - timedelta(days=config.pre_fire_buffer_days)
    ).strftime("%Y-%m-%d")
    end_date_pre = (
        datetime.strptime(fire_date, "%Y-%m-%dT%H:%M:%S.%fZ") - timedelta(days=1)
    ).strftime("%Y-%m-%d")

    start_date_post = datetime.strptime(extinguish_date, "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%Y-%m-%d")

    end_date_post = (
        datetime.strptime(start_date_post, "%Y-%m-%d")
        + timedelta(days=config.post_fire_window_days)
    ).strftime("%Y-%m-%d")

    landcover_year = str(int(fire_date[0:4]) - 1)

    baseline, closest_bl = load_baseline_stack(
        dc, gpgon, time=(start_date_pre, end_date_pre), config=config
    )
    if closest_bl is None:
        if log_path:
            append_log(
                log_path,
                f"{fire_display_name}\tbaseline_scenes=0\tpost_scenes=0\tgrid=0x0"
                "\ttotal_px=0\tvalid_px=0\tburn_px=0\tmasked_px=0",
            )
        print("No baseline data for this fire. Skipping.")
        return None

    post = load_ard_with_fallback(
        dc,
        gpgon,
        time=(start_date_post, end_date_post),
        config=config,
        min_gooddata_thresholds=(0.90,0.50),
    )
    if post.time.size == 0:
        if log_path:
            yy = int(closest_bl.sizes.get("y", 0))
            xx = int(closest_bl.sizes.get("x", 0))
            total_px = yy * xx
            bl_valid_px = (
                int((closest_bl.oa_nbart_contiguity == 1).sum().compute().item())
                if "oa_nbart_contiguity" in closest_bl.data_vars
                else 0
            )
            append_log(
                log_path,
                f"{fire_display_name}\tbaseline_scenes={baseline.time.size}\tpost_scenes=0"
                f"\tgrid={yy}x{xx}\ttotal_px={total_px}\tvalid_px_baseline={bl_valid_px}"
                "\tburn_px=0\tmasked_px=0",
            )
        print("No post-fire data for this fire. Skipping.")
        return None

    landcover = dc.load(
        product="ga_ls_landcover_class_cyear_3",
        geopolygon=gpgon,
        time=(landcover_year),
        output_crs=config.output_crs,
        resolution=config.resolution,
        group_by="solar_day",
        dask_chunks={},
    )
    if landcover.time.size == 0:
        if log_path:
            yy = int(closest_bl.sizes.get("y", 0))
            xx = int(closest_bl.sizes.get("x", 0))
            total_px = yy * xx
            append_log(
                log_path,
                f"{fire_display_name}\tbaseline_scenes={baseline.time.size}"
                f"\tpost_scenes={post.time.size}\tgrid={yy}x{xx}"
                f"\ttotal_px={total_px}\tvalid_px=0\tburn_px=0\tmasked_px=0"
                "\tlandcover=missing",
            )
        print(f"No landcover data for year {landcover_year}. Skipping.")
        return None
    landcover = landcover.isel(time=0)

    pre_nbr = calculate_indices(closest_bl, index="NBR", collection="ga_s2_3", drop=True)
    post_nbr = calculate_indices(post, index="NBR", collection="ga_s2_3", drop=True)
    min_post_nbr = post_nbr.min("time")
    delta_nbr = pre_nbr - min_post_nbr

    severity = calculate_severity(delta_nbr, landcover, config.grass_classes)

    debug_mask = create_debug_mask(closest_bl, post)
    final_severity = severity.where(debug_mask == 0, 6)
    final_severity.name = "burn_severity"

    try:
        yy = int(final_severity.sizes.get("y", 0))
        xx = int(final_severity.sizes.get("x", 0))
        total_px = yy * xx
    except Exception:
        yy = xx = total_px = 0

    try:
        valid_px = int((debug_mask == 0).sum().item())
    except Exception:
        valid_px = 0

    try:
        burn_px = int(final_severity.isin([1, 2, 3, 4, 5]).sum().item())
    except Exception:
        burn_px = 0

    try:
        masked_px = int((final_severity == 6).sum().item())
    except Exception:
        masked_px = 0

    try:
        bl_valid_px = (
            int((closest_bl.oa_nbart_contiguity == 1).sum().item())
            if "oa_nbart_contiguity" in closest_bl.data_vars
            else 0
        )
        post_any_contig = (
            post.oa_nbart_contiguity.max("time") if "oa_nbart_contiguity" in post.data_vars else None
        )
        post_valid_px_any = (
            int((post_any_contig == 1).sum().item()) if post_any_contig is not None else 0
        )
    except Exception:
        bl_valid_px = 0
        post_valid_px_any = 0

    if log_path:
        append_log(
            log_path,
            (
                f"{fire_display_name}"
                f"\tbaseline_scenes={baseline.time.size}"
                f"\tpost_scenes={post.time.size}"
                f"\tgrid={yy}x{xx}"
                f"\ttotal_px={total_px}"
                f"\tvalid_px={valid_px}"
                f"\tburn_px={burn_px}"
                f"\tmasked_px={masked_px}"
                f"\tvalid_px_baseline={bl_valid_px}"
                f"\tvalid_px_post_any={post_valid_px_any}"
            ),
        )

    print("Vectorizing severity raster...")
    severity_vectors = xr_vectorize(
        final_severity,
        attribute_col="severity_rating",
        crs=config.output_crs,
        mask=final_severity != 0,
    )
    if severity_vectors.empty:
        print("No burn area detected for this fire.")
        return None

    clipped = severity_vectors.clip(gpd.GeoDataFrame([fire_series], crs=poly_crs).to_crs("EPSG:3577"))

    aggregated = clipped.dissolve(by="severity_rating").reset_index()

    agregated["severity_rating"] = agregated["severity_rating"].astype(int)

    severity_class_lables = config.severity_list_dict
    
    agregated["severity_class"] = aggrigated_severity["severity_rating"].map(severity_class_lables)

        #calculate and add area 
    aggregated['area_ha'] = round((aggregated["geometry"].area)/10000, 2)
    
    #calculate and add perimiter (length measures perimiter of a multipoly)
    aggregated['perim_km'] =round((aggregated["geometry"].length)/1000, 2)

    

    if fire_id is not None:
        aggregated["fire_id"] = fire_id
    aggregated["fire_name"] = fire_name_value
    aggregated["ignition_date"] = fire_date
    aggregated["extinguish_date"] = extinguish_date
    aggregated["fire_type"] = attribute.get("fire_type")
    aggregated["capt_date"] = attribute.get("capt_date")
    aggregated["capt_method"] = attribute.get("capt_method")
    aggregated["state"] = attribute.get("state")
    aggregated["agency"] = attribute.get("agency")

    
    fire_id_for_save, vector_filename = _build_vector_filename(
        fire_series=fire_series, attributes=attributes
    )

    base_dir = out_dir if out_dir else config.output_dir
    os.makedirs(base_dir, exist_ok=True)
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # This matches the required DEA_burn_severity_<fire_id>_<date>.geojson naming.
    out_vec = os.path.join(results_dir, vector_filename)
    aggregated.to_file(out_vec, driver="GeoJSON")
    print(f"Saved per-fire severity GeoJSON ({fire_id_for_save}): {out_vec}")

    out_cog_preview = os.path.join(base_dir, f"s2_postfire_preview_{fire_id_for_save}.tif")
    write_cog(post.isel(time=0).to_array().compute(), fname=out_cog_preview, overwrite=True)
    print(f"Saved post-fire preview COG: {out_cog_preview}")

    out_cog_debug = os.path.join(base_dir, f"debug_mask_raster_{fire_id_for_save}.tif")
    write_cog(debug_mask.compute(), fname=out_cog_debug, overwrite=True)
    print(f"Saved debug mask COG: {out_cog_debug}")

    print(f"Successfully processed fire: {fire_id_for_save}")
    return aggregated


def main(config: RuntimeConfig | None = None) -> None:
    """
    Entry point for processing burn severity polygons.
    """
    runtime = config or get_default_runtime_config()

    os.makedirs(runtime.output_dir, exist_ok=True)
    print(f"All outputs will be saved to: {runtime.output_dir}")

    upload_to_s3 = runtime.upload_to_s3
    s3_prefix = runtime.upload_to_s3_prefix
    s3_fs = None
    if upload_to_s3:
        try:
            import s3fs  # type: ignore

            s3_fs = s3fs.S3FileSystem(anon=False)
        except Exception as exc:
            print("Warning: '--upload-to-s3-prefix' requested but 's3fs' not available.")
            print("         Install with: pip install s3fs")
            print(f"Details: {exc}")
            upload_to_s3 = False

    dc = datacube.Datacube(app=runtime.app_name)

    all_polys = load_and_prepare_polygons(runtime)
    if all_polys is None or all_polys.empty:
        print("No polygons loaded. Exiting.")
        return

    print(f"Found {len(all_polys)} total features. Processing all of them.")
    fire_success = fire_fail = fire_skip = 0

    print("\nBeginning per-fire processing (single polygon per feature)...")
    for idx, fire_series in all_polys.iterrows():
        fire_attrs = _extract_attribute_values(fire_series)
        if fire_attrs.get("fire_name"):
            base_fire_name = str(fire_attrs["fire_name"]).strip()
        elif fire_attrs.get("fire_id") is not None:
            base_fire_name = f"fire_id_{fire_attrs['fire_id']}"
        else:
            base_fire_name = f"fire_{idx}"

        base_fire_slug = clean_fire_slug(base_fire_name)
        if not base_fire_slug:
            base_fire_slug = f"fire_{idx}"
        base_fire_slug = base_fire_slug.replace(os.sep, "_")
        if os.altsep:
            base_fire_slug = base_fire_slug.replace(os.altsep, "_")

        fire_dir = os.path.join(runtime.output_dir, base_fire_slug)
        os.makedirs(fire_dir, exist_ok=True)

        fire_id_for_save, vector_filename = _build_vector_filename(
            fire_series=fire_series, attributes=fire_attrs)
        results_dir = os.path.join(fire_dir, "results")
        final_vector_path = os.path.join(results_dir, vector_filename)
        log_path = os.path.join(fire_dir, f"{ base_fire_slug}_processing.log")

        if not os.path.exists(log_path):
            append_log(
                log_path,
                f"# Processing log for {base_fire_name} (feature index {idx})",
            )
            append_log(
                log_path,
                "# fire_name\tbaseline_scenes\tpost_scenes\tgrid\ttotal_px\tvalid_px"
                "\tburn_px\tmasked_px\tvalid_px_baseline\tvalid_px_post_any",
            )

        skip_due_to_output = False
        if not runtime.force_rebuild:
            if is_valid_geojson(final_vector_path):
                print(
                    f"[Fire '{base_fire_name}' ({fire_id_for_save})] Local output exists & valid. Skipping."
                )
                fire_skip += 1
                skip_due_to_output = True
            elif upload_to_s3 and s3_fs is not None:
                bucket, prefix = parse_s3_uri(s3_prefix)
                prefix = prefix.strip("/")
                prefix_part = f"{prefix}/" if prefix else ""
                remote_key = (
                    f"{prefix_part}results/{vector_filename}"
                )
                if s3_key_exists_and_nonempty(s3_fs, bucket, remote_key):
                    print(
                        f"[Fire '{base_fire_name}' ({fire_id_for_save})] Output exists in S3. Skipping."
                    )
                    fire_skip += 1
                    skip_due_to_output = True

        if skip_due_to_output:
            continue

        print("\n" + "=" * 80)
        print(f"Processing fire polygon: '{base_fire_name}'")
        print("=" * 80)

        try:
            gdf_fire = process_single_fire(
                fire_series=fire_series,
                poly_crs=all_polys.crs,
                dc=dc,
                unique_fire_name=base_fire_slug,
                config=runtime,
                log_path=log_path,
                out_dir=fire_dir,
            )
            if gdf_fire is not None and len(gdf_fire) > 0:
                fire_success += 1
        except Exception as exc:
            fire_fail += 1
            print(f"!!! FAILED to process fire '{base_fire_name}': {exc}")
            traceback.print_exc()
            continue

        if upload_to_s3:
            ok = upload_dir_to_s3_and_cleanup(fire_dir, s3_prefix)
            if not ok:
                print(
                    f"[S3 upload] WARNING: '{fire_dir}' was not removed (upload verification failed)."
                )

    print("\n" + "=" * 80)
    print("Batch processing complete.")
    print(f"  Fire success: {fire_success}")
    print(f"  Fire failed:  {fire_fail}")
    print(f"  Fire skipped: {fire_skip}")
    print("=" * 88)

def _decorate_help(text: str, default_value: Any) -> str:
    return f"{text} (default: {default_value})"


@click.command(context_settings={"auto_envvar_prefix": "DEA_BURN_SEVERITY"})
@click.option(
    "--config",
    type=str,
    default=None,
    help="Path or URL to a YAML configuration file overriding packaged defaults.",
)
@click.option(
    "--output-dir",
    type=str,
    default=None,
    help=_decorate_help("Output base directory", DEFAULT_CONFIG_DICT["output_dir"]),
)
@click.option(
    "--force-rebuild",
    type=bool,
    default=None,
    help=_decorate_help("Rebuild even if outputs exist", DEFAULT_CONFIG_DICT["force_rebuild"]),
)
@click.option(
    "--upload-to-s3-prefix",
    type=str,
    default=None,
    help=_decorate_help(
        "S3 prefix to upload all outputs (shared folder)", DEFAULT_CONFIG_DICT["upload_to_s3_prefix"]
    ),
)
@click.option(
    "--upload-to-s3",
    type=bool,
    default=None,
    help=_decorate_help(
        "Enable S3 upload and local cleanup", DEFAULT_CONFIG_DICT["upload_to_s3"]
    ),
)
@click.option(
    "--app-name",
    type=str,
    default=None,
    help=_decorate_help("Datacube app name", DEFAULT_CONFIG_DICT["app_name"]),
)
@click.option(
    "--db-table",
    type=str,
    default=None,
    help=_decorate_help("Database table containing polygons", DEFAULT_CONFIG_DICT["db_table"]),
)
def cli(**kwargs: Any) -> None:
    """
    Console script entry point.
    """
    config_path = kwargs.pop("config", None)

    try:
        runtime_config = build_runtime_config(kwargs, config_path)
    except Exception as exc:
        raise SystemExit(f"Failed to load configuration: {exc}") from exc

    main(runtime_config)


if __name__ == "__main__":
    cli()
