"""
Command-line interface for the DEA burn severity classification workflow.

The original monolithic script has been reorganised so configuration,
severity analytics and CLI entry points live in dedicated modules.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import traceback
from datetime import datetime, timedelta
from typing import Any, Iterable

import click
import datacube
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from datacube.utils.cog import write_cog
from datacube.utils.geometry import CRS, Geometry
from shapely.geometry import shape

from dea_tools.bandindices import calculate_indices
from dea_tools.datahandling import load_ard
from dea_tools.spatial import xr_vectorize

from .configuration import (
    DEFAULT_CONFIG_DICT,
    RuntimeConfig,
    build_runtime_config,
    get_default_runtime_config,
)
from .severity import calculate_severity, create_debug_mask


LOAD_DASK_CHUNKS: dict[str, int] = {"x": 2048, "y": 2048}

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

DB_SCHEMA = "public"


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


def _is_valid_date_string(value: str) -> bool:
    return bool(re.match(r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$", value))


def process_date(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (datetime, pd.Timestamp)):
        return pd.to_datetime(value).strftime("%Y-%m-%d")
    if isinstance(value, np.datetime64):
        return pd.to_datetime(value).strftime("%Y-%m-%d")
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return None
    if _is_valid_date_string(text):
        return text
    digits = re.sub(r"[^0-9]", "", text)
    if len(digits) >= 8:
        return f"{digits[0:4]}-{digits[4:6]}-{digits[6:8]}"
    return None


def clean_fire_slug(name: str | None) -> str:
    if not name:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9\s]", "", str(name))
    cleaned = re.sub(r"\s+", "_", cleaned).strip("_")
    return cleaned


def _format_results_filename(
    identifier_source: Any | None, date_source: Any | None, fallback_slug: str
) -> tuple[str, str]:
    """Return file-friendly identifier/date tuple for DEA outputs."""
    identifier = clean_fire_slug(str(identifier_source).strip()) if identifier_source not in (None, "") else ""
    if not identifier:
        identifier = fallback_slug
    date_value = process_date(date_source)
    if not date_value:
        date_value = datetime.utcnow().strftime("%Y-%m-%d")
    fire_id_for_save = f"{identifier}_{date_value}"
    return fire_id_for_save, f"DEA_burn_severity_{fire_id_for_save}.json"


def _build_vector_filename(
    fire_series: pd.Series, attributes: dict[str, Any], fallback_slug: str
) -> tuple[str, str]:
    identifier_src = attributes.get("fire_id")
    if identifier_src in (None, ""):
        identifier_src = _first_valid_value(fire_series, FIRE_ID_FIELDS)
    processed_date_src = attributes.get("date_processed")
    if processed_date_src in (None, ""):
        processed_date_src = _first_valid_value(fire_series, ("date_processed", "date_proce"))
    return _format_results_filename(identifier_src, processed_date_src, fallback_slug)


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
    if "time" not in dataset.dims:
        raise ValueError("Dataset must include a 'time' dimension.")
    ref_band = _select_reference_band(dataset)
    time_numeric = xr.DataArray(
        dataset["time"].astype("datetime64[ns]").astype(np.int64), dims="time"
    )
    valid_mask = dataset[ref_band].notnull()
    valid_times = valid_mask * time_numeric
    latest_time = valid_times.max(dim="time")
    latest_mask = valid_times == latest_time
    latest_values = dataset.where(latest_mask).max(dim="time")
    return latest_values


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

def _read_geojson_maybe_s3(path: str) -> gpd.GeoDataFrame:
    """
    Read a GeoJSON from a local path or S3 URI into a GeoDataFrame.
    For S3, streams to a temporary local file (requires s3fs).
    """
    if isinstance(path, str) and path.lower().startswith("s3://"):
        try:
            import s3fs  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Reading from s3:// requires the 's3fs' package. "
                "Install with: pip install s3fs"
            ) from exc

        gdf = None
        errs: list[str] = []
        for anon in (True, False):
            try:
                fs = s3fs.S3FileSystem(anon=anon)
                with fs.open(path, "rb") as fsrc, tempfile.NamedTemporaryFile(
                    suffix=".geojson", delete=False
                ) as tmp:
                    shutil.copyfileobj(fsrc, tmp)
                    tmp_path = tmp.name
                gdf = gpd.read_file(tmp_path)
                os.remove(tmp_path)
                break
            except Exception as inner_exc:
                errs.append(str(inner_exc))
                gdf = None
        if gdf is None:
            raise RuntimeError(
                f"Failed to read GeoJSON from S3 path '{path}'. Errors: {errs}"
            )
        return gdf
    return gpd.read_file(path)


def _load_polygons_from_database(config: RuntimeConfig) -> gpd.GeoDataFrame:
    """
    Load polygons directly from the configured PostgreSQL/PostGIS table.
    """
    if not config.db_table:
        raise ValueError("Configuration 'db_table' must be provided for database polygon loading.")
    if not config.db_columns:
        raise ValueError("Configuration 'db_columns' must include at least one attribute column.")

    try:
        import psycopg2  # type: ignore
        from psycopg2 import sql  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Database polygon loading requires the 'psycopg2-binary' package. "
            "Install with: pip install psycopg2-binary"
        ) from exc

    conn = psycopg2.connect(
        host=config.db_host,
        dbname=config.db_name,
        port=str(config.db_port),
        user=config.db_user,
        password=config.db_password,
    )

    table_identifier = (
        sql.Identifier(DB_SCHEMA, config.db_table)
        if DB_SCHEMA
        else sql.Identifier(config.db_table)
    )
    select_clause = sql.SQL(", ").join(sql.Identifier(col) for col in config.db_columns)
    query = sql.SQL(
        "SELECT {fields}, ST_AsGeoJSON({geom}) AS geom_geojson FROM {table}"
    ).format(
        fields=select_clause,
        geom=sql.Identifier(config.db_geom_column),
        table=table_identifier,
    )

    print(
        f"Querying polygons from table '{DB_SCHEMA + '.' if DB_SCHEMA else ''}{config.db_table}'..."
    )
    records: list[dict[str, Any]] = []
    failures = 0

    try:
        with conn, conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
            print(f"Retrieved {len(rows)} rows from database.")
            for idx, row in enumerate(rows):
                attr_values = row[:-1]
                geom_raw = row[-1]
                try:
                    json_start = geom_raw.find("{")
                    geom_str = geom_raw[json_start:] if json_start >= 0 else geom_raw
                    geom_mapping = json.loads(geom_str)
                    geom_obj = shape(geom_mapping)
                except Exception as exc:
                    failures += 1
                    print(f"Warning: Failed to parse geometry for row {idx}: {exc}")
                    continue

                record = {col: val for col, val in zip(config.db_columns, attr_values)}
                record["geometry"] = geom_obj
                records.append(record)
    finally:
        conn.close()

    if failures:
        print(f"Skipped {failures} rows due to geometry parsing errors.")
    if not records:
        return gpd.GeoDataFrame(
            columns=[*config.db_columns, "geometry"], crs=config.db_output_crs
        )

    return gpd.GeoDataFrame(records, crs=config.db_output_crs)


def load_and_prepare_polygons(config: RuntimeConfig) -> gpd.GeoDataFrame | None:
    """
    Loads fire polygons from the configured database and prepares them for processing.
    Dissolves by 'fire_id' if available to ensure one row per fire.
    """
    try:
        poly_gdf = _load_polygons_from_database(config)
    except Exception as exc:
        print(f"Error: Failed loading polygons from database: {exc}")
        return None

    try:
        if len(poly_gdf) > 1 and "fire_id" in poly_gdf.columns:
            print("Dissolving polygons by 'fire_id'...")
            poly_gdf = poly_gdf.dissolve(by="fire_id", aggfunc="first")
        elif "fire_id" not in poly_gdf.columns:
            print("Warning: 'fire_id' not in columns. Skipping dissolve.")
    except TypeError as exc:
        print(f"Warning: Could not dissolve polygon ({exc}). Continuing with loaded data.")

    if "fire_id" not in poly_gdf.columns:
        poly_gdf["fire_id"] = list(poly_gdf.index)

    return poly_gdf


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
        "measurements": config.s2_measurements,
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
        if getattr(data, "time", xr.DataArray()).size > 0:
            print(f"Success: Loaded {data.time.size} time slices.")
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
        "measurements": config.s2_measurements,
        "output_crs": config.output_crs,
        "resolution": config.resolution,
        "group_by": "solar_day",
        "cloud_mask": "s2cloudless",
        "dask_chunks": LOAD_DASK_CHUNKS,
        "min_gooddata": 0.99,
    }

    baseline = load_ard(**base_params)
    if baseline.time.size > 0:
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


def _append_log(log_path: str, line: str) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as file:
        file.write(line.rstrip("\n") + "\n")
        file.flush()
        os.fsync(file.fileno())


def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    if not s3_uri.lower().startswith("s3://"):
        raise ValueError(f"Not an S3 URI: {s3_uri}")
    no_scheme = s3_uri[5:]
    bucket, _, key = no_scheme.partition("/")
    return bucket, key.rstrip("/")


def _s3_key_exists_and_nonempty(fs, bucket: str, key: str) -> bool:
    s3_path = f"{bucket}/{key}"
    try:
        if fs.exists(s3_path):
            info = fs.info(s3_path)
            return info.get("Size", 0) > 0
        return False
    except Exception:
        return False


def _upload_dir_to_s3_and_cleanup(local_dir: str, s3_prefix: str) -> bool:
    """
    Upload local_dir recursively to <s3_prefix>/<basename(local_dir)>/*
    and, on success (existence + size checks), delete local_dir.
    """
    if not os.path.isdir(local_dir):
        print(f"[S3 upload] Local directory does not exist: {local_dir}")
        return False

    import s3fs  # type: ignore

    bucket, key_prefix = _parse_s3_uri(s3_prefix)
    dest_base = key_prefix.strip("/")
    slug = os.path.basename(os.path.normpath(local_dir))
    dest_dir_key = f"{dest_base}/{slug}"

    fs = s3fs.S3FileSystem(anon=False)

    local_files: list[tuple[str, int, str]] = []
    for root, _, files in os.walk(local_dir):
        for name in files:
            full = os.path.join(root, name)
            rel = os.path.relpath(full, local_dir).replace("\\", "/")
            size = os.path.getsize(full)
            local_files.append((rel, size, full))

    if not local_files:
        print(f"[S3 upload] Nothing to upload from {local_dir}")
        return False

    print(f"[S3 upload] Uploading '{local_dir}' -> 's3://{bucket}/{dest_dir_key}/' ...")
    for rel, _, full in local_files:
        remote_key = f"{dest_dir_key}/{rel}"
        fs.put(full, f"{bucket}/{remote_key}")

    shutil.rmtree(local_dir, ignore_errors=True)
    return True


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
        reprojected to 'EPSG:4283', or None if nothing to save.
    """

    os.environ["AWS_NO_SIGN_REQUEST"] = "Yes"

    gpgon = Geometry(fire_series.geometry, crs=poly_crs)

    poly = gpd.GeoDataFrame([fire_series], crs=poly_crs).copy()
    poly = poly.to_crs("EPSG:4283")

    attributes = _extract_attribute_values(fire_series)

    fire_id = attributes.get("fire_id")
    fire_name_value = attributes.get("fire_name")
    fire_slug = unique_fire_name
    fire_display_name = str(fire_name_value).strip() if fire_name_value else fire_slug

    fire_date = attributes.get("ignition_date")
    if not fire_date:
        fallback_raw = _first_valid_value(fire_series, ("capt_date", "capture_date", "capture_da"))
        fire_date = process_date(fallback_raw)
    if not fire_date:
        raise ValueError(
            f"Could not determine ignition date for fire '{fire_display_name}'."
        )

    extinguish_date_value = attributes.get("date_retrieved") or attributes.get("extinguish_date")
    extinguish_date = extinguish_date_value if extinguish_date_value else "None"

    start_date_pre = (
        datetime.strptime(fire_date, "%Y-%m-%d") - timedelta(days=config.pre_fire_buffer_days)
    ).strftime("%Y-%m-%d")
    end_date_pre = (
        datetime.strptime(fire_date, "%Y-%m-%d") - timedelta(days=1)
    ).strftime("%Y-%m-%d")

    if extinguish_date == "None":
        start_date_post = (
            datetime.strptime(fire_date, "%Y-%m-%d") + timedelta(days=config.post_fire_start_days)
        ).strftime("%Y-%m-%d")
    else:
        start_date_post = extinguish_date

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
            _append_log(
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
        min_gooddata_thresholds=(0.90,),
    )
    if post.time.size == 0:
        if log_path:
            yy = int(closest_bl.sizes.get("y", 0))
            xx = int(closest_bl.sizes.get("x", 0))
            total_px = yy * xx
            bl_valid_px = (
                int((closest_bl.oa_nbart_contiguity == 1).sum().item())
                if "oa_nbart_contiguity" in closest_bl.data_vars
                else 0
            )
            _append_log(
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
            _append_log(
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
        _append_log(
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
        attribute_col="severity",
        crs=config.output_crs,
        mask=final_severity != 0,
    )
    if severity_vectors.empty:
        print("No burn area detected for this fire.")
        return None

    severity_vectors = severity_vectors.to_crs("EPSG:4283")
    clipped = severity_vectors.clip(gpd.GeoDataFrame([fire_series], crs=poly_crs).to_crs("EPSG:4283"))

    aggregated = clipped.dissolve(by="severity").reset_index()

    if fire_id is not None:
        aggregated["fire_id"] = fire_id
    aggregated["fire_name"] = fire_name_value or fire_slug
    aggregated["ignition_date"] = fire_date
    aggregated["extinguish_date"] = extinguish_date
    fire_id_for_save, vector_filename = _build_vector_filename(
        fire_series=fire_series, attributes=attributes, fallback_slug=fire_slug
    )

    for key, value in attributes.items():
        if key in {"fire_id", "fire_name", "ignition_date", "extinguish_date"}:
            continue
        aggregated[key] = value

    base_dir = out_dir if out_dir else config.output_dir
    os.makedirs(base_dir, exist_ok=True)
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # This matches the required DEA_burn_severity_<fire_id>_<date>.json naming.
    out_vec = os.path.join(results_dir, vector_filename)
    aggregated.to_file(out_vec, driver="GeoJSON")
    print(f"Saved per-fire severity GeoJSON ({fire_id_for_save}): {out_vec}")

    out_cog_preview = os.path.join(base_dir, f"s2_postfire_preview_{fire_slug}.tif")
    write_cog(post.isel(time=0).to_array().compute(), fname=out_cog_preview, overwrite=True)
    print(f"Saved post-fire preview COG: {out_cog_preview}")

    out_cog_debug = os.path.join(base_dir, f"debug_mask_raster_{fire_slug}.tif")
    write_cog(debug_mask.compute(), fname=out_cog_debug, overwrite=True)
    print(f"Saved debug mask COG: {out_cog_debug}")

    print(f"Successfully processed fire: {fire_display_name}")
    return aggregated


def _is_valid_geojson(path: str) -> bool:
    """
    Check if a GeoJSON exists, is non-empty, and has severity column.
    """
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return False
        gdf = gpd.read_file(path)
        return (len(gdf) > 0) and ("severity" in gdf.columns)
    except Exception:
        return False


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
            fire_series=fire_series, attributes=fire_attrs, fallback_slug=base_fire_slug
        )
        results_dir = os.path.join(fire_dir, "results")
        final_vector_path = os.path.join(results_dir, vector_filename)
        log_path = os.path.join(fire_dir, f"{base_fire_slug}_processing.log")

        if not os.path.exists(log_path):
            _append_log(
                log_path,
                f"# Processing log for {base_fire_name} (feature index {idx})",
            )
            _append_log(
                log_path,
                "# fire_name\tbaseline_scenes\tpost_scenes\tgrid\ttotal_px\tvalid_px"
                "\tburn_px\tmasked_px\tvalid_px_baseline\tvalid_px_post_any",
            )

        skip_due_to_output = False
        if not runtime.force_rebuild:
            if _is_valid_geojson(final_vector_path):
                print(
                    f"[Fire '{base_fire_name}' ({fire_id_for_save})] Local output exists & valid. Skipping."
                )
                fire_skip += 1
                skip_due_to_output = True
            elif upload_to_s3 and s3_fs is not None:
                bucket, prefix = _parse_s3_uri(s3_prefix)
                remote_key = (
                    f"{prefix}/{base_fire_slug}/results/{vector_filename}"
                )
                if _s3_key_exists_and_nonempty(s3_fs, bucket, remote_key):
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
            ok = _upload_dir_to_s3_and_cleanup(fire_dir, s3_prefix)
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
        "S3 prefix to upload each per-fire subfolder", DEFAULT_CONFIG_DICT["upload_to_s3_prefix"]
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
