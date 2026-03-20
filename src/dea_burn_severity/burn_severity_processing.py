import os
import re
from datetime import datetime

import geopandas as gpd
import pandas as pd
import numpy as np

from typing import Any, Iterable

import dea_burn_severity.burn_severity_config as burn_config
from dea_burn_severity.database import InputDatabase


def prep_outputs() -> None:
    # TODO this needs to be configurable for the notebook
    os.makedirs(burn_config.output_dir, exist_ok=True)
    print(f"All outputs will be saved to: {burn_config.output_dir}")


def _clean_fire_slug(name: str | None) -> str:
    if not name:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9\s]", "", str(name))
    cleaned = re.sub(r"\s+", "_", cleaned).strip("_")
    return cleaned


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
    if bool(re.match(r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$", text)):
        return text
    digits = re.sub(r"[^0-9]", "", text)
    if len(digits) >= 8:
        return f"{digits[0:4]}-{digits[4:6]}-{digits[6:8]}"
    return None


def _format_results_filename(
    identifier_source: Any | None, date_source: Any | None, fallback_slug: str
) -> tuple[str, str]:
    """Return file-friendly identifier/date tuple for DEA outputs."""
    identifier = _clean_fire_slug(str(identifier_source).strip()) if identifier_source not in (None, "") else ""
    if not identifier:
        identifier = fallback_slug
    date_value = process_date(date_source)
    if not date_value:
        date_value = datetime.utcnow().strftime("%Y-%m-%d")
    fire_id_for_save = f"{identifier}_{date_value}"
    return fire_id_for_save, f"DEA_burn_severity_{fire_id_for_save}.json"


def _build_vector_filename(
    fire_series: pd.Series, fallback_slug: str
) -> tuple[str, str]:
    identifier_src = fire_series.get("fire_id")
    print("ID:", identifier_src)
    if identifier_src in (None, ""):
        identifier_src = _first_valid_value(fire_series, ("fire_id",))
    processed_date_src = fire_series.get("date_processed")
    if processed_date_src in (None, ""):
        processed_date_src = _first_valid_value(fire_series, ("date_processed", "date_proce"))
    return _format_results_filename(identifier_src, processed_date_src, fallback_slug)



def process_all_polygons(polygons: gpd.GeoDataFrame) -> None:
    for idx, entry in polygons[:100].iterrows():
        # Construct the output name based on the attributes.
        # TODO there will be dupes this way, everything can be repeated except the last thing
        # TODO maybe put the idx/rowid in the thing always?
        base_fire_slug = entry["fire_name"] or entry["fire_id"] or f"fire_{idx}"
        base_fire_slug = _clean_fire_slug(base_fire_slug)
        print(base_fire_slug)

        fire_dir = os.path.join(burn_config.output_dir, base_fire_slug)
        os.makedirs(fire_dir, exist_ok=True)
        
        fire_id_for_save, vector_filename = _build_vector_filename(
            fire_series=entry, fallback_slug=base_fire_slug
        )
        results_dir = os.path.join(fire_dir, "results")
        final_vector_path = os.path.join(results_dir, vector_filename)
        log_path = os.path.join(fire_dir, f"{base_fire_slug}_processing.log")


        # TODO finish the output path setup on line 470ish of cli.py

        # ...

        # map_burn_severity(entry)

        # ...

        # ... write result to db and anywhere else it needs to be


# for fire in list...
def map_burn_severity() -> None:
    pass


def write_output() -> None:
    # write cog
    # write polyg on
    # write to log
    pass
