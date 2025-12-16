"""
Database access helpers for burn severity processing.
"""

from __future__ import annotations
from datetime import datetime, timedelta
import json
from typing import Any

import geopandas as gpd
from shapely.geometry import shape
from datacube.utils.geometry import Geometry
from .configuration import RuntimeConfig
from .date_time import process_date

DB_SCHEMA = "public"

def perform_spatial_dissolve(poly, id_list) -> gpd.GeoDataFrame:
    """takes a list of ids and the corrosponding geopandas dataframe and checks for spatial overlap 
    to see if polygons are actually the same fire or not"""
    for fire in is_list:
        subset = poly[poly['fire_id'] == fire].copy()
        
        # Find largest
        area_list = subset['area_ha'].unique()
        largest = subset[subset['area_ha'] == area_list.max()]
        largest_dc = Geometry(largest.geometry.iloc[0], crs=poly.crs)
        
         # Check overlaps
        do_they_overlap = subset.intersects(largest_dc.geom)
        
        # Non-overlapping shapes
        unique_shapes = subset[~do_they_overlap]
        for idx, row in unique_shapes.iterrows():
            geom_to_process.append(gpd.GeoDataFrame(row.to_frame().T, geometry='geometry', crs=poly.crs))  # Convert Series to DataFrame
        
        # Overlapping shapes dissolved
        non_unique_shapes = subset[do_they_overlap]
        
        agg = {col: "first" for col in subset.columns if col not in ["geometry", "ignition_date", "capt_date", "date_processed"]}
        agg.update({"ignition_date": "min", "capt_date": "min", "date_processed": "max"})
        
        combine_to_one = non_unique_shapes.dissolve(aggfunc=agg)
        geom_to_process.append(combine_to_one)
    return final_gdf_to_process = gpd.GeoDataFrame(pd.concat(geom_to_process, ignore_index=True), crs=poly.crs)

def load_polygons_from_database(config: RuntimeConfig) -> gpd.GeoDataFrame:
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
    filters polygons by date -> only process mature polygons.
    drops polygons with no area
    dissolves polygons with same id and spatial overlap.
    Dissolves by 'fire_id' if available to ensure one row per fire.
    """
    try:
        poly_gdf = load_polygons_from_database(config)
    except Exception as exc:
        print(f"Error: Failed loading polygons from database: {exc}")
        return None

    # drop very small polygons
    poly_gdf = poly_gdf.drop(poly_gdf[poly_gdf.area_ha < 1].index)
    
    if "fire_id" not in poly_gdf.columns:
        poly_gdf["fire_id"] = list(poly_gdf.index)

    else:
        #make a list of id's 
        list_of_fires = list(set(poly_gdf['fire_id']))
    
        poly_gpd = perform_spatial_dissolve(poly_gdf, list_of_fires)

    today_date = datetime.now()

    cut_off_date = date.strftime((today_date - timedelta(days=70)), "%Y-%m-%dT%H:%M:%S.%fZ")

    poly_gpd["date_processed"] = poly_gpd["date_processed"].apply(process_date)
    
    mature_polygons = poly_gdf.drop(poly_gdf[poly_gdf.date_processed < cut_off_date].index)

    return mature_polygons


__all__ = ["DB_SCHEMA", "load_and_prepare_polygons", "load_polygons_from_database"]
