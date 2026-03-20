"""
Database access helpers for burn severity processing.
"""

from __future__ import annotations
from datetime import date, timedelta

import json
from typing import Any

import geopandas as gpd
import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import connection
from shapely.geometry import shape

import dea_burn_severity.burn_severity_config as burn_config


class InputDatabase:
    DB_SCHEMA = "public"

    DB_COLUMNS = [
        "fire_id",
        "fire_name",
        "fire_type",
        "ignition_date",
        "capt_date",
        "capt_method",
        "area_ha",
        "perim_km",
        "state",
        "agency",
        "date_retrieved",
        "date_processed",
    ]

    DB_GEO_COLUMN = "geom"
    DB_OUTPUT_CRS: str = "EPSG:4283"

    def _get_conn(self) -> connection:
        return psycopg2.connect(
            host=self.db_host,
            dbname=self.db_name,
            port=str(self.db_port),
            user=self.db_user,
            password=self.db_password,
        )

    def __init__(
        self,
        # *,
        db_host: str | None = burn_config.db_host,
        db_name: str | None = burn_config.db_name,
        db_port: int | None = burn_config.db_port,
        db_user: str | None = burn_config.db_user,
        db_password: str | None = burn_config.db_password,
    ) -> None:

        # Anything not passed through will be loaded from the env. Anything missing at this stage will
        # cause issues so will throw errors.

        if not db_host:
            raise ValueError(
                "Configuration 'db_host' must be provided for database polygon loading."
            )
        if not db_name:
            raise ValueError(
                "Configuration 'db_name' must be provided for database polygon loading."
            )
        if not db_port:
            raise ValueError(
                "Configuration 'db_port' must be provided for database polygon loading."
            )
        if not db_user:
            raise ValueError(
                "Configuration 'db_user' must be provided for database polygon loading."
            )
        if not db_password:
            raise ValueError(
                "Configuration 'db_password' must be provided for database polygon loading."
            )

        self.db_host = db_host
        self.db_name = db_name
        self.db_port = db_port
        self.db_user = db_user
        self.db_password = db_password
        self.table_identifier = sql.Identifier(
            InputDatabase.DB_SCHEMA, burn_config.db_table
        )

    def load_polygons_from_database(self) -> gpd.GeoDataFrame:
        """
        Load all polygons directly from the configured PostgreSQL/PostGIS table.
        """

        select_clause = sql.SQL(", ").join(
            sql.Identifier(col) for col in InputDatabase.DB_COLUMNS
        )
        query = sql.SQL(
            "SELECT {fields}, ST_AsGeoJSON({geom}) AS geom_geojson FROM {table}"
        ).format(
            fields=select_clause,
            geom=sql.Identifier(InputDatabase.DB_GEO_COLUMN),
            table=self.table_identifier,
        )

        print(f"Querying polygons from table '{self.table_identifier}'...")
        records: list[dict[str, Any]] = []
        failures = 0

        with self._get_conn() as con, con.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            print(f"Retrieved {len(rows)} rows from database.")
            # TODO can we parse on get instead of this?
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

                record = {
                    col: val for col, val in zip(InputDatabase.DB_COLUMNS, attr_values)
                }
                record["geometry"] = geom_obj
                records.append(record)

        if failures:
            print(f"Skipped {failures} rows due to geometry parsing errors.")
        if not records:
            return gpd.GeoDataFrame(
                columns=[*InputDatabase.DB_COLUMNS, "geometry"],
                crs=InputDatabase.DB_OUTPUT_CRS,
            )

        return gpd.GeoDataFrame(records, crs=InputDatabase.DB_OUTPUT_CRS)

    def load_and_prepare_polygons(self) -> gpd.GeoDataFrame | None:
        """
        Loads fire polygons from the configured database and prepares them for processing.
        Dissolves by 'fire_id' if available to ensure one row per fire.
        """
        try:
            # Fetch all rows.
            poly_gdf = self.load_polygons_from_database()
        except Exception as exc:
            print(f"Error: Failed loading polygons from database: {exc}")
            return None

        # Run pre-filtering
        return perform_pre_filter(poly_gdf)


def perform_pre_filter(poly: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Filter results.

    Does the following:
        - Remove any that have area < 1ha
        - Removes any entries with invalid geometry
        - For all sets of fire_id (entries with the same id)
            - Find overlapping polygons
            - Create a single merged polygon for all entries
        - Removes any entries that are less than 65 days old
    """

    # filter on size
    # TODO mark everything filtered out of here as a "won't do" with the right reason.
    print(f"Started with: {len(poly)}")
    
    filtered = poly.drop(poly[poly.area_ha < 1].index)
    print(f"After removing small entries: {len(filtered)}")
    
    # Filter out invalid polygons
    # TODO investigate why there are so many bad ones?
    filtered = filtered.loc[filtered.geometry.is_valid]
    print(f"After removing invalid entries: {len(filtered)}")

    unique_fire_ids = list(set(filtered["fire_id"]))

    # TODO we need to capture the ones that got dropped so we can mark them as won't do
    filtered = perform_spatial_dissolve(filtered, unique_fire_ids)

    print(f"After dissolving overlapping entries: {len(filtered)}")

    # Age filtering
    today_date = date.today()

    cutoff_date = today_date - timedelta(days=65)

    filtered = filtered[filtered.date_processed <= cutoff_date]

    # Make all nan 0 to eliminate cate's date filtering headache!!
    filtered = filtered.fillna(0)
    
    print(f"After removing fires processed < 65 days ago: {len(filtered)}")

    return filtered


# Taken from notebook helper
def perform_spatial_dissolve(
    poly: pd.DataFrame, id_list: list[str]
) -> gpd.GeoDataFrame:
    """takes a list of ids and the corrosponding geopandas dataframe and checks for spatial overlap
    to see if polygons are actually the same fire or not"""

    geom_to_process = []

    for fire in id_list:
        subset: pd.DataFrame = poly[poly["fire_id"] == fire].copy()

        # Find largest
        area_list = subset["area_ha"].unique()
        largest = subset[subset["area_ha"] == area_list.max()]
        largest_dc = largest.geometry.iloc[0]

        # Check overlaps
        do_they_overlap = subset.geometry.intersects(largest_dc)

        # Non-overlapping shapes
        unique_shapes = subset[~do_they_overlap]
        for idx, row in unique_shapes.iterrows():
            geom_to_process.append(
                gpd.GeoDataFrame(row.to_frame().T, geometry="geometry", crs=poly.crs)
            )  # Convert Series to DataFrame

        # Overlapping shapes dissolved
        non_unique_shapes = subset[do_they_overlap]

        agg = {
            col: "first"
            for col in subset.columns
            if col not in ["geometry", "ignition_date", "capt_date", "date_processed"]
        }
        agg.update(
            {"ignition_date": "min", "capt_date": "min", "date_processed": "max"}
        )

        combine_to_one = non_unique_shapes.dissolve(aggfunc=agg)
        geom_to_process.append(combine_to_one)
    return gpd.GeoDataFrame(pd.concat(geom_to_process, ignore_index=True), crs=poly.crs)
