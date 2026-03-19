"""
Database access helpers for burn severity processing.
"""

from __future__ import annotations

import json
from typing import Any

import geopandas as gpd
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import connection
from shapely.geometry import shape

from .configuration import RuntimeConfig


class InputDatabase:
    DB_SCHEMA = "public"

    def _get_conn(self) -> connection:
        return psycopg2.connect(
            host=self.db_host,
            dbname=self.db_name,
            port=str(self.db_port),
            user=self.db_user,
            password=self.db_password,
        )

    def __init__(self, config: RuntimeConfig) -> None:
        if not config.db_table:
            raise ValueError(
                "Configuration 'db_table' must be provided for database polygon loading."
            )
        if not config.db_columns:
            raise ValueError(
                "Configuration 'db_columns' must include at least one attribute column."
            )

        self.db_host = config.db_host
        self.db_name = config.db_name
        self.db_port = config.db_port
        self.db_user = config.db_user
        self.db_password = config.db_password
        self.db_columns = config.db_columns
        self.db_geom_column = config.db_geom_column
        self.table_identifier = sql.Identifier(InputDatabase.DB_SCHEMA, config.db_table)
        self.db_output_crs = config.db_output_crs

    def _load_polygons_from_database(self) -> gpd.GeoDataFrame:
        """
        Load polygons directly from the configured PostgreSQL/PostGIS table.
        """

        # TODO Do filtering and status checking
        
        
        select_clause = sql.SQL(", ").join(
            sql.Identifier(col) for col in self.db_columns
        )
        query = sql.SQL(
            "SELECT {fields}, ST_AsGeoJSON({geom}) AS geom_geojson FROM {table}"
        ).format(
            fields=select_clause,
            geom=sql.Identifier(self.db_geom_column),
            table=self.table_identifier,
        )

        print(f"Querying polygons from table '{self.table_identifier}'...")
        records: list[dict[str, Any]] = []
        failures = 0

        with self._get_conn() as con, con.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            print(f"Retrieved {len(rows)} rows from database.")
            for idx, row in enumerate(rows):
                attr_values = row[:-1]
                geom_raw = row[-1]
                try:
                    json_start = geom_raw.find("{")
                    geom_str = (
                        geom_raw[json_start:] if json_start >= 0 else geom_raw
                    )
                    geom_mapping = json.loads(geom_str)
                    geom_obj = shape(geom_mapping)
                except Exception as exc:
                    failures += 1
                    print(f"Warning: Failed to parse geometry for row {idx}: {exc}")
                    continue

                record = {
                    col: val for col, val in zip(self.db_columns, attr_values)
                }
                record["geometry"] = geom_obj
                records.append(record)

        if failures:
            print(f"Skipped {failures} rows due to geometry parsing errors.")
        if not records:
            return gpd.GeoDataFrame(
                columns=[*self.db_columns, "geometry"], crs=self.db_output_crs
            )

        return gpd.GeoDataFrame(records, crs=self.db_output_crs)


    def load_and_prepare_polygons(self) -> gpd.GeoDataFrame | None:
        """
        Loads fire polygons from the configured database and prepares them for processing.
        Dissolves by 'fire_id' if available to ensure one row per fire.
        """
        try:
            poly_gdf = self._load_polygons_from_database()
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
            print(
                f"Warning: Could not dissolve polygon ({exc}). Continuing with loaded data."
            )

        if "fire_id" not in poly_gdf.columns:
            poly_gdf["fire_id"] = list(poly_gdf.index)

        return poly_gdf


    def pre_process_polygons(self) -> None:
        pass
