"""
Database access helpers for burn severity processing.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any

import geopandas as gpd
import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import connection
from psycopg2.extras import execute_values
from shapely.geometry import shape

from dea_burn_severity.burn_severity_config import RuntimeBurnConfig


class JobStatus:
    UNPROCESSED: str = "UNPROCESSED"
    COMPLETE: str = "COMPLETE"
    SKIPPED_INVALID: str = "SKIPPED_INVALID"
    SKIPPED: str = "SKIPPED"
    FAILED: str = "FAILED"


class JobStatusTable:
    """Class for interfacing with the job database. Requires a write connection to function correctly."""

    """Creation of this database can be done with the following SQL snippet. This will always be performed manually so
    python code will not be provided.
    
    ALTER TABLE nli_lastboundaries_trigger ADD CONSTRAINT unique_uid UNIQUE (uid);

    CREATE TABLE processing_status (
        trigger_uid integer PRIMARY KEY
            REFERENCES nli_lastboundaries_trigger(uid)
            ON DELETE CASCADE,

        status text NOT NULL,
            
        lastmod timestamptz NOT NULL DEFAULT now(),

        message text
    );
    
    GRANT SELECT, INSERT, UPDATE, DELETE ON processing_status TO pipeline_user;
    
    ---
    Valid status values:
        - UNPROCESSED (new or not ready)
        - COMPLETE
        - SKIPPED_INVALID
        - SKIPPED (valid reason like merged together)
        - FAILED (with reason)
    """

    def __init__(self, db: InputDatabase):
        self.db = db
        self.status_table_identifier = sql.Identifier(
            InputDatabase.DB_SCHEMA, db.burn_config.db_status_table
        )

    def update_available_jobs(self):
        """Find any new entries from the trigger table that have not got a status associated and add them as pending."""
        with self.db._get_conn() as con, con.cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    """
                    INSERT INTO {status_table} (trigger_uid, status, lastmod, message)
                    SELECT t.uid, {status_value}, now(), NULL
                    FROM {trigger_table} t
                    WHERE NOT EXISTS (
                    SELECT 1
                    FROM {status_table} js
                    WHERE js.trigger_uid = t.uid
                    );
                """
                ).format(
                    status_value=sql.Literal(JobStatus.UNPROCESSED),
                    trigger_table=self.db.table_identifier,
                    status_table=self.status_table_identifier,
                )
            )

    def set_job_status(self, uid: int, status: str, message: str | None):
        return self.set_job_status_bulk([([uid], status, message)])

    def set_job_status_bulk(self, entries: list[tuple[list[int], str, str | None]]):
        """Set the status and message for a job."""

        if not self.db.burn_config.db_use_status_table:
            return

        # Flatten the input entries
        rows: list[tuple[int, str, str | None]] = []

        for uids, status, message in entries:
            rows.extend((uid, status, message) for uid in uids)

        with self.db._get_conn() as con, con.cursor() as cursor:
            query = sql.SQL("""
                UPDATE {status_table} AS js
                SET status  = v.status,
                    message = v.message,
                    lastmod = now()
                FROM (VALUES %s) AS v(trigger_uid, status, message)
                WHERE js.trigger_uid = v.trigger_uid
            """).format(status_table=self.status_table_identifier)

            execute_values(
                cursor,
                query.as_string(cursor),
                rows,
                page_size=1000,
            )

            con.commit()

        print(f"Updated status for {len(rows)} entries")

    def report_status(self):
        """Return a textual report with stats on what the current state of the system is."""
        # TODO implement
        pass


class InputDatabase:
    DB_SCHEMA = "public"

    DB_COLUMNS = [
        "uid",
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
        {
            "target": "capt_method",
            "sources": ("capt_method", "capt_metho"),
            "date": False,
        },
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

    FIRE_ID_FIELDS: tuple[str, ...] = ("fire_id",)

    def _get_conn(self) -> connection:
        return psycopg2.connect(
            host=self.db_host,
            dbname=self.db_name,
            port=str(self.db_port),
            user=self.db_user,
            password=self.db_password,
        )

    def __init__(self, config: RuntimeBurnConfig) -> None:

        # Anything not passed through will be loaded from the env. Anything missing at this stage will
        # cause issues so will throw errors.
        if not config.db_host:
            raise ValueError(
                "Configuration 'db_host' must be provided for database polygon loading."
            )
        if not config.db_name:
            raise ValueError(
                "Configuration 'db_name' must be provided for database polygon loading."
            )
        if not config.db_port:
            raise ValueError(
                "Configuration 'db_port' must be provided for database polygon loading."
            )
        if not config.db_user:
            raise ValueError(
                "Configuration 'db_user' must be provided for database polygon loading."
            )
        if not config.db_password:
            raise ValueError(
                "Configuration 'db_password' must be provided for database polygon loading."
            )

        self.burn_config = config
        self.db_host = config.db_host
        self.db_name = config.db_name
        self.db_port = config.db_port
        self.db_user = config.db_user
        self.db_password = config.db_password
        self.table_identifier = sql.Identifier(InputDatabase.DB_SCHEMA, config.db_table)
        self.job_status_table = JobStatusTable(self)
        self.db_use_status_table = config.db_use_status_table

    def load_polygons_from_database(self) -> gpd.GeoDataFrame:
        """
        Load all polygons directly from the configured PostgreSQL/PostGIS table.
        """
        if self.db_use_status_table:
            self.job_status_table.update_available_jobs()

            where_clause = sql.SQL("""
                JOIN {status_table} js ON js.trigger_uid = t.uid
                WHERE js.status = '{unprocessed_value}';""").format(
                status_table=self.job_status_table.status_table_identifier,
                unprocessed_value=sql.Literal(JobStatus.UNPROCESSED)
            )
        else:
            where_clause = sql.SQL("")

        select_clause = sql.SQL(", ").join(
            sql.Identifier(col) for col in InputDatabase.DB_COLUMNS
        )

        query = sql.SQL(
            "SELECT {fields}, ST_AsGeoJSON({geom}) AS geom_geojson FROM {table} t {where_clause}"
        ).format(
            fields=select_clause,
            geom=sql.Identifier(InputDatabase.DB_GEO_COLUMN),
            table=self.table_identifier,
            where_clause=where_clause,
        )

        print(f"Querying polygons from table '{self.table_identifier}'...")
        records: list[dict[str, Any]] = []

        with self._get_conn() as con, con.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            print(f"Retrieved {len(rows)} rows from database.")

            failed_geo = []
            # TODO can we parse on get instead of this?
            for row in rows:
                attr_values = row[:-1]
                geom_raw = row[-1]
                row_id = row[0]
                try:
                    json_start = geom_raw.find("{")
                    geom_str = geom_raw[json_start:] if json_start >= 0 else geom_raw
                    geom_mapping = json.loads(geom_str)
                    geom_obj = shape(geom_mapping)
                except Exception as exc:
                    failed_geo.append(row_id)
                    print(
                        f"Warning: Failed to parse geometry for row with id {row_id}: {exc}"
                    )
                    continue

                record = {
                    col: val for col, val in zip(InputDatabase.DB_COLUMNS, attr_values)
                }
                record["geometry"] = geom_obj
                records.append(record)

        if len(failed_geo) > 0:
            print(f"Skipped {len(failed_geo)} rows due to geometry parsing errors.")
            self.job_status_table.set_job_status_bulk(
                [(failed_geo, JobStatus.SKIPPED_INVALID, "Unparsable geometry")]
            )
        if not records:
            return gpd.GeoDataFrame(
                columns=[*InputDatabase.DB_COLUMNS, "geometry"],
                crs=InputDatabase.DB_OUTPUT_CRS,
            )

        return gpd.GeoDataFrame(records, crs=InputDatabase.DB_OUTPUT_CRS)

    def load_filtered_polygons(self) -> gpd.GeoDataFrame | None:
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
        return self.perform_pre_filter(poly_gdf)

    def perform_pre_filter(self, poly: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
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
        removed_entries = []
        # filter on size

        print(f"Started with: {len(poly)}")

        small_entries = poly[poly.area_ha < self.burn_config.fire_area_minimum_ha]
        removed_entries.append(
            (
                small_entries.uid.tolist(),
                JobStatus.SKIPPED,
                f"Area smaller than {self.burn_config.fire_area_minimum_ha} ha",
            )
        )

        filtered = poly.drop(small_entries.index)
        print(
            f"After removing small entries (<{self.burn_config.fire_area_minimum_ha}ha): {len(filtered)}"
        )

        # Filter out invalid polygons
        # TODO investigate why there are so many bad ones?
        removed_entries.append(
            (
                filtered.loc[~filtered.geometry.is_valid].uid.tolist(),
                JobStatus.SKIPPED_INVALID,
                "Invalid geometry",
            )
        )
        filtered = filtered.loc[filtered.geometry.is_valid]
        print(f"After removing invalid entries: {len(filtered)}")

        unique_fire_ids = list(set(filtered["fire_id"]))

        filtered, dissolved_entries = perform_spatial_dissolve(
            filtered, unique_fire_ids
        )
        removed_entries.extend(dissolved_entries)
        print(f"After dissolving overlapping entries: {len(filtered)}")

        # Age filtering
        today_date = date.today()

        cutoff_date = today_date - timedelta(
            days=self.burn_config.post_fire_window_days
        )

        removed_entries.append(
            (
                filtered[~(filtered.date_processed <= cutoff_date)].uid.tolist(),
                JobStatus.UNPROCESSED,
                "Too new",
            )
        )
        filtered = filtered[filtered.date_processed <= cutoff_date]

        # Make all nan 0 to eliminate cate's date filtering headache!!
        filtered = filtered.fillna(0)

        print(
            f"After removing fires processed <{self.burn_config.post_fire_window_days} days ago: {len(filtered)}"
        )

        self.job_status_table.set_job_status_bulk(removed_entries)

        return filtered


# Taken from notebook helper
def perform_spatial_dissolve(
    poly: pd.DataFrame, id_list: list[str]
) -> tuple[gpd.GeoDataFrame, list[tuple[list[int], str, str]]]:
    """takes a list of ids and the corrosponding geopandas dataframe and checks for spatial overlap
    to see if polygons are actually the same fire or not"""

    geom_to_process = []

    removed_entries = []

    for fire in id_list:
        subset: pd.DataFrame = poly[poly["fire_id"] == fire].copy()

        # Find largest
        area_list = subset["area_ha"].unique()
        largest = subset[subset["area_ha"] == area_list.max()].iloc[0]
        largest_dc = largest.geometry

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

        # Track the dissolved uids. Remove the uid of the new "parent" entry.
        dissolved_uids = non_unique_shapes.uid.tolist()
        dissolved_uids.remove(largest.uid)
        if len(dissolved_uids) > 0:
            removed_entries.append(
                (dissolved_uids, JobStatus.SKIPPED, f"Dissolved into {largest.uid}")
            )

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
    return gpd.GeoDataFrame(
        pd.concat(geom_to_process, ignore_index=True), crs=poly.crs
    ), removed_entries
