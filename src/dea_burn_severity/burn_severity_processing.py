import os
import re
import shutil
import traceback
from datetime import datetime, timedelta
from typing import Any, Iterable

import datacube
import geopandas as gpd
import numpy as np
import pandas as pd
from datacube.utils.cog import write_cog
from datacube.utils.geometry import CRS, Geometry
from dea_tools.bandindices import calculate_indices
from dea_tools.spatial import xr_vectorize

from .burn_severity_config import RuntimeBurnConfig, StaticBurnConfig
from .database import InputDatabase, JobStatus

from .data_loading import load_ard_with_fallback, load_baseline_stack
from .severity import calculate_severity, create_debug_mask


class BurnSeverityProcessor:
    def __init__(self, config: RuntimeBurnConfig):
        self.burn_config = config
        self.job_status_table = InputDatabase(config).job_status_table

    def process_single_fire(
        self,
        fire_series: pd.Series,
        poly_crs: CRS,
        dc: datacube.Datacube,
        unique_fire_name: str,
        log_path: str | None = None,
        out_dir: str | None = None,
    ) -> gpd.GeoDataFrame | None:
        """
        Full burn mapping workflow for a single fire polygon.
        Returns:
            GeoDataFrame dissolved by 'severity_rating' (with 'severity_rating' column),
            reprojected to 'EPSG:3577', or None if nothing to save.
        """

        os.environ["AWS_NO_SIGN_REQUEST"] = "Yes"

        gpgon = Geometry(fire_series.geometry, crs=poly_crs)

        poly = gpd.GeoDataFrame([fire_series], crs=poly_crs).copy()

        # Transform geometry to EPSG:4283 if needed (they should all be 4283 allready
        if poly.crs.to_epsg() != 4283:
            poly = poly.to_crs("EPSG:4283")

        attributes = self._extract_attribute_values(fire_series)

        # Unique fire id for status writing.
        fire_uid = fire_series.uid

        # TODO need to clean up ids and dates, they are all over the place between both of them.
        fire_id = attributes.get("fire_id")
        fire_name_value = attributes.get("fire_name")
        fire_slug = unique_fire_name
        fire_display_name = (
            str(fire_name_value).strip() if fire_name_value else fire_slug
        )

        fire_date: Any | None = attributes.get("ignition_date")
        # use
        if not fire_date:
            fallback_raw = self._first_valid_value(
                fire_series, ("capt_date", "capture_date", "capture_da")
            )
            # flag we are using capture date not ignition date
            pre_fire_buffer = self.burn_config.pre_fire_buffer_days
            fire_date = self._process_date(fallback_raw)
            if fire_date:
                # adjust fire date IF using capture rather than ignition
                # I hope this works if capture date is missing rather than just failling
                fire_date = (
                    datetime.strptime(fire_date, "%Y-%m-%d")
                    - timedelta(days=self.burn_config.adjustment_missing_ignit_date)
                ).strftime("%Y-%m-%d")

        if not fire_date:
            self.job_status_table.set_job_status(
                fire_uid, JobStatus.FAILED, "No ignition date"
            )
            raise ValueError(
                f"Could not determine ignition date for fire '{fire_display_name}'."
            )

        process_date_value = attributes.get("date_processed")

        if process_date_value:
            extinguish_date = (
                datetime.strptime(process_date_value, "%Y-%m-%d") - timedelta(days=7)
            ).strftime("%Y-%m-%d")
        else:
            self.job_status_table.set_job_status(
                fire_uid, JobStatus.FAILED, "No ignition date"
            )
            raise ValueError(
                f"Could not determine extinguish date for fire '{fire_display_name}'."
            )

        start_date_pre = (
            datetime.strptime(fire_date, "%Y-%m-%d")
            - timedelta(days=self.burn_config.pre_fire_buffer_days)
        ).strftime("%Y-%m-%d")
        end_date_pre = (
            datetime.strptime(fire_date, "%Y-%m-%d") - timedelta(days=1)
        ).strftime("%Y-%m-%d")

        if extinguish_date == "None":
            start_date_post = (
                datetime.strptime(fire_date, "%Y-%m-%d")
                + timedelta(days=self.burn_config.post_fire_start_days)
            ).strftime("%Y-%m-%d")
        else:
            start_date_post = extinguish_date

        end_date_post = (
            datetime.strptime(start_date_post, "%Y-%m-%d")
            + timedelta(days=self.burn_config.post_fire_window_days)
        ).strftime("%Y-%m-%d")

        landcover_year = str(int(fire_date[0:4]) - 1)

        # add in saftynet for landcover when new year starts. remove when 2025 landcover is published
        # TODO Cate - review this, it wasn't in the old code
        if int(landcover_year) > 2024:
            landcover_year = "2024"

        baseline, closest_bl = load_baseline_stack(
            dc, gpgon, time=(start_date_pre, end_date_pre)
        )
        if closest_bl is None:
            if log_path:
                self._append_log(
                    log_path,
                    f"{fire_display_name}\tbaseline_scenes=0\tpost_scenes=0\tgrid=0x0"
                    "\ttotal_px=0\tvalid_px=0\tburn_px=0\tmasked_px=0",
                )
            # TODO Cate - should this be retried in the future or will it never succeed?
            print("No baseline data for this fire. Skipping.")
            self.job_status_table.set_job_status(
                fire_uid, JobStatus.FAILED, "No baseline data"
            )
            return None

        post = load_ard_with_fallback(
            dc,
            gpgon,
            time=(start_date_post, end_date_post),
            min_gooddata_thresholds=(0.90, 0.50, 0.20),
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
                self._append_log(
                    log_path,
                    f"{fire_display_name}\tbaseline_scenes={baseline.time.size}\tpost_scenes=0"
                    f"\tgrid={yy}x{xx}\ttotal_px={total_px}\tvalid_px_baseline={bl_valid_px}"
                    "\tburn_px=0\tmasked_px=0",
                )
            print("No post-fire data for this fire. Skipping.")

            # TODO Cate - should this be retried in the future or will it never succeed?
            self.job_status_table.set_job_status(
                fire_uid, JobStatus.FAILED, "No post-fire data"
            )
            return None

        landcover = dc.load(
            product="ga_ls_landcover_class_cyear_3",
            geopolygon=gpgon,
            time=(landcover_year),
            output_crs=StaticBurnConfig.output_crs,
            resolution=StaticBurnConfig.resolution,
            group_by="solar_day",
            dask_chunks={},
        )
        # if dc.load fails to load landcover unfortunatly checking via time.size will throw error :(
        if not landcover.dims:
            # add a new landcover check that trys the year before incase anual updates are late :/
            prev_landcover_year = str(int(landcover_year) - 1)
            # try to load the previous callendar year of landcover data if it didn't work.
            # we only try this once, not indefinantly
            landcover = dc.load(
                product="ga_ls_landcover_class_cyear_3",
                geopolygon=gpgon,
                time=(prev_landcover_year),
                output_crs=StaticBurnConfig.output_crs,
                resolution=StaticBurnConfig.resolution,
                group_by="solar_day",
                dask_chunks={},
            )
            # if landcover still empty allow fail.
        if not landcover.dims:
            if log_path:
                yy = int(closest_bl.sizes.get("y", 0))
                xx = int(closest_bl.sizes.get("x", 0))
                total_px = yy * xx
                self._append_log(
                    log_path,
                    f"{fire_display_name}\tbaseline_scenes={baseline.time.size}"
                    f"\tpost_scenes={post.time.size}\tgrid={yy}x{xx}"
                    f"\ttotal_px={total_px}\tvalid_px=0\tburn_px=0\tmasked_px=0"
                    "\tlandcover=missing",
                )
            print(f"No landcover data for year {landcover_year}. Skipping.")
            # TODO Cate - should this be retried in the future or will it never succeed?
            self.job_status_table.set_job_status(
                fire_uid, JobStatus.FAILED, "No landcover data"
            )
            return None
        landcover = landcover.isel(time=0)

        pre_nbr = calculate_indices(
            closest_bl, index="NBR", collection="ga_s2_3", drop=True
        )
        post_nbr = calculate_indices(post, index="NBR", collection="ga_s2_3", drop=True)
        min_post_nbr = post_nbr.min("time")
        delta_nbr = pre_nbr - min_post_nbr

        severity = calculate_severity(
            delta_nbr, landcover, StaticBurnConfig.grass_classes
        )

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
                post.oa_nbart_contiguity.max("time")
                if "oa_nbart_contiguity" in post.data_vars
                else None
            )
            post_valid_px_any = (
                int((post_any_contig == 1).sum().item())
                if post_any_contig is not None
                else 0
            )
        except Exception:
            bl_valid_px = 0
            post_valid_px_any = 0

        if log_path:
            self._append_log(
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
            crs=StaticBurnConfig.output_crs,
        )
        # remove mask by 0, we actually want to vectorise all values even 0
        if severity_vectors.empty:
            print("No burn area detected for this fire.")
            # TODO Cate - should this be retried in the future or will it never succeed?
            self.job_status_table.set_job_status(
                fire_uid, JobStatus.FAILED, "No burn area detected"
            )
            return None

        # reproject burn extent
        albers_extent = gpd.GeoDataFrame([fire_series], crs=poly_crs).to_crs(
            "EPSG:3577"
        )

        # check severity is albers
        if severity_vectors.crs.to_epsg() != 3577:
            severity_vectors = severity_vectors.to_crs("EPSG:3577")
        # clip severity output to fire extent in albers
        clipped = severity_vectors.clip(albers_extent)

        aggregated = clipped.dissolve(by="severity_rating").reset_index()

        # calculate and add area
        aggregated["area_ha"] = round((aggregated["geometry"].area) / 10000, 2)

        # calculate and add perimiter (length measures perimiter of a multipoly)
        aggregated["perim_km"] = round((aggregated["geometry"].length) / 1000, 2)

        # add severity lable attribues as desiered
        aggregated["severity_class"] = aggregated["severity_rating"].map(
            StaticBurnConfig.severity_class_name
        )

        # add our assumed extinguish date to shapefile
        aggregated["extinguish_date"] = extinguish_date

        if fire_id is not None:
            aggregated["fire_id"] = fire_id
        aggregated["fire_name"] = fire_name_value or fire_slug
        aggregated["ignition_date"] = fire_date
        aggregated["extinguish_date"] = extinguish_date
        fire_id_for_save, vector_filename = self._build_vector_filename(
            fire_series=fire_series, attributes=attributes, fallback_slug=fire_slug
        )

        for key, value in attributes.items():
            if key in {
                "fire_id",
                "fire_name",
                "ignition_date",
                "extinguish_date",
                "area_ha",
                "perim_km",
                "date_retrieved",
                "date_processed",
            }:
                continue
            aggregated[key] = value

        base_dir = out_dir if out_dir else self.burn_config.output_dir
        os.makedirs(base_dir, exist_ok=True)
        results_dir = os.path.join(base_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        # This matches the required DEA_burn_severity_<fire_id>_<date>.json naming.
        out_vec = os.path.join(results_dir, vector_filename)
        aggregated.to_file(out_vec, driver="GeoJSON")
        print(f"Saved per-fire severity GeoJSON ({fire_id_for_save}): {out_vec}")

        out_cog_preview = os.path.join(base_dir, f"s2_postfire_preview_{fire_slug}.tif")
        write_cog(
            post.isel(time=0).to_array().compute(),
            fname=out_cog_preview,
            overwrite=True,
        )
        print(f"Saved post-fire preview COG: {out_cog_preview}")

        out_cog_debug = os.path.join(base_dir, f"debug_mask_raster_{fire_slug}.tif")
        write_cog(debug_mask.compute(), fname=out_cog_debug, overwrite=True)
        print(f"Saved debug mask COG: {out_cog_debug}")

        print(f"Successfully processed fire: {fire_display_name}")
        return aggregated

    def prep_outputs(self) -> None:
        os.makedirs(self.burn_config.output_dir, exist_ok=True)
        print(f"All outputs will be saved to: {self.burn_config.output_dir}")

    def _is_valid_geojson(self, path: str) -> bool:
        """
        Check if a GeoJSON exists, is non-empty, and has severity column.
        """
        try:
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                return False
            gdf = gpd.read_file(path)
            return (len(gdf) > 0) and ("severity_rating" in gdf.columns)
        except Exception:
            return False

    def _clean_fire_slug(self, name: str | None) -> str:
        if not name:
            return ""
        cleaned = re.sub(r"[^A-Za-z0-9\s]", "", str(name))
        cleaned = re.sub(r"\s+", "_", cleaned).strip("_")
        return cleaned

    def _first_valid_value(self, series: pd.Series, keys: Iterable[str]) -> Any | None:
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

    def _append_log(self, log_path: str, line: str) -> None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as file:
            file.write(line.rstrip("\n") + "\n")
            file.flush()
            os.fsync(file.fileno())

    def _process_date(self, value: Any) -> str | None:
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

    def _upload_product_to_s3(self, file_path: str, s3_prefix: str):
        if not os.path.exists(file_path):
            print(f"[S3 upload] File does not exist: {file_path}")
            return False

        import boto3

        bucket, key_prefix = self._parse_s3_uri(s3_prefix)
        dest_dir_key = key_prefix.strip("/")
        remote_key = f"{dest_dir_key}/{os.path.basename(file_path)}"

        s3_client = boto3.client("s3")
        s3_client.upload_file(Filename=str(file_path), Bucket=bucket, Key=remote_key)
        print(f"[S3 upload] File uploaded to: {remote_key}")
        return True

    def _cleanup(self, local_dir: str) -> bool:
        """
        Clean up the local_dir.
        """
        if not os.path.isdir(local_dir):
            print(f"[S3 upload] Local directory does not exist: {local_dir}")
            return False

        shutil.rmtree(local_dir, ignore_errors=True)
        return True

    def _format_results_filename(
        self, identifier_source: Any | None, date_source: Any | None, fallback_slug: str
    ) -> tuple[str, str]:
        """Return file-friendly identifier/date tuple for DEA outputs."""
        identifier = (
            self._clean_fire_slug(str(identifier_source).strip())
            if identifier_source not in (None, "")
            else ""
        )
        if not identifier:
            identifier = fallback_slug
        date_value = self._process_date(date_source)
        if not date_value:
            date_value = datetime.utcnow().strftime("%Y-%m-%d")
        fire_id_for_save = f"{identifier}_{date_value}"
        return fire_id_for_save, f"DEA_burn_severity_{fire_id_for_save}.geojson"

    def _build_vector_filename(
        self, fire_series: pd.Series, attributes: dict[str, Any], fallback_slug: str
    ) -> tuple[str, str]:
        identifier_src = attributes.get("fire_id")
        if identifier_src in (None, ""):
            identifier_src = self._first_valid_value(
                fire_series, InputDatabase.FIRE_ID_FIELDS
            )
        processed_date_src = attributes.get("date_processed")
        if processed_date_src in (None, ""):
            processed_date_src = self._first_valid_value(
                fire_series, ("date_processed", "date_proce")
            )
        return self._format_results_filename(
            identifier_src, processed_date_src, fallback_slug
        )

    def _extract_attribute_values(self, series: pd.Series) -> dict[str, Any]:
        attributes: dict[str, Any] = {}
        for rule in InputDatabase.ATTRIBUTE_COPY_RULES:
            raw_value = self._first_valid_value(series, rule["sources"])
            if raw_value is None:
                continue
            if rule["date"]:
                formatted = self._process_date(raw_value)
                if formatted is None:
                    continue
                attributes[rule["target"]] = formatted
            else:
                attributes[rule["target"]] = raw_value
        return attributes

    def _parse_s3_uri(self, s3_uri: str) -> tuple[str, str]:
        if not s3_uri.lower().startswith("s3://"):
            raise ValueError(f"Not an S3 URI: {s3_uri}")
        no_scheme = s3_uri[5:]
        bucket, _, key = no_scheme.partition("/")
        return bucket, key.rstrip("/")

    def _s3_key_exists_and_nonempty(self, fs: Any, bucket: str, key: str) -> bool:
        s3_path = f"{bucket}/{key}"
        try:
            if fs.exists(s3_path):
                info = fs.info(s3_path)
                return info.get("Size", 0) > 0
            return False
        except Exception:
            return False

    def process_all_polygons(self, all_polys: gpd.GeoDataFrame) -> None:

        os.makedirs(self.burn_config.output_dir, exist_ok=True)
        print(f"All outputs will be saved to: {self.burn_config.output_dir}")

        upload_to_s3 = self.burn_config.upload_to_s3
        s3_prefix = f"s3://{self.burn_config.upload_to_s3_bucket}/{self.burn_config.upload_to_s3_path}"

        s3_fs = None
        if upload_to_s3:
            try:
                import s3fs  # type: ignore

                s3_fs = s3fs.S3FileSystem(anon=False)
            except Exception as exc:
                print(
                    "Warning: '--upload-to-s3-prefix' requested but 's3fs' not available."
                )
                print("         Install with: pip install s3fs")
                print(f"Details: {exc}")
                upload_to_s3 = False

        dc = datacube.Datacube(app=self.burn_config.dc_app_name)

        if all_polys is None or all_polys.empty:
            print("No polygons loaded. Exiting.")
            return

        print(f"Found {len(all_polys)} total features. Processing all of them.")
        fire_success = fire_fail = fire_skip = 0

        print("\nBeginning per-fire processing (single polygon per feature)...")
        for _, fire_series in all_polys.iterrows():
            fire_uid = fire_series["uid"]
            fire_attrs = self._extract_attribute_values(fire_series)
            if fire_attrs.get("fire_name"):
                base_fire_name = str(fire_attrs["fire_name"]).strip()
            elif fire_attrs.get("fire_id") is not None:
                base_fire_name = f"fire_id_{fire_attrs['fire_id']}"
            else:
                base_fire_name = "fire"

            # Always append the fire UID to the name so it is unique for the row, even if the fire name is reused
            base_fire_name += f" {fire_uid}"

            base_fire_slug = self._clean_fire_slug(base_fire_name)
            if not base_fire_slug:
                base_fire_slug = f"fire_{fire_uid}"
            base_fire_slug = base_fire_slug.replace(os.sep, "_")
            if os.altsep:
                base_fire_slug = base_fire_slug.replace(os.altsep, "_")

            fire_dir = os.path.join(self.burn_config.output_dir, base_fire_slug)
            os.makedirs(fire_dir, exist_ok=True)

            fire_id_for_save, vector_filename = self._build_vector_filename(
                fire_series=fire_series,
                attributes=fire_attrs,
                fallback_slug=base_fire_slug,
            )
            results_dir = os.path.join(fire_dir, "results")
            final_vector_path = os.path.join(results_dir, vector_filename)
            log_path = os.path.join(fire_dir, f"{base_fire_slug}_processing.log")

            if not os.path.exists(log_path):
                self._append_log(
                    log_path,
                    f"# Processing log for {base_fire_name} (uid {fire_uid})",
                )
                self._append_log(
                    log_path,
                    "# fire_name\tbaseline_scenes\tpost_scenes\tgrid\ttotal_px\tvalid_px"
                    "\tburn_px\tmasked_px\tvalid_px_baseline\tvalid_px_post_any",
                )

            skip_due_to_output = False
            if not self.burn_config.force_rebuild:
                if self._is_valid_geojson(final_vector_path):
                    print(
                        f"[Fire '{base_fire_name}' ({fire_id_for_save})] Local output exists & valid. Skipping."
                    )
                    fire_skip += 1
                    skip_due_to_output = True
                elif upload_to_s3 and s3_fs is not None:
                    bucket, prefix = self._parse_s3_uri(s3_prefix)
                    prefix = prefix.strip("/")
                    prefix_part = f"{prefix}/" if prefix else ""
                    remote_key = f"{prefix_part}results/{vector_filename}"
                    if self._s3_key_exists_and_nonempty(s3_fs, bucket, remote_key):
                        print(
                            f"[Fire '{base_fire_name}' ({fire_id_for_save})] Output exists in S3. Skipping."
                        )
                        fire_skip += 1
                        skip_due_to_output = True

            if skip_due_to_output:
                # TODO check the status we want here, maybe compelte?
                self.job_status_table.set_job_status(
                    fire_series.uid, JobStatus.SKIPPED, "Output already exists"
                )
                continue

            print("\n" + "=" * 80)
            print(f"Processing fire polygon: '{base_fire_name}'")
            print("=" * 80)

            try:
                gdf_fire = self.process_single_fire(
                    fire_series=fire_series,
                    poly_crs=all_polys.crs,
                    dc=dc,
                    unique_fire_name=base_fire_slug,
                    log_path=log_path,
                    out_dir=fire_dir,
                )
                if gdf_fire is not None and len(gdf_fire) > 0:
                    fire_success += 1
            except Exception as exc:
                fire_fail += 1
                print(f"!!! FAILED to process fire '{base_fire_name}': {exc}")
                traceback.print_exc()
                self.job_status_table.set_job_status(
                    fire_series.uid, JobStatus.FAILED, str(exc)
                )
                continue

            if upload_to_s3:
                ok = self._upload_product_to_s3(final_vector_path, s3_prefix)
                self._cleanup(fire_dir)
                if not ok:
                    self.job_status_table.set_job_status(
                        fire_series.uid, JobStatus.UNPROCESSED, "Upload failed"
                    )
                    print(
                        f"[S3 upload] WARNING: '{fire_dir}' was not removed (upload verification failed)."
                    )
                    continue
            self.job_status_table.set_job_status(
                fire_series.uid, JobStatus.COMPLETE, None
            )

        print("\n" + "=" * 80)
        print("Batch processing complete.")
        print(f"  Fire success: {fire_success}")
        print(f"  Fire failed:  {fire_fail}")
        print(f"  Fire skipped: {fire_skip}")
        print("=" * 88)
