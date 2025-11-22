"""
Result I/O helpers (S3 interactions, GeoJSON checks, and related utilities).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any

import geopandas as gpd


def read_geojson_maybe_s3(path: str) -> gpd.GeoDataFrame:
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


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    if not s3_uri.lower().startswith("s3://"):
        raise ValueError(f"Not an S3 URI: {s3_uri}")
    no_scheme = s3_uri[5:]
    bucket, _, key = no_scheme.partition("/")
    return bucket, key.rstrip("/")


def s3_key_exists_and_nonempty(fs: Any, bucket: str, key: str) -> bool:
    s3_path = f"{bucket}/{key}"
    try:
        if fs.exists(s3_path):
            info = fs.info(s3_path)
            return info.get("Size", 0) > 0
        return False
    except Exception:
        return False


def upload_dir_to_s3_and_cleanup(local_dir: str, s3_prefix: str) -> bool:
    """
    Upload local_dir recursively into the supplied S3 prefix and, on success,
    delete local_dir. All files land directly under the configured prefix, so
    there is a single folder on S3 for every fire's artefacts.
    """
    if not os.path.isdir(local_dir):
        print(f"[S3 upload] Local directory does not exist: {local_dir}")
        return False

    import s3fs  # type: ignore

    bucket, key_prefix = parse_s3_uri(s3_prefix)
    dest_dir_key = key_prefix.strip("/")

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

    if dest_dir_key:
        target_display = f"s3://{bucket}/{dest_dir_key}/"
    else:
        target_display = f"s3://{bucket}/"

    print(f"[S3 upload] Uploading '{local_dir}' -> '{target_display}' ...")
    for rel, _, full in local_files:
        if dest_dir_key:
            remote_key = f"{dest_dir_key}/{rel}"
        else:
            remote_key = rel
        fs.put(full, f"{bucket}/{remote_key}")

    shutil.rmtree(local_dir, ignore_errors=True)
    return True


def is_valid_geojson(path: str) -> bool:
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


__all__ = [
    "is_valid_geojson",
    "parse_s3_uri",
    "read_geojson_maybe_s3",
    "s3_key_exists_and_nonempty",
    "upload_dir_to_s3_and_cleanup",
]
