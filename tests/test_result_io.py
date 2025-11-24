import json
import os
import sys
from types import SimpleNamespace

import geopandas as gpd
import pytest
from shapely.geometry import Point

from dea_burn_severity import result_io


def test_parse_s3_uri_and_exists_helpers():
    bucket, key = result_io.parse_s3_uri("s3://my-bucket/path/to/object")
    assert bucket == "my-bucket"
    assert key == "path/to/object"

    class DummyFS:
        def __init__(self, exists=True, size=5):
            self.exists_called = False
            self.info_called = False
            self._exists = exists
            self._size = size

        def exists(self, path):
            self.exists_called = True
            return self._exists

        def info(self, path):
            self.info_called = True
            return {"Size": self._size}

    fs = DummyFS()
    assert result_io.s3_key_exists_and_nonempty(fs, "bucket", "key") is True
    assert fs.exists_called and fs.info_called

    fs_empty = DummyFS(size=0)
    assert result_io.s3_key_exists_and_nonempty(fs_empty, "bucket", "key") is False

    with pytest.raises(ValueError):
        result_io.parse_s3_uri("not-an-s3-uri")


def test_is_valid_geojson_and_reader(tmp_path):
    gdf = gpd.GeoDataFrame(
        {"severity": [1], "value": [5]},
        geometry=[Point(0, 0)],
        crs="EPSG:4326",
    )
    path = tmp_path / "test.geojson"
    gdf.to_file(path, driver="GeoJSON")

    assert result_io.is_valid_geojson(path)

    read_back = result_io.read_geojson_maybe_s3(str(path))
    assert "severity" in read_back.columns
    assert len(read_back) == 1

    empty_path = tmp_path / "empty.geojson"
    empty_path.write_text(json.dumps({}))
    assert result_io.is_valid_geojson(str(empty_path)) is False


def test_upload_dir_to_s3_and_cleanup(monkeypatch, tmp_path):
    local_dir = tmp_path / "fire"
    local_dir.mkdir()
    data_file = local_dir / "file.txt"
    data_file.write_text("hello")

    puts: list[tuple[str, str]] = []

    class DummyFS:
        def __init__(self, anon=False):
            self.anon = anon

        def put(self, src, dest):
            puts.append((src, dest))

    dummy_module = SimpleNamespace(S3FileSystem=lambda anon=False: DummyFS(anon=anon))
    monkeypatch.setitem(sys.modules, "s3fs", dummy_module)

    ok = result_io.upload_dir_to_s3_and_cleanup(str(local_dir), "s3://bucket/prefix")

    assert ok is True
    assert not local_dir.exists()
    assert puts == [(str(data_file), "bucket/prefix/file.txt")]
