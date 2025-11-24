"""
Configuration helpers and runtime settings for the DEA burn severity workflow.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse
from urllib.request import urlopen

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
PACKAGE_CONFIG_PATH = PACKAGE_ROOT / "config" / "dea_burn_severity_processing.yaml"
DEFAULT_CONFIG_URL = (
    "https://raw.githubusercontent.com/GeoscienceAustralia/dea-burn-severity/refs/heads/main/"
    "config/dea_burn_severity_processing.yaml"
)

CLI_CONFIG_KEYS = (
    "output_dir",
    "force_rebuild",
    "upload_to_s3_prefix",
    "upload_to_s3",
    "app_name",
    "db_table",
)

REQUIRED_DB_ENV_VARS: dict[str, str] = {
    "db_host": "FIRE_DB_HOSTNAME",
    "db_name": "FIRE_DB_NAME",
    "db_password": "FIRE_DB_PASSWORD",
    "db_user": "FIRE_DB_USERNAME",
    "db_port": "DB_PORT",
}


@dataclass(frozen=True)
class RuntimeConfig:
    """
    Immutable runtime configuration consumed by the processing pipeline.
    """

    output_dir: str
    force_rebuild: bool
    upload_to_s3: bool
    upload_to_s3_prefix: str
    app_name: str
    output_crs: str
    resolution: tuple[float, float]
    s2_products: tuple[str, ...]
    s2_measurements: tuple[str, ...]
    pre_fire_buffer_days: int
    post_fire_start_days: int
    post_fire_window_days: int
    grass_classes: tuple[int, ...]
    db_table: str
    db_columns: tuple[str, ...]
    db_geom_column: str
    db_host: str
    db_name: str
    db_password: str
    db_user: str
    db_port: int
    db_output_crs: str

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> "RuntimeConfig":
        """
        Build a RuntimeConfig from a mapping, validating required keys.
        """

        def _as_tuple(name: str) -> tuple[Any, ...]:
            value = config.get(name, ())
            if isinstance(value, Sequence) and not isinstance(value, str):
                return tuple(value)
            raise ValueError(f"Configuration '{name}' must be a sequence.")

        resolution = list(config.get("resolution", []))
        if len(resolution) != 2:
            raise ValueError("Configuration 'resolution' must contain exactly two values.")

        return cls(
            output_dir=str(config["output_dir"]),
            force_rebuild=bool(config.get("force_rebuild", False)),
            upload_to_s3=bool(config["upload_to_s3"]),
            upload_to_s3_prefix=str(config["upload_to_s3_prefix"]),
            app_name=str(config["app_name"]),
            output_crs=str(config["output_crs"]),
            resolution=(float(resolution[0]), float(resolution[1])),
            s2_products=tuple(str(v) for v in _as_tuple("s2_products")),
            s2_measurements=tuple(str(v) for v in _as_tuple("s2_measurements")),
            pre_fire_buffer_days=int(config["pre_fire_buffer_days"]),
            post_fire_start_days=int(config["post_fire_start_days"]),
            post_fire_window_days=int(config["post_fire_window_days"]),
            grass_classes=tuple(int(v) for v in _as_tuple("grass_classes")),
            db_table=str(config.get("db_table", "")),
            db_columns=tuple(str(v) for v in _as_tuple("db_columns")),
            db_geom_column=str(config.get("db_geom_column", "geom")),
            db_host=str(config.get("db_host", "")),
            db_name=str(config.get("db_name", "")),
            db_password=str(config.get("db_password", "")),
            db_user=str(config.get("db_user", "")),
            db_port=int(config.get("db_port", 5432)),
            db_output_crs=str(config.get("db_output_crs", "EPSG:4283")),
        )

    def as_dict(self) -> dict[str, Any]:
        """
        Return a JSON-serialisable representation of the configuration.
        """
        return {
            "output_dir": self.output_dir,
            "force_rebuild": self.force_rebuild,
            "upload_to_s3": self.upload_to_s3,
            "upload_to_s3_prefix": self.upload_to_s3_prefix,
            "app_name": self.app_name,
            "output_crs": self.output_crs,
            "resolution": list(self.resolution),
            "s2_products": list(self.s2_products),
            "s2_measurements": list(self.s2_measurements),
            "pre_fire_buffer_days": self.pre_fire_buffer_days,
            "post_fire_start_days": self.post_fire_start_days,
            "post_fire_window_days": self.post_fire_window_days,
            "grass_classes": list(self.grass_classes),
            "db_table": self.db_table,
            "db_columns": list(self.db_columns),
            "db_geom_column": self.db_geom_column,
            "db_host": self.db_host,
            "db_name": self.db_name,
            "db_password": self.db_password,
            "db_user": self.db_user,
            "db_port": self.db_port,
            "db_output_crs": self.db_output_crs,
        }


def load_default_config() -> dict[str, Any]:
    """
    Load the packaged default YAML configuration (with optional remote fallback).
    """
    source: str | Path = PACKAGE_CONFIG_PATH
    try:
        raw_config = load_yaml_config(source)
    except OSError:
        source = DEFAULT_CONFIG_URL
        raw_config = load_yaml_config(source)
    return copy.deepcopy(raw_config)


def _load_text_from_source(source: str | Path) -> str:
    """
    Load a text file from local disk, HTTP(S) or S3.
    """
    if isinstance(source, Path):
        return source.expanduser().read_text(encoding="utf-8")
    parsed = urlparse(str(source))
    scheme = parsed.scheme.lower()
    if scheme in ("http", "https"):
        with urlopen(str(source)) as response:
            return response.read().decode("utf-8")
    if scheme == "s3":
        try:
            import s3fs  # type: ignore
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "Reading s3:// configs requires the 's3fs' package. "
                "Install with: pip install s3fs"
            ) from exc
        fs = s3fs.S3FileSystem(anon=False)
        with fs.open(str(source), "rb") as stream:
            return stream.read().decode("utf-8")
    if scheme in ("", "file"):
        path = Path(parsed.path) if scheme else Path(str(source))
        return path.expanduser().read_text(encoding="utf-8")
    raise ValueError(f"Unsupported configuration scheme '{scheme}' for '{source}'.")


def load_yaml_config(source: str | Path) -> dict[str, Any]:
    """
    Load a YAML configuration from the provided source.
    """
    text = _load_text_from_source(source)
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("Configuration YAML must define a mapping at the top level.")
    return data


DEFAULT_CONFIG_DICT = load_default_config()
DEFAULT_RUNTIME_CONFIG: RuntimeConfig | None = None


def merge_config(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    """
    Return a shallow merge of base + override (override wins).
    """
    merged = copy.deepcopy(base)
    if override:
        merged.update(override)
    return merged


def build_runtime_config(
    cli_kwargs: Mapping[str, Any],
    config_path: str | Path | None,
) -> RuntimeConfig:
    """
    Merge defaults, optional YAML and CLI overrides into a RuntimeConfig.
    """
    user_config: dict[str, Any] | None = None
    if config_path:
        user_config = load_yaml_config(config_path)

    merged = merge_config(DEFAULT_CONFIG_DICT, user_config)
    for key in CLI_CONFIG_KEYS:
        value = cli_kwargs.get(key)
        if value is not None:
            merged[key] = value

    for config_key, env_var in REQUIRED_DB_ENV_VARS.items():
        merged[config_key] = os.environ.get(env_var, "")

    return RuntimeConfig.from_dict(merged)


def get_default_runtime_config() -> RuntimeConfig:
    """
    Lazily build the default runtime configuration using DB values from env vars.
    """
    global DEFAULT_RUNTIME_CONFIG
    if DEFAULT_RUNTIME_CONFIG is None:
        DEFAULT_RUNTIME_CONFIG = build_runtime_config({}, None)
    return DEFAULT_RUNTIME_CONFIG


__all__ = [
    "CLI_CONFIG_KEYS",
    "DEFAULT_CONFIG_DICT",
    "DEFAULT_RUNTIME_CONFIG",
    "DEFAULT_CONFIG_URL",
    "PACKAGE_CONFIG_PATH",
    "RuntimeConfig",
    "build_runtime_config",
    "get_default_runtime_config",
    "load_yaml_config",
    "merge_config",
]
