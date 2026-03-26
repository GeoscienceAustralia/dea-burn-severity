import os
from dataclasses import dataclass


@dataclass
class RuntimeBurnConfig:
    force_rebuild = False
    output_dir: str = "products"
    dc_app_name = "Burn_Severity"
    upload_to_s3: bool = False
    upload_to_s3_prefix: str = "s3://dea-public-data-dev/projects/burn_cube/derivative/dea_burn_severity/result"

    fire_area_minimum_ha: int = 1
    pre_fire_buffer_days: int = 50
    adjustment_missing_ignit_date: int = 30
    post_fire_window_days: int = 60
    post_fire_start_days: int = 15

    # Env var loaded - used by pipeline
    db_host: str | None = os.getenv("FIRE_DB_HOSTNAME")
    db_password: str | None = os.getenv("FIRE_DB_PASSWORD")
    db_user: str | None = os.getenv("FIRE_DB_USERNAME")
    
    db_name: str | None = os.getenv("FIRE_DB_NAME", "fire_severity_product")
    db_port: int | None = int(os.getenv("DB_PORT", 5432))

    db_table: str = "nli_lastboundaries_trigger"


class StaticBurnConfig:
    """Configuration for burn severity that will not be normally changed at runtime."""

    resolution = (-10, 10)
    output_crs = "EPSG:3577"

    s2_measurements: list[str] = [
        "nbart_blue",
        "nbart_green",
        "nbart_red",
        "nbart_nir_1",
        "nbart_nir_2",
        "nbart_swir_2",
        "nbart_swir_3",
        "oa_nbart_contiguity",
        "oa_s2cloudless_mask",
    ]

    s2_products: list[str] = ["ga_s2am_ard_3", "ga_s2bm_ard_3", "ga_s2cm_ard_3"]

    grass_classes: list[int] = [
        3,
        14,
        15,
        16,
        17,
        18,
        21,
        32,
        33,
        34,
        35,
        36,
        39,
        50,
        51,
        52,
        53,
        54,
        57,
        78,
        79,
        80,
        81,
        82,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
        91,
        92,
        94,
        95,
        96,
        97,
    ]

    severity_class_name: dict[int: str] = {
        0 : 'Unburnt',
        1 : 'Grass_extent',
        2 : 'Low',
        3 : 'Medium',
        4 : 'High',
        5 : 'Extreme',
        6 : 'No_data'}
