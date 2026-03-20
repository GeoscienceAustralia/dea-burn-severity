import os
from typing import Any

# RUNTIME options
force_rebuild = False
output_dir: str = "products"
dc_app_name = "Burn_Severity"
upload_to_s3: bool = False
upload_to_s3_prefix: str = (
    "s3://dea-public-data-dev/projects/burn_cube/derivative/dea_burn_severity/result"
)


# Static Configuration
# TODO check this, the helper has Burnt_severity in it.

pre_fire_buffer_days: int = 50
post_fire_window_days: int = 60
post_fire_start_days: int = 15

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

# TODO migrate this
measurements = s2_measurements

grass_classes = [
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

# Env vars
db_host: str | None = os.getenv("FIRE_DB_HOSTNAME")
db_name: str | None = os.getenv("FIRE_DB_NAME")
db_password: str | None = os.getenv("FIRE_DB_PASSWORD")
db_user: str | None = os.getenv("FIRE_DB_USERNAME")
db_port: int | None = int(os.getenv("DB_PORT", 0))

db_table: str = "nli_lastboundaries_trigger"


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
    {"target": "capt_method", "sources": ("capt_method", "capt_metho"), "date": False},
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
