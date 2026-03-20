import os

# RUNTIME options
force_rebuild = False
output_dir: str = "products"
dc_app_name = "Burn_Severity"
upload_to_s3: bool = True
upload_to_s3_prefix: str | None = (
    "s3://dea-public-data-dev/projects/burn_cube/derivative/dea_burn_severity/result"
)


# Static Configuration
# TODO check this, the helper has Burnt_severity in it.

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


# Env vars
db_host: str | None = os.getenv("FIRE_DB_HOSTNAME")
db_name: str | None = os.getenv("FIRE_DB_NAME")
db_password: str | None = os.getenv("FIRE_DB_PASSWORD")
db_user: str | None = os.getenv("FIRE_DB_USERNAME")
db_port: int | None = int(os.getenv("DB_PORT", 0))

db_table: str = "nli_lastboundaries_trigger"
