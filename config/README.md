# Config folder 📑

Holds the packaged defaults for the burn-severity CLI. The main file is `dea_burn_severity_processing.yaml`, which mirrors the legacy shipped YAML and is baked into the Python package.

## How configs are applied
- Defaults from `src/dea_burn_severity/config/dea_burn_severity_processing.yaml` are loaded first (identical contents to this folder’s YAML).
- Optional external YAML (`--config /path/or/url.yaml`) overrides those defaults.
- CLI flags override both; the Click command also honours `DEA_BURN_SEVERITY_*` env vars for the same flags.
- Database credentials always come from env (`FIRE_DB_HOSTNAME`, `FIRE_DB_NAME`, `FIRE_DB_USERNAME`, `FIRE_DB_PASSWORD`, `DB_PORT`).

## Key fields to tune
- `output_dir`, `upload_to_s3`, `upload_to_s3_prefix` — output location and publishing.
- `s2_products`, `s2_measurements` — Sentinel-2 collections/bands passed into `dea_tools.load_ard` (kept as lists to satisfy `load_ard`’s expected mutability).
- Temporal windows: `pre_fire_buffer_days`, `post_fire_start_days`, `post_fire_window_days`.
- `grass_classes` — landcover codes treated as grass; drives severity thresholding.
- DB shape metadata: `db_table`, `db_columns`, `db_geom_column`, `db_output_crs`.

## Minimal override example
```yaml
output_dir: /data/burns
upload_to_s3: false
db_table: nli_lastboundaries_trigger
db_columns: [fire_id, fire_name, ignition_date, capt_date, state, agency, date_processed]
db_geom_column: geom
```
