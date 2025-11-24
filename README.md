# dea-burn-severity 🔥🛰️🌿

Fully automated burn-severity workflow for Digital Earth Australia. This CLI pulls fire footprints from Postgres, loads Sentinel‑2 ARD, computes delta NBR with landcover-aware thresholds, vectorises severity classes, and optionally ships results to S3—idempotently and with rich QA logging.

## What this tool delivers
- 🛰️ **Data ingest**: Sentinel‑2 ARD (`ga_s2am_ard_3`, `ga_s2bm_ard_3`, `ga_s2cm_ard_3`) with S2 Cloudless masking, loaded via `dea_tools.load_ard`.
- 🧭 **Temporal windows**: Configurable pre/post windows (defaults: `pre_fire_buffer_days=50`, `post_fire_start_days=15`, `post_fire_window_days=60`) tuned for burn assessments.
- 🌿 **Landcover-aware severity**: Delta NBR against `ga_ls_landcover_class_cyear_3`, with grass-class thresholds defined in `grass_classes`.
- 🧪 **Quality signals**: Cloud/contiguity/water masking, pixel counts, and per-fire logs for fast QA.
- 🧩 **Geo outputs**: Severity polygons dissolved by class, plus preview/debug COGs. Ready for downstream GIS or dashboards.
- 🚛 **Distribution**: Local outputs under `products/` by default, optional S3 upload + local cleanup when `upload_to_s3` is enabled.

## Quick start
```bash
# 1) Install
pip install -e .  # editable makes iteration easy; requires dea_tools + datacube deps

# 2) Set DB credentials (env var names are fixed)
export FIRE_DB_HOSTNAME=...
export FIRE_DB_NAME=...
export FIRE_DB_USERNAME=...
export FIRE_DB_PASSWORD=...
export DB_PORT=5432

# 3) Run with defaults (uses packaged YAML)
dea-burn-severity

# 4) Or point at your own config
dea-burn-severity --config /path/to/dea_burn_severity_processing.yaml
```
Tip: `DEA_BURN_SEVERITY_*` env vars mirror CLI flags (e.g. `DEA_BURN_SEVERITY_OUTPUT_DIR`), since Click’s `auto_envvar_prefix` is set.

## Inputs & prerequisites
- ✅ Postgres/PostGIS table containing fire footprints; geometry is read via `ST_AsGeoJSON`.
- ✅ DB creds from env: `FIRE_DB_HOSTNAME`, `FIRE_DB_NAME`, `FIRE_DB_USERNAME`, `FIRE_DB_PASSWORD`, `DB_PORT` (defaults to 5432).
- ✅ Optional S3 credentials if uploading outputs.
- ✅ Datacube configured with Sentinel‑2 ARD + landcover products; `dea_tools` available on the Python path.
- ✅ `psycopg2-binary` installed when using DB loading (the CLI does not vendor it).

## Configuration (YAML + CLI + env)
- 📦 **Packaged defaults**: `src/dea_burn_severity/config/dea_burn_severity_processing.yaml` mirrors the legacy shipped YAML.
- 🔄 **Merge order**: defaults → optional YAML (`--config` local/http(s)/s3) → CLI flags → `DEA_BURN_SEVERITY_*` env for those flags. DB creds always come from the fixed env var names above.
- 🔑 **Key fields** (see `config/dea_burn_severity_processing.yaml` for all):
  - `output_dir`: base folder (`products` by default).
  - `s2_products`, `s2_measurements`: Sentinel‑2 collections + bands (passed as lists when calling `load_ard`).
  - `output_crs`, `resolution`: reprojection and pixel size (default EPSG:3577, -10/10 m).
  - `pre_fire_buffer_days`, `post_fire_start_days`, `post_fire_window_days`: temporal windows.
  - `grass_classes`: landcover codes treated as grass; determines thresholds.
  - `db_table`, `db_columns`, `db_geom_column`, `db_output_crs`: how footprints are read and reprojected.
  - `upload_to_s3`, `upload_to_s3_prefix`: enable S3 publishing + cleanup of local run dirs.

Minimal custom YAML example:
```yaml
output_dir: /data/burns
upload_to_s3: true
upload_to_s3_prefix: s3://dea-public-data-dev/projects/burn_cube/derivative/dea_burn_severity/result
db_table: nli_lastboundaries_trigger
db_columns: [fire_id, fire_name, ignition_date, capt_date, capt_method, state, agency, date_retrieved, date_processed]
db_geom_column: geom
```

## How the pipeline runs (per fire) 🧭
1. **Polygon ingest**: Load GeoDataFrame from Postgres, dissolve by `fire_id` when present; ensure `fire_id` exists for downstream naming.
2. **Date wiring**: Derive `ignition_date` (or fallback capture date) and `extinguish_date`. Compute pre/post windows from config.
3. **Baseline stack**: Call `load_ard` with `min_gooddata=0.99`; if empty, retry with mask dilation + `min_gooddata=0.20` and build a latest‑valid composite.
4. **Post-fire stack**: Call `load_ard_with_fallback` with decreasing `min_gooddata` thresholds (0.99 → 0.90 by default).
5. **Landcover**: Load `ga_ls_landcover_class_cyear_3` for the year before ignition.
6. **Indices**: Compute pre/post NBR via `calculate_indices`; derive delta NBR.
7. **Severity classification**: Apply grass/woody thresholds (`calculate_severity`), generate debug mask (cloud/water/contiguity), and set masked pixels to class `6`.
8. **Vectorisation**: Convert severity raster to vectors, clip to fire footprint, dissolve by class, and attach metadata (`fire_id`, `fire_name`, dates, plus all other attributes from `db_columns`).
9. **Outputs**:
   - GeoJSON: `products/results/DEA_burn_severity_<fire_id>_<date>.json`
   - Preview COG: `products/s2_postfire_preview_<fire_slug>.tif` (first post-fire scene)
   - Debug COG: `products/debug_mask_raster_<fire_slug>.tif`
   - Optional run log: per-fire pixel counts, baseline/post scene counts, masked/valid stats.
10. **Distribution**: If `upload_to_s3` is true, the per-fire folder is uploaded to `upload_to_s3_prefix` and removed locally after verification.

## CLI flags (all optional thanks to defaults)
- `--config PATH|URL` — external YAML to merge.
- `--output-dir PATH` — override base output folder.
- `--force-rebuild true|false` — ignore existing outputs.
- `--upload-to-s3 true|false` — toggle publishing + cleanup.
- `--upload-to-s3-prefix s3://bucket/prefix` — target prefix.
- `--app-name NAME` — datacube app name.
- `--db-table NAME` — override table (columns come from YAML).
Use `dea-burn-severity --help` for the live list; `DEA_BURN_SEVERITY_OUTPUT_DIR` etc. mirror these options.

## Outputs & QA 📂
- **GeoJSON**: severity polygons dissolved by class, CRS `EPSG:4283`.
- **COGs**: one post-fire preview (first time slice) and one debug mask per fire.
- **Logs**: When `log_path` is provided internally, each fire logs scene counts, grid size, valid/burn/masked pixel totals, and baseline/post contiguity stats.
- **Idempotency**: Existing per-fire outputs are skipped unless `--force-rebuild` is set.

## Development & troubleshooting 🛠️
- Install: `pip install -e .` (ensure `dea_tools`, `datacube`, `psycopg2-binary`, GDAL stack available).
- Docker: `docker build -t dea-burn-severity .` then `docker run --rm dea-burn-severity dea-burn-severity --help`.
- Common hiccups:
  - 🔑 Missing DB creds → set `FIRE_DB_*` env vars.
  - 🌥️ No baseline scenes → falls back to relaxed composite; still skips if empty.
  - 📦 Missing `dea_tools`/`datacube` imports → install into the same environment.
  - 📡 S3 upload failures → outputs remain on disk for manual retry.

Happy mapping! 🎉
