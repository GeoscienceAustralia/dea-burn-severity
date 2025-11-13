# dea-burn-severity

CLI wrapper for the DEA burn severity classification workflow.

## Installation

Install directly from source (editable mode recommended while developing):

```bash
pip install -e .
```

Ensure `dea_tools` is installed or otherwise available on the Python path; the CLI
expects to import it directly from the environment.

## Usage

```bash
dea-burn-severity \
  --config https://example.com/dea_burn_severity_processing.yaml \
  --save-per-part-vectors false
```

Run `dea-burn-severity --help` to inspect all options.

## Processing Overview

- 🔄 **Config merge**: Baked-in defaults (mirroring the former packaged YAML) are merged with any external YAML and CLI flags, wiring options like `output_dir`, `resolution`, acquisition windows, and S3 upload behaviour.
- 🗺️ **Polygon prep**: Fire footprints (`polygons`) are loaded from local paths/`s3://` URIs or pulled straight from a Postgres table (pass `--polygon-source database`), dissolved to one row per `fire_id` if available, and assumed to be single polygons (no explode/merge stage required).
- 🛰️ **Baseline vs post-fire stacks**: For each fire polygon the CLI instantiates a `datacube.Datacube`, first attempts a pristine (99% clear) Sentinel-2 baseline, then falls back to a dilated-cloud composite that picks the latest valid pixel if necessary; post-fire loads retain the looser `min_gooddata` behaviour.
- 🌿 **Landcover-aware severity**: Landcover tiles (`ga_ls_landcover_class_cyear_3`) provide grass vs woody masks. The workflow computes delta NBR (`calculate_indices`) and applies class-specific thresholds to yield categorical severity rasters.
- 🧪 **Quality masks & stats**: A composite debug mask flags water, cloud, and contiguity issues across the time series; per-fire logs capture pixel counts, valid baselines, and missing data to support QA.
- 📝 **Metadata carry-through**: Output severity vectors inherit key fire metadata (ID, name, type, capture dates, agencies, etc.) using the same attribute mapping as the reference notebook, keeping downstream consumers aligned.
- 🧩 **Vectorisation & exports**: Severity rasters are vectorised (`xr_vectorize`) to GeoJSON and optionally saved as Cloud-Optimised GeoTIFFs. Outputs can stay local or be uploaded to S3 (with graceful skipping when artefacts already exist).
- ✅ **Idempotent execution**: Existing results are respected unless `--force-rebuild` is supplied, avoiding redundant reprocessing when rerunning the pipeline.

## Configuration

The CLI ships with sensible defaults compiled directly into the source. Provide a custom YAML file—local path, `http(s)://`
URL, or `s3://` URI—via `--config` to override any value. CLI flags continue to override
both the built-in and external configuration values. The YAML holds every CLI option
(e.g. `polygons`, `output_dir`, S3 settings), so supplying an external config alone is often
enough to run the pipeline. Boolean CLI overrides accept `true`/`false`.

### Database polygon loading

Pass `--polygon-source database` to read footprints from Postgres instead of the filesystem. The schema is fixed to `public`; configure the table/column metadata via `db_table`, `db_columns`, `db_geom_column`, and ensure the environment supplies credentials via the names in `db_host_env`, `db_name_env`, and `db_password_env` (defaults match the snippet provided: `DB_HOSTNAME`, `DB_NAME`, `DB_PASSWORD`). The reader uses `psycopg2-binary`, so install it alongside the CLI when enabling this path.

Example YAML overrides (matching the sample row you shared):

```yaml
db_table: nli_lastboundaries_trigger
db_columns:
  - fire_id
  - fire_name
  - fire_type
  - ignition_date
  - capt_date
  - capt_method
  - area_ha
  - perim_km
  - state
  - agency
  - date_retrieved
  - date_processed
db_geom_column: geom
```

With these settings the CLI expects rows like:

| fire_id | fire_name | fire_type | ignition_date | capt_date | capt_method | area_ha | perim_km | state | agency | date_retrieved | date_processed |
|---------|-----------|-----------|---------------|-----------|-------------|---------|----------|-------|--------|----------------|----------------|
| `a518823f-bdea-4299-877b-7b44328d243d` | `None` | `None` | `2025-10-26 20:18:56` | `2025-10-26 21:28:18` | `FIREMAPPER` | `5625.38` | `29.78` | `QLD` | `Qld Fire and Emergency Services` | `2025-10-27 08:15:00` | `2025-11-03` |

The geometry column is read via `ST_AsGeoJSON(geom)` and converted into the GeoDataFrame used by the pipeline.

## Docker

Build the container image (uses requirements/constraints mirrored from dea-fmc):

```bash
docker build -t dea-burn-severity .
```

Run the CLI inside the container:

```bash
docker run --rm dea-burn-severity dea-burn-severity --help
```
