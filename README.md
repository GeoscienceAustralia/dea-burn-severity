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

- 🔄 **Config merge**: Defaults from `dea_burn_severity/config/dea_burn_severity_processing.yaml` are merged with any external YAML and CLI flags, wiring options like `output_dir`, `resolution`, acquisition windows, and S3 upload behaviour.
- 🗺️ **Polygon prep**: Fire footprints (`polygons`) are loaded from local paths or `s3://` URIs, dissolved to one row per `fire_id` if available, and exploded into individual parts so multi-part incidents are handled piece by piece.
- 🛰️ **Baseline vs post-fire stacks**: For each part the CLI instantiates a `datacube.Datacube`, loads Sentinel-2 ARD using `dea_tools.load_ard`, and automatically retries with looser `min_gooddata` thresholds until it finds valid pre- and post-fire observations.
- 🌿 **Landcover-aware severity**: Landcover tiles (`ga_ls_landcover_class_cyear_3`) provide grass vs woody masks. The workflow computes delta NBR (`calculate_indices`) and applies class-specific thresholds to yield categorical severity rasters.
- 🧪 **Quality masks & stats**: A composite debug mask flags water, cloud, and contiguity issues across the time series; per-part logs capture pixel counts, valid baselines, and missing data to support QA.
- 🧩 **Vectorisation & exports**: Severity rasters are vectorised (`xr_vectorize`) to GeoJSON, combined per original fire, and optionally saved as Cloud-Optimised GeoTIFFs. Outputs can stay local or be uploaded to S3 (with graceful skipping when artefacts already exist).
- ✅ **Idempotent execution**: Existing results are respected unless `--force-rebuild` is supplied, avoiding redundant reprocessing when rerunning the pipeline.

## Configuration

The packaged defaults live in `dea_burn_severity/config/dea_burn_severity_processing.yaml`. If `--config`
is omitted those defaults are used. Provide a custom YAML file—local path, `http(s)://`
URL, or `s3://` URI—via `--config` to override any value. CLI flags continue to override
both the bundled and external configuration values. The YAML holds every CLI option
(e.g. `polygons`, `output_dir`, S3 settings), so supplying an external config alone is often
enough to run the pipeline. Boolean CLI overrides accept `true`/`false`.

## Docker

Build the container image (uses requirements/constraints mirrored from dea-fmc):

```bash
docker build -t dea-burn-severity .
```

Run the CLI inside the container:

```bash
docker run --rm dea-burn-severity dea-burn-severity --help
```
