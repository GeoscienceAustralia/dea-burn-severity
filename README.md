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
  --polygons s3://bucket/path/to/fire_polygons.geojson \
  --output-dir ./products \
  --max-fires 5 \
  --upload-to-s3-prefix s3://bucket/path/to/results \
  --app-name Burnt_Area_Mapping
```Burn_Severity

Run `dea-burn-severity --help` to inspect all options.

## Docker

Build the container image (uses requirements/constraints mirrored from dea-fmc):

```bash
docker build -t dea-burn-severity .
```

Run the CLI inside the container:

```bash
docker run --rm dea-burn-severity dea-burn-severity --help
```
