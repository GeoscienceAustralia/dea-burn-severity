# dea_burn_severity package map 🗺️

This package wraps the burn-severity workflow into a reusable CLI. Use this guide to jump to the right module when you need to tweak behaviour or debug a run.

## File-by-file cheat sheet
- `cli.py` — user-facing Click command; wires runtime config, spins up the Datacube, loops over fire footprints, and orchestrates per-fire processing + optional S3 uploads.
- `configuration.py` — loads the packaged YAML defaults, merges user YAML + CLI + env overrides, and produces an immutable `RuntimeConfig`.
- `database.py` — reads fire footprints from Postgres/PostGIS using `psycopg2`, dissolves by `fire_id` when present, and guarantees a `fire_id` column for downstream naming.
- `data_loading.py` — convenience wrappers around `dea_tools.load_ard`: strict/relaxed baseline loader, post-fire loader with threshold fallbacks, and a “latest valid pixel” compositing helper.
- `severity.py` — NBR-based severity classification and debug mask creation (water/cloud/contiguity handling); this is where threshold tweaks live.
- `result_io.py` — saving utilities (e.g. COG/GeoJSON writing) and S3 upload helpers used by the CLI.
- `logging_utils.py` — small helpers for appending per-fire QA stats to a log file.

## End-to-end data flow
1) Load config (`configuration.py`) → 2) Pull fire polygons from DB (`database.py`) → 3) For each fire, load baseline/post stacks (`data_loading.py`) → 4) Compute delta NBR + severity (`severity.py`) → 5) Vectorise/save/upload (`result_io.py`) → 6) Log QA metrics (`logging_utils.py`).

## Development tips
- Run locally with `dea-burn-severity --help` to see live defaults; env vars use the `DEA_BURN_SEVERITY_` prefix.
- Baseline masking/compositing knobs live in `data_loading.py`; severity thresholds live in `severity.py`.
- If you need to add new fire metadata to outputs, update `db_columns` in the YAML and ensure `result_io.py` propagates the new fields when writing vectors.
