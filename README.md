# 311Analyzer

`311Analyzer` is an end-to-end NYC 311 analysis project focused on `2025-2026` complaints.

The repo turns raw 311 records into a cleaned analytic dataset, enriches it with complaint-text taxonomy and neighborhood context, builds reusable geospatial and operations scorecards, evaluates neighborhood equity patterns, and finishes with a presentable interactive dashboard.

## What The Project Does

The workflow is organized as a sequence of reusable data products rather than one giant notebook.

1. Raw 311 data is stored locally in SQLite and yearly parquet files.
2. A cleaned analytic parquet is built for the scoped `2025-2026` window.
3. Complaint text is enriched with issue families, issue subtypes, and closure-language outputs.
4. Geographic, operational, fairness, and predictive artifacts are written under `data/analytics/`.
5. The notebooks explore and explain each layer.
6. `311dashboard.py` reads those outputs and turns them into an interactive final product.

## Main Data Flow

### 1. Local storage and ingestion

The raw CSV is not meant to be reparsed every time.

Instead, the repo uses:

- `data/build_sqlite.py` to build `data/311_requests.sqlite`
- `data/build_parquet.py` to build yearly parquet files under `data/311_requests_parquet/`

These local artifacts make later analysis much faster and keep the workflow reproducible.

### 2. Clean analytic dataset

The main cleaned analysis base is:

- `data/analytics/requests_2025_2026_analytic.parquet`

This dataset standardizes timestamps, preserves data-quality flags, adds date/time features, and keeps the `2025-2026` analysis window consistent across the project.

Important project conventions:

- `2026` is partial-year data and should be labeled as `YTD`
- response-time metrics only use rows with valid closure intervals
- QA flags are preserved instead of dropping messy rows globally

### 3. NLP enrichment

The complaint-text layer lives in:

- `data/analytics/requests_2025_2026_issue_subtypes.parquet`

This adds:

- `issue_family`
- `issue_subtype`
- `subtype_modeled_flag`
- closure-language groupings such as `resolution_outcome_group`

That NLP output is reused by the geography, operations, fairness, and predictive layers.

## Analysis Modules

The repo now uses stable script names instead of phase-numbered module names.

### `geo.py`

Builds the geographic analysis outputs used for burden mapping, hotspot persistence, and community-board views.

Main outputs include:

- `requests_2025_2026_zcta_metrics.parquet`
- `requests_2025_2026_community_board_metrics.parquet`
- `requests_2025_2026_community_board_monthly.parquet`
- `requests_2025_2026_grid_monthly.parquet`
- `requests_2025_2026_grid_persistence.parquet`

Public builder:

- `build_geo_outputs()`

### `operations.py`

Builds service-performance scorecards and trend outputs.

Main outputs include:

- `requests_2025_2026_agency_metrics.parquet`
- `requests_2025_2026_complaint_type_metrics.parquet`
- `requests_2025_2026_agency_issue_metrics.parquet`
- `requests_2025_2026_operations_monthly.parquet`
- `requests_2025_2026_community_board_operations.parquet`

Public builder:

- `build_operations_outputs()`

### `fairness.py`

Builds neighborhood disparity and adjusted comparison outputs using ZCTA demographics and community-board sensitivity checks.

Main outputs include:

- `requests_2025_2026_zcta_fairness_metrics.parquet`
- `requests_2025_2026_zcta_fairness_stratified.parquet`
- `requests_2025_2026_zcta_fairness_monthly.parquet`
- `requests_2025_2026_community_board_fairness_sensitivity.parquet`
- `requests_2025_2026_fairness_model_results.parquet`

Public builder:

- `build_fairness_outputs()`

### `predictive.py`

Builds the resolution-bucket forecasting outputs.

The current predictive task is multiclass prediction of `resolution_bucket`, evaluated on `2025` train and `2026` YTD test data.

Main outputs include:

- `requests_2025_2026_resolution_bucket_model_metrics.parquet`
- `requests_2025_2026_resolution_bucket_predictions.parquet`
- `requests_2025_2026_resolution_bucket_feature_importance.parquet`
- `requests_2025_2026_resolution_bucket_error_slices.parquet`
- `requests_2025_2026_resolution_bucket_confusion_matrix.parquet`

Public builder:

- `build_predictive_outputs()`

### `dashboard.py` and `311dashboard.py`

`dashboard.py` contains the actual Streamlit app.

`311dashboard.py` is the clean entrypoint used to run it.

The dashboard brings together:

- city patterns
- text insights
- geography
- operations
- neighborhood equity
- forecasting

It primarily reads the reusable artifacts above, while also scanning targeted columns from the analytic and NLP parquet files for richer EDA and text-summary views.

## Notebooks

The notebooks are still the clearest narrative walk-throughs of each layer:

- `EDA.ipynb`
- `NLP.ipynb`
- `Geo.ipynb`
- `Operations.ipynb`
- `Fairness.ipynb`
- `Predictive.ipynb`

They now import the renamed modules (`geo`, `operations`, `fairness`, `predictive`) instead of the old phase-numbered script names.

## How To Run The Project

### Build local reference data

```bash
./.venv/bin/python data/build_reference_geo.py
./.venv/bin/python data/build_reference_demographics.py
```

### Rebuild local storage from raw data if needed

```bash
./.venv/bin/python data/build_sqlite.py
./.venv/bin/python data/build_parquet.py
```

### Run a builder directly

```bash
./.venv/bin/python geo.py
./.venv/bin/python operations.py
./.venv/bin/python fairness.py
./.venv/bin/python predictive.py
```

### Launch the dashboard

```bash
./.venv/bin/python -m pip install -r requirements-dashboard.txt
./.venv/bin/streamlit run 311dashboard.py
```

## Repo Conventions

- large local data artifacts are intentionally ignored by git
- `data/` is local storage, not a place for committed heavy outputs
- notebooks and lightweight build scripts are meant to stay in version control
- the dashboard is the presentable final surface, while the notebooks remain the analytical deep dives

## Publish Notes

Before publishing, make sure you do **not** commit:

- raw CSV files
- local parquet outputs under `data/analytics/`
- SQLite databases
- virtual environments
- IDE settings
- local model or dashboard cache artifacts

Those are all treated as local build products rather than source code.
