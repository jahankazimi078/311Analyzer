# AGENTS

## Project
This repository analyzes NYC 311 complaint data to identify service patterns, complaint hotspots, and response-time bottlenecks.

## Current Scope
- Primary analysis window: `2025-2026`
- Main analysis artifact: `EDA.ipynb`
- Cleaned analytic output: `data/analytics/requests_2025_2026_analytic.parquet`

## Data Handling
- Treat `data/` as local storage for raw and generated artifacts.
- Keep only lightweight documentation and build scripts under version control.
- Do not commit large parquet, sqlite, or raw CSV files.

## Analysis Notes
- `2026` is partial-year data and should be labeled that way in trend analysis.
- Response-time metrics should exclude missing or invalid close timestamps.
- Preserve data-quality flags instead of dropping problematic rows globally.

## Workflow
- Use `EDA.ipynb` for scoped exploratory analysis.
- Prefer building from existing yearly parquet files instead of reparsing the raw CSV.
- Keep changes minimal and document new assumptions in the notebook or README.
