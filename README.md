# 311Analyzer

NYC 311 complaints analysis focused on `2025-2026` complaint patterns, operational response metrics, and follow-on hotspot/fairness work.

## Current Artifacts
- `EDA.ipynb`: cleaned analytic workflow and exploratory analysis
- `data/build_sqlite.py`: chunked SQLite build from the raw CSV
- `data/build_parquet.py`: chunked yearly parquet build from the raw CSV
- `ag/project_spec.md`: project specification

## Notes
- Large local data files are intentionally ignored by git.
- The main cleaned output is written locally to `data/analytics/requests_2025_2026_analytic.parquet`.
