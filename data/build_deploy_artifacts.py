from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard_artifacts import (
    DASHBOARD_SUMMARY_FILENAMES,
    build_dashboard_summary_artifacts,
)


ANALYTICS_DIR = PROJECT_ROOT / "data" / "analytics"
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
DEPLOY_ANALYTICS_DIR = PROJECT_ROOT / "data" / "deploy" / "analytics"
DEPLOY_REFERENCE_DIR = PROJECT_ROOT / "data" / "deploy" / "reference"

ANALYTIC_SOURCE = ANALYTICS_DIR / "requests_2025_2026_analytic.parquet"
NLP_SOURCE = ANALYTICS_DIR / "requests_2025_2026_issue_subtypes.parquet"

DEPLOY_ANALYTIC_CORE = (
    DEPLOY_ANALYTICS_DIR / "requests_2025_2026_analytic_dashboard_core.parquet"
)
DEPLOY_ANALYTIC_GEO = (
    DEPLOY_ANALYTICS_DIR / "requests_2025_2026_analytic_dashboard_geo.parquet"
)
DEPLOY_NLP = (
    DEPLOY_ANALYTICS_DIR / "requests_2025_2026_issue_subtypes_dashboard.parquet"
)

ANALYTIC_CORE_COLUMNS = [
    "created_date",
    "created_year",
    "created_month_start",
    "created_hour",
    "created_weekday",
    "created_season",
    "borough",
    "agency",
    "complaint_type",
    "descriptor",
    "status",
    "is_closed_status",
    "closed_status_missing_date_flag",
    "nonclosed_status_has_date_flag",
    "negative_resolution_flag",
    "resolved_with_valid_date",
    "resolution_days",
    "resolution_bucket",
]

ANALYTIC_GEO_COLUMNS = [
    "unique_key",
    "borough",
    "complaint_type",
    "latitude",
    "longitude",
]

NLP_COLUMNS = [
    "unique_key",
    "complaint_type",
    "issue_family",
    "issue_subtype",
    "subtype_modeled_flag",
    "potential_label_mismatch_flag",
    "residual_cluster_label",
    "resolution_outcome_group",
    "resolution_outcome_confidence",
    "subtype_source",
    "issue_subtype_confidence",
    "agency",
    "borough",
]

STATIC_ANALYTIC_FILES = [
    "requests_2025_2026_agency_issue_metrics.parquet",
    "requests_2025_2026_agency_metrics.parquet",
    "requests_2025_2026_community_board_fairness_sensitivity.parquet",
    "requests_2025_2026_community_board_metrics.parquet",
    "requests_2025_2026_community_board_monthly.parquet",
    "requests_2025_2026_community_board_operations.parquet",
    "requests_2025_2026_complaint_type_metrics.parquet",
    "requests_2025_2026_fairness_model_results.parquet",
    "requests_2025_2026_grid_monthly.parquet",
    "requests_2025_2026_grid_persistence.parquet",
    "requests_2025_2026_operations_monthly.parquet",
    "requests_2025_2026_resolution_bucket_confusion_matrix.parquet",
    "requests_2025_2026_resolution_bucket_error_slices.parquet",
    "requests_2025_2026_resolution_bucket_feature_importance.parquet",
    "requests_2025_2026_resolution_bucket_model_metrics.parquet",
    "requests_2025_2026_resolution_bucket_predictions.parquet",
    "requests_2025_2026_zcta_fairness_metrics.parquet",
    "requests_2025_2026_zcta_fairness_monthly.parquet",
    "requests_2025_2026_zcta_fairness_stratified.parquet",
    "requests_2025_2026_zcta_metrics.parquet",
]

STATIC_REFERENCE_FILES = [
    "nyc_zcta_demographics.parquet",
    "nyc_zcta_reference.parquet",
]


def ensure_inputs_exist(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"Copied {source.name} -> {destination.relative_to(PROJECT_ROOT)}")


def write_projection(source: Path, columns: list[str], destination: Path) -> None:
    df = pd.read_parquet(source, columns=columns)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destination, index=False)
    size_mb = destination.stat().st_size / (1024 * 1024)
    print(f"Wrote {destination.relative_to(PROJECT_ROOT)} ({size_mb:.2f} MB)")


def build_deploy_artifacts() -> None:
    ensure_inputs_exist(
        [
            ANALYTIC_SOURCE,
            NLP_SOURCE,
            *(ANALYTICS_DIR / name for name in STATIC_ANALYTIC_FILES),
            *(REFERENCE_DIR / name for name in STATIC_REFERENCE_FILES),
        ]
    )

    generated_summary_paths = build_dashboard_summary_artifacts(
        ANALYTIC_SOURCE,
        NLP_SOURCE,
        ANALYTICS_DIR,
    )
    for path in generated_summary_paths:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"Built {path.relative_to(PROJECT_ROOT)} ({size_mb:.2f} MB)")

    write_projection(ANALYTIC_SOURCE, ANALYTIC_CORE_COLUMNS, DEPLOY_ANALYTIC_CORE)
    write_projection(ANALYTIC_SOURCE, ANALYTIC_GEO_COLUMNS, DEPLOY_ANALYTIC_GEO)
    write_projection(NLP_SOURCE, NLP_COLUMNS, DEPLOY_NLP)

    for name in STATIC_ANALYTIC_FILES:
        copy_file(ANALYTICS_DIR / name, DEPLOY_ANALYTICS_DIR / name)

    for name in DASHBOARD_SUMMARY_FILENAMES:
        copy_file(ANALYTICS_DIR / name, DEPLOY_ANALYTICS_DIR / name)

    for name in STATIC_REFERENCE_FILES:
        copy_file(REFERENCE_DIR / name, DEPLOY_REFERENCE_DIR / name)


if __name__ == "__main__":
    build_deploy_artifacts()
