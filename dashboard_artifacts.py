from __future__ import annotations

import gc
from pathlib import Path

import pandas as pd

from geo import NYC_LAT_RANGE, NYC_LON_RANGE, summarize_hotspot_grids


WEEKDAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
SEASON_ORDER = ["Winter", "Spring", "Summer", "Fall"]
RESOLUTION_BUCKET_ORDER = [
    "<1 day",
    "1-3 days",
    "3-7 days",
    "7-30 days",
    "30+ days",
    "Missing",
]

AGENCY_METRICS_FILENAME = "requests_2025_2026_agency_metrics.parquet"
COMMUNITY_BOARD_METRICS_FILENAME = "requests_2025_2026_community_board_metrics.parquet"
GRID_MONTHLY_FILENAME = "requests_2025_2026_grid_monthly.parquet"

EDA_SUMMARY_FILENAMES = {
    "quality": "requests_2025_2026_dashboard_eda_quality.parquet",
    "monthly": "requests_2025_2026_dashboard_eda_monthly.parquet",
    "monthly_by_type": "requests_2025_2026_dashboard_eda_monthly_by_type.parquet",
    "top_complaint_types": "requests_2025_2026_dashboard_eda_top_complaint_types.parquet",
    "seasonal": "requests_2025_2026_dashboard_eda_seasonal.parquet",
    "hourly": "requests_2025_2026_dashboard_eda_hourly.parquet",
    "weekday_hour": "requests_2025_2026_dashboard_eda_weekday_hour.parquet",
    "borough_counts": "requests_2025_2026_dashboard_eda_borough_counts.parquet",
    "borough_mix": "requests_2025_2026_dashboard_eda_borough_mix.parquet",
    "resolution_bucket_counts": "requests_2025_2026_dashboard_eda_resolution_bucket_counts.parquet",
    "descriptor_counts": "requests_2025_2026_dashboard_eda_descriptor_counts.parquet",
    "sample_rows": "requests_2025_2026_dashboard_eda_sample_rows.parquet",
}

NLP_SUMMARY_FILENAMES = {
    "overall": "requests_2025_2026_dashboard_nlp_overall.parquet",
    "issue_families": "requests_2025_2026_dashboard_nlp_issue_families.parquet",
    "modeled_subtypes": "requests_2025_2026_dashboard_nlp_modeled_subtypes.parquet",
    "subtype_source": "requests_2025_2026_dashboard_nlp_subtype_source.parquet",
    "outcome_groups": "requests_2025_2026_dashboard_nlp_outcome_groups.parquet",
    "outcome_confidence": "requests_2025_2026_dashboard_nlp_outcome_confidence.parquet",
    "complaint_type_coverage": "requests_2025_2026_dashboard_nlp_complaint_type_coverage.parquet",
    "residual_counts": "requests_2025_2026_dashboard_nlp_residual_counts.parquet",
    "outcome_by_complaint_type": "requests_2025_2026_dashboard_nlp_outcome_by_complaint_type.parquet",
    "outcome_by_agency": "requests_2025_2026_dashboard_nlp_outcome_by_agency.parquet",
    "board_top_subtypes": "requests_2025_2026_dashboard_nlp_board_top_subtypes.parquet",
    "agency_top_subtypes": "requests_2025_2026_dashboard_nlp_agency_top_subtypes.parquet",
}

GEO_SUMMARY_FILENAMES = {
    "borough_coverage": "requests_2025_2026_dashboard_geo_borough_coverage.parquet",
    "hotspot_monthly": "requests_2025_2026_dashboard_geo_hotspot_monthly.parquet",
    "complaint_hotspots": "requests_2025_2026_dashboard_geo_complaint_hotspots.parquet",
    "subtype_hotspots": "requests_2025_2026_dashboard_geo_subtype_hotspots.parquet",
}

DASHBOARD_SUMMARY_FILENAMES = [
    *EDA_SUMMARY_FILENAMES.values(),
    *NLP_SUMMARY_FILENAMES.values(),
    *GEO_SUMMARY_FILENAMES.values(),
]


def _summary_paths(base_dir: Path, filenames: dict[str, str]) -> dict[str, Path]:
    return {key: base_dir / value for key, value in filenames.items()}


def eda_summary_paths(base_dir: Path) -> dict[str, Path]:
    return _summary_paths(base_dir, EDA_SUMMARY_FILENAMES)


def nlp_summary_paths(base_dir: Path) -> dict[str, Path]:
    return _summary_paths(base_dir, NLP_SUMMARY_FILENAMES)


def geo_summary_paths(base_dir: Path) -> dict[str, Path]:
    return _summary_paths(base_dir, GEO_SUMMARY_FILENAMES)


def add_month_label(
    df: pd.DataFrame, source_column: str = "created_month_start"
) -> pd.DataFrame:
    labeled = df.copy()
    labeled[source_column] = pd.to_datetime(labeled[source_column], errors="coerce")
    labeled["month_label"] = labeled[source_column].dt.strftime("%Y-%m")
    return labeled


def first_mode(series: pd.Series, default: str = "Unknown") -> str:
    mode = series.dropna().astype("string").mode()
    if mode.empty:
        return default
    return str(mode.iloc[0])


def write_table(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def build_eda_summary_artifacts(analytic_path: Path, output_dir: Path) -> list[Path]:
    paths = eda_summary_paths(output_dir)
    columns = [
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
    df = pd.read_parquet(analytic_path, columns=columns)
    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    df["created_month_start"] = pd.to_datetime(
        df["created_month_start"], errors="coerce"
    )
    df["year_label"] = (
        df["created_year"]
        .map({2025: "2025", 2026: "2026 YTD"})
        .fillna(df["created_year"].astype("string"))
    )

    quality = pd.DataFrame(
        [
            {
                "complaints": len(df.index),
                "resolved_with_valid_date": int(df["resolved_with_valid_date"].sum()),
                "resolved_with_valid_date_share": float(
                    df["resolved_with_valid_date"].mean()
                ),
                "closed_status_missing_date_count": int(
                    df["closed_status_missing_date_flag"].sum()
                ),
                "nonclosed_status_has_date_count": int(
                    df["nonclosed_status_has_date_flag"].sum()
                ),
                "negative_resolution_count": int(df["negative_resolution_flag"].sum()),
                "median_resolution_days": float(
                    df.loc[df["resolved_with_valid_date"], "resolution_days"].median()
                ),
                "max_created_date": df["created_date"].max(),
            }
        ]
    )

    monthly = add_month_label(
        df.groupby(["created_month_start", "year_label"], observed=True)
        .size()
        .reset_index(name="complaints")
        .sort_values("created_month_start")
    )

    top_complaint_types = (
        df["complaint_type"]
        .astype("string")
        .value_counts()
        .head(12)
        .rename_axis("complaint_type")
        .reset_index(name="complaints")
    )
    top_type_values = top_complaint_types["complaint_type"].tolist()

    monthly_by_type = add_month_label(
        df.loc[df["complaint_type"].astype("string").isin(top_type_values)]
        .groupby(["created_month_start", "complaint_type"], observed=True)
        .size()
        .reset_index(name="complaints")
        .sort_values("created_month_start")
    )

    seasonal = (
        df.groupby(["created_season", "year_label"], observed=True)
        .size()
        .reset_index(name="complaints")
    )
    seasonal["created_season"] = pd.Categorical(
        seasonal["created_season"], categories=SEASON_ORDER, ordered=True
    )
    seasonal = seasonal.sort_values(["created_season", "year_label"]).reset_index(
        drop=True
    )

    hourly = (
        df.groupby("created_hour", observed=True)
        .size()
        .reset_index(name="complaints")
        .sort_values("created_hour")
    )

    weekday_hour = (
        df.groupby(["created_weekday", "created_hour"], observed=True)
        .size()
        .reset_index(name="complaints")
    )
    weekday_hour["created_weekday"] = pd.Categorical(
        weekday_hour["created_weekday"].astype("string"),
        categories=WEEKDAY_ORDER,
        ordered=True,
    )
    weekday_hour = weekday_hour.sort_values(
        ["created_weekday", "created_hour"]
    ).reset_index(drop=True)

    borough_counts = (
        df.groupby("borough", observed=True)
        .size()
        .reset_index(name="complaints")
        .sort_values("complaints", ascending=False)
    )

    borough_mix = (
        df.loc[df["complaint_type"].astype("string").isin(top_type_values[:8])]
        .groupby(["borough", "complaint_type"], observed=True)
        .size()
        .reset_index(name="complaints")
        .sort_values(["borough", "complaints"], ascending=[True, False])
    )

    resolution_bucket_counts = (
        df["resolution_bucket"]
        .astype("string")
        .fillna("Missing")
        .replace({"<NA>": "Missing"})
        .value_counts()
        .reindex(RESOLUTION_BUCKET_ORDER)
        .fillna(0)
        .rename_axis("resolution_bucket")
        .reset_index(name="complaints")
    )

    descriptor_counts = (
        df["descriptor"]
        .astype("string")
        .fillna("Unknown")
        .value_counts()
        .head(15)
        .rename_axis("descriptor")
        .reset_index(name="complaints")
    )

    sample_rows = (
        df.loc[:, ["created_date", "borough", "complaint_type", "descriptor", "status"]]
        .dropna(subset=["created_date"])
        .sample(min(15, len(df.index)), random_state=42)
        .sort_values("created_date")
        .reset_index(drop=True)
    )

    tables = {
        "quality": quality,
        "monthly": monthly,
        "monthly_by_type": monthly_by_type,
        "top_complaint_types": top_complaint_types,
        "seasonal": seasonal,
        "hourly": hourly,
        "weekday_hour": weekday_hour,
        "borough_counts": borough_counts,
        "borough_mix": borough_mix,
        "resolution_bucket_counts": resolution_bucket_counts,
        "descriptor_counts": descriptor_counts,
        "sample_rows": sample_rows,
    }
    return [write_table(table, paths[name]) for name, table in tables.items()]


def build_nlp_summary_artifacts(
    nlp_path: Path, analytics_dir: Path, output_dir: Path
) -> list[Path]:
    paths = nlp_summary_paths(output_dir)
    columns = [
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
    df = pd.read_parquet(nlp_path, columns=columns)

    overall = pd.DataFrame(
        [
            {
                "complaints": len(df.index),
                "modeled_share": float(df["subtype_modeled_flag"].mean()),
                "mismatch_share": float(df["potential_label_mismatch_flag"].mean()),
                "resolution_text_share": float(
                    df["resolution_outcome_group"]
                    .astype("string")
                    .ne("no_resolution_text")
                    .mean()
                ),
                "high_confidence_subtype_share": float(
                    df["issue_subtype_confidence"].astype("string").eq("high").mean()
                ),
            }
        ]
    )

    issue_families = (
        df["issue_family"]
        .astype("string")
        .value_counts()
        .head(15)
        .rename_axis("issue_family")
        .reset_index(name="complaints")
    )
    modeled_subtypes = (
        df.loc[df["subtype_modeled_flag"], "issue_subtype"]
        .astype("string")
        .value_counts()
        .head(15)
        .rename_axis("issue_subtype")
        .reset_index(name="complaints")
    )
    subtype_source = (
        df["subtype_source"]
        .astype("string")
        .value_counts()
        .rename_axis("subtype_source")
        .reset_index(name="complaints")
    )
    outcome_groups = (
        df["resolution_outcome_group"]
        .astype("string")
        .value_counts()
        .rename_axis("resolution_outcome_group")
        .reset_index(name="complaints")
    )
    outcome_confidence = (
        df["resolution_outcome_confidence"]
        .astype("string")
        .value_counts()
        .rename_axis("resolution_outcome_confidence")
        .reset_index(name="complaints")
    )

    complaint_type_coverage = (
        df.groupby("complaint_type", observed=True)
        .agg(
            complaints=("issue_family", "size"),
            subtype_modeled_share=("subtype_modeled_flag", "mean"),
            mismatch_share=("potential_label_mismatch_flag", "mean"),
            top_issue_family=("issue_family", first_mode),
        )
        .reset_index()
        .sort_values("complaints", ascending=False)
    )
    top_modeled_by_type = (
        df.loc[df["subtype_modeled_flag"]]
        .groupby("complaint_type", observed=True)["issue_subtype"]
        .agg(top_issue_subtype=first_mode)
        .reset_index()
    )
    complaint_type_coverage = complaint_type_coverage.merge(
        top_modeled_by_type, on="complaint_type", how="left"
    )

    residual_counts = (
        df["residual_cluster_label"]
        .astype("string")
        .fillna("missing")
        .replace({"<NA>": "missing"})
        .value_counts()
        .rename_axis("residual_cluster_label")
        .reset_index(name="complaints")
    )

    outcome_by_complaint_type = (
        df.groupby(["resolution_outcome_group", "complaint_type"], observed=True)
        .size()
        .reset_index(name="complaints")
        .sort_values(
            ["resolution_outcome_group", "complaints"], ascending=[True, False]
        )
    )
    outcome_by_agency = (
        df.groupby(["resolution_outcome_group", "agency"], observed=True)
        .size()
        .reset_index(name="complaints")
        .sort_values(
            ["resolution_outcome_group", "complaints"], ascending=[True, False]
        )
    )

    community_board_metrics = pd.read_parquet(
        analytics_dir / COMMUNITY_BOARD_METRICS_FILENAME
    )
    agency_metrics = pd.read_parquet(analytics_dir / AGENCY_METRICS_FILENAME)
    board_top_subtype_rows = community_board_metrics.loc[
        community_board_metrics["top_issue_subtype"].notna()
        & community_board_metrics["top_issue_subtype"]
        .astype("string")
        .ne("not_modeled")
    ]
    board_top_subtypes = (
        board_top_subtype_rows["top_issue_subtype"]
        .astype("string")
        .value_counts()
        .rename_axis("issue_subtype")
        .reset_index(name="boards")
    )
    agency_top_subtypes = agency_metrics.loc[
        :, ["agency", "complaints", "top_modeled_subtype"]
    ].copy()

    tables = {
        "overall": overall,
        "issue_families": issue_families,
        "modeled_subtypes": modeled_subtypes,
        "subtype_source": subtype_source,
        "outcome_groups": outcome_groups,
        "outcome_confidence": outcome_confidence,
        "complaint_type_coverage": complaint_type_coverage,
        "residual_counts": residual_counts,
        "outcome_by_complaint_type": outcome_by_complaint_type,
        "outcome_by_agency": outcome_by_agency,
        "board_top_subtypes": board_top_subtypes,
        "agency_top_subtypes": agency_top_subtypes,
    }
    return [write_table(table, paths[name]) for name, table in tables.items()]


def build_geo_summary_artifacts(
    analytic_path: Path, nlp_path: Path, analytics_dir: Path, output_dir: Path
) -> list[Path]:
    paths = geo_summary_paths(output_dir)
    analytic = pd.read_parquet(
        analytic_path,
        columns=["unique_key", "latitude", "longitude", "borough", "complaint_type"],
    )
    analytic["valid_coordinate_flag"] = (
        analytic["latitude"].notna()
        & analytic["longitude"].notna()
        & analytic["latitude"].between(*NYC_LAT_RANGE)
        & analytic["longitude"].between(*NYC_LON_RANGE)
    )

    borough_coverage = (
        analytic.groupby("borough", observed=True)
        .agg(
            complaints=("valid_coordinate_flag", "size"),
            geocoded_complaints=("valid_coordinate_flag", "sum"),
        )
        .reset_index()
    )
    borough_coverage["geocoded_share"] = (
        borough_coverage["geocoded_complaints"] / borough_coverage["complaints"]
    )
    borough_coverage = borough_coverage.sort_values(
        "complaints", ascending=False
    ).reset_index(drop=True)

    grid_monthly = pd.read_parquet(analytics_dir / GRID_MONTHLY_FILENAME)
    grid_monthly["created_month_start"] = pd.to_datetime(
        grid_monthly["created_month_start"], errors="coerce"
    )
    grid_monthly["hotspot_threshold"] = grid_monthly.groupby("created_month_start")[
        "complaints"
    ].transform(lambda series: series.quantile(0.9))
    grid_monthly["is_hotspot"] = grid_monthly["complaints"].ge(
        grid_monthly["hotspot_threshold"]
    )
    hotspot_monthly = (
        grid_monthly.groupby("created_month_start", observed=True)
        .agg(
            hotspot_cells=("is_hotspot", "sum"),
            hotspot_complaints=(
                "complaints",
                lambda series: int(
                    series[grid_monthly.loc[series.index, "is_hotspot"]].sum()
                ),
            ),
            total_complaints=("complaints", "sum"),
        )
        .reset_index()
    )
    hotspot_monthly["hotspot_complaint_share"] = (
        hotspot_monthly["hotspot_complaints"] / hotspot_monthly["total_complaints"]
    )
    hotspot_monthly = add_month_label(hotspot_monthly)

    subtypes = pd.read_parquet(
        nlp_path, columns=["unique_key", "issue_subtype", "subtype_modeled_flag"]
    )
    hotspot_frame = analytic.merge(
        subtypes, on="unique_key", how="left", validate="one_to_one"
    )
    complaint_hotspots = summarize_hotspot_grids(
        hotspot_frame, "complaint_type", top_n_categories=6, top_n_grids=3
    )
    modeled_only = hotspot_frame.loc[
        hotspot_frame["subtype_modeled_flag"]
        & hotspot_frame["issue_subtype"].astype("string").ne("not_modeled")
    ].copy()
    subtype_hotspots = summarize_hotspot_grids(
        modeled_only, "issue_subtype", top_n_categories=6, top_n_grids=3
    )

    tables = {
        "borough_coverage": borough_coverage,
        "hotspot_monthly": hotspot_monthly,
        "complaint_hotspots": complaint_hotspots,
        "subtype_hotspots": subtype_hotspots,
    }
    return [write_table(table, paths[name]) for name, table in tables.items()]


def build_dashboard_summary_artifacts(
    analytic_path: Path, nlp_path: Path, analytics_dir: Path
) -> list[Path]:
    generated = []
    generated.extend(build_eda_summary_artifacts(analytic_path, analytics_dir))
    gc.collect()
    generated.extend(
        build_nlp_summary_artifacts(nlp_path, analytics_dir, analytics_dir)
    )
    gc.collect()
    generated.extend(
        build_geo_summary_artifacts(
            analytic_path, nlp_path, analytics_dir, analytics_dir
        )
    )
    gc.collect()
    return generated
