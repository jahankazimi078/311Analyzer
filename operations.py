from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from geo import add_community_board_fields, first_mode, percentile_90


PROJECT_ROOT = Path(__file__).resolve().parent
ANALYTIC_PATH = PROJECT_ROOT / "data" / "analytics" / "requests_2025_2026_analytic.parquet"
NLP_PATH = PROJECT_ROOT / "data" / "analytics" / "requests_2025_2026_issue_subtypes.parquet"

OUTPUT_DIR = PROJECT_ROOT / "data" / "analytics"
AGENCY_METRICS_PATH = OUTPUT_DIR / "requests_2025_2026_agency_metrics.parquet"
COMPLAINT_TYPE_METRICS_PATH = OUTPUT_DIR / "requests_2025_2026_complaint_type_metrics.parquet"
AGENCY_ISSUE_METRICS_PATH = OUTPUT_DIR / "requests_2025_2026_agency_issue_metrics.parquet"
OPERATIONS_MONTHLY_PATH = OUTPUT_DIR / "requests_2025_2026_operations_monthly.parquet"
COMMUNITY_BOARD_OPERATIONS_PATH = OUTPUT_DIR / "requests_2025_2026_community_board_operations.parquet"


def load_operations_analysis_frame() -> pd.DataFrame:
    if not ANALYTIC_PATH.exists():
        raise FileNotFoundError(f"Missing analytic parquet: {ANALYTIC_PATH}")
    if not NLP_PATH.exists():
        raise FileNotFoundError(f"Missing NLP parquet: {NLP_PATH}")

    analytic_columns = [
        "unique_key",
        "created_date",
        "created_year",
        "created_month_start",
        "agency",
        "agency_name",
        "complaint_type",
        "descriptor",
        "status",
        "borough",
        "community_board",
        "is_closed_status",
        "has_closed_date",
        "closed_status_missing_date_flag",
        "nonclosed_status_has_date_flag",
        "negative_resolution_flag",
        "resolution_days",
        "resolved_with_valid_date",
        "resolution_bucket",
    ]
    nlp_columns = [
        "unique_key",
        "issue_family",
        "issue_subtype",
        "subtype_modeled_flag",
        "potential_label_mismatch_flag",
        "resolution_outcome_group",
    ]

    analytic_df = pd.read_parquet(ANALYTIC_PATH, columns=analytic_columns)
    nlp_df = pd.read_parquet(NLP_PATH, columns=nlp_columns)
    df = analytic_df.merge(nlp_df, on="unique_key", how="left", validate="one_to_one")

    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    df["created_month_start"] = pd.to_datetime(df["created_month_start"], errors="coerce")
    df["created_year"] = pd.to_numeric(df["created_year"], errors="coerce").astype("Int64")
    df["agency"] = df["agency"].astype("string").str.upper().fillna("Unknown")
    df["agency_name"] = df["agency_name"].astype("string").fillna("Unknown")
    df["complaint_type"] = df["complaint_type"].astype("string").fillna("Unknown")
    df["status"] = df["status"].astype("string").str.title().fillna("Unknown")
    df["resolution_days"] = pd.to_numeric(df["resolution_days"], errors="coerce")
    df["resolved_with_valid_date"] = df["resolved_with_valid_date"].fillna(False)
    df["is_closed_status"] = df["is_closed_status"].fillna(df["status"].eq("Closed"))
    df["has_closed_date"] = df["has_closed_date"].fillna(False)
    df["closed_status_missing_date_flag"] = df["closed_status_missing_date_flag"].fillna(False)
    df["nonclosed_status_has_date_flag"] = df["nonclosed_status_has_date_flag"].fillna(False)
    df["negative_resolution_flag"] = df["negative_resolution_flag"].fillna(False)
    df["unresolved_flag"] = ~df["resolved_with_valid_date"]
    df["status_backlog_flag"] = ~df["is_closed_status"]

    df["issue_family"] = df["issue_family"].fillna(df["complaint_type"])
    df["issue_subtype"] = df["issue_subtype"].fillna("not_modeled")
    df["subtype_modeled_flag"] = df["subtype_modeled_flag"].fillna(False)
    df["potential_label_mismatch_flag"] = df["potential_label_mismatch_flag"].fillna(False)
    df["resolution_outcome_group"] = df["resolution_outcome_group"].fillna("unknown")

    return add_community_board_fields(df)


def aligned_ytd_cutoff(df: pd.DataFrame, partial_year: int = 2026) -> pd.Timestamp | None:
    partial_year_dates = df.loc[df["created_year"].eq(partial_year), "created_date"].dropna()
    if partial_year_dates.empty:
        return None
    return partial_year_dates.max().normalize()


def filter_aligned_ytd(
    df: pd.DataFrame,
    base_year: int = 2025,
    partial_year: int = 2026,
) -> pd.DataFrame:
    cutoff = aligned_ytd_cutoff(df, partial_year=partial_year)
    if cutoff is None:
        return df.loc[df["created_year"].isin([base_year, partial_year])].copy()

    base_cutoff = cutoff.replace(year=base_year)
    mask = (
        df["created_year"].eq(base_year) & df["created_date"].le(base_cutoff)
    ) | (
        df["created_year"].eq(partial_year) & df["created_date"].le(cutoff)
    )
    return df.loc[mask].copy()


def _build_group_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = (
        df.groupby(group_cols, dropna=False, observed=True)
        .agg(
            complaints=("unique_key", "size"),
            resolved_complaints=("resolved_with_valid_date", "sum"),
            unresolved_complaints=("unresolved_flag", "sum"),
            status_backlog_complaints=("status_backlog_flag", "sum"),
            closed_status_missing_date_count=("closed_status_missing_date_flag", "sum"),
            nonclosed_status_has_date_count=("nonclosed_status_has_date_flag", "sum"),
            negative_resolution_count=("negative_resolution_flag", "sum"),
            mismatch_flagged_complaints=("potential_label_mismatch_flag", "sum"),
        )
        .reset_index()
    )

    resolution_metrics = (
        df.loc[df["resolved_with_valid_date"]]
        .groupby(group_cols, dropna=False, observed=True)["resolution_days"]
        .agg(median_resolution_days="median", p90_resolution_days=percentile_90)
        .reset_index()
    )

    summary = grouped.merge(resolution_metrics, on=group_cols, how="left")
    summary["resolved_share"] = summary["resolved_complaints"] / summary["complaints"]
    summary["unresolved_share"] = summary["unresolved_complaints"] / summary["complaints"]
    summary["status_backlog_share"] = summary["status_backlog_complaints"] / summary["complaints"]
    summary["closed_status_missing_date_share"] = (
        summary["closed_status_missing_date_count"] / summary["complaints"]
    )
    summary["nonclosed_status_has_date_share"] = (
        summary["nonclosed_status_has_date_count"] / summary["complaints"]
    )
    summary["negative_resolution_share"] = summary["negative_resolution_count"] / summary["complaints"]
    summary["mismatch_flag_share"] = summary["mismatch_flagged_complaints"] / summary["complaints"]

    float_columns = [
        "median_resolution_days",
        "p90_resolution_days",
        "resolved_share",
        "unresolved_share",
        "status_backlog_share",
        "closed_status_missing_date_share",
        "nonclosed_status_has_date_share",
        "negative_resolution_share",
        "mismatch_flag_share",
    ]
    for column in float_columns:
        if column in summary.columns:
            summary[column] = pd.to_numeric(summary[column], errors="coerce").astype(float)
    return summary


def _top_modeled_subtypes(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    modeled = df.loc[df["subtype_modeled_flag"] & df["issue_subtype"].ne("not_modeled")]
    if modeled.empty:
        return pd.DataFrame(columns=[*group_cols, "top_modeled_subtype"])
    return (
        modeled.groupby(group_cols, dropna=False, observed=True)["issue_subtype"]
        .agg(top_modeled_subtype=first_mode)
        .reset_index()
    )


def build_agency_metrics(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["agency", "agency_name"]
    summary = _build_group_summary(df, group_cols)

    descriptors = (
        df.groupby(group_cols, dropna=False, observed=True)
        .agg(
            agencies_boroughs=("borough", lambda series: int(series.dropna().nunique())),
            top_complaint_type=("complaint_type", first_mode),
            top_issue_family=("issue_family", first_mode),
            top_resolution_outcome_group=("resolution_outcome_group", first_mode),
        )
        .reset_index()
    )
    summary = summary.merge(descriptors, on=group_cols, how="left")
    summary = summary.merge(_top_modeled_subtypes(df, group_cols), on=group_cols, how="left")
    summary["borough_coverage"] = summary["agencies_boroughs"]
    return summary.sort_values("complaints", ascending=False).reset_index(drop=True)


def build_complaint_type_metrics(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["complaint_type"]
    summary = _build_group_summary(df, group_cols)

    descriptors = (
        df.groupby(group_cols, dropna=False, observed=True)
        .agg(
            agencies_responding=("agency", lambda series: int(series.dropna().nunique())),
            top_agency=("agency", first_mode),
            top_issue_family=("issue_family", first_mode),
            top_resolution_outcome_group=("resolution_outcome_group", first_mode),
        )
        .reset_index()
    )
    summary = summary.merge(descriptors, on=group_cols, how="left")
    summary = summary.merge(_top_modeled_subtypes(df, group_cols), on=group_cols, how="left")
    return summary.sort_values("complaints", ascending=False).reset_index(drop=True)


def build_agency_issue_metrics(
    df: pd.DataFrame,
    issue_levels: tuple[str, ...] = ("complaint_type", "issue_family", "issue_subtype"),
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for issue_level in issue_levels:
        issue_df = df.copy()
        if issue_level == "issue_subtype":
            issue_df = issue_df.loc[
                issue_df["subtype_modeled_flag"] & issue_df["issue_subtype"].ne("not_modeled")
            ].copy()
        if issue_df.empty:
            continue

        group_cols = [issue_level, "agency", "agency_name"]
        summary = _build_group_summary(issue_df, group_cols)
        summary = summary.rename(columns={issue_level: "issue_value"})
        summary["issue_level"] = issue_level

        issue_totals = _build_group_summary(issue_df, [issue_level]).rename(columns={issue_level: "issue_value"})
        issue_totals = issue_totals.rename(
            columns={
                "complaints": "issue_complaints",
                "resolved_complaints": "issue_resolved_complaints",
                "unresolved_complaints": "issue_unresolved_complaints",
                "status_backlog_complaints": "issue_status_backlog_complaints",
                "median_resolution_days": "issue_median_resolution_days",
                "p90_resolution_days": "issue_p90_resolution_days",
                "resolved_share": "issue_resolved_share",
                "unresolved_share": "issue_unresolved_share",
                "status_backlog_share": "issue_status_backlog_share",
            }
        )
        keep_columns = [
            "issue_value",
            "issue_complaints",
            "issue_resolved_complaints",
            "issue_unresolved_complaints",
            "issue_status_backlog_complaints",
            "issue_median_resolution_days",
            "issue_p90_resolution_days",
            "issue_resolved_share",
            "issue_unresolved_share",
            "issue_status_backlog_share",
        ]
        summary = summary.merge(issue_totals[keep_columns], on="issue_value", how="left")
        summary["issue_volume_share"] = summary["complaints"] / summary["issue_complaints"]
        summary["median_resolution_gap_days"] = (
            summary["median_resolution_days"] - summary["issue_median_resolution_days"]
        )
        summary["p90_resolution_gap_days"] = summary["p90_resolution_days"] - summary["issue_p90_resolution_days"]
        summary["unresolved_share_gap"] = summary["unresolved_share"] - summary["issue_unresolved_share"]
        summary["status_backlog_share_gap"] = (
            summary["status_backlog_share"] - summary["issue_status_backlog_share"]
        )
        float_columns = [
            "issue_median_resolution_days",
            "issue_p90_resolution_days",
            "issue_resolved_share",
            "issue_unresolved_share",
            "issue_status_backlog_share",
            "issue_volume_share",
            "median_resolution_gap_days",
            "p90_resolution_gap_days",
            "unresolved_share_gap",
            "status_backlog_share_gap",
        ]
        for column in float_columns:
            summary[column] = pd.to_numeric(summary[column], errors="coerce").astype(float)
        frames.append(summary)

    if not frames:
        return pd.DataFrame()

    agency_issue_metrics = pd.concat(frames, ignore_index=True)
    agency_issue_metrics["slower_than_issue_median_flag"] = agency_issue_metrics[
        "median_resolution_gap_days"
    ].gt(0)
    return agency_issue_metrics.sort_values(
        ["issue_level", "issue_complaints", "complaints"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def build_operations_monthly(df: pd.DataFrame) -> pd.DataFrame:
    monthly = _build_group_summary(df, ["created_month_start", "created_year"])
    cutoff = aligned_ytd_cutoff(df)
    monthly["aligned_ytd_cutoff"] = cutoff
    monthly["year_type"] = np.where(monthly["created_year"].eq(2026), "YTD", "full_or_aligned")
    return monthly.sort_values("created_month_start").reset_index(drop=True)


def build_community_board_operations(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["community_board", "community_board_borough", "community_board_code"]
    summary = _build_group_summary(df, group_cols)
    descriptors = (
        df.groupby(group_cols, dropna=False, observed=True)
        .agg(
            top_agency=("agency", first_mode),
            top_complaint_type=("complaint_type", first_mode),
            top_issue_family=("issue_family", first_mode),
        )
        .reset_index()
    )
    summary = summary.merge(descriptors, on=group_cols, how="left")
    summary = summary.merge(_top_modeled_subtypes(df, group_cols), on=group_cols, how="left")
    return summary.sort_values(["community_board_borough", "complaints"], ascending=[True, False]).reset_index(drop=True)


def build_segment_monthly_metrics(df: pd.DataFrame, segment_column: str) -> pd.DataFrame:
    monthly = _build_group_summary(df, ["created_month_start", "created_year", segment_column])
    return monthly.sort_values([segment_column, "created_month_start"]).reset_index(drop=True)


def export_operations_outputs(
    agency_metrics: pd.DataFrame,
    complaint_type_metrics: pd.DataFrame,
    agency_issue_metrics: pd.DataFrame,
    operations_monthly: pd.DataFrame,
    community_board_operations: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    agency_metrics.to_parquet(AGENCY_METRICS_PATH, index=False)
    complaint_type_metrics.to_parquet(COMPLAINT_TYPE_METRICS_PATH, index=False)
    agency_issue_metrics.to_parquet(AGENCY_ISSUE_METRICS_PATH, index=False)
    operations_monthly.to_parquet(OPERATIONS_MONTHLY_PATH, index=False)
    community_board_operations.to_parquet(COMMUNITY_BOARD_OPERATIONS_PATH, index=False)


def build_operations_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_operations_analysis_frame()
    agency_metrics = build_agency_metrics(df)
    complaint_type_metrics = build_complaint_type_metrics(df)
    agency_issue_metrics = build_agency_issue_metrics(df)
    operations_monthly = build_operations_monthly(df)
    community_board_operations = build_community_board_operations(df)
    export_operations_outputs(
        agency_metrics,
        complaint_type_metrics,
        agency_issue_metrics,
        operations_monthly,
        community_board_operations,
    )
    return (
        df,
        agency_metrics,
        complaint_type_metrics,
        agency_issue_metrics,
        operations_monthly,
        community_board_operations,
    )


if __name__ == "__main__":
    build_operations_outputs()
