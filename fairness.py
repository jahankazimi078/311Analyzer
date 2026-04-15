from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score

from geo import add_community_board_fields, clean_zip_series, first_mode, percentile_90


PROJECT_ROOT = Path(__file__).resolve().parent
ANALYTIC_PATH = PROJECT_ROOT / "data" / "analytics" / "requests_2025_2026_analytic.parquet"
NLP_PATH = PROJECT_ROOT / "data" / "analytics" / "requests_2025_2026_issue_subtypes.parquet"
ZCTA_DEMOGRAPHICS_PATH = PROJECT_ROOT / "data" / "reference" / "nyc_zcta_demographics.parquet"

OUTPUT_DIR = PROJECT_ROOT / "data" / "analytics"
ZCTA_FAIRNESS_METRICS_PATH = OUTPUT_DIR / "requests_2025_2026_zcta_fairness_metrics.parquet"
ZCTA_FAIRNESS_STRATIFIED_PATH = OUTPUT_DIR / "requests_2025_2026_zcta_fairness_stratified.parquet"
ZCTA_FAIRNESS_MONTHLY_PATH = OUTPUT_DIR / "requests_2025_2026_zcta_fairness_monthly.parquet"
COMMUNITY_BOARD_FAIRNESS_SENSITIVITY_PATH = (
    OUTPUT_DIR / "requests_2025_2026_community_board_fairness_sensitivity.parquet"
)
FAIRNESS_MODEL_RESULTS_PATH = OUTPUT_DIR / "requests_2025_2026_fairness_model_results.parquet"

DEMOGRAPHIC_COLUMNS = [
    "population",
    "median_household_income",
    "poverty_share",
    "renter_share",
    "nonwhite_share",
    "bachelors_or_higher_share",
    "median_gross_rent",
    "acs_missing_flag",
]
QUINTILE_COLUMNS = [
    "income_quintile",
    "poverty_quintile",
    "renter_quintile",
    "nonwhite_quintile",
    "education_quintile",
]


def load_zcta_demographics() -> pd.DataFrame:
    if not ZCTA_DEMOGRAPHICS_PATH.exists():
        raise FileNotFoundError(
            f"Missing ZCTA demographics parquet: {ZCTA_DEMOGRAPHICS_PATH}. Run `./.venv/bin/python data/build_reference_demographics.py` first."
        )
    return pd.read_parquet(ZCTA_DEMOGRAPHICS_PATH)


def add_quintile_column(
    df: pd.DataFrame,
    source_column: str,
    output_column: str,
    labels: list[str],
) -> pd.DataFrame:
    values = pd.to_numeric(df[source_column], errors="coerce")
    non_null = values.dropna()
    if non_null.nunique() < len(labels):
        df[output_column] = pd.Series(pd.NA, index=df.index, dtype="string")
        return df

    ranked = non_null.rank(method="first")
    quintiles = pd.qcut(ranked, q=len(labels), labels=labels)
    df[output_column] = pd.Series(pd.NA, index=df.index, dtype="string")
    df.loc[non_null.index, output_column] = quintiles.astype("string")
    return df


def load_fairness_analysis_frame() -> pd.DataFrame:
    if not ANALYTIC_PATH.exists():
        raise FileNotFoundError(f"Missing analytic parquet: {ANALYTIC_PATH}")
    if not NLP_PATH.exists():
        raise FileNotFoundError(f"Missing NLP parquet: {NLP_PATH}")

    analytic_columns = [
        "unique_key",
        "created_date",
        "created_year",
        "created_month_start",
        "created_season",
        "agency",
        "complaint_type",
        "descriptor",
        "borough",
        "incident_zip",
        "community_board",
        "status",
        "is_closed_status",
        "resolution_days",
        "resolved_with_valid_date",
    ]
    nlp_columns = [
        "unique_key",
        "issue_family",
        "issue_subtype",
        "subtype_modeled_flag",
        "potential_label_mismatch_flag",
    ]

    analytic_df = pd.read_parquet(ANALYTIC_PATH, columns=analytic_columns)
    nlp_df = pd.read_parquet(NLP_PATH, columns=nlp_columns)
    demographics_df = load_zcta_demographics()
    df = analytic_df.merge(nlp_df, on="unique_key", how="left", validate="one_to_one")

    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    df["created_month_start"] = pd.to_datetime(df["created_month_start"], errors="coerce")
    df["created_year"] = pd.to_numeric(df["created_year"], errors="coerce").astype("Int64")
    df["agency"] = df["agency"].astype("string").str.upper().fillna("Unknown")
    df["complaint_type"] = df["complaint_type"].astype("string").fillna("Unknown")
    df["descriptor"] = df["descriptor"].astype("string").fillna("Unknown")
    df["borough"] = df["borough"].astype("string").str.upper().fillna("Unknown")
    df["status"] = df["status"].astype("string").str.title().fillna("Unknown")
    df["resolution_days"] = pd.to_numeric(df["resolution_days"], errors="coerce")
    df["resolved_with_valid_date"] = df["resolved_with_valid_date"].fillna(False)
    df["is_closed_status"] = df["is_closed_status"].fillna(df["status"].eq("Closed"))
    df["unresolved_flag"] = ~df["resolved_with_valid_date"]
    df["status_backlog_flag"] = ~df["is_closed_status"]
    df["issue_family"] = df["issue_family"].fillna(df["complaint_type"])
    df["issue_subtype"] = df["issue_subtype"].fillna("not_modeled")
    df["subtype_modeled_flag"] = df["subtype_modeled_flag"].fillna(False)
    df["potential_label_mismatch_flag"] = df["potential_label_mismatch_flag"].fillna(False)
    df["zcta"] = clean_zip_series(df["incident_zip"])

    merged = df.merge(demographics_df, on="zcta", how="left")
    merged = add_quintile_column(
        merged,
        source_column="median_household_income",
        output_column="income_quintile",
        labels=["Q1 lowest income", "Q2", "Q3", "Q4", "Q5 highest income"],
    )
    merged = add_quintile_column(
        merged,
        source_column="poverty_share",
        output_column="poverty_quintile",
        labels=["Q1 lowest poverty", "Q2", "Q3", "Q4", "Q5 highest poverty"],
    )
    merged = add_quintile_column(
        merged,
        source_column="renter_share",
        output_column="renter_quintile",
        labels=["Q1 lowest renter", "Q2", "Q3", "Q4", "Q5 highest renter"],
    )
    merged = add_quintile_column(
        merged,
        source_column="nonwhite_share",
        output_column="nonwhite_quintile",
        labels=["Q1 lowest nonwhite", "Q2", "Q3", "Q4", "Q5 highest nonwhite"],
    )
    merged = add_quintile_column(
        merged,
        source_column="bachelors_or_higher_share",
        output_column="education_quintile",
        labels=["Q1 lowest education", "Q2", "Q3", "Q4", "Q5 highest education"],
    )
    merged["zcta_demographic_match_flag"] = merged["median_household_income"].notna()
    return add_community_board_fields(merged)


def matched_zcta_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["zcta"].notna() & df["zcta_demographic_match_flag"]].copy()


def _group_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = (
        df.groupby(group_cols, dropna=False, observed=True)
        .agg(
            complaints=("unique_key", "size"),
            resolved_complaints=("resolved_with_valid_date", "sum"),
            unresolved_complaints=("unresolved_flag", "sum"),
            status_backlog_complaints=("status_backlog_flag", "sum"),
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
    summary["mismatch_flag_share"] = summary["mismatch_flagged_complaints"] / summary["complaints"]

    float_columns = [
        "median_resolution_days",
        "p90_resolution_days",
        "resolved_share",
        "unresolved_share",
        "status_backlog_share",
        "mismatch_flag_share",
    ]
    for column in float_columns:
        summary[column] = pd.to_numeric(summary[column], errors="coerce").astype(float)
    return summary


def build_zcta_fairness_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = matched_zcta_frame(df)
    group_cols = ["zcta"]
    summary = _group_summary(df, group_cols)
    descriptors = (
        df
        .groupby(group_cols, dropna=False, observed=True)
        .agg(
            borough_count=("borough", lambda series: int(series.dropna().nunique())),
            top_borough=("borough", first_mode),
            top_agency=("agency", first_mode),
            top_complaint_type=("complaint_type", first_mode),
            top_issue_family=("issue_family", first_mode),
        )
        .reset_index()
    )
    zcta_demographics = df.loc[:, ["zcta", *DEMOGRAPHIC_COLUMNS, *QUINTILE_COLUMNS]].drop_duplicates(subset=["zcta"]).copy()

    summary = summary.merge(descriptors, on="zcta", how="left")
    summary = summary.merge(zcta_demographics, on="zcta", how="left", validate="one_to_one")
    summary["complaints_per_10k"] = np.where(
        summary["population"].gt(0),
        summary["complaints"] / summary["population"] * 10_000,
        np.nan,
    )
    summary["unresolved_per_10k"] = np.where(
        summary["population"].gt(0),
        summary["unresolved_complaints"] / summary["population"] * 10_000,
        np.nan,
    )
    return summary.sort_values("complaints", ascending=False).reset_index(drop=True)


def build_zcta_fairness_stratified(
    df: pd.DataFrame,
    issue_levels: tuple[str, ...] = ("complaint_type", "issue_family", "issue_subtype"),
) -> pd.DataFrame:
    df = matched_zcta_frame(df)
    frames: list[pd.DataFrame] = []
    zcta_demographics = df.loc[:, ["zcta", *DEMOGRAPHIC_COLUMNS, *QUINTILE_COLUMNS]].drop_duplicates(subset=["zcta"]).copy()

    for issue_level in issue_levels:
        issue_df = df.copy()
        if issue_level == "issue_subtype":
            issue_df = issue_df.loc[
                issue_df["subtype_modeled_flag"] & issue_df["issue_subtype"].ne("not_modeled")
            ].copy()
        if issue_df.empty:
            continue

        group_cols = ["zcta", issue_level, "agency"]
        summary = _group_summary(issue_df, group_cols).rename(columns={issue_level: "issue_value"})
        summary["issue_level"] = issue_level

        benchmark = _group_summary(issue_df, [issue_level, "agency"]).rename(columns={issue_level: "issue_value"})
        benchmark = benchmark.rename(
            columns={
                "complaints": "benchmark_complaints",
                "resolved_complaints": "benchmark_resolved_complaints",
                "unresolved_complaints": "benchmark_unresolved_complaints",
                "status_backlog_complaints": "benchmark_status_backlog_complaints",
                "median_resolution_days": "benchmark_median_resolution_days",
                "p90_resolution_days": "benchmark_p90_resolution_days",
                "resolved_share": "benchmark_resolved_share",
                "unresolved_share": "benchmark_unresolved_share",
                "status_backlog_share": "benchmark_status_backlog_share",
            }
        )
        keep_columns = [
            "issue_value",
            "agency",
            "benchmark_complaints",
            "benchmark_resolved_complaints",
            "benchmark_unresolved_complaints",
            "benchmark_status_backlog_complaints",
            "benchmark_median_resolution_days",
            "benchmark_p90_resolution_days",
            "benchmark_resolved_share",
            "benchmark_unresolved_share",
            "benchmark_status_backlog_share",
        ]
        summary = summary.merge(benchmark[keep_columns], on=["issue_value", "agency"], how="left")
        summary = summary.merge(zcta_demographics, on="zcta", how="left")
        summary["cell_volume_share_of_benchmark"] = summary["complaints"] / summary["benchmark_complaints"]
        summary["median_resolution_gap_days"] = (
            summary["median_resolution_days"] - summary["benchmark_median_resolution_days"]
        )
        summary["p90_resolution_gap_days"] = summary["p90_resolution_days"] - summary["benchmark_p90_resolution_days"]
        summary["unresolved_share_gap"] = summary["unresolved_share"] - summary["benchmark_unresolved_share"]
        summary["status_backlog_share_gap"] = (
            summary["status_backlog_share"] - summary["benchmark_status_backlog_share"]
        )
        float_columns = [
            "benchmark_median_resolution_days",
            "benchmark_p90_resolution_days",
            "benchmark_resolved_share",
            "benchmark_unresolved_share",
            "benchmark_status_backlog_share",
            "cell_volume_share_of_benchmark",
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
    return pd.concat(frames, ignore_index=True).sort_values(
        ["issue_level", "benchmark_complaints", "complaints"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def build_zcta_fairness_monthly(df: pd.DataFrame) -> pd.DataFrame:
    df = matched_zcta_frame(df)
    monthly = _group_summary(df, ["created_month_start", "created_year", "zcta"])
    zcta_demographics = df.loc[:, ["zcta", *DEMOGRAPHIC_COLUMNS, *QUINTILE_COLUMNS]].drop_duplicates(subset=["zcta"]).copy()
    monthly = monthly.merge(zcta_demographics, on="zcta", how="left", validate="many_to_one")
    monthly["complaints_per_10k"] = np.where(
        monthly["population"].gt(0),
        monthly["complaints"] / monthly["population"] * 10_000,
        np.nan,
    )
    return monthly.sort_values(["zcta", "created_month_start"]).reset_index(drop=True)


def build_community_board_fairness_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["community_board", "community_board_borough", "community_board_code"]
    summary = _group_summary(df, group_cols)
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
    return summary.sort_values(["community_board_borough", "complaints"], ascending=[True, False]).reset_index(drop=True)


def build_fairness_model_results(df: pd.DataFrame, sample_size: int = 250_000) -> pd.DataFrame:
    df = matched_zcta_frame(df)
    demographic_features = [
        "income_quintile",
        "poverty_quintile",
        "renter_quintile",
        "nonwhite_quintile",
        "education_quintile",
    ]
    control_features = [
        "complaint_type",
        "agency",
        "borough",
        "created_year",
        "created_season",
    ]
    feature_columns = [*demographic_features, *control_features]
    rows = []

    resolved_df = df.loc[df["resolved_with_valid_date"] & df["zcta"].notna(), feature_columns + ["resolution_days"]].copy()
    resolved_df = resolved_df.dropna(subset=feature_columns + ["resolution_days"])
    if len(resolved_df.index) > sample_size:
        resolved_df = resolved_df.sample(sample_size, random_state=42)
    resolved_df["log_resolution_days"] = np.log1p(resolved_df["resolution_days"])

    unresolved_df = df.loc[df["zcta"].notna(), feature_columns + ["unresolved_flag"]].copy()
    unresolved_df = unresolved_df.dropna(subset=feature_columns)
    if len(unresolved_df.index) > sample_size:
        unresolved_df = unresolved_df.sample(sample_size, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                feature_columns,
            )
        ]
    )

    duration_model = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", LinearRegression())]
    )
    duration_model.fit(resolved_df[feature_columns], resolved_df["log_resolution_days"])
    duration_features = duration_model.named_steps["preprocessor"].get_feature_names_out()
    duration_coefficients = duration_model.named_steps["model"].coef_
    duration_r2 = duration_model.score(resolved_df[feature_columns], resolved_df["log_resolution_days"])

    backlog_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1_000, solver="lbfgs")),
        ]
    )
    backlog_model.fit(unresolved_df[feature_columns], unresolved_df["unresolved_flag"].astype(int))
    backlog_features = backlog_model.named_steps["preprocessor"].get_feature_names_out()
    backlog_coefficients = backlog_model.named_steps["model"].coef_[0]
    backlog_probabilities = backlog_model.predict_proba(unresolved_df[feature_columns])[:, 1]
    backlog_auc = roc_auc_score(unresolved_df["unresolved_flag"].astype(int), backlog_probabilities)

    for feature_name, coefficient in zip(duration_features, duration_coefficients):
        if any(token in feature_name for token in demographic_features):
            rows.append(
                {
                    "model_name": "log_resolution_days",
                    "feature": feature_name,
                    "coefficient": float(coefficient),
                    "metric": "r2",
                    "metric_value": float(duration_r2),
                    "sample_rows": int(len(resolved_df.index)),
                }
            )
    for feature_name, coefficient in zip(backlog_features, backlog_coefficients):
        if any(token in feature_name for token in demographic_features):
            rows.append(
                {
                    "model_name": "unresolved_log_odds",
                    "feature": feature_name,
                    "coefficient": float(coefficient),
                    "metric": "roc_auc",
                    "metric_value": float(backlog_auc),
                    "sample_rows": int(len(unresolved_df.index)),
                }
            )
    return pd.DataFrame(rows).sort_values(["model_name", "feature"]).reset_index(drop=True)


def export_fairness_outputs(
    zcta_fairness_metrics: pd.DataFrame,
    zcta_fairness_stratified: pd.DataFrame,
    zcta_fairness_monthly: pd.DataFrame,
    community_board_fairness_sensitivity: pd.DataFrame,
    fairness_model_results: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    zcta_fairness_metrics.to_parquet(ZCTA_FAIRNESS_METRICS_PATH, index=False)
    zcta_fairness_stratified.to_parquet(ZCTA_FAIRNESS_STRATIFIED_PATH, index=False)
    zcta_fairness_monthly.to_parquet(ZCTA_FAIRNESS_MONTHLY_PATH, index=False)
    community_board_fairness_sensitivity.to_parquet(COMMUNITY_BOARD_FAIRNESS_SENSITIVITY_PATH, index=False)
    fairness_model_results.to_parquet(FAIRNESS_MODEL_RESULTS_PATH, index=False)


def build_fairness_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_fairness_analysis_frame()
    zcta_fairness_metrics = build_zcta_fairness_metrics(df)
    zcta_fairness_stratified = build_zcta_fairness_stratified(df)
    zcta_fairness_monthly = build_zcta_fairness_monthly(df)
    community_board_fairness_sensitivity = build_community_board_fairness_sensitivity(df)
    fairness_model_results = build_fairness_model_results(df)
    export_fairness_outputs(
        zcta_fairness_metrics,
        zcta_fairness_stratified,
        zcta_fairness_monthly,
        community_board_fairness_sensitivity,
        fairness_model_results,
    )
    return (
        df,
        zcta_fairness_metrics,
        zcta_fairness_stratified,
        zcta_fairness_monthly,
        community_board_fairness_sensitivity,
        fairness_model_results,
    )


if __name__ == "__main__":
    build_fairness_outputs()
