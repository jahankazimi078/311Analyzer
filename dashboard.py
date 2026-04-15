from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from geo import NYC_LAT_RANGE, NYC_LON_RANGE, summarize_hotspot_grids


STREAMLIT_IMPORT_ERROR: ModuleNotFoundError | None = None

try:
    import streamlit as st
except ModuleNotFoundError as exc:  # pragma: no cover - import guard for local verification
    STREAMLIT_IMPORT_ERROR = exc

    class _StreamlitShim:
        @staticmethod
        def cache_data(func=None, **_: Any):
            if func is None:
                return lambda inner: inner
            return func

    st = _StreamlitShim()


PROJECT_ROOT = Path(__file__).resolve().parent
ANALYTIC_PATH = PROJECT_ROOT / "data" / "analytics" / "requests_2025_2026_analytic.parquet"
NLP_PATH = PROJECT_ROOT / "data" / "analytics" / "requests_2025_2026_issue_subtypes.parquet"
ANALYTICS_DIR = PROJECT_ROOT / "data" / "analytics"
DEPLOY_ROOT = PROJECT_ROOT / "data" / "deploy"
DEPLOY_ANALYTICS_DIR = DEPLOY_ROOT / "analytics"
DEPLOY_REFERENCE_DIR = DEPLOY_ROOT / "reference"

DEPLOY_ANALYTIC_CORE_PATH = DEPLOY_ANALYTICS_DIR / "requests_2025_2026_analytic_dashboard_core.parquet"
DEPLOY_ANALYTIC_GEO_PATH = DEPLOY_ANALYTICS_DIR / "requests_2025_2026_analytic_dashboard_geo.parquet"
DEPLOY_NLP_PATH = DEPLOY_ANALYTICS_DIR / "requests_2025_2026_issue_subtypes_dashboard.parquet"

OPERATIONS_MONTHLY_PATH = ANALYTICS_DIR / "requests_2025_2026_operations_monthly.parquet"
AGENCY_METRICS_PATH = ANALYTICS_DIR / "requests_2025_2026_agency_metrics.parquet"
COMPLAINT_TYPE_METRICS_PATH = ANALYTICS_DIR / "requests_2025_2026_complaint_type_metrics.parquet"
AGENCY_ISSUE_METRICS_PATH = ANALYTICS_DIR / "requests_2025_2026_agency_issue_metrics.parquet"
ZCTA_METRICS_PATH = ANALYTICS_DIR / "requests_2025_2026_zcta_metrics.parquet"
GRID_MONTHLY_PATH = ANALYTICS_DIR / "requests_2025_2026_grid_monthly.parquet"
GRID_PERSISTENCE_PATH = ANALYTICS_DIR / "requests_2025_2026_grid_persistence.parquet"
COMMUNITY_BOARD_METRICS_PATH = ANALYTICS_DIR / "requests_2025_2026_community_board_metrics.parquet"
COMMUNITY_BOARD_MONTHLY_PATH = ANALYTICS_DIR / "requests_2025_2026_community_board_monthly.parquet"
COMMUNITY_BOARD_OPERATIONS_PATH = ANALYTICS_DIR / "requests_2025_2026_community_board_operations.parquet"
ZCTA_FAIRNESS_METRICS_PATH = ANALYTICS_DIR / "requests_2025_2026_zcta_fairness_metrics.parquet"
ZCTA_FAIRNESS_STRATIFIED_PATH = ANALYTICS_DIR / "requests_2025_2026_zcta_fairness_stratified.parquet"
ZCTA_FAIRNESS_MONTHLY_PATH = ANALYTICS_DIR / "requests_2025_2026_zcta_fairness_monthly.parquet"
COMMUNITY_BOARD_FAIRNESS_SENSITIVITY_PATH = (
    ANALYTICS_DIR / "requests_2025_2026_community_board_fairness_sensitivity.parquet"
)
FAIRNESS_MODEL_RESULTS_PATH = ANALYTICS_DIR / "requests_2025_2026_fairness_model_results.parquet"
RESOLUTION_MODEL_METRICS_PATH = ANALYTICS_DIR / "requests_2025_2026_resolution_bucket_model_metrics.parquet"
RESOLUTION_PREDICTIONS_PATH = ANALYTICS_DIR / "requests_2025_2026_resolution_bucket_predictions.parquet"
RESOLUTION_CONFUSION_PATH = ANALYTICS_DIR / "requests_2025_2026_resolution_bucket_confusion_matrix.parquet"
RESOLUTION_ERROR_SLICES_PATH = ANALYTICS_DIR / "requests_2025_2026_resolution_bucket_error_slices.parquet"
RESOLUTION_FEATURE_IMPORTANCE_PATH = ANALYTICS_DIR / "requests_2025_2026_resolution_bucket_feature_importance.parquet"

PAGE_OPTIONS = [
    "Overview",
    "City Patterns",
    "Text Insights",
    "Geography",
    "Operations",
    "Neighborhood Equity",
    "Forecasting",
]
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SEASON_ORDER = ["Winter", "Spring", "Summer", "Fall"]
RESOLUTION_BUCKET_ORDER = ["<1 day", "1-3 days", "3-7 days", "7-30 days", "30+ days", "Missing"]
MAP_METRIC_OPTIONS = {
    "Complaints": "complaints",
    "Complaints per 10k": "complaints_per_10k",
    "Unresolved share": "unresolved_share",
    "Median resolution days": "median_resolution_days",
    "P90 resolution days": "p90_resolution_days",
}
HOTSPOT_METRIC_OPTIONS = {
    "Hotspot month share": "hotspot_month_share",
    "Total complaints": "total_complaints",
    "Unresolved share": "unresolved_share",
}
FAIRNESS_MAP_METRICS = {
    "Complaints per 10k": "complaints_per_10k",
    "Median resolution days": "median_resolution_days",
    "Unresolved share": "unresolved_share",
    "Status backlog share": "status_backlog_share",
}
FAIRNESS_QUINTILE_OPTIONS = {
    "Income quintile": "income_quintile",
    "Poverty quintile": "poverty_quintile",
    "Renter quintile": "renter_quintile",
    "Nonwhite quintile": "nonwhite_quintile",
    "Education quintile": "education_quintile",
}
MODEL_METRIC_LABELS = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro F1",
    "weighted_f1": "Weighted F1",
}
MODEL_NAME_LABELS = {
    "most_frequent_baseline": "Most frequent baseline",
    "multinomial_logistic": "Multinomial logistic",
    "hist_gradient_boosting": "HistGradientBoosting",
}
FEATURE_SET_LABELS = {
    "shared": "Shared baseline",
    "post_routing": "Post-routing",
    "intake_only": "Intake-only",
}


def _ensure_exists(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required dashboard input: {path}")
    return path


def resolve_dashboard_input(path: Path) -> Path:
    if path.exists():
        return path

    try:
        relative_path = path.relative_to(PROJECT_ROOT / "data")
    except ValueError:
        return _ensure_exists(path)

    deploy_path = DEPLOY_ROOT / relative_path
    if deploy_path.exists():
        return deploy_path
    return _ensure_exists(path)


def _missing_columns_message(label: str, requested: list[str], available: set[str]) -> str:
    missing = sorted(set(requested) - available)
    return f"{label} is missing required columns: {', '.join(missing)}"


def _read_parquet_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    return pd.read_parquet(_ensure_exists(path), columns=columns)


def _parquet_columns(path: Path) -> set[str]:
    return set(pq.read_schema(_ensure_exists(path)).names)


@st.cache_data(show_spinner=False)
def read_analytic_columns(columns: list[str]) -> pd.DataFrame:
    if ANALYTIC_PATH.exists():
        return _read_parquet_columns(ANALYTIC_PATH, columns)

    deploy_sources = [
        ("deployment analytic core parquet", DEPLOY_ANALYTIC_CORE_PATH),
        ("deployment analytic geo parquet", DEPLOY_ANALYTIC_GEO_PATH),
    ]
    source_columns_by_path: list[tuple[Path, set[str]]] = []
    available_columns: set[str] = set()
    frames: list[pd.DataFrame] = []
    remaining = list(dict.fromkeys(columns))

    for _, path in deploy_sources:
        if not path.exists():
            continue
        source_columns = _parquet_columns(path)
        source_columns_by_path.append((path, source_columns))
        available_columns.update(source_columns)
        if set(remaining).issubset(source_columns):
            return _read_parquet_columns(path, remaining)

    for path, source_columns in source_columns_by_path:
        selected = [column for column in remaining if column in source_columns]
        if not selected:
            continue
        frame = _read_parquet_columns(path, selected)
        if "unique_key" in frame.columns:
            frame = frame.drop_duplicates(subset="unique_key", keep="first")
        frames.append(frame)
        remaining = [column for column in remaining if column not in selected]

    if remaining:
        raise FileNotFoundError(_missing_columns_message("Deployment analytic artifacts", columns, available_columns))
    if len(frames) == 1:
        return frames[0]

    merged = frames[0]
    for frame in frames[1:]:
        if "unique_key" not in merged.columns or "unique_key" not in frame.columns:
            raise ValueError(
                "Requested analytic columns span multiple deployment artifacts that cannot be joined safely."
            )
        merged = merged.merge(frame, on="unique_key", how="inner", validate="one_to_one")
    return merged.loc[:, columns]


@st.cache_data(show_spinner=False)
def read_nlp_columns(columns: list[str]) -> pd.DataFrame:
    if NLP_PATH.exists():
        return _read_parquet_columns(NLP_PATH, columns)

    if not DEPLOY_NLP_PATH.exists():
        raise FileNotFoundError(f"Missing required dashboard input: {NLP_PATH}")

    available_columns = _parquet_columns(DEPLOY_NLP_PATH)
    missing = [column for column in columns if column not in available_columns]
    if missing:
        raise FileNotFoundError(_missing_columns_message("Deployment NLP artifact", columns, available_columns))
    return _read_parquet_columns(DEPLOY_NLP_PATH, columns)


@st.cache_data(show_spinner=False)
def load_table(path: str) -> pd.DataFrame:
    return pd.read_parquet(resolve_dashboard_input(Path(path)))


@st.cache_data(show_spinner=False)
def load_geo_table(path: str) -> gpd.GeoDataFrame:
    return gpd.read_parquet(resolve_dashboard_input(Path(path)))


def format_int(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    return f"{int(round(float(value))):,}"


def format_float(value: Any, decimals: int = 2) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):,.{decimals}f}"


def format_pct(value: Any, decimals: int = 1) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.{decimals}f}%"


def format_days(value: Any, decimals: int = 2) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{decimals}f} days"


def pretty_model_name(value: Any) -> str:
    return MODEL_NAME_LABELS.get(str(value), str(value))


def pretty_feature_set(value: Any) -> str:
    return FEATURE_SET_LABELS.get(str(value), str(value))


def previous_month_start(timestamp: pd.Timestamp) -> pd.Timestamp:
    month_start = timestamp.to_period("M").to_timestamp()
    return month_start - pd.offsets.MonthBegin(1)


def add_month_label(df: pd.DataFrame, source_column: str = "created_month_start") -> pd.DataFrame:
    df = df.copy()
    df[source_column] = pd.to_datetime(df[source_column], errors="coerce")
    df["month_label"] = df[source_column].dt.strftime("%Y-%m")
    return df


def first_mode(series: pd.Series, default: str = "Unknown") -> str:
    mode = series.dropna().astype("string").mode()
    if mode.empty:
        return default
    return str(mode.iloc[0])


def safe_share(numerator: float, denominator: float) -> float:
    if denominator in [0, None] or pd.isna(denominator):
        return np.nan
    return float(numerator) / float(denominator)


def series_bar_chart(series: pd.Series, title: str | None = None) -> None:
    if title:
        st.markdown(f"**{title}**")
    st.bar_chart(series, width="stretch")


def pivot_line_chart(
    df: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    title: str | None = None,
) -> None:
    if title:
        st.markdown(f"**{title}**")
    st.line_chart(df.pivot(index=index, columns=columns, values=values), width="stretch")


def line_chart_from_series(df: pd.DataFrame, index: str, value: str, title: str | None = None) -> None:
    if title:
        st.markdown(f"**{title}**")
    chart_df = df.set_index(index)[value]
    st.line_chart(chart_df, width="stretch")


def findings_box(lines: list[str]) -> None:
    st.markdown("**Key findings**")
    st.markdown("\n".join(f"- {line}" for line in lines))


def quality_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def encode_metric_colors(values: pd.Series, reverse_scale: bool = False) -> list[list[int]]:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return [[80, 140, 220, 180] for _ in range(len(values.index))]

    lower = float(valid.quantile(0.05))
    upper = float(valid.quantile(0.95))
    if upper <= lower:
        normalized = pd.Series(0.5, index=values.index, dtype=float)
    else:
        normalized = ((numeric - lower) / (upper - lower)).clip(0, 1)
    normalized = normalized.fillna(0.25)
    if reverse_scale:
        normalized = 1 - normalized

    colors: list[list[int]] = []
    for value in normalized:
        red = int(60 + 175 * value)
        green = int(170 - 110 * value)
        blue = int(220 - 120 * value)
        colors.append([red, green, blue, 190])
    return colors


def scale_marker_sizes(values: pd.Series, min_radius: int = 800, max_radius: int = 8000) -> list[float]:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return [float(min_radius)] * len(values.index)
    lower = float(valid.quantile(0.1))
    upper = float(valid.quantile(0.95))
    if upper <= lower:
        normalized = pd.Series(0.5, index=values.index, dtype=float)
    else:
        normalized = ((numeric - lower) / (upper - lower)).clip(0, 1)
    normalized = normalized.fillna(0.2)
    return (min_radius + normalized * (max_radius - min_radius)).tolist()


def map_points_from_geometries(
    geo_df: gpd.GeoDataFrame,
    metric_column: str,
    label_column: str,
    tooltip_columns: list[str],
    reverse_scale: bool = False,
) -> pd.DataFrame:
    plot_df = geo_df.loc[geo_df.geometry.notna()].copy()
    centroids = plot_df.to_crs(epsg=3857).geometry.centroid.to_crs(epsg=4326)
    plot_df["latitude"] = centroids.y.astype(float)
    plot_df["longitude"] = centroids.x.astype(float)
    plot_df["marker_color"] = encode_metric_colors(plot_df[metric_column], reverse_scale=reverse_scale)
    plot_df["marker_radius"] = scale_marker_sizes(plot_df[metric_column])
    ordered_columns = [
        label_column,
        metric_column,
        "latitude",
        "longitude",
        "marker_color",
        "marker_radius",
        *tooltip_columns,
    ]
    unique_columns = list(dict.fromkeys(ordered_columns))
    return plot_df.loc[:, unique_columns].reset_index(drop=True)


def render_point_map(plot_df: pd.DataFrame, label_column: str, metric_column: str, metric_label: str) -> None:
    try:
        import pydeck as pdk
    except ModuleNotFoundError:
        st.info("Install `streamlit` to enable interactive map rendering.")
        st.dataframe(plot_df.drop(columns=["marker_color", "marker_radius"], errors="ignore"), width="stretch")
        return

    view_state = pdk.ViewState(
        latitude=float(plot_df["latitude"].mean()),
        longitude=float(plot_df["longitude"].mean()),
        zoom=9.3,
        pitch=0,
    )
    tooltip = {
        "html": (
            f"<b>{label_column.replace('_', ' ').title()}:</b> {{{label_column}}}<br/>"
            f"<b>{metric_label}:</b> {{{metric_column}}}"
        )
    }
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=plot_df,
        get_position="[longitude, latitude]",
        get_fill_color="marker_color",
        get_radius="marker_radius",
        pickable=True,
        opacity=0.75,
        stroked=True,
        line_width_min_pixels=1,
        get_line_color=[255, 255, 255, 120],
    )
    deck = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip, map_style="light")
    st.pydeck_chart(deck, width="stretch")


def format_metric_for_map(metric_column: str, series: pd.Series) -> pd.Series:
    if "share" in metric_column:
        return series.map(lambda value: format_pct(value, 1))
    if "days" in metric_column:
        return series.map(lambda value: format_float(value, 2))
    if "per_10k" in metric_column:
        return series.map(lambda value: format_float(value, 1))
    return series.map(format_int)


@st.cache_data(show_spinner=False)
def build_eda_summary() -> dict[str, Any]:
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
    df = read_analytic_columns(columns)
    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    df["created_month_start"] = pd.to_datetime(df["created_month_start"], errors="coerce")
    df["year_label"] = df["created_year"].map({2025: "2025", 2026: "2026 YTD"}).fillna(df["created_year"].astype("string"))

    quality = {
        "complaints": len(df.index),
        "resolved_with_valid_date": int(df["resolved_with_valid_date"].sum()),
        "resolved_with_valid_date_share": float(df["resolved_with_valid_date"].mean()),
        "closed_status_missing_date_count": int(df["closed_status_missing_date_flag"].sum()),
        "nonclosed_status_has_date_count": int(df["nonclosed_status_has_date_flag"].sum()),
        "negative_resolution_count": int(df["negative_resolution_flag"].sum()),
        "median_resolution_days": float(df.loc[df["resolved_with_valid_date"], "resolution_days"].median()),
        "max_created_date": df["created_date"].max(),
    }

    monthly = (
        df.groupby(["created_month_start", "year_label"], observed=True)
        .size()
        .reset_index(name="complaints")
        .sort_values("created_month_start")
    )
    monthly = add_month_label(monthly)

    top_complaint_types = (
        df["complaint_type"].astype("string").value_counts().head(12).rename_axis("complaint_type").reset_index(name="complaints")
    )
    top_type_values = top_complaint_types["complaint_type"].tolist()
    monthly_by_type = (
        df.loc[df["complaint_type"].astype("string").isin(top_type_values)]
        .groupby(["created_month_start", "complaint_type"], observed=True)
        .size()
        .reset_index(name="complaints")
        .sort_values("created_month_start")
    )
    monthly_by_type = add_month_label(monthly_by_type)

    seasonal = (
        df.groupby(["created_season", "year_label"], observed=True)
        .size()
        .reset_index(name="complaints")
    )
    seasonal["created_season"] = pd.Categorical(seasonal["created_season"], categories=SEASON_ORDER, ordered=True)
    seasonal = seasonal.sort_values(["created_season", "year_label"]).reset_index(drop=True)

    hourly = (
        df.groupby("created_hour", observed=True).size().reset_index(name="complaints").sort_values("created_hour")
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
    weekday_hour = weekday_hour.sort_values(["created_weekday", "created_hour"]).reset_index(drop=True)

    borough_counts = (
        df.groupby("borough", observed=True).size().reset_index(name="complaints").sort_values("complaints", ascending=False)
    )

    borough_mix = (
        df.loc[df["complaint_type"].astype("string").isin(top_type_values[:8])]
        .groupby(["borough", "complaint_type"], observed=True)
        .size()
        .reset_index(name="complaints")
        .sort_values(["borough", "complaints"], ascending=[True, False])
    )

    resolution_bucket_counts = (
        df["resolution_bucket"].astype("string").fillna("Missing").replace({"<NA>": "Missing"})
        .value_counts()
        .reindex(RESOLUTION_BUCKET_ORDER)
        .fillna(0)
        .rename_axis("resolution_bucket")
        .reset_index(name="complaints")
    )

    descriptor_counts = (
        df["descriptor"].astype("string").fillna("Unknown").value_counts().head(15).rename_axis("descriptor").reset_index(name="complaints")
    )

    sample_rows = (
        df.loc[:, ["created_date", "borough", "complaint_type", "descriptor", "status"]]
        .dropna(subset=["created_date"]) 
        .sample(min(15, len(df.index)), random_state=42)
        .sort_values("created_date")
        .reset_index(drop=True)
    )

    return {
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


@st.cache_data(show_spinner=False)
def build_nlp_summary() -> dict[str, Any]:
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
    df = read_nlp_columns(columns)

    overall = {
        "complaints": len(df.index),
        "modeled_share": float(df["subtype_modeled_flag"].mean()),
        "mismatch_share": float(df["potential_label_mismatch_flag"].mean()),
        "resolution_text_share": float(df["resolution_outcome_group"].astype("string").ne("no_resolution_text").mean()),
        "high_confidence_subtype_share": float(df["issue_subtype_confidence"].astype("string").eq("high").mean()),
    }

    issue_families = (
        df["issue_family"].astype("string").value_counts().head(15).rename_axis("issue_family").reset_index(name="complaints")
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
        df["subtype_source"].astype("string").value_counts().rename_axis("subtype_source").reset_index(name="complaints")
    )
    outcome_groups = (
        df["resolution_outcome_group"].astype("string").value_counts().rename_axis("resolution_outcome_group").reset_index(name="complaints")
    )
    outcome_confidence = (
        df["resolution_outcome_confidence"].astype("string").value_counts().rename_axis("resolution_outcome_confidence").reset_index(name="complaints")
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
    complaint_type_coverage = complaint_type_coverage.merge(top_modeled_by_type, on="complaint_type", how="left")

    residual_counts = (
        df["residual_cluster_label"].astype("string").fillna("missing").replace({"<NA>": "missing"})
        .value_counts()
        .rename_axis("residual_cluster_label")
        .reset_index(name="complaints")
    )
    nonmissing_residual = residual_counts.loc[~residual_counts["residual_cluster_label"].isin(["missing", "<NA>", "nan", "None"])]

    outcome_by_complaint_type = (
        df.groupby(["resolution_outcome_group", "complaint_type"], observed=True)
        .size()
        .reset_index(name="complaints")
        .sort_values(["resolution_outcome_group", "complaints"], ascending=[True, False])
    )
    outcome_by_agency = (
        df.groupby(["resolution_outcome_group", "agency"], observed=True)
        .size()
        .reset_index(name="complaints")
        .sort_values(["resolution_outcome_group", "complaints"], ascending=[True, False])
    )

    community_board_metrics = load_table(str(COMMUNITY_BOARD_METRICS_PATH))
    agency_metrics = load_table(str(AGENCY_METRICS_PATH))
    board_top_subtypes = (
        community_board_metrics.loc[community_board_metrics["top_issue_subtype"].notna()]
        .loc[community_board_metrics["top_issue_subtype"].astype("string").ne("not_modeled")]
        ["top_issue_subtype"]
        .astype("string")
        .value_counts()
        .rename_axis("issue_subtype")
        .reset_index(name="boards")
    )
    agency_top_subtypes = agency_metrics.loc[:, ["agency", "complaints", "top_modeled_subtype"]].copy()

    return {
        "overall": overall,
        "issue_families": issue_families,
        "modeled_subtypes": modeled_subtypes,
        "subtype_source": subtype_source,
        "outcome_groups": outcome_groups,
        "outcome_confidence": outcome_confidence,
        "complaint_type_coverage": complaint_type_coverage,
        "residual_counts": residual_counts,
        "nonmissing_residual": nonmissing_residual,
        "outcome_by_complaint_type": outcome_by_complaint_type,
        "outcome_by_agency": outcome_by_agency,
        "board_top_subtypes": board_top_subtypes,
        "agency_top_subtypes": agency_top_subtypes,
    }


@st.cache_data(show_spinner=False)
def build_geo_summary() -> dict[str, Any]:
    analytic = read_analytic_columns(["unique_key", "latitude", "longitude", "borough", "complaint_type"])
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
    borough_coverage["geocoded_share"] = borough_coverage["geocoded_complaints"] / borough_coverage["complaints"]
    borough_coverage = borough_coverage.sort_values("complaints", ascending=False).reset_index(drop=True)

    grid_monthly = load_table(str(GRID_MONTHLY_PATH)).copy()
    grid_monthly["created_month_start"] = pd.to_datetime(grid_monthly["created_month_start"], errors="coerce")
    grid_monthly["hotspot_threshold"] = grid_monthly.groupby("created_month_start")["complaints"].transform(
        lambda series: series.quantile(0.9)
    )
    grid_monthly["is_hotspot"] = grid_monthly["complaints"].ge(grid_monthly["hotspot_threshold"])
    hotspot_monthly = (
        grid_monthly.groupby("created_month_start", observed=True)
        .agg(
            hotspot_cells=("is_hotspot", "sum"),
            hotspot_complaints=("complaints", lambda series: int(series[grid_monthly.loc[series.index, "is_hotspot"]].sum())),
            total_complaints=("complaints", "sum"),
        )
        .reset_index()
    )
    hotspot_monthly["hotspot_complaint_share"] = hotspot_monthly["hotspot_complaints"] / hotspot_monthly["total_complaints"]
    hotspot_monthly = add_month_label(hotspot_monthly)

    subtypes = read_nlp_columns(["unique_key", "issue_subtype", "subtype_modeled_flag"])
    hotspot_frame = analytic.merge(subtypes, on="unique_key", how="left", validate="one_to_one")
    complaint_hotspots = summarize_hotspot_grids(hotspot_frame, "complaint_type", top_n_categories=6, top_n_grids=3)
    modeled_only = hotspot_frame.loc[
        hotspot_frame["subtype_modeled_flag"] & hotspot_frame["issue_subtype"].astype("string").ne("not_modeled")
    ].copy()
    subtype_hotspots = summarize_hotspot_grids(modeled_only, "issue_subtype", top_n_categories=6, top_n_grids=3)

    return {
        "borough_coverage": borough_coverage,
        "hotspot_monthly": hotspot_monthly,
        "complaint_hotspots": complaint_hotspots,
        "subtype_hotspots": subtype_hotspots,
    }


def aligned_complete_months(monthly: pd.DataFrame) -> pd.DataFrame:
    monthly = monthly.copy()
    monthly["created_month_start"] = pd.to_datetime(monthly["created_month_start"], errors="coerce")
    cutoff = pd.to_datetime(monthly["aligned_ytd_cutoff"], errors="coerce").dropna().max()
    if pd.isna(cutoff):
        return monthly.sort_values("created_month_start").reset_index(drop=True)

    latest_complete_month = previous_month_start(cutoff)
    latest_complete_month_number = int(latest_complete_month.month)
    filtered = monthly.loc[monthly["created_month_start"].dt.month.le(latest_complete_month_number)].copy()
    return filtered.sort_values("created_month_start").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def build_operations_summary() -> dict[str, Any]:
    operations_monthly = load_table(str(OPERATIONS_MONTHLY_PATH)).copy()
    complaint_type_metrics = load_table(str(COMPLAINT_TYPE_METRICS_PATH)).copy()
    agency_metrics = load_table(str(AGENCY_METRICS_PATH)).copy()
    agency_issue_metrics = load_table(str(AGENCY_ISSUE_METRICS_PATH)).copy()
    community_board_operations = load_table(str(COMMUNITY_BOARD_OPERATIONS_PATH)).copy()

    aligned_monthly = add_month_label(aligned_complete_months(operations_monthly))
    qa_totals = {
        "closed_status_missing_date_count": int(operations_monthly["closed_status_missing_date_count"].sum()),
        "nonclosed_status_has_date_count": int(operations_monthly["nonclosed_status_has_date_count"].sum()),
        "negative_resolution_count": int(operations_monthly["negative_resolution_count"].sum()),
    }

    return {
        "operations_monthly": operations_monthly,
        "aligned_monthly": aligned_monthly,
        "complaint_type_metrics": complaint_type_metrics,
        "agency_metrics": agency_metrics,
        "agency_issue_metrics": agency_issue_metrics,
        "community_board_operations": community_board_operations,
        "qa_totals": qa_totals,
    }


@st.cache_data(show_spinner=False)
def build_fairness_summary() -> dict[str, Any]:
    fairness_metrics = load_table(str(ZCTA_FAIRNESS_METRICS_PATH)).copy()
    fairness_stratified = load_table(str(ZCTA_FAIRNESS_STRATIFIED_PATH)).copy()
    fairness_monthly = load_table(str(ZCTA_FAIRNESS_MONTHLY_PATH)).copy()
    fairness_model_results = load_table(str(FAIRNESS_MODEL_RESULTS_PATH)).copy()
    board_sensitivity = load_table(str(COMMUNITY_BOARD_FAIRNESS_SENSITIVITY_PATH)).copy()

    fairness_monthly["created_month_start"] = pd.to_datetime(fairness_monthly["created_month_start"], errors="coerce")
    fairness_monthly = add_month_label(fairness_monthly)

    quintile_monthly_frames: list[pd.DataFrame] = []
    for label, column in FAIRNESS_QUINTILE_OPTIONS.items():
        quintile_monthly = (
            fairness_monthly.loc[fairness_monthly[column].notna()]
            .groupby(["month_label", column], observed=True)
            .agg(
                complaints=("complaints", "sum"),
                median_resolution_days=("median_resolution_days", "median"),
                unresolved_share=("unresolved_share", "median"),
                status_backlog_share=("status_backlog_share", "median"),
            )
            .reset_index()
            .rename(columns={column: "quintile_value"})
        )
        quintile_monthly["quintile_label"] = label
        quintile_monthly_frames.append(quintile_monthly)
    quintile_monthly = pd.concat(quintile_monthly_frames, ignore_index=True)

    return {
        "fairness_metrics": fairness_metrics,
        "fairness_stratified": fairness_stratified,
        "fairness_monthly": fairness_monthly,
        "quintile_monthly": quintile_monthly,
        "fairness_model_results": fairness_model_results,
        "board_sensitivity": board_sensitivity,
    }


@st.cache_data(show_spinner=False)
def build_predictive_summary() -> dict[str, Any]:
    model_metrics = load_table(str(RESOLUTION_MODEL_METRICS_PATH)).copy()
    predictions = load_table(str(RESOLUTION_PREDICTIONS_PATH)).copy()
    confusion = load_table(str(RESOLUTION_CONFUSION_PATH)).copy()
    error_slices = load_table(str(RESOLUTION_ERROR_SLICES_PATH)).copy()
    feature_importance = load_table(str(RESOLUTION_FEATURE_IMPORTANCE_PATH)).copy()

    predictions["created_date"] = pd.to_datetime(predictions["created_date"], errors="coerce")
    predictions["actual_bucket"] = predictions["resolution_bucket"].astype("string")
    predictions["predicted_bucket"] = predictions["predicted_resolution_bucket"].astype("string")
    high_confidence_misses = predictions.loc[
        ~predictions["correct_prediction_flag"] & predictions["predicted_probability"].ge(0.80)
    ].copy()
    high_confidence_misses = high_confidence_misses.sort_values("predicted_probability", ascending=False)

    overall_metrics = model_metrics.loc[model_metrics["metric_scope"].eq("overall")].copy()
    class_metrics = model_metrics.loc[model_metrics["metric_scope"].eq("class")].copy()

    return {
        "model_metrics": model_metrics,
        "overall_metrics": overall_metrics,
        "class_metrics": class_metrics,
        "predictions": predictions,
        "confusion": confusion,
        "error_slices": error_slices,
        "feature_importance": feature_importance,
        "high_confidence_misses": high_confidence_misses,
    }


def render_page_intro(title: str, caption: str) -> None:
    st.title("311Dashboard")
    st.subheader(title)
    st.caption(caption)


def render_overview_page() -> None:
    operations = build_operations_summary()
    geo = load_geo_table(str(ZCTA_METRICS_PATH))
    grid_persistence = load_geo_table(str(GRID_PERSISTENCE_PATH))
    fairness = build_fairness_summary()
    predictive = build_predictive_summary()
    eda = build_eda_summary()
    nlp = build_nlp_summary()

    render_page_intro(
        "Overview",
        "A polished executive view of city demand, issue taxonomy, neighborhood burden, service performance, equity signals, and resolution-time forecasting.",
    )

    total_complaints = eda["quality"]["complaints"]
    top_complaint_type = eda["top_complaint_types"].iloc[0]
    top_burden_zcta = geo.sort_values("complaints_per_10k", ascending=False).iloc[0]
    top_hotspot = grid_persistence.sort_values("hotspot_month_share", ascending=False).iloc[0]
    best_macro = (
        predictive["overall_metrics"].loc[predictive["overall_metrics"]["metric"].eq("macro_f1")]
        .sort_values("metric_value", ascending=False)
        .iloc[0]
    )

    cards = st.columns(4)
    cards[0].metric("Complaints in scope", format_int(total_complaints), "2025-2026")
    cards[1].metric("Largest complaint type", str(top_complaint_type["complaint_type"]), format_int(top_complaint_type["complaints"]))
    cards[2].metric("Highest burden ZIP", str(top_burden_zcta["zcta"]), f"{format_float(top_burden_zcta['complaints_per_10k'], 0)} per 10k")
    cards[3].metric(
        "Best predictive benchmark",
        pretty_model_name(best_macro["model_name"]),
        f"{pretty_feature_set(best_macro['feature_set'])} macro F1 {format_float(best_macro['metric_value'], 3)}",
    )

    findings_box(
        [
            f"Complaint demand is concentrated in a few very high-volume categories led by {top_complaint_type['complaint_type']}.",
            f"The dashboard brings the full project together in one interface, from exploratory analysis through forecasting.",
            "2026 remains labeled as YTD in trend and forecasting views so partial-year movement is not overstated.",
            "Fairness pages keep the repo's interpretation guardrails: area-level disparities are signals for follow-up, not causal proof.",
        ]
    )

    st.markdown(
        "Each section below highlights one analytic layer with a real chart and a short takeaway. The sidebar pages expand each section into the full set of views."
    )

    phase_tabs = st.tabs(["Demand", "Text", "Geography", "Operations", "Equity", "Forecasting"])
    with phase_tabs[0]:
        st.markdown(
            "The citywide demand picture is dominated by a small number of recurring complaint families, with strong concentration by borough and by time of day."
        )
        series_bar_chart(
            eda["top_complaint_types"].head(10).set_index("complaint_type")["complaints"],
            title="Largest complaint categories",
        )
        cols = st.columns(2)
        cols[0].markdown(
            f"`{top_complaint_type['complaint_type']}` is the largest category in scope with {format_int(top_complaint_type['complaints'])} complaints."
        )
        cols[1].markdown(
            f"{eda['borough_counts'].iloc[0]['borough']} contributes the highest raw complaint volume across the city."
        )
    with phase_tabs[1]:
        st.markdown(
            "The text layer turns broad complaint categories into more actionable issue subtypes and closure-language patterns that can be carried into later phases."
        )
        series_bar_chart(
            nlp["modeled_subtypes"].head(10).set_index("issue_subtype")["complaints"],
            title="Largest modeled issue subtypes",
        )
        st.markdown(
            f"Subtype assignment covers {format_pct(nlp['overall']['modeled_share'], 1)} of complaints, led by `{nlp['modeled_subtypes'].iloc[0]['issue_subtype']}`."
        )
    with phase_tabs[2]:
        st.markdown(
            "The geography layer distinguishes simple raw volume from resident-normalized burden and recurring hotspot persistence."
        )
        map_df = map_points_from_geometries(
            geo,
            metric_column="complaints_per_10k",
            label_column="zcta",
            tooltip_columns=["complaints", "population"],
        )
        map_df["complaints_per_10k"] = format_metric_for_map("complaints_per_10k", map_df["complaints_per_10k"])
        render_point_map(map_df, label_column="zcta", metric_column="complaints_per_10k", metric_label="Complaints per 10k")
        st.markdown(
            f"The highest-burden ZIP in the current outputs is `{top_burden_zcta['zcta']}`, and the most persistent hotspot cell is active in {format_pct(top_hotspot['hotspot_month_share'], 1)} of observed months."
        )
    with phase_tabs[3]:
        st.markdown(
            "Operations views show how service speed and backlog shift over time and where agencies run slower than their own issue-specific benchmark."
        )
        pivot_line_chart(
            operations["aligned_monthly"],
            index="month_label",
            columns="created_year",
            values="status_backlog_share",
            title="Status backlog share: aligned 2025 vs 2026 YTD",
        )
        slowest_type = operations["complaint_type_metrics"].loc[
            operations["complaint_type_metrics"]["complaints"].ge(10_000)
        ].sort_values("median_resolution_days", ascending=False).iloc[0]
        st.markdown(
            f"Among higher-volume categories, `{slowest_type['complaint_type']}` has the slowest median resolution time at {format_days(slowest_type['median_resolution_days'], 2)}."
        )
    with phase_tabs[4]:
        st.markdown(
            "Neighborhood equity views show where burden and outcome gaps are visible, and where those gaps remain after comparing like-with-like issue and agency slices."
        )
        income_quintiles = (
            fairness["fairness_metrics"].loc[fairness["fairness_metrics"]["income_quintile"].notna()]
            .groupby("income_quintile", observed=True)["median_resolution_days"]
            .median()
            .sort_index()
        )
        series_bar_chart(income_quintiles, title="Median resolution time by income quintile")
        st.markdown(
            f"The highest resident-normalized reported burden in the fairness layer appears in ZCTA `{fairness['fairness_metrics'].sort_values('complaints_per_10k', ascending=False).iloc[0]['zcta']}`."
        )
    with phase_tabs[5]:
        st.markdown(
            "The forecasting layer compares a baseline, an interpretable multinomial logistic model, and a stronger tree benchmark across both post-routing and intake-only feature sets."
        )
        score_pivot = predictive["overall_metrics"].loc[predictive["overall_metrics"]["metric"].eq("macro_f1")].pivot(
            index="model_name",
            columns="feature_set",
            values="metric_value",
        )
        score_pivot = score_pivot.rename(index=pretty_model_name, columns=pretty_feature_set)
        st.dataframe(
            score_pivot.style.format(lambda value: format_float(value, 3)).background_gradient(cmap="Blues"),
            width="stretch",
        )
        st.markdown(
            f"The strongest macro-F1 result in the current benchmark is `{pretty_model_name(best_macro['model_name'])}` on the `{pretty_feature_set(best_macro['feature_set'])}` feature set at {format_float(best_macro['metric_value'], 3)}."
        )


def render_eda_page() -> None:
    eda = build_eda_summary()
    complaint_type_metrics = load_table(str(COMPLAINT_TYPE_METRICS_PATH))
    zcta_metrics = load_geo_table(str(ZCTA_METRICS_PATH))

    render_page_intro(
        "City Patterns",
        "Exploratory analysis of complaint volume, timing, category mix, data quality, and resolution behavior across the scoped NYC 311 dataset.",
    )

    findings_box(
        [
            f"The largest complaint family in scope is {eda['top_complaint_types'].iloc[0]['complaint_type']} with {format_int(eda['top_complaint_types'].iloc[0]['complaints'])} complaints.",
            f"The median valid resolution interval remains short overall at {format_days(eda['quality']['median_resolution_days'], 2)}.",
            f"{eda['borough_counts'].iloc[0]['borough']} carries the largest raw complaint volume in the scoped window.",
            "The cleaned dataset preserves quality flags instead of dropping inconsistent rows globally.",
        ]
    )

    tabs = st.tabs(
        [
            "Data Quality Summary",
            "Complaint Volume Trends",
            "Seasonal and Hourly Patterns",
            "Complaint Mix and Geographic Concentration",
            "Operations and Response Time Analysis",
            "Text Preview",
        ]
    )

    with tabs[0]:
        quality = eda["quality"]
        metrics = st.columns(4)
        metrics[0].metric("Complaints", format_int(quality["complaints"]))
        metrics[1].metric("Valid resolution rows", format_pct(quality["resolved_with_valid_date_share"]), format_int(quality["resolved_with_valid_date"]))
        metrics[2].metric("Median valid resolution", format_days(quality["median_resolution_days"], 2))
        metrics[3].metric("Latest complaint in scope", quality["max_created_date"].strftime("%Y-%m-%d") if pd.notna(quality["max_created_date"]) else "NA")

        st.dataframe(
            quality_table(
                [
                    {"check": "Closed status missing close date", "rows": quality["closed_status_missing_date_count"], "share_of_total": quality["closed_status_missing_date_count"] / quality["complaints"]},
                    {"check": "Non-closed status with close date", "rows": quality["nonclosed_status_has_date_count"], "share_of_total": quality["nonclosed_status_has_date_count"] / quality["complaints"]},
                    {"check": "Negative resolution interval", "rows": quality["negative_resolution_count"], "share_of_total": quality["negative_resolution_count"] / quality["complaints"]},
                ]
            ).style.format({"rows": format_int, "share_of_total": lambda value: format_pct(value, 2)}),
            width="stretch",
        )

    with tabs[1]:
        pivot_line_chart(eda["monthly"], index="month_label", columns="year_label", values="complaints", title="Complaint volume by month")
        selected_types = st.multiselect(
            "Complaint types to compare",
            options=eda["top_complaint_types"]["complaint_type"].tolist(),
            default=eda["top_complaint_types"]["complaint_type"].tolist()[:4],
        )
        if selected_types:
            monthly_by_type = eda["monthly_by_type"].loc[eda["monthly_by_type"]["complaint_type"].isin(selected_types)]
            pivot_line_chart(monthly_by_type, index="month_label", columns="complaint_type", values="complaints", title="Top complaint-type trends")
        series_bar_chart(
            eda["top_complaint_types"].set_index("complaint_type")["complaints"],
            title="Top complaint types in scope",
        )

    with tabs[2]:
        season_pivot = eda["seasonal"].pivot(index="created_season", columns="year_label", values="complaints")
        st.markdown("**Seasonal complaint volume**")
        st.bar_chart(season_pivot, width="stretch")
        series_bar_chart(eda["hourly"].set_index("created_hour")["complaints"], title="Complaints by hour of day")
        heatmap = eda["weekday_hour"].pivot(index="created_weekday", columns="created_hour", values="complaints").reindex(WEEKDAY_ORDER)
        st.markdown("**Weekday by hour heatmap**")
        st.dataframe(heatmap.style.background_gradient(cmap="Blues").format(format_int), width="stretch")

    with tabs[3]:
        series_bar_chart(eda["borough_counts"].set_index("borough")["complaints"], title="Complaint volume by borough")
        borough_mix_pivot = eda["borough_mix"].pivot(index="borough", columns="complaint_type", values="complaints").fillna(0)
        st.markdown("**Borough complaint mix for the largest categories**")
        st.bar_chart(borough_mix_pivot, width="stretch")
        zcta_cols = st.columns(2)
        zcta_cols[0].markdown("**Highest raw-volume ZIPs**")
        zcta_cols[0].dataframe(
            zcta_metrics.sort_values("complaints", ascending=False).head(10).loc[:, ["zcta", "complaints", "unresolved_share"]].style.format({"complaints": format_int, "unresolved_share": lambda value: format_pct(value, 1)}),
            width="stretch",
        )
        zcta_cols[1].markdown("**Highest per-capita burden ZIPs**")
        zcta_cols[1].dataframe(
            zcta_metrics.sort_values("complaints_per_10k", ascending=False).head(10).loc[:, ["zcta", "complaints_per_10k", "complaints"]].style.format({"complaints_per_10k": lambda value: format_float(value, 1), "complaints": format_int}),
            width="stretch",
        )

    with tabs[4]:
        series_bar_chart(
            eda["resolution_bucket_counts"].set_index("resolution_bucket")["complaints"],
            title="Resolution bucket distribution",
        )
        st.dataframe(
            complaint_type_metrics.sort_values("median_resolution_days", ascending=False)
            .head(20)
            .loc[:, ["complaint_type", "complaints", "median_resolution_days", "p90_resolution_days", "unresolved_share", "top_agency"]]
            .style.format(
                {
                    "complaints": format_int,
                    "median_resolution_days": lambda value: format_days(value, 2),
                    "p90_resolution_days": lambda value: format_days(value, 2),
                    "unresolved_share": lambda value: format_pct(value, 1),
                }
            ),
            width="stretch",
        )

    with tabs[5]:
        series_bar_chart(eda["descriptor_counts"].set_index("descriptor")["complaints"], title="Most common descriptors")
        st.dataframe(
            eda["sample_rows"].style.format({"created_date": lambda value: value.strftime("%Y-%m-%d %H:%M") if pd.notna(value) else "NA"}),
            width="stretch",
        )


def render_nlp_page() -> None:
    nlp = build_nlp_summary()

    render_page_intro(
        "Text Insights",
        "Issue taxonomy, subtype coverage, closure-language patterns, and the operational signals that came out of the complaint text layer.",
    )

    findings_box(
        [
            f"Subtype modeling covers {format_pct(nlp['overall']['modeled_share'], 1)} of complaints and most assigned subtypes come from deterministic rules rather than opaque embeddings.",
            f"The largest modeled subtype is {nlp['modeled_subtypes'].iloc[0]['issue_subtype']} with {format_int(nlp['modeled_subtypes'].iloc[0]['complaints'])} complaints.",
            f"Resolution language is present on {format_pct(nlp['overall']['resolution_text_share'], 1)} of complaints, which makes closure-language NLP useful but incomplete.",
            f"Potential label mismatch flags remain extremely rare at {format_pct(nlp['overall']['mismatch_share'], 4)} of complaints.",
        ]
    )

    tabs = st.tabs(
        [
            "Rule-Based Subtype Taxonomy",
            "Model-Assisted Residual Discovery",
            "Resolution Language NLP",
            "Operations and Geography Tie-Ins",
        ]
    )

    with tabs[0]:
        metrics = st.columns(4)
        metrics[0].metric("Subtype modeled share", format_pct(nlp["overall"]["modeled_share"], 1))
        metrics[1].metric("Resolution text share", format_pct(nlp["overall"]["resolution_text_share"], 1))
        metrics[2].metric("High-confidence subtype share", format_pct(nlp["overall"]["high_confidence_subtype_share"], 1))
        metrics[3].metric("Mismatch flag share", format_pct(nlp["overall"]["mismatch_share"], 4))
        series_bar_chart(nlp["issue_families"].set_index("issue_family")["complaints"], title="Top issue families")
        series_bar_chart(nlp["modeled_subtypes"].set_index("issue_subtype")["complaints"], title="Top modeled issue subtypes")
        st.markdown("**Subtype source mix**")
        st.bar_chart(nlp["subtype_source"].set_index("subtype_source")["complaints"], width="stretch")
        st.dataframe(
            nlp["complaint_type_coverage"].head(20).style.format(
                {
                    "complaints": format_int,
                    "subtype_modeled_share": lambda value: format_pct(value, 1),
                    "mismatch_share": lambda value: format_pct(value, 3),
                }
            ),
            width="stretch",
        )

    with tabs[1]:
        if nlp["nonmissing_residual"].empty:
            st.warning(
                "No stable residual clusters were retained in the current text output, so this section is shown as a guardrail rather than a populated chart."
            )
        else:
            series_bar_chart(nlp["nonmissing_residual"].set_index("residual_cluster_label")["complaints"], title="Residual clusters")
        st.dataframe(nlp["residual_counts"].head(10).style.format({"complaints": format_int}), width="stretch")

    with tabs[2]:
        series_bar_chart(nlp["outcome_groups"].set_index("resolution_outcome_group")["complaints"], title="Resolution outcome groups")
        series_bar_chart(nlp["outcome_confidence"].set_index("resolution_outcome_confidence")["complaints"], title="Outcome confidence mix")
        selected_outcome = st.selectbox(
            "Inspect one outcome group",
            options=nlp["outcome_groups"]["resolution_outcome_group"].tolist(),
        )
        cols = st.columns(2)
        cols[0].markdown("**Top complaint types for this outcome**")
        cols[0].dataframe(
            nlp["outcome_by_complaint_type"].loc[nlp["outcome_by_complaint_type"]["resolution_outcome_group"].eq(selected_outcome)]
            .head(12)
            .style.format({"complaints": format_int}),
            width="stretch",
        )
        cols[1].markdown("**Top agencies for this outcome**")
        cols[1].dataframe(
            nlp["outcome_by_agency"].loc[nlp["outcome_by_agency"]["resolution_outcome_group"].eq(selected_outcome)]
            .head(12)
            .style.format({"complaints": format_int}),
            width="stretch",
        )

    with tabs[3]:
        if not nlp["board_top_subtypes"].empty:
            series_bar_chart(nlp["board_top_subtypes"].set_index("issue_subtype")["boards"], title="Most common top subtype across community boards")
        st.markdown("**Agency-level dominant modeled subtype**")
        st.dataframe(
            nlp["agency_top_subtypes"].sort_values("complaints", ascending=False).style.format({"complaints": format_int}),
            width="stretch",
        )


def render_geospatial_page() -> None:
    geo_summary = build_geo_summary()
    zcta_metrics = load_geo_table(str(ZCTA_METRICS_PATH))
    grid_persistence = load_geo_table(str(GRID_PERSISTENCE_PATH))
    community_board_metrics = load_table(str(COMMUNITY_BOARD_METRICS_PATH))
    community_board_monthly = load_table(str(COMMUNITY_BOARD_MONTHLY_PATH))

    render_page_intro(
        "Geography",
        "Spatial coverage, burden maps, persistent hotspots, and neighborhood-level operational context across ZIPs, grids, and community boards.",
    )

    findings_box(
        [
            f"Coordinate coverage remains high across boroughs, with {format_pct(geo_summary['borough_coverage']['geocoded_share'].mean(), 1)} average borough-level geocoding coverage.",
            f"The highest per-capita ZIP burden is {zcta_metrics.sort_values('complaints_per_10k', ascending=False).iloc[0]['zcta']} at {format_float(zcta_metrics.sort_values('complaints_per_10k', ascending=False).iloc[0]['complaints_per_10k'], 0)} complaints per 10k residents.",
            f"The most persistent hotspot cell appears in {format_pct(grid_persistence['hotspot_month_share'].max(), 1)} of active months.",
            "The spatial analysis shows not only where complaints are dense, but where resident-normalized burden and recurring hotspot persistence diverge from simple raw counts.",
        ]
    )

    tabs = st.tabs(
        [
            "Executive Summary",
            "Geographic Coverage By Borough",
            "Counts Versus Per-Capita Burden",
            "Community Board Operational View",
            "Persistent Hotspot Zones",
            "Hotspot Summaries By Category",
            "Findings",
        ]
    )

    with tabs[0]:
        metrics = st.columns(4)
        metrics[0].metric("ZIP / ZCTA rows", format_int(len(zcta_metrics.index)))
        metrics[1].metric("Community boards", format_int(len(community_board_metrics.index)))
        metrics[2].metric("Persistent hotspot cells", format_int(len(grid_persistence.index)))
        metrics[3].metric("Mean geocoded share", format_pct(geo_summary["borough_coverage"]["geocoded_share"].mean(), 1))
        line_chart_from_series(
            geo_summary["hotspot_monthly"],
            index="month_label",
            value="hotspot_complaint_share",
            title="Share of monthly complaints inside hotspot cells",
        )

    with tabs[1]:
        coverage = geo_summary["borough_coverage"].copy()
        coverage["geocoded_share_display"] = coverage["geocoded_share"].map(lambda value: format_pct(value, 1))
        st.bar_chart(coverage.set_index("borough")[["complaints", "geocoded_complaints"]], width="stretch")
        st.dataframe(
            coverage.loc[:, ["borough", "complaints", "geocoded_complaints", "geocoded_share"]].style.format(
                {"complaints": format_int, "geocoded_complaints": format_int, "geocoded_share": lambda value: format_pct(value, 1)}
            ),
            width="stretch",
        )

    with tabs[2]:
        map_metric_label = st.selectbox("ZIP map metric", options=list(MAP_METRIC_OPTIONS.keys()))
        map_metric = MAP_METRIC_OPTIONS[map_metric_label]
        plot_df = map_points_from_geometries(
            zcta_metrics,
            metric_column=map_metric,
            label_column="zcta",
            tooltip_columns=["population", "complaints"],
            reverse_scale=False,
        )
        plot_df[map_metric] = format_metric_for_map(map_metric, plot_df[map_metric])
        render_point_map(plot_df, label_column="zcta", metric_column=map_metric, metric_label=map_metric_label)
        cols = st.columns(2)
        cols[0].markdown("**Highest raw-volume ZIPs**")
        cols[0].dataframe(
            zcta_metrics.sort_values("complaints", ascending=False).head(15).loc[:, ["zcta", "complaints", "unresolved_share", "median_resolution_days"]].style.format(
                {"complaints": format_int, "unresolved_share": lambda value: format_pct(value, 1), "median_resolution_days": lambda value: format_days(value, 2)}
            ),
            width="stretch",
        )
        cols[1].markdown("**Highest per-capita burden ZIPs**")
        cols[1].dataframe(
            zcta_metrics.sort_values("complaints_per_10k", ascending=False).head(15).loc[:, ["zcta", "complaints_per_10k", "complaints", "unresolved_per_10k"]].style.format(
                {"complaints_per_10k": lambda value: format_float(value, 1), "complaints": format_int, "unresolved_per_10k": lambda value: format_float(value, 1)}
            ),
            width="stretch",
        )

    with tabs[3]:
        board_sort_metric = st.selectbox(
            "Board ranking metric",
            options=["complaints", "unresolved_share", "median_resolution_days", "p90_resolution_days"],
            format_func=lambda value: value.replace("_", " ").title(),
        )
        st.dataframe(
            community_board_metrics.sort_values(board_sort_metric, ascending=False).loc[:, [
                "community_board",
                "community_board_borough",
                "complaints",
                "unresolved_share",
                "median_resolution_days",
                "p90_resolution_days",
                "top_complaint_type",
                "top_issue_subtype",
            ]].style.format(
                {
                    "complaints": format_int,
                    "unresolved_share": lambda value: format_pct(value, 1),
                    "median_resolution_days": lambda value: format_days(value, 2),
                    "p90_resolution_days": lambda value: format_days(value, 2),
                }
            ),
            width="stretch",
        )
        top_boards = community_board_metrics.sort_values("complaints", ascending=False)["community_board"].head(8).tolist()
        selected_boards = st.multiselect("Board trend comparison", options=top_boards, default=top_boards[:3])
        if selected_boards:
            board_monthly = add_month_label(
                community_board_monthly.loc[community_board_monthly["community_board"].isin(selected_boards)].copy()
            )
            metric = st.selectbox(
                "Board trend metric",
                options=["complaints", "unresolved_share", "mismatch_flag_share"],
                format_func=lambda value: value.replace("_", " ").title(),
            )
            pivot_line_chart(board_monthly, index="month_label", columns="community_board", values=metric, title="Board monthly comparison")

    with tabs[4]:
        hotspot_metric_label = st.selectbox("Hotspot map metric", options=list(HOTSPOT_METRIC_OPTIONS.keys()))
        hotspot_metric = HOTSPOT_METRIC_OPTIONS[hotspot_metric_label]
        hotspot_df = map_points_from_geometries(
            grid_persistence,
            metric_column=hotspot_metric,
            label_column="grid_id",
            tooltip_columns=["total_complaints", "hotspot_months", "active_months"],
            reverse_scale=hotspot_metric == "unresolved_share",
        )
        hotspot_df[hotspot_metric] = format_metric_for_map(hotspot_metric, hotspot_df[hotspot_metric])
        render_point_map(hotspot_df, label_column="grid_id", metric_column=hotspot_metric, metric_label=hotspot_metric_label)
        pivot_line_chart(
            geo_summary["hotspot_monthly"].assign(series="Hotspot complaint share"),
            index="month_label",
            columns="series",
            values="hotspot_complaint_share",
            title="Share of monthly complaints inside hotspot cells",
        )
        st.dataframe(
            grid_persistence.sort_values([hotspot_metric, "total_complaints"], ascending=[False, False]).head(20).loc[:, [
                "grid_id",
                "total_complaints",
                "total_unresolved_complaints",
                "hotspot_months",
                "active_months",
                "hotspot_month_share",
                "unresolved_share",
            ]].style.format(
                {
                    "total_complaints": format_int,
                    "total_unresolved_complaints": format_int,
                    "hotspot_months": format_int,
                    "active_months": format_int,
                    "hotspot_month_share": lambda value: format_pct(value, 1),
                    "unresolved_share": lambda value: format_pct(value, 1),
                }
            ),
            width="stretch",
        )

    with tabs[5]:
        cols = st.columns(2)
        cols[0].markdown("**Top hotspot grids by complaint type**")
        cols[0].dataframe(geo_summary["complaint_hotspots"].style.format({"complaints": format_int}), width="stretch")
        cols[1].markdown("**Top hotspot grids by modeled subtype**")
        cols[1].dataframe(geo_summary["subtype_hotspots"].style.format({"complaints": format_int}), width="stretch")

    with tabs[6]:
        st.markdown(
            "The practical message is straightforward: the project now shows not just what residents report, but where pressure concentrates, where burden looks disproportionately high, and which zones stay operationally hot over time."
        )


def render_operations_page() -> None:
    ops = build_operations_summary()

    render_page_intro(
        "Operations",
        "Service-speed scorecards, backlog trends, and within-issue benchmarking for agencies and neighborhood operations.",
    )

    complaint_type_metrics = ops["complaint_type_metrics"]
    agency_metrics = ops["agency_metrics"]
    agency_issue_metrics = ops["agency_issue_metrics"]
    aligned_monthly = ops["aligned_monthly"]
    community_board_operations = ops["community_board_operations"]

    high_volume_slowest = complaint_type_metrics.loc[complaint_type_metrics["complaints"].ge(10_000)].sort_values("median_resolution_days", ascending=False).iloc[0]
    highest_backlog_agency = agency_metrics.sort_values("status_backlog_share", ascending=False).iloc[0]
    slowest_board = community_board_operations.loc[community_board_operations["complaints"].ge(10_000)].sort_values("median_resolution_days", ascending=False).iloc[0]

    findings_box(
        [
            f"Among high-volume categories, {high_volume_slowest['complaint_type']} has the slowest median resolution time at {format_days(high_volume_slowest['median_resolution_days'], 2)}.",
            f"{highest_backlog_agency['agency']} has the highest status backlog share at {format_pct(highest_backlog_agency['status_backlog_share'], 1)}.",
            f"The slowest high-volume community board in median resolution time is {slowest_board['community_board']} at {format_days(slowest_board['median_resolution_days'], 2)}.",
            "Backlog and duration QA fields remain visible instead of hiding messy source-system behavior.",
        ]
    )

    tabs = st.tabs(
        [
            "Data Quality Guardrails",
            "Slowest Complaint Categories",
            "Agency Scorecards",
            "Within-Issue Bottlenecks",
            "Monthly Operations Trends",
            "Neighborhood Operations View",
            "Findings",
        ]
    )

    with tabs[0]:
        st.info(
            "Headline metrics use two backlog views: unresolved share is based on missing valid closure intervals, while status backlog share is based on complaints whose current status is not Closed."
        )
        st.dataframe(
            quality_table(
                [
                    {"check": "Closed status missing close date", "rows": ops["qa_totals"]["closed_status_missing_date_count"]},
                    {"check": "Non-closed status with close date", "rows": ops["qa_totals"]["nonclosed_status_has_date_count"]},
                    {"check": "Negative resolution interval", "rows": ops["qa_totals"]["negative_resolution_count"]},
                ]
            ).style.format({"rows": format_int}),
            width="stretch",
        )

    with tabs[1]:
        min_rows = st.slider("Minimum complaint rows", min_value=1_000, max_value=int(complaint_type_metrics["complaints"].max()), value=10_000, step=1_000, key="ops_min_rows")
        rank_metric = st.selectbox(
            "Rank complaint categories by",
            options=["median_resolution_days", "p90_resolution_days", "unresolved_share", "status_backlog_share"],
            format_func=lambda value: value.replace("_", " ").title(),
        )
        filtered = complaint_type_metrics.loc[complaint_type_metrics["complaints"].ge(min_rows)].sort_values(rank_metric, ascending=False).head(20)
        series_bar_chart(filtered.set_index("complaint_type")[rank_metric], title="Top complaint categories by selected operations metric")
        st.dataframe(
            filtered.loc[:, ["complaint_type", "complaints", "median_resolution_days", "p90_resolution_days", "unresolved_share", "status_backlog_share", "top_agency"]].style.format(
                {
                    "complaints": format_int,
                    "median_resolution_days": lambda value: format_days(value, 2),
                    "p90_resolution_days": lambda value: format_days(value, 2),
                    "unresolved_share": lambda value: format_pct(value, 1),
                    "status_backlog_share": lambda value: format_pct(value, 1),
                }
            ),
            width="stretch",
        )

    with tabs[2]:
        agency_metric = st.selectbox(
            "Agency scorecard metric",
            options=["complaints", "median_resolution_days", "p90_resolution_days", "unresolved_share", "status_backlog_share"],
            format_func=lambda value: value.replace("_", " ").title(),
        )
        series_bar_chart(agency_metrics.sort_values(agency_metric, ascending=False).set_index("agency")[agency_metric].head(15), title="Agency ranking")
        st.dataframe(
            agency_metrics.sort_values(agency_metric, ascending=False).loc[:, ["agency", "agency_name", "complaints", "median_resolution_days", "p90_resolution_days", "unresolved_share", "status_backlog_share", "top_complaint_type", "top_modeled_subtype"]].style.format(
                {
                    "complaints": format_int,
                    "median_resolution_days": lambda value: format_days(value, 2),
                    "p90_resolution_days": lambda value: format_days(value, 2),
                    "unresolved_share": lambda value: format_pct(value, 1),
                    "status_backlog_share": lambda value: format_pct(value, 1),
                }
            ),
            width="stretch",
        )

    with tabs[3]:
        issue_level = st.selectbox(
            "Issue comparison level",
            options=sorted(agency_issue_metrics["issue_level"].dropna().unique().tolist()),
            key="ops_issue_level",
        )
        min_issue_rows = st.slider("Minimum rows for issue slice", min_value=500, max_value=int(agency_issue_metrics["complaints"].max()), value=2_000, step=500, key="ops_issue_rows")
        direction = st.radio("Show", options=["Slower than benchmark", "Faster than benchmark"], horizontal=True)
        filtered_gap = agency_issue_metrics.loc[
            agency_issue_metrics["issue_level"].eq(issue_level) & agency_issue_metrics["complaints"].ge(min_issue_rows)
        ].copy()
        ascending = direction == "Faster than benchmark"
        filtered_gap = filtered_gap.sort_values(["median_resolution_gap_days", "unresolved_share_gap"], ascending=[ascending, ascending]).head(25)
        series_bar_chart(filtered_gap.set_index("agency")["median_resolution_gap_days"], title="Median resolution gap by agency")
        st.dataframe(
            filtered_gap.loc[:, ["issue_value", "agency", "agency_name", "complaints", "issue_complaints", "issue_volume_share", "median_resolution_gap_days", "unresolved_share_gap", "status_backlog_share_gap"]].style.format(
                {
                    "complaints": format_int,
                    "issue_complaints": format_int,
                    "issue_volume_share": lambda value: format_pct(value, 1),
                    "median_resolution_gap_days": lambda value: format_days(value, 2),
                    "unresolved_share_gap": lambda value: format_pct(value, 1),
                    "status_backlog_share_gap": lambda value: format_pct(value, 1),
                }
            ).background_gradient(cmap="RdYlGn_r", subset=["median_resolution_gap_days", "unresolved_share_gap", "status_backlog_share_gap"]),
            width="stretch",
        )

    with tabs[4]:
        trend_metric = st.selectbox(
            "Trend metric",
            options=["complaints", "median_resolution_days", "p90_resolution_days", "unresolved_share", "status_backlog_share"],
            format_func=lambda value: value.replace("_", " ").title(),
        )
        pivot_line_chart(aligned_monthly, index="month_label", columns="created_year", values=trend_metric, title="Aligned 2025 vs 2026 YTD complete-month trends")
        qa_monthly = aligned_monthly.loc[:, ["month_label", "created_year", "closed_status_missing_date_count", "nonclosed_status_has_date_count", "negative_resolution_count"]].copy()
        qa_monthly["year_label"] = qa_monthly["created_year"].map({2025: "2025 aligned", 2026: "2026 YTD"})
        qa_pivot = qa_monthly.pivot(index="month_label", columns="year_label", values="closed_status_missing_date_count")
        st.markdown("**Closed-status-without-date QA counts**")
        st.line_chart(qa_pivot, width="stretch")

    with tabs[5]:
        board_metric = st.selectbox(
            "Community board ranking metric",
            options=["complaints", "median_resolution_days", "p90_resolution_days", "unresolved_share", "status_backlog_share"],
            format_func=lambda value: value.replace("_", " ").title(),
        )
        st.dataframe(
            community_board_operations.sort_values(board_metric, ascending=False).loc[:, ["community_board", "community_board_borough", "complaints", "median_resolution_days", "p90_resolution_days", "unresolved_share", "status_backlog_share", "top_agency", "top_complaint_type"]].style.format(
                {
                    "complaints": format_int,
                    "median_resolution_days": lambda value: format_days(value, 2),
                    "p90_resolution_days": lambda value: format_days(value, 2),
                    "unresolved_share": lambda value: format_pct(value, 1),
                    "status_backlog_share": lambda value: format_pct(value, 1),
                }
            ),
            width="stretch",
        )

    with tabs[6]:
        st.markdown(
            "This operations readout is intentionally structured rather than overwhelming: slow categories, agency comparisons, within-issue bottlenecks, aligned trends, and neighborhood performance are shown as separate views instead of one undifferentiated table."
        )


def render_fairness_page() -> None:
    fairness = build_fairness_summary()
    zcta_metrics = load_geo_table(str(ZCTA_METRICS_PATH))

    render_page_intro(
        "Neighborhood Equity",
        "Area-level disparity analysis across ZIPs and community boards, with demographic context and like-for-like matched comparisons.",
    )

    fairness_metrics = fairness["fairness_metrics"]
    fairness_stratified = fairness["fairness_stratified"]
    quintile_monthly = fairness["quintile_monthly"]
    fairness_model_results = fairness["fairness_model_results"]
    board_sensitivity = fairness["board_sensitivity"]

    top_burden = fairness_metrics.sort_values("complaints_per_10k", ascending=False).iloc[0]
    top_delay = fairness_metrics.loc[fairness_metrics["complaints"].ge(5_000)].sort_values("median_resolution_days", ascending=False).iloc[0]
    top_gap = fairness_stratified.loc[fairness_stratified["complaints"].ge(100)].sort_values("median_resolution_gap_days", ascending=False).iloc[0]

    findings_box(
        [
            f"The highest reported burden per resident appears in ZCTA {top_burden['zcta']} at {format_float(top_burden['complaints_per_10k'], 0)} complaints per 10k residents.",
            f"Among higher-volume ZCTAs, {top_delay['zcta']} shows the slowest median resolution time at {format_days(top_delay['median_resolution_days'], 2)}.",
            f"The largest adjusted matched-slice gap in the current artifact is {format_days(top_gap['median_resolution_gap_days'], 2)} for ZCTA {top_gap['zcta']} within {top_gap['issue_level']} = {top_gap['issue_value']}.",
            "These are signals of outcome disparity after observable adjustment, not proof of discriminatory intent.",
        ]
    )

    tabs = st.tabs(
        [
            "Guardrails",
            "Baseline Neighborhood Disparities",
            "Demographic Quintile Comparisons",
            "Adjusted Neighborhood Gaps",
            "Monthly Fairness Trends",
            "Lightweight Adjusted Models",
            "Community Board Sensitivity Check",
            "Findings",
        ]
    )

    with tabs[0]:
        st.info(
            "The strongest claims here are descriptive: some neighborhoods show higher reported burden, slower resolution, or larger backlog shares even after observable adjustment. The weakest claim would be causation, and the dashboard avoids making it."
        )

    with tabs[1]:
        fairness_map = zcta_metrics.loc[:, ["zcta", "geometry"]].merge(fairness_metrics, on="zcta", how="inner")
        fairness_metric_label = st.selectbox("Fairness map metric", options=list(FAIRNESS_MAP_METRICS.keys()))
        fairness_metric = FAIRNESS_MAP_METRICS[fairness_metric_label]
        plot_df = map_points_from_geometries(
            fairness_map,
            metric_column=fairness_metric,
            label_column="zcta",
            tooltip_columns=["top_borough", "complaints"],
            reverse_scale=False,
        )
        plot_df[fairness_metric] = format_metric_for_map(fairness_metric, plot_df[fairness_metric])
        render_point_map(plot_df, label_column="zcta", metric_column=fairness_metric, metric_label=fairness_metric_label)
        st.dataframe(
            fairness_metrics.sort_values(fairness_metric, ascending=False).head(20).loc[:, [
                "zcta",
                "complaints",
                "complaints_per_10k",
                "median_resolution_days",
                "unresolved_share",
                "status_backlog_share",
                "top_borough",
            ]].style.format(
                {
                    "complaints": format_int,
                    "complaints_per_10k": lambda value: format_float(value, 1),
                    "median_resolution_days": lambda value: format_days(value, 2),
                    "unresolved_share": lambda value: format_pct(value, 1),
                    "status_backlog_share": lambda value: format_pct(value, 1),
                }
            ),
            width="stretch",
        )

    with tabs[2]:
        quintile_label = st.selectbox("Demographic quintile view", options=list(FAIRNESS_QUINTILE_OPTIONS.keys()))
        quintile_column = FAIRNESS_QUINTILE_OPTIONS[quintile_label]
        quintile_summary = (
            fairness_metrics.loc[fairness_metrics[quintile_column].notna()]
            .groupby(quintile_column, observed=True)
            .agg(
                zctas=("zcta", "nunique"),
                complaints=("complaints", "sum"),
                median_resolution_days=("median_resolution_days", "median"),
                unresolved_share=("unresolved_share", "median"),
                status_backlog_share=("status_backlog_share", "median"),
            )
            .reset_index()
        )
        quintile_metric = st.selectbox(
            "Quintile comparison metric",
            options=["median_resolution_days", "unresolved_share", "status_backlog_share", "complaints"],
            format_func=lambda value: value.replace("_", " ").title(),
        )
        series_bar_chart(quintile_summary.set_index(quintile_column)[quintile_metric], title="Quintile comparison")
        st.dataframe(
            quintile_summary.style.format(
                {
                    "zctas": format_int,
                    "complaints": format_int,
                    "median_resolution_days": lambda value: format_days(value, 2),
                    "unresolved_share": lambda value: format_pct(value, 1),
                    "status_backlog_share": lambda value: format_pct(value, 1),
                }
            ),
            width="stretch",
        )

    with tabs[3]:
        issue_level = st.selectbox(
            "Adjusted gap issue level",
            options=sorted(fairness_stratified["issue_level"].dropna().unique().tolist()),
            key="fairness_issue_level",
        )
        min_rows = st.slider("Minimum matched complaints", min_value=25, max_value=int(fairness_stratified["complaints"].quantile(0.95)), value=100, step=25)
        gap_view = st.radio("Gap ranking", options=["Slowest matched slices", "Fastest matched slices"], horizontal=True)
        adjusted_gap = fairness_stratified.loc[
            fairness_stratified["issue_level"].eq(issue_level) & fairness_stratified["complaints"].ge(min_rows)
        ].copy()
        ascending = gap_view == "Fastest matched slices"
        adjusted_gap = adjusted_gap.sort_values(["median_resolution_gap_days", "unresolved_share_gap"], ascending=[ascending, ascending]).head(30)
        series_bar_chart(adjusted_gap.set_index("zcta")["median_resolution_gap_days"], title="Adjusted median-resolution gap by ZCTA slice")
        st.dataframe(
            adjusted_gap.loc[:, ["zcta", "issue_value", "agency", "complaints", "benchmark_complaints", "median_resolution_gap_days", "unresolved_share_gap", "status_backlog_share_gap", "income_quintile", "poverty_quintile"]].style.format(
                {
                    "complaints": format_int,
                    "benchmark_complaints": format_int,
                    "median_resolution_gap_days": lambda value: format_days(value, 2),
                    "unresolved_share_gap": lambda value: format_pct(value, 1),
                    "status_backlog_share_gap": lambda value: format_pct(value, 1),
                }
            ).background_gradient(cmap="RdYlGn_r", subset=["median_resolution_gap_days", "unresolved_share_gap", "status_backlog_share_gap"]),
            width="stretch",
        )

    with tabs[4]:
        quintile_label = st.selectbox("Trend quintile family", options=list(FAIRNESS_QUINTILE_OPTIONS.keys()), key="fairness_trend_quintile")
        trend_metric = st.selectbox(
            "Trend metric",
            options=["median_resolution_days", "unresolved_share", "status_backlog_share", "complaints"],
            format_func=lambda value: value.replace("_", " ").title(),
            key="fairness_trend_metric",
        )
        trend_df = quintile_monthly.loc[quintile_monthly["quintile_label"].eq(quintile_label)]
        pivot_line_chart(trend_df, index="month_label", columns="quintile_value", values=trend_metric, title="Monthly disparity trends by quintile")

    with tabs[5]:
        model_name = st.selectbox("Adjusted model", options=sorted(fairness_model_results["model_name"].unique().tolist()))
        model_df = fairness_model_results.loc[fairness_model_results["model_name"].eq(model_name)].copy()
        model_score = model_df["metric_value"].iloc[0] if not model_df.empty else np.nan
        st.metric("Model fit metric", f"{model_df['metric'].iloc[0] if not model_df.empty else 'metric'} {format_float(model_score, 3)}")
        top_coefficients = model_df.loc[model_df["feature"].str.contains("quintile", na=False)].copy()
        top_coefficients["abs_coefficient"] = top_coefficients["coefficient"].abs()
        top_coefficients = top_coefficients.sort_values("abs_coefficient", ascending=False).head(20)
        series_bar_chart(top_coefficients.set_index("feature")["coefficient"], title="Quintile-related coefficients")
        st.dataframe(top_coefficients.loc[:, ["feature", "coefficient", "metric", "metric_value", "sample_rows"]].style.format({"coefficient": lambda value: format_float(value, 4), "metric_value": lambda value: format_float(value, 3), "sample_rows": format_int}), width="stretch")

    with tabs[6]:
        st.dataframe(
            board_sensitivity.sort_values("median_resolution_days", ascending=False).loc[:, ["community_board", "community_board_borough", "complaints", "median_resolution_days", "p90_resolution_days", "unresolved_share", "status_backlog_share", "top_agency", "top_issue_family"]].style.format(
                {
                    "complaints": format_int,
                    "median_resolution_days": lambda value: format_days(value, 2),
                    "p90_resolution_days": lambda value: format_days(value, 2),
                    "unresolved_share": lambda value: format_pct(value, 1),
                    "status_backlog_share": lambda value: format_pct(value, 1),
                }
            ),
            width="stretch",
        )

    with tabs[7]:
        st.markdown(
            "The equity analysis is not a single fairness verdict. It is a careful read on where disparity signals appear, where they weaken after adjustment, and where caution still matters."
        )


def render_predictive_page() -> None:
    predictive = build_predictive_summary()

    render_page_intro(
        "Forecasting",
        "Resolution-time forecasting benchmarks, class-level performance, interpretation layers, and scored miss analysis for 2026 YTD complaints.",
    )

    overall_metrics = predictive["overall_metrics"]
    class_metrics = predictive["class_metrics"]
    confusion = predictive["confusion"]
    error_slices = predictive["error_slices"]
    feature_importance = predictive["feature_importance"]
    high_confidence_misses = predictive["high_confidence_misses"]

    post_routing_best = overall_metrics.loc[
        overall_metrics["metric"].eq("macro_f1") & overall_metrics["feature_set"].astype("string").eq("post_routing")
    ].sort_values("metric_value", ascending=False).iloc[0]
    intake_best = overall_metrics.loc[
        overall_metrics["metric"].eq("macro_f1") & overall_metrics["feature_set"].astype("string").eq("intake_only")
    ].sort_values("metric_value", ascending=False).iloc[0]

    findings_box(
        [
            f"Best post-routing macro F1: {pretty_model_name(post_routing_best['model_name'])} at {format_float(post_routing_best['metric_value'], 3)}.",
            f"Best intake-only macro F1: {pretty_model_name(intake_best['model_name'])} at {format_float(intake_best['metric_value'], 3)}.",
            f"The intake-only benchmark stays close to the post-routing benchmark, which means the incoming complaint record already carries substantial duration signal.",
            f"There are {format_int(len(high_confidence_misses.index))} high-confidence misses at probability >= 0.80 in the scored 2026 YTD prediction artifact.",
        ]
    )

    tabs = st.tabs(
        [
            "Guardrails",
            "Model Comparison",
            "Confusion Matrix",
            "Class Metrics",
            "Logistic Interpretation Layer",
            "Error Slices",
            "High-Confidence Misses",
            "Takeaways",
        ]
    )

    with tabs[0]:
        st.info(
            "This model is designed for operational forecasting, not causal explanation. The test window is 2026 YTD, and the target reflects observed resolution outcomes rather than intrinsic issue severity."
        )

    with tabs[1]:
        selected_metric = st.selectbox(
            "Model ranking metric",
            options=list(MODEL_METRIC_LABELS.keys()),
            format_func=lambda value: MODEL_METRIC_LABELS[value],
        )
        score_pivot = overall_metrics.loc[overall_metrics["metric"].eq(selected_metric)].pivot(index="model_name", columns="feature_set", values="metric_value")
        score_pivot = score_pivot.rename(index=pretty_model_name, columns=pretty_feature_set)
        st.dataframe(score_pivot.style.format(lambda value: format_float(value, 3)).background_gradient(cmap="Blues"), width="stretch")

        cards = st.columns(2)
        cards[0].metric("Best post-routing", pretty_model_name(post_routing_best["model_name"]), f"Macro F1 {format_float(post_routing_best['metric_value'], 3)}")
        cards[1].metric("Best intake-only", pretty_model_name(intake_best["model_name"]), f"Macro F1 {format_float(intake_best['metric_value'], 3)}")

    with tabs[2]:
        confusion_feature_set = st.selectbox(
            "Confusion matrix feature set",
            options=sorted(confusion["feature_set"].dropna().astype(str).unique().tolist()),
            format_func=pretty_feature_set,
        )
        confusion_model = st.selectbox(
            "Confusion matrix model",
            options=sorted(confusion["model_name"].dropna().astype(str).unique().tolist()),
            format_func=pretty_model_name,
        )
        confusion_view = confusion.loc[
            confusion["feature_set"].astype(str).eq(confusion_feature_set)
            & confusion["model_name"].astype(str).eq(confusion_model)
        ].copy()
        confusion_pivot = confusion_view.pivot(index="actual_resolution_bucket", columns="predicted_resolution_bucket", values="actual_bucket_share")
        st.dataframe(confusion_pivot.style.format(lambda value: format_pct(value, 1)).background_gradient(cmap="Blues"), width="stretch")

    with tabs[3]:
        class_metric = st.selectbox(
            "Class-level metric",
            options=["precision", "recall", "f1", "support"],
            format_func=lambda value: value.upper() if value != "support" else "Support",
        )
        class_feature_set = st.selectbox(
            "Class-metric feature set",
            options=sorted(class_metrics["feature_set"].dropna().astype(str).unique().tolist()),
            format_func=pretty_feature_set,
        )
        class_model = st.selectbox(
            "Class-metric model",
            options=sorted(class_metrics["model_name"].dropna().astype(str).unique().tolist()),
            format_func=pretty_model_name,
        )
        class_view = class_metrics.loc[
            class_metrics["feature_set"].astype(str).eq(class_feature_set)
            & class_metrics["model_name"].astype(str).eq(class_model)
            & class_metrics["metric"].eq(class_metric)
        ].copy()
        class_view["target_class"] = pd.Categorical(class_view["target_class"].astype("string"), categories=RESOLUTION_BUCKET_ORDER[:-1], ordered=True)
        class_view = class_view.sort_values("target_class")
        series_bar_chart(class_view.set_index("target_class")["metric_value"], title="Per-class metric view")
        st.dataframe(class_view.loc[:, ["target_class", "metric_value"]].style.format({"metric_value": format_int if class_metric == "support" else lambda value: format_float(value, 3)}), width="stretch")

    with tabs[4]:
        importance_feature_set = st.selectbox(
            "Feature importance set",
            options=sorted(feature_importance["feature_set"].dropna().astype(str).unique().tolist()),
            format_func=pretty_feature_set,
        )
        top_n = st.slider("Top features", min_value=5, max_value=30, value=15)
        importance_view = feature_importance.loc[
            feature_importance["feature_set"].astype(str).eq(importance_feature_set)
            & feature_importance["model_name"].astype(str).eq("multinomial_logistic")
        ].sort_values("mean_abs_coefficient", ascending=False).head(top_n)
        series_bar_chart(importance_view.set_index("feature")["mean_abs_coefficient"], title="Top logistic features by mean absolute coefficient")
        st.dataframe(
            importance_view.loc[:, ["feature_group", "feature", "mean_abs_coefficient", "max_abs_coefficient", "strongest_class"]].style.format(
                {
                    "mean_abs_coefficient": lambda value: format_float(value, 3),
                    "max_abs_coefficient": lambda value: format_float(value, 3),
                }
            ),
            width="stretch",
        )

    with tabs[5]:
        slice_column = st.selectbox("Error slice segment", options=sorted(error_slices["segment_column"].dropna().unique().tolist()))
        slice_feature_set = st.selectbox(
            "Error slice feature set",
            options=sorted(error_slices["feature_set"].dropna().astype(str).unique().tolist()),
            format_func=pretty_feature_set,
        )
        slice_view = error_slices.loc[
            error_slices["segment_column"].eq(slice_column)
            & error_slices["feature_set"].astype(str).eq(slice_feature_set)
        ].sort_values(["accuracy", "complaints"], ascending=[True, False])
        series_bar_chart(slice_view.set_index("segment_value")["accuracy"], title="Accuracy by error slice")
        st.dataframe(
            slice_view.loc[:, ["segment_value", "complaints", "accuracy", "median_confidence", "top_actual_bucket", "top_predicted_bucket"]].style.format(
                {
                    "complaints": format_int,
                    "accuracy": lambda value: format_pct(value, 1),
                    "median_confidence": lambda value: format_pct(value, 1),
                }
            ),
            width="stretch",
        )

    with tabs[6]:
        miss_feature_set = st.selectbox(
            "Miss table feature set",
            options=sorted(high_confidence_misses["feature_set"].dropna().astype(str).unique().tolist()),
            format_func=pretty_feature_set,
        )
        confidence_threshold = st.slider("Minimum predicted probability", min_value=0.80, max_value=0.99, value=0.90, step=0.01)
        misses = high_confidence_misses.loc[
            high_confidence_misses["feature_set"].astype(str).eq(miss_feature_set)
            & high_confidence_misses["predicted_probability"].ge(confidence_threshold)
        ].head(100)
        st.dataframe(
            misses.loc[:, ["created_date", "complaint_type", "descriptor", "agency", "borough", "actual_bucket", "predicted_bucket", "predicted_probability"]].style.format(
                {"created_date": lambda value: value.strftime("%Y-%m-%d %H:%M") if pd.notna(value) else "NA", "predicted_probability": lambda value: format_pct(value, 1)}
            ),
            width="stretch",
        )

    with tabs[7]:
        st.markdown(
            "The forecasting layer supports two practical use cases: `post_routing` includes `agency` and performs best as an operational forecast after routing is known, while `intake_only` removes `agency` and still stays close enough to be useful earlier in the complaint lifecycle."
        )


def main() -> None:
    if STREAMLIT_IMPORT_ERROR is not None:
        raise SystemExit(
            "Streamlit is not installed. Install it with `pip install streamlit` and run `streamlit run 311dashboard.py`."
        )

    st.set_page_config(page_title="311Dashboard", layout="wide")
    st.sidebar.title("311Dashboard")
    st.sidebar.caption("NYC 311 demand, service performance, neighborhood burden, equity signals, and forecasting in one interactive report.")
    page = st.sidebar.radio("Page", options=PAGE_OPTIONS)
    st.sidebar.markdown("**Scope**")
    st.sidebar.write("Primary analysis window: `2025-2026`")
    st.sidebar.write("`2026` is partial-year and labeled as YTD where applicable.")
    st.sidebar.markdown("**Pages**")
    st.sidebar.write("Overview, city patterns, text insights, geography, operations, neighborhood equity, and forecasting.")

    if page == "Overview":
        render_overview_page()
    elif page == "City Patterns":
        render_eda_page()
    elif page == "Text Insights":
        render_nlp_page()
    elif page == "Geography":
        render_geospatial_page()
    elif page == "Operations":
        render_operations_page()
    elif page == "Neighborhood Equity":
        render_fairness_page()
    else:
        render_predictive_page()


if __name__ == "__main__":
    main()
