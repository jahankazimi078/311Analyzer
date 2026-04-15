from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box


PROJECT_ROOT = Path(__file__).resolve().parent
ANALYTIC_PATH = PROJECT_ROOT / "data" / "analytics" / "requests_2025_2026_analytic.parquet"
NLP_PATH = PROJECT_ROOT / "data" / "analytics" / "requests_2025_2026_issue_subtypes.parquet"
ZCTA_REFERENCE_PATH = PROJECT_ROOT / "data" / "reference" / "nyc_zcta_reference.parquet"

OUTPUT_DIR = PROJECT_ROOT / "data" / "analytics"
ZCTA_METRICS_PATH = OUTPUT_DIR / "requests_2025_2026_zcta_metrics.parquet"
COMMUNITY_BOARD_METRICS_PATH = OUTPUT_DIR / "requests_2025_2026_community_board_metrics.parquet"
COMMUNITY_BOARD_MONTHLY_PATH = OUTPUT_DIR / "requests_2025_2026_community_board_monthly.parquet"
GRID_MONTHLY_PATH = OUTPUT_DIR / "requests_2025_2026_grid_monthly.parquet"
GRID_PERSISTENCE_PATH = OUTPUT_DIR / "requests_2025_2026_grid_persistence.parquet"
GRID_PERSISTENCE_GEOJSON_PATH = OUTPUT_DIR / "requests_2025_2026_grid_persistence.geojson"

NYC_LAT_RANGE = (40.49, 40.93)
NYC_LON_RANGE = (-74.28, -73.68)
GRID_SIZE_DEGREES = 0.01


def clean_zip_series(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.extract(r"(\d{5})", expand=False)
        .astype("string")
    )


def percentile_90(series: pd.Series) -> float:
    valid = series.dropna()
    if valid.empty:
        return np.nan
    return float(valid.quantile(0.9))


def first_mode(series: pd.Series) -> str:
    mode = series.dropna().mode()
    if mode.empty:
        return "Unknown"
    return str(mode.iloc[0])


def add_community_board_fields(df: pd.DataFrame) -> pd.DataFrame:
    community_board = df["community_board"].astype("string").str.strip()
    df["community_board"] = community_board
    df["community_board_code"] = community_board.str.extract(r"^(\d{2})", expand=False).astype("string")
    df["community_board_borough"] = community_board.str.extract(r"([A-Z][A-Z ]+)$", expand=False).astype("string")
    df["community_board_borough"] = df["community_board_borough"].fillna(df["borough"])
    df["community_board_label"] = df["community_board"]
    return df


def load_geo_analysis_frame() -> pd.DataFrame:
    if not ANALYTIC_PATH.exists():
        raise FileNotFoundError(f"Missing analytic parquet: {ANALYTIC_PATH}")
    if not NLP_PATH.exists():
        raise FileNotFoundError(f"Missing NLP parquet: {NLP_PATH}")

    analytic_columns = [
        "unique_key",
        "created_date",
        "created_year",
        "complaint_type",
        "borough",
        "incident_zip",
        "community_board",
        "council_district",
        "latitude",
        "longitude",
        "status",
        "resolution_days",
        "resolved_with_valid_date",
    ]
    nlp_columns = [
        "unique_key",
        "issue_family",
        "subtype_modeled_flag",
        "issue_subtype",
        "potential_label_mismatch_flag",
        "resolution_outcome_group",
    ]

    analytic_df = pd.read_parquet(ANALYTIC_PATH, columns=analytic_columns)
    nlp_df = pd.read_parquet(NLP_PATH, columns=nlp_columns)
    df = analytic_df.merge(nlp_df, on="unique_key", how="left", validate="one_to_one")

    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    df["created_month_start"] = df["created_date"].dt.to_period("M").dt.to_timestamp()
    df["borough"] = df["borough"].astype("string").str.upper()
    df["incident_zip_clean"] = clean_zip_series(df["incident_zip"])
    df["status"] = df["status"].astype("string").str.title()
    df["resolution_days"] = pd.to_numeric(df["resolution_days"], errors="coerce")
    df["resolved_with_valid_date"] = df["resolved_with_valid_date"].fillna(False)
    df["unresolved_flag"] = ~df["resolved_with_valid_date"]

    df["valid_coordinate_flag"] = (
        df["latitude"].notna()
        & df["longitude"].notna()
        & df["latitude"].between(*NYC_LAT_RANGE)
        & df["longitude"].between(*NYC_LON_RANGE)
    )
    df["issue_subtype"] = df["issue_subtype"].fillna("not_modeled")
    df["issue_family"] = df["issue_family"].fillna(df["complaint_type"])
    df["subtype_modeled_flag"] = df["subtype_modeled_flag"].fillna(False)
    df["potential_label_mismatch_flag"] = df["potential_label_mismatch_flag"].fillna(False)
    return add_community_board_fields(df)


def load_zcta_reference() -> gpd.GeoDataFrame:
    if not ZCTA_REFERENCE_PATH.exists():
        raise FileNotFoundError(
            "Missing ZCTA reference parquet. Run `./.venv/bin/python data/build_reference_geo.py` first."
        )
    return gpd.read_parquet(ZCTA_REFERENCE_PATH)


def build_zcta_metrics(df: pd.DataFrame, zcta_reference: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    resolution_subset = df.loc[df["resolved_with_valid_date"], ["incident_zip_clean", "resolution_days"]].copy()
    resolution_metrics = (
        resolution_subset.groupby("incident_zip_clean")["resolution_days"]
        .agg(median_resolution_days="median", p90_resolution_days=percentile_90)
        .reset_index()
    )

    zcta_metrics = (
        df.groupby("incident_zip_clean", dropna=False)
        .agg(
            complaints=("unique_key", "size"),
            unresolved_complaints=("unresolved_flag", "sum"),
            geocoded_complaints=("valid_coordinate_flag", "sum"),
        )
        .reset_index()
        .rename(columns={"incident_zip_clean": "zcta"})
    )
    resolution_metrics = resolution_metrics.rename(columns={"incident_zip_clean": "zcta"})
    zcta_metrics = zcta_metrics.merge(resolution_metrics, on="zcta", how="left")

    zcta_metrics["unresolved_share"] = zcta_metrics["unresolved_complaints"] / zcta_metrics["complaints"]
    zcta_metrics["geocoded_share"] = zcta_metrics["geocoded_complaints"] / zcta_metrics["complaints"]

    merged = zcta_reference.merge(zcta_metrics, on="zcta", how="left", validate="one_to_one")
    for column in ["complaints", "unresolved_complaints", "geocoded_complaints"]:
        merged[column] = merged[column].fillna(0).astype("int64")

    merged["complaints_per_10k"] = np.where(
        merged["population"].gt(0),
        merged["complaints"] / merged["population"] * 10_000,
        np.nan,
    )
    merged["unresolved_per_10k"] = np.where(
        merged["population"].gt(0),
        merged["unresolved_complaints"] / merged["population"] * 10_000,
        np.nan,
    )
    return merged


def build_community_board_metrics(df: pd.DataFrame) -> pd.DataFrame:
    resolution_subset = df.loc[
        df["resolved_with_valid_date"],
        ["community_board", "resolution_days"],
    ].copy()
    resolution_metrics = (
        resolution_subset.groupby("community_board", observed=True)["resolution_days"]
        .agg(median_resolution_days="median", p90_resolution_days=percentile_90)
        .reset_index()
    )

    board_metrics = (
        df.groupby(
            ["community_board", "community_board_borough", "community_board_code"],
            observed=True,
            dropna=False,
        )
        .agg(
            complaints=("unique_key", "size"),
            unresolved_complaints=("unresolved_flag", "sum"),
            geocoded_complaints=("valid_coordinate_flag", "sum"),
            modeled_subtype_complaints=("subtype_modeled_flag", "sum"),
            mismatch_flagged_complaints=("potential_label_mismatch_flag", "sum"),
            top_complaint_type=("complaint_type", first_mode),
            top_issue_subtype=("issue_subtype", first_mode),
        )
        .reset_index()
    )
    board_metrics = board_metrics.merge(resolution_metrics, on="community_board", how="left")
    board_metrics["unresolved_share"] = board_metrics["unresolved_complaints"] / board_metrics["complaints"]
    board_metrics["geocoded_share"] = board_metrics["geocoded_complaints"] / board_metrics["complaints"]
    board_metrics["modeled_subtype_share"] = (
        board_metrics["modeled_subtype_complaints"] / board_metrics["complaints"]
    )
    board_metrics["mismatch_flag_share"] = (
        board_metrics["mismatch_flagged_complaints"] / board_metrics["complaints"]
    )
    board_metrics["borough_complaint_rank"] = board_metrics.groupby("community_board_borough")["complaints"].rank(
        method="dense",
        ascending=False,
    )
    return board_metrics.sort_values(["community_board_borough", "complaints"], ascending=[True, False]).reset_index(drop=True)


def build_community_board_monthly(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.groupby(
            ["created_month_start", "created_year", "community_board", "community_board_borough"],
            observed=True,
            dropna=False,
        )
        .agg(
            complaints=("unique_key", "size"),
            unresolved_complaints=("unresolved_flag", "sum"),
            mismatch_flagged_complaints=("potential_label_mismatch_flag", "sum"),
        )
        .reset_index()
    )
    monthly["unresolved_share"] = monthly["unresolved_complaints"] / monthly["complaints"]
    monthly["mismatch_flag_share"] = monthly["mismatch_flagged_complaints"] / monthly["complaints"]
    return monthly


def assign_grid_bins(df: pd.DataFrame, grid_size_degrees: float = GRID_SIZE_DEGREES) -> pd.DataFrame:
    geo_df = df.loc[df["valid_coordinate_flag"]].copy()
    lon_min = NYC_LON_RANGE[0]
    lat_min = NYC_LAT_RANGE[0]

    geo_df["grid_lon_bin"] = np.floor((geo_df["longitude"] - lon_min) / grid_size_degrees).astype("int16")
    geo_df["grid_lat_bin"] = np.floor((geo_df["latitude"] - lat_min) / grid_size_degrees).astype("int16")
    geo_df["grid_id"] = geo_df["grid_lon_bin"].astype("string") + ":" + geo_df["grid_lat_bin"].astype("string")
    geo_df["grid_lon_min"] = lon_min + geo_df["grid_lon_bin"] * grid_size_degrees
    geo_df["grid_lat_min"] = lat_min + geo_df["grid_lat_bin"] * grid_size_degrees
    geo_df["grid_lon_max"] = geo_df["grid_lon_min"] + grid_size_degrees
    geo_df["grid_lat_max"] = geo_df["grid_lat_min"] + grid_size_degrees
    return geo_df


def build_monthly_grid_metrics(df: pd.DataFrame, grid_size_degrees: float = GRID_SIZE_DEGREES) -> pd.DataFrame:
    geo_df = assign_grid_bins(df, grid_size_degrees=grid_size_degrees)
    monthly_grid = (
        geo_df.groupby(
            [
                "created_month_start",
                "grid_id",
                "grid_lon_bin",
                "grid_lat_bin",
                "grid_lon_min",
                "grid_lon_max",
                "grid_lat_min",
                "grid_lat_max",
            ],
            observed=True,
        )
        .agg(
            complaints=("unique_key", "size"),
            unresolved_complaints=("unresolved_flag", "sum"),
        )
        .reset_index()
    )
    monthly_grid["unresolved_share"] = monthly_grid["unresolved_complaints"] / monthly_grid["complaints"]
    return monthly_grid


def build_grid_persistence(
    monthly_grid: pd.DataFrame,
    hotspot_quantile: float = 0.9,
) -> gpd.GeoDataFrame:
    monthly_grid = monthly_grid.copy()
    monthly_grid["hotspot_threshold"] = monthly_grid.groupby("created_month_start")["complaints"].transform(
        lambda series: series.quantile(hotspot_quantile)
    )
    monthly_grid["is_hotspot"] = monthly_grid["complaints"].ge(monthly_grid["hotspot_threshold"])

    persistence = (
        monthly_grid.groupby(
            [
                "grid_id",
                "grid_lon_bin",
                "grid_lat_bin",
                "grid_lon_min",
                "grid_lon_max",
                "grid_lat_min",
                "grid_lat_max",
            ],
            observed=True,
        )
        .agg(
            total_complaints=("complaints", "sum"),
            total_unresolved_complaints=("unresolved_complaints", "sum"),
            hotspot_months=("is_hotspot", "sum"),
            active_months=("created_month_start", "nunique"),
        )
        .reset_index()
    )
    persistence["hotspot_month_share"] = persistence["hotspot_months"] / persistence["active_months"]
    persistence["unresolved_share"] = (
        persistence["total_unresolved_complaints"] / persistence["total_complaints"]
    )

    geometry = [
        box(row.grid_lon_min, row.grid_lat_min, row.grid_lon_max, row.grid_lat_max)
        for row in persistence.itertuples(index=False)
    ]
    return gpd.GeoDataFrame(persistence, geometry=geometry, crs="EPSG:4326")


def top_complaint_types(df: pd.DataFrame, top_n: int = 4) -> list[str]:
    return df["complaint_type"].value_counts().head(top_n).index.tolist()


def top_modeled_subtypes(df: pd.DataFrame, top_n: int = 4) -> list[str]:
    modeled = df.loc[df["subtype_modeled_flag"] & df["issue_subtype"].ne("not_modeled"), "issue_subtype"]
    return modeled.value_counts().head(top_n).index.tolist()


def summarize_hotspot_grids(
    df: pd.DataFrame,
    category_column: str,
    top_n_categories: int = 6,
    top_n_grids: int = 3,
) -> pd.DataFrame:
    geo_df = assign_grid_bins(df)
    top_categories = geo_df[category_column].value_counts().head(top_n_categories).index.tolist()
    summary = (
        geo_df.loc[geo_df[category_column].isin(top_categories)]
        .groupby([category_column, "grid_id"], observed=True)
        .agg(
            complaints=("unique_key", "size"),
            boroughs=("borough", lambda series: ", ".join(sorted(set(series.dropna().astype(str)))[:3])),
        )
        .reset_index()
        .sort_values([category_column, "complaints"], ascending=[True, False])
    )
    return summary.groupby(category_column, observed=True).head(top_n_grids).reset_index(drop=True)


def sample_geocoded_rows(df: pd.DataFrame, sample_size: int = 75_000, query: str | None = None) -> pd.DataFrame:
    sample_df = df.loc[df["valid_coordinate_flag"], ["longitude", "latitude", "complaint_type", "issue_subtype"]].copy()
    if query:
        sample_df = sample_df.query(query)
    if len(sample_df.index) <= sample_size:
        return sample_df
    return sample_df.sample(sample_size, random_state=42)


def export_geo_outputs(
    zcta_metrics: gpd.GeoDataFrame,
    community_board_metrics: pd.DataFrame,
    community_board_monthly: pd.DataFrame,
    monthly_grid: pd.DataFrame,
    grid_persistence: gpd.GeoDataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    zcta_metrics.to_parquet(ZCTA_METRICS_PATH, index=False)
    community_board_metrics.to_parquet(COMMUNITY_BOARD_METRICS_PATH, index=False)
    community_board_monthly.to_parquet(COMMUNITY_BOARD_MONTHLY_PATH, index=False)
    monthly_grid.to_parquet(GRID_MONTHLY_PATH, index=False)
    grid_persistence.to_parquet(GRID_PERSISTENCE_PATH, index=False)
    grid_persistence.to_file(GRID_PERSISTENCE_GEOJSON_PATH, driver="GeoJSON")


def build_geo_outputs() -> tuple[pd.DataFrame, gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame]:
    df = load_geo_analysis_frame()
    zcta_reference = load_zcta_reference()
    zcta_metrics = build_zcta_metrics(df, zcta_reference)
    community_board_metrics = build_community_board_metrics(df)
    community_board_monthly = build_community_board_monthly(df)
    monthly_grid = build_monthly_grid_metrics(df)
    grid_persistence = build_grid_persistence(monthly_grid)
    export_geo_outputs(zcta_metrics, community_board_metrics, community_board_monthly, monthly_grid, grid_persistence)
    return df, zcta_metrics, community_board_metrics, community_board_monthly, monthly_grid, grid_persistence


if __name__ == "__main__":
    build_geo_outputs()
