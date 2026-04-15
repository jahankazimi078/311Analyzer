from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import geopandas as gpd
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYTIC_PATH = PROJECT_ROOT / "data" / "analytics" / "requests_2025_2026_analytic.parquet"
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
RAW_DIR = REFERENCE_DIR / "raw"
EXTRACT_DIR = RAW_DIR / "zcta_2024"
ZCTA_REFERENCE_PATH = REFERENCE_DIR / "nyc_zcta_reference.parquet"
ZCTA_GEOJSON_PATH = REFERENCE_DIR / "nyc_zcta_reference.geojson"
ACS_POPULATION_PATH = REFERENCE_DIR / "nyc_zcta_population.parquet"

ZCTA_BOUNDARY_URL = "https://www2.census.gov/geo/tiger/TIGER2024/ZCTA520/tl_2024_us_zcta520.zip"
ACS_POPULATION_URL = "https://api.census.gov/data/2023/acs/acs5?get=NAME,B01003_001E&for=zip%20code%20tabulation%20area:*"
NYC_BOROUGHS = {"BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND"}


def clean_zip_series(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.extract(r"(\d{5})", expand=False)
        .astype("string")
    )


def download_binary(url: str) -> bytes:
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    return response.content


def ensure_zcta_shapefile() -> Path:
    shapefile_path = EXTRACT_DIR / "tl_2024_us_zcta520.shp"
    if shapefile_path.exists():
        return shapefile_path

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    boundary_bytes = download_binary(ZCTA_BOUNDARY_URL)
    with ZipFile(BytesIO(boundary_bytes)) as archive:
        archive.extractall(EXTRACT_DIR)
    return shapefile_path


def load_nyc_request_zips() -> set[str]:
    if not ANALYTIC_PATH.exists():
        raise FileNotFoundError(f"Missing analytic parquet: {ANALYTIC_PATH}")

    df = pd.read_parquet(ANALYTIC_PATH, columns=["incident_zip", "borough"])
    request_zips = clean_zip_series(
        df.loc[df["borough"].astype("string").str.upper().isin(NYC_BOROUGHS), "incident_zip"]
    )
    return set(request_zips.dropna().unique().tolist())


def load_population_table() -> pd.DataFrame:
    payload = requests.get(ACS_POPULATION_URL, timeout=180)
    payload.raise_for_status()
    rows = payload.json()

    population = pd.DataFrame(rows[1:], columns=rows[0]).rename(
        columns={
            "zip code tabulation area": "zcta",
            "B01003_001E": "population",
            "NAME": "zcta_name",
        }
    )
    population["zcta"] = population["zcta"].astype("string").str.zfill(5)
    population["population"] = pd.to_numeric(population["population"], errors="coerce")
    return population


def detect_zcta_column(columns: list[str]) -> str:
    for candidate in ("ZCTA5CE20", "GEOID20", "ZCTA5CE10", "GEOID10"):
        if candidate in columns:
            return candidate
    raise KeyError(f"Could not locate a ZCTA column in: {columns}")


def build_reference_data() -> None:
    shapefile_path = ensure_zcta_shapefile()
    request_zips = load_nyc_request_zips()
    population = load_population_table()

    zcta_gdf = gpd.read_file(shapefile_path)
    zcta_column = detect_zcta_column(list(zcta_gdf.columns))
    zcta_gdf = zcta_gdf.rename(columns={zcta_column: "zcta"})
    zcta_gdf["zcta"] = zcta_gdf["zcta"].astype("string").str.zfill(5)
    zcta_gdf = zcta_gdf.loc[zcta_gdf["zcta"].isin(request_zips), ["zcta", "geometry"]].copy()
    zcta_gdf = zcta_gdf.to_crs(4326)

    zcta_reference = zcta_gdf.merge(population, on="zcta", how="left", validate="one_to_one")
    zcta_reference["population_missing_flag"] = zcta_reference["population"].isna()

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    zcta_reference.to_parquet(ZCTA_REFERENCE_PATH, index=False)
    zcta_reference.to_file(ZCTA_GEOJSON_PATH, driver="GeoJSON")
    zcta_reference.drop(columns="geometry").to_parquet(ACS_POPULATION_PATH, index=False)

    print(f"Wrote ZCTA reference parquet: {ZCTA_REFERENCE_PATH}")
    print(f"Wrote ZCTA reference geojson: {ZCTA_GEOJSON_PATH}")
    print(f"Wrote ZCTA population parquet: {ACS_POPULATION_PATH}")
    print(f"NYC request ZIPs matched to ZCTAs: {len(zcta_reference):,}")


if __name__ == "__main__":
    build_reference_data()
