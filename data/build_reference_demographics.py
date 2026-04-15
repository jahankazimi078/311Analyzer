from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
ZCTA_REFERENCE_PATH = REFERENCE_DIR / "nyc_zcta_reference.parquet"
ZCTA_DEMOGRAPHICS_PATH = REFERENCE_DIR / "nyc_zcta_demographics.parquet"

ACS_YEAR = 2023
ACS_PROFILE_URL = f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5"

ACS_VARIABLES = {
    "median_household_income": "B19013_001E",
    "poverty_universe": "B17001_001E",
    "poverty_count": "B17001_002E",
    "occupied_housing_units": "B25003_001E",
    "renter_occupied_units": "B25003_003E",
    "race_ethnicity_total": "B03002_001E",
    "non_hispanic_white_alone": "B03002_003E",
    "education_total_25_plus": "B15003_001E",
    "bachelors_degree": "B15003_022E",
    "masters_degree": "B15003_023E",
    "professional_degree": "B15003_024E",
    "doctorate_degree": "B15003_025E",
    "median_gross_rent": "B25064_001E",
}


def fetch_acs_table() -> pd.DataFrame:
    variable_list = ["NAME", *ACS_VARIABLES.values()]
    params = {
        "get": ",".join(variable_list),
        "for": "zip code tabulation area:*",
    }
    response = requests.get(ACS_PROFILE_URL, params=params, timeout=180)
    response.raise_for_status()
    rows = response.json()
    return pd.DataFrame(rows[1:], columns=rows[0])


def coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def build_demographic_reference() -> pd.DataFrame:
    if not ZCTA_REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Missing ZCTA reference parquet: {ZCTA_REFERENCE_PATH}. Run `./.venv/bin/python data/build_reference_geo.py` first."
        )

    zcta_reference = pd.read_parquet(ZCTA_REFERENCE_PATH, columns=["zcta", "zcta_name", "population"])
    acs = fetch_acs_table().rename(columns={"zip code tabulation area": "zcta", "NAME": "acs_name"})
    acs["zcta"] = acs["zcta"].astype("string").str.zfill(5)
    acs = coerce_numeric(acs, list(ACS_VARIABLES.values()))

    acs["bachelors_or_higher_count"] = acs[
        [
            ACS_VARIABLES["bachelors_degree"],
            ACS_VARIABLES["masters_degree"],
            ACS_VARIABLES["professional_degree"],
            ACS_VARIABLES["doctorate_degree"],
        ]
    ].sum(axis=1, min_count=1)

    demographics = acs.rename(
        columns={
            ACS_VARIABLES["median_household_income"]: "median_household_income",
            ACS_VARIABLES["poverty_universe"]: "poverty_universe",
            ACS_VARIABLES["poverty_count"]: "poverty_count",
            ACS_VARIABLES["occupied_housing_units"]: "occupied_housing_units",
            ACS_VARIABLES["renter_occupied_units"]: "renter_occupied_units",
            ACS_VARIABLES["race_ethnicity_total"]: "race_ethnicity_total",
            ACS_VARIABLES["non_hispanic_white_alone"]: "non_hispanic_white_alone",
            ACS_VARIABLES["education_total_25_plus"]: "education_total_25_plus",
            ACS_VARIABLES["median_gross_rent"]: "median_gross_rent",
        }
    )[
        [
            "zcta",
            "acs_name",
            "median_household_income",
            "poverty_universe",
            "poverty_count",
            "occupied_housing_units",
            "renter_occupied_units",
            "race_ethnicity_total",
            "non_hispanic_white_alone",
            "education_total_25_plus",
            "bachelors_or_higher_count",
            "median_gross_rent",
        ]
    ]

    demographics["poverty_share"] = demographics["poverty_count"] / demographics["poverty_universe"]
    demographics["renter_share"] = demographics["renter_occupied_units"] / demographics["occupied_housing_units"]
    demographics["nonwhite_share"] = (
        demographics["race_ethnicity_total"] - demographics["non_hispanic_white_alone"]
    ) / demographics["race_ethnicity_total"]
    demographics["bachelors_or_higher_share"] = (
        demographics["bachelors_or_higher_count"] / demographics["education_total_25_plus"]
    )

    merged = zcta_reference.merge(demographics, on="zcta", how="left", validate="one_to_one")
    merged["acs_missing_flag"] = merged["median_household_income"].isna()
    merged["population_gap_flag"] = merged["population"].isna()

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(ZCTA_DEMOGRAPHICS_PATH, index=False)

    print(f"Wrote ZCTA demographics parquet: {ZCTA_DEMOGRAPHICS_PATH}")
    print(f"Matched ZCTAs: {len(merged):,}")
    print(f"ACS missing rows: {int(merged['acs_missing_flag'].sum()):,}")
    return merged


if __name__ == "__main__":
    build_demographic_reference()
