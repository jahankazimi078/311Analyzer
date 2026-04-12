from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "311_Service_Requests_from_2020_to_Present.csv"
CHUNK_SIZE = 100_000
DATE_COLUMNS = [
    "created_date",
    "closed_date",
    "due_date",
    "resolution_action_updated_date",
]
SOURCE_DATE_FORMAT = "%m/%d/%Y %I:%M:%S %p"


def normalize_column(name: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "_" for char in name)
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def iter_clean_chunks() -> Iterator[pd.DataFrame]:
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Missing raw CSV: {RAW_CSV}")

    for chunk in pd.read_csv(
        RAW_CSV,
        chunksize=CHUNK_SIZE,
        low_memory=False,
        dtype="string",
    ):
        chunk.columns = [normalize_column(name) for name in chunk.columns]
        for column in DATE_COLUMNS:
            if column in chunk.columns:
                parsed = pd.to_datetime(
                    chunk[column],
                    errors="coerce",
                    format=SOURCE_DATE_FORMAT,
                )
                chunk[column] = parsed.dt.strftime("%Y-%m-%d %H:%M:%S")
        yield chunk
