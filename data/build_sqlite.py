from __future__ import annotations

import sqlite3
from pathlib import Path

from _build_utils import CHUNK_SIZE, PROJECT_ROOT, RAW_CSV, iter_clean_chunks


OUTPUT_DB = PROJECT_ROOT / "data" / "311_requests.sqlite"
TABLE_NAME = "requests"


def build_database() -> None:
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Missing raw CSV: {RAW_CSV}")

    OUTPUT_DB.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_DB.exists():
        OUTPUT_DB.unlink()

    with sqlite3.connect(OUTPUT_DB) as connection:
        connection.execute("PRAGMA journal_mode = WAL;")
        connection.execute("PRAGMA synchronous = NORMAL;")
        connection.execute("PRAGMA temp_store = MEMORY;")
        connection.execute("PRAGMA cache_size = -200000;")

        total_rows = 0

        for chunk_number, chunk in enumerate(iter_clean_chunks(), start=1):
            chunk.to_sql(TABLE_NAME, connection, if_exists="append", index=False)
            total_rows += len(chunk.index)
            print(f"loaded chunk {chunk_number}: {total_rows:,} rows")

        connection.execute(
            "CREATE INDEX idx_requests_unique_key ON requests (unique_key);"
        )
        connection.execute(
            "CREATE INDEX idx_requests_created_date ON requests (created_date);"
        )
        connection.execute("CREATE INDEX idx_requests_borough ON requests (borough);")
        connection.execute("ANALYZE;")

    size_gb = OUTPUT_DB.stat().st_size / (1024**3)
    print(f"sqlite database ready: {OUTPUT_DB} ({size_gb:.2f} GB)")


if __name__ == "__main__":
    build_database()
