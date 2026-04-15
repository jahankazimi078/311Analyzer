This folder contains chunk-built alternatives to the raw 311 CSV:

- `311_requests.sqlite`: SQLite database for indexed SQL queries.
- `311_requests_parquet/`: one compressed Parquet file per `created_year`.

Why this is better than the raw CSV:
- SQLite avoids reparsing a 12+ GB text file for every analysis.
- You can query subsets directly with SQL instead of loading the whole dataset.
- Basic indexes make common filters faster.
- Parquet is columnar and compressed, which is better for analytics scans and storage.
- The yearly split keeps file counts low while still letting you load a single year directly.

Build or rebuild it with:

```bash
./.venv/bin/python data/build_sqlite.py
./.venv/bin/python data/build_parquet.py
./.venv/bin/python data/build_reference_geo.py
```

`build_reference_geo.py` downloads two lightweight external sources for Phase 5 geospatial analysis:

- Census TIGER/Line ZCTA boundaries
- ACS total population by ZCTA

Those local outputs are written under `data/reference/` and are used by `Geo.ipynb`.

Quick example:

```bash
./.venv/bin/python - <<'PY'
import sqlite3

with sqlite3.connect("data/311_requests.sqlite") as conn:
    rows = conn.execute(
        """
        SELECT borough, COUNT(*)
        FROM requests
        WHERE created_date >= '2025-01-01 00:00:00'
        GROUP BY borough
        ORDER BY COUNT(*) DESC
        LIMIT 10
        """
    ).fetchall()
    print(rows)
PY
```

Parquet example:

```bash
./.venv/bin/python - <<'PY'
import pyarrow.dataset as ds

dataset = ds.dataset("data/311_requests_parquet/2025.parquet", format="parquet")
table = dataset.to_table(
    columns=["borough", "created_date"],
)
print(table.num_rows)
PY
```
