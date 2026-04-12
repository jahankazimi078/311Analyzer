from __future__ import annotations

import shutil

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from _build_utils import PROJECT_ROOT, RAW_CSV, iter_clean_chunks


OUTPUT_DIR = PROJECT_ROOT / "data" / "311_requests_parquet"


def build_dataset() -> None:
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Missing raw CSV: {RAW_CSV}")

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    writers: dict[str, pq.ParquetWriter] = {}

    try:
        for chunk_number, chunk in enumerate(iter_clean_chunks(), start=1):
            chunk = chunk.assign(
                created_year=chunk["created_date"].str.slice(0, 4).fillna("unknown")
            )

            for created_year, year_chunk in chunk.groupby("created_year", dropna=False):
                output_path = OUTPUT_DIR / f"{created_year}.parquet"
                table = pa.Table.from_pandas(year_chunk, preserve_index=False)

                if created_year not in writers:
                    writers[created_year] = pq.ParquetWriter(
                        output_path,
                        table.schema,
                        compression="zstd",
                    )

                writers[created_year].write_table(table)

            total_rows += len(chunk.index)
            print(f"written chunk {chunk_number}: {total_rows:,} rows")
    finally:
        for writer in writers.values():
            writer.close()

    dataset = ds.dataset(sorted(str(path) for path in OUTPUT_DIR.glob("*.parquet")), format="parquet")
    size_gb = sum(path.stat().st_size for path in OUTPUT_DIR.glob("*.parquet")) / (1024**3)
    print(f"parquet dataset ready: {OUTPUT_DIR} ({size_gb:.2f} GB, {dataset.count_rows():,} rows)")


if __name__ == "__main__":
    build_dataset()
