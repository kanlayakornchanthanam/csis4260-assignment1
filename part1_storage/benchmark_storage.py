"""
Part 1 - CSV vs Parquet storage benchmark at 1x, 10x and 100x scale.

For each scale we build a dataset by repeating the original 619,040 rows
(tickers get a suffix, e.g. AAPL_3, so every copy stays a distinct company),
then measure for every format:
  * file size on disk
  * write time
  * full scan time (stream every row & column, compute mean close)
  * selective read time (2 columns + filter to one ticker) - typical dashboard query

Formats: CSV (plain), CSV (gzip), Parquet with none / snappy / gzip / zstd compression.

Usage:
    python part1_storage/benchmark_storage.py                 # 1x, 10x, 100x
    python part1_storage/benchmark_storage.py --scales 1 10   # skip 100x on a small laptop
"""
import argparse
import gc
import os
import shutil
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "all_stocks_5yr.csv"
WORK = ROOT / "data" / "scaled"
OUT = ROOT / "results"

PARQUET_CODECS = ["none", "snappy", "gzip", "zstd"]
REPEATS = 3  # timings are the median of REPEATS runs (1 run at 100x)


def load_base() -> pa.Table:
    t = pacsv.read_csv(SRC)
    return t.cast(pa.schema([
        ("date", pa.date32()), ("open", pa.float64()), ("high", pa.float64()),
        ("low", pa.float64()), ("close", pa.float64()), ("volume", pa.int64()),
        ("name", pa.string())]))


def scaled_chunks(base: pa.Table, scale: int):
    """Yield one copy of the data at a time so 100x never sits in RAM at once."""
    for i in range(scale):
        if i == 0:
            yield base
        else:
            names = pc.binary_join_element_wise(base["name"], pa.scalar(f"_{i}"), "")
            yield base.set_column(6, "name", names)


def timed(fn, repeats):
    times = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
        del out
    return sorted(times)[len(times) // 2]


def write_csv(base, scale, path, gzip=False):
    stream = pa.CompressedOutputStream(str(path), "gzip") if gzip else pa.OSFile(str(path), "wb")
    with stream, pacsv.CSVWriter(stream, base.schema) as w:
        for chunk in scaled_chunks(base, scale):
            w.write_table(chunk)


def write_parquet(base, scale, path, codec):
    with pq.ParquetWriter(str(path), base.schema, compression=codec) as w:
        for chunk in scaled_chunks(base, scale):
            w.write_table(chunk, row_group_size=1_000_000)


def _batches(path, kind, columns=None):
    """Stream record batches so even the 100x file (~62M rows) never has to fit in RAM."""
    if kind.startswith("csv"):
        opts = pacsv.ConvertOptions(include_columns=columns) if columns else None
        yield from pacsv.open_csv(str(path), convert_options=opts)
    else:
        yield from pq.ParquetFile(str(path)).iter_batches(batch_size=1_000_000, columns=columns)


def read_full(path, kind):
    """Full scan: parse every column of every row, compute the average close."""
    total, n = 0.0, 0
    for b in _batches(path, kind):
        total += pc.sum(b.column("close")).as_py() or 0.0
        n += b.num_rows
    return total / n


def read_selective(path, kind, ticker="AAPL"):
    """Typical dashboard query: 3 columns, one ticker."""
    cols = ["date", "close", "name"]
    if kind.startswith("parquet"):  # parquet can skip row groups using min/max statistics
        return pq.read_table(str(path), columns=cols, filters=[("name", "=", ticker)])
    parts = [b.filter(pc.equal(b.column("name"), ticker)) for b in _batches(path, kind, cols)]
    return pa.Table.from_batches(parts)


def main(scales):
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(exist_ok=True)
    base = load_base()
    formats = [("csv", None), ("csv_gzip", None)] + [(f"parquet_{c}", c) for c in PARQUET_CODECS]
    rows = []
    for scale in scales:
        reps = 1 if scale >= 100 else REPEATS
        print(f"\n=== scale {scale}x  ({base.num_rows * scale:,} rows) ===")
        for kind, codec in formats:
            ext = ".csv.gz" if kind == "csv_gzip" else (".csv" if kind == "csv" else ".parquet")
            path = WORK / f"stocks_{scale}x_{kind}{ext}"
            if kind == "csv":
                w = timed(lambda: write_csv(base, scale, path), 1)
            elif kind == "csv_gzip":
                w = timed(lambda: write_csv(base, scale, path, gzip=True), 1)
            else:
                w = timed(lambda: write_parquet(base, scale, path, codec), 1)
            size_mb = path.stat().st_size / 1e6
            r_full = timed(lambda: read_full(path, kind), reps)
            r_sel = timed(lambda: read_selective(path, kind), reps)
            # pandas read for reference (what most course notebooks use); skip at 100x - too big for RAM
            if scale < 100:
                pd_fn = (lambda: pd.read_csv(path)) if kind.startswith("csv") else (lambda: pd.read_parquet(path))
                r_pd = timed(pd_fn, reps)
            else:
                r_pd = None
            row = dict(scale=f"{scale}x", rows=base.num_rows * scale, format=kind,
                       size_mb=round(size_mb, 1), write_s=round(w, 3), read_full_s=round(r_full, 3),
                       read_selective_s=round(r_sel, 3),
                       pandas_read_s=None if r_pd is None else round(r_pd, 3))
            print(row)
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT / "part1_storage_benchmark.csv", index=False)  # save as we go
            path.unlink()  # free disk before the next format
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "part1_storage_benchmark.csv", index=False)
    shutil.rmtree(WORK, ignore_errors=True)
    print("\nSaved results/part1_storage_benchmark.csv")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", nargs="+", type=int, default=[1, 10, 100])
    main(ap.parse_args().scales)
