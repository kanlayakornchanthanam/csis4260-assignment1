"""
Part 2 - Pandas vs Polars benchmark + next-day closing price prediction (1x data).

Steps
 1. Benchmark Pandas vs Polars on the same analysis pipeline:
      load CSV -> add technical indicators -> clean -> 80/20 chronological split
      -> summary analytics (per-ticker stats, yearly returns) -> export model matrices
 2. Train 2 algorithms per company (505 companies) on the first 80% of each
    company's history, back-test on the last 20%:
      * Linear Regression
      * Random Forest Regressor
    Both predict next-day *return*; predicted close = today's close * (1 + predicted return).
    (Tree models cannot extrapolate beyond prices seen in training, so predicting the
    return instead of the raw price keeps them usable when the test period hits new highs.)
    A naive "tomorrow = today" baseline is reported for context.
 3. Save predictions + metrics to results/ for the Part 3 dashboard.

Usage:  python part2_analysis/run_part2.py
"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from indicators import FEATURES, add_indicators_pandas, add_indicators_polars

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "all_stocks_5yr.csv"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
TRAIN_FRAC = 0.8
REPEATS = 3


# ------------------------------------------------------------------ pipelines
def pandas_pipeline():
    t = {}
    s = time.perf_counter(); df = pd.read_csv(DATA, parse_dates=["date"]); t["1 load csv"] = time.perf_counter() - s
    s = time.perf_counter(); df = add_indicators_pandas(df); t["2 technical indicators"] = time.perf_counter() - s
    s = time.perf_counter()
    df = df.dropna(subset=FEATURES + ["target_close"]).reset_index(drop=True)
    df["row_no"] = df.groupby("name").cumcount()
    n = df.groupby("name")["row_no"].transform("max") + 1
    df["split"] = np.where(df["row_no"] < (n * TRAIN_FRAC).astype(int), "train", "test")
    t["3 clean + 80/20 split"] = time.perf_counter() - s
    s = time.perf_counter()
    stats = df.groupby("name").agg(avg_close=("close", "mean"), volatility=("return_1d", "std"),
                                   avg_volume=("volume", "mean"), avg_rsi=("RSI_14", "mean"))
    yearly = (df.assign(year=df["date"].dt.year).groupby(["name", "year"])["close"]
              .agg(["first", "last"]).assign(ret=lambda x: x["last"] / x["first"] - 1))
    t["4 groupby analytics"] = time.perf_counter() - s
    s = time.perf_counter()
    mats = {k: (g[FEATURES].to_numpy(), g["split"].to_numpy()) for k, g in df.groupby("name")}
    t["5 per-ticker model matrices"] = time.perf_counter() - s
    return df, t


def polars_pipeline():
    t = {}
    s = time.perf_counter(); df = pl.read_csv(DATA, try_parse_dates=True); t["1 load csv"] = time.perf_counter() - s
    s = time.perf_counter(); df = add_indicators_polars(df); t["2 technical indicators"] = time.perf_counter() - s
    s = time.perf_counter()
    df = df.drop_nulls(FEATURES + ["target_close"]).filter(pl.all_horizontal(pl.col(FEATURES).is_finite()))
    df = df.with_columns(pl.int_range(pl.len()).over("name").alias("row_no"),
                         pl.len().over("name").alias("_n"))
    df = df.with_columns(pl.when(pl.col("row_no") < (pl.col("_n") * TRAIN_FRAC).cast(pl.Int64))
                         .then(pl.lit("train")).otherwise(pl.lit("test")).alias("split")).drop("_n")
    t["3 clean + 80/20 split"] = time.perf_counter() - s
    s = time.perf_counter()
    stats = df.group_by("name").agg(pl.col("close").mean().alias("avg_close"),
                                    pl.col("return_1d").std().alias("volatility"),
                                    pl.col("volume").mean().alias("avg_volume"),
                                    pl.col("RSI_14").mean().alias("avg_rsi"))
    yearly = (df.with_columns(pl.col("date").dt.year().alias("year")).group_by(["name", "year"])
              .agg(pl.col("close").first().alias("first"), pl.col("close").last().alias("last"))
              .with_columns((pl.col("last") / pl.col("first") - 1).alias("ret")))
    t["4 groupby analytics"] = time.perf_counter() - s
    s = time.perf_counter()
    mats = {k[0]: (g.select(FEATURES).to_numpy(), g["split"].to_numpy()) for k, g in df.group_by("name")}
    t["5 per-ticker model matrices"] = time.perf_counter() - s
    return df, t


def benchmark():
    rows = []
    for lib, fn in (("pandas", pandas_pipeline), ("polars", polars_pipeline)):
        runs = [fn()[1] for _ in range(REPEATS)]
        for step in runs[0]:
            rows.append(dict(library=lib, step=step, seconds=float(np.median([r[step] for r in runs]))))
    bench = pd.DataFrame(rows)
    total = bench.groupby("library", as_index=False)["seconds"].sum().assign(step="TOTAL")
    bench = pd.concat([bench, total], ignore_index=True)
    wide = bench.pivot(index="step", columns="library", values="seconds")
    wide["speedup (pandas/polars)"] = wide["pandas"] / wide["polars"]
    return wide.round(4)


# ------------------------------------------------------------------ modelling
def metrics(y, p, prev):
    err = p - y
    return dict(rmse=float(np.sqrt(np.mean(err ** 2))), mae=float(np.mean(np.abs(err))),
                mape=float(np.mean(np.abs(err / y)) * 100),
                direction_acc=float(np.mean(np.sign(p - prev) == np.sign(y - prev)) * 100))


def train_models(df: pd.DataFrame):
    preds, mets = [], []
    t_lr = t_rf = 0.0
    for name, g in df.groupby("name", sort=True):
        X = g[FEATURES].to_numpy()
        close = g["close"].to_numpy()
        y_ret = g["target_close"].to_numpy() / close - 1
        tr, te = (g["split"] == "train").to_numpy(), (g["split"] == "test").to_numpy()

        s = time.perf_counter()
        lr = LinearRegression().fit(X[tr], y_ret[tr]); p_lr = close * (1 + lr.predict(X))
        t_lr += time.perf_counter() - s
        s = time.perf_counter()
        rf = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=5,
                                   n_jobs=-1, random_state=42).fit(X[tr], y_ret[tr])
        p_rf = close * (1 + rf.predict(X))
        t_rf += time.perf_counter() - s

        out = g[["date", "name", "close", "target_close", "split", "SMA_20", "EMA_20",
                 "RSI_14", "MACD", "MACD_signal"]].copy()
        out["pred_linear"], out["pred_rf"] = p_lr, p_rf
        preds.append(out)

        y, prev = g["target_close"].to_numpy()[te], close[te]
        for model, p in (("Naive (today=tomorrow)", prev), ("Linear Regression", p_lr[te]),
                         ("Random Forest", p_rf[te])):
            row = dict(name=name, model=model, **metrics(y, p, prev))
            if model.startswith("Naive"):
                row["direction_acc"] = np.nan  # naive never predicts a move, so direction is undefined
            mets.append(row)
    return pd.concat(preds, ignore_index=True), pd.DataFrame(mets), dict(linear=t_lr, rf=t_rf)


def main():
    print("Benchmarking Pandas vs Polars ...")
    bench = benchmark()
    print(bench.to_string())
    bench.to_csv(OUT / "part2_pandas_vs_polars.csv")

    # sanity check: both libraries produce the same indicators
    pdf, _ = pandas_pipeline()
    pldf, _ = polars_pipeline()
    m = pdf.merge(pldf.to_pandas(), on=["name", "date"], suffixes=("_pd", "_pl"))
    for col in ["SMA_20", "EMA_20", "RSI_14", "MACD", "MACD_signal"]:
        diff = (m[f"{col}_pd"] - m[f"{col}_pl"]).abs().max()
        print(f"max |pandas - polars| for {col}: {diff:.2e}")

    print("\nTraining Linear Regression + Random Forest for", pdf["name"].nunique(), "companies ...")
    preds, mets, train_t = train_models(pdf)
    print("training time (s):", {k: round(v, 1) for k, v in train_t.items()})
    preds.to_parquet(OUT / "predictions.parquet", index=False)
    mets.to_csv(OUT / "metrics_by_ticker.csv", index=False)
    summary = mets.groupby("model")[["rmse", "mae", "mape", "direction_acc"]].mean().round(3)
    summary["train_seconds_all_505"] = pd.Series({"Linear Regression": train_t["linear"],
                                                  "Random Forest": train_t["rf"]}).round(1)
    summary.to_csv(OUT / "metrics_summary.csv")
    print("\nAverage test metrics across companies:\n", summary.to_string())


if __name__ == "__main__":
    main()
