"""
Technical indicators implemented twice - once in Pandas, once in Polars - so we can
benchmark the two libraries on identical work.

Indicators (all computed per company, in date order):
  * SMA_20  - 20-day simple moving average of close
  * EMA_20  - 20-day exponential moving average of close
  * RSI_14  - 14-day Relative Strength Index (Wilder smoothing)
  * MACD    - EMA_12 - EMA_26, plus MACD_signal = 9-day EMA of MACD
Extra model features: daily return, 1/2/3-day lagged close.
Target: next-day close (`target_close`).
"""
import pandas as pd
import polars as pl

FEATURES = ["close", "SMA_20", "EMA_20", "RSI_14", "MACD", "MACD_signal",
            "return_1d", "close_lag1", "close_lag2", "close_lag3", "volume"]


# ----------------------------------------------------------------------------- pandas
def add_indicators_pandas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["name", "date"]).reset_index(drop=True)
    g = df.groupby("name")["close"]
    df["SMA_20"] = g.transform(lambda s: s.rolling(20).mean())
    df["EMA_20"] = g.transform(lambda s: s.ewm(span=20, adjust=False).mean())
    ema12 = g.transform(lambda s: s.ewm(span=12, adjust=False).mean())
    ema26 = g.transform(lambda s: s.ewm(span=26, adjust=False).mean())
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df.groupby("name")["MACD"].transform(lambda s: s.ewm(span=9, adjust=False).mean())

    delta = g.diff()
    df["_gain"] = delta.clip(lower=0)
    df["_loss"] = -delta.clip(upper=0)
    avg_gain = df.groupby("name")["_gain"].transform(lambda s: s.ewm(alpha=1 / 14, adjust=False).mean())
    avg_loss = df.groupby("name")["_loss"].transform(lambda s: s.ewm(alpha=1 / 14, adjust=False).mean())
    df["RSI_14"] = 100 - 100 / (1 + avg_gain / avg_loss)
    df = df.drop(columns=["_gain", "_loss"])

    df["return_1d"] = g.pct_change()
    for k in (1, 2, 3):
        df[f"close_lag{k}"] = g.shift(k)
    df["target_close"] = g.shift(-1)
    return df


# ----------------------------------------------------------------------------- polars
def add_indicators_polars(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(["name", "date"])
    c = pl.col("close")
    delta = c.diff().over("name")
    df = df.with_columns(
        c.rolling_mean(20).over("name").alias("SMA_20"),
        c.ewm_mean(span=20, adjust=False).over("name").alias("EMA_20"),
        (c.ewm_mean(span=12, adjust=False) - c.ewm_mean(span=26, adjust=False)).over("name").alias("MACD"),
        delta.clip(lower_bound=0).alias("_gain"),
        (-delta.clip(upper_bound=0)).alias("_loss"),
        c.pct_change().over("name").alias("return_1d"),
        c.shift(1).over("name").alias("close_lag1"),
        c.shift(2).over("name").alias("close_lag2"),
        c.shift(3).over("name").alias("close_lag3"),
        c.shift(-1).over("name").alias("target_close"),
    )
    df = df.with_columns(
        pl.col("MACD").ewm_mean(span=9, adjust=False).over("name").alias("MACD_signal"),
        (100 - 100 / (1 + pl.col("_gain").ewm_mean(alpha=1 / 14, adjust=False, ignore_nulls=True).over("name")
                      / pl.col("_loss").ewm_mean(alpha=1 / 14, adjust=False, ignore_nulls=True).over("name"))
         ).alias("RSI_14"),
    ).drop(["_gain", "_loss"])
    return df
