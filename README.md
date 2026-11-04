# CSIS 4260 – Assignment 1: S&P 500 Storage, Analysis & Prediction Dashboard

**Student:** Kandy Chanthanam · Douglas College · Fall 2026

Dataset: daily prices for 505 S&P 500 companies, 2013-02-08 → 2018-02-07 (619,040 rows, ~29 MB CSV).

```
├── data/all_stocks_5yr.csv            # provided dataset
├── part1_storage/benchmark_storage.py # CSV vs Parquet at 1x / 10x / 100x
├── part2_analysis/
│   ├── indicators.py                  # SMA, EMA, RSI, MACD in Pandas AND Polars
│   └── run_part2.py                   # Pandas vs Polars benchmark + LR / RF models
├── part3_dashboard/app.py             # Streamlit dashboard
├── results/                           # benchmark tables, metrics, predictions.parquet
└── requirements.txt
```

## How to run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python part1_storage/benchmark_storage.py          # ~45 min incl. 100x (use --scales 1 10 for a quick run)
python part2_analysis/run_part2.py                 # ~3 min
streamlit run part3_dashboard/app.py               # opens http://localhost:8501
```

Results already in `results/` were produced on a 2-core / 7 GB RAM Linux machine (Python 3, pandas 3.0, polars 1.x, pyarrow 26, scikit-learn 1.9).

---

## Part 1 – Storing and retrieving data: CSV vs Parquet

### Recommendation

| Scale | Rows | Recommended format | Why |
|---|---|---|---|
| **1x** | 0.6 M | **Parquet (snappy)** — CSV still acceptable | Everything is under 0.4 s; CSV is fine for sharing/inspection, but Parquet is already 3× smaller and ~10× faster to read |
| **10x** | 6.2 M | **Parquet (snappy or zstd)** | CSV full read 2.3 s (pandas 3.6 s) vs 0.3 s; 318 MB vs ~90–105 MB |
| **100x** | 62 M | **Parquet (zstd)** | CSV is 3.25 GB and 23 s to scan; zstd Parquet is 0.87 GB (−73%) and ~3.5 s. Loading the whole CSV into memory crashed our 7 GB machine, so columnar + streaming/selective reads become a necessity |

Avoid **gzip** at any scale: it gives only ~0.5% smaller files than zstd but writes are ~35× slower (566 s vs 17 s at 100x).

### Benchmark results (median of 3 runs; 1 run at 100x)

| Scale | Format | Size (MB) | Write (s) | Full scan (s) | Selective read* (s) | pandas read (s) |
|---|---|---:|---:|---:|---:|---:|
| 1x | CSV | 30.7 | 0.35 | 0.224 | 0.126 | 0.322 |
| 1x | CSV gzip | 10.0 | 6.07 | 0.240 | 0.156 | 0.433 |
| 1x | Parquet none | 13.1 | 0.12 | 0.029 | 0.013 | 0.038 |
| 1x | Parquet snappy | 10.5 | 0.13 | 0.027 | 0.013 | 0.035 |
| 1x | Parquet gzip | 8.7 | 5.67 | 0.050 | 0.020 | 0.058 |
| 1x | Parquet zstd | 8.7 | 0.14 | 0.029 | 0.014 | 0.039 |
| 10x | CSV | 318.3 | 3.30 | 2.319 | 1.209 | 3.588 |
| 10x | CSV gzip | 100.4 | 59.59 | 2.331 | 1.462 | 4.430 |
| 10x | Parquet none | 131.4 | 1.26 | 0.254 | 0.107 | 0.282 |
| 10x | Parquet snappy | 105.2 | 1.41 | 0.294 | 0.113 | 0.293 |
| 10x | Parquet gzip | 86.7 | 57.78 | 0.524 | 0.162 | 0.604 |
| 10x | Parquet zstd | 87.2 | 1.67 | 0.299 | 0.114 | 0.325 |
| 100x | CSV | 3250.1 | 33.77 | 23.327 | 12.694 | out of memory |
| 100x | CSV gzip | 1005.2 | 585.01 | 23.275 | 14.391 | out of memory |
| 100x | Parquet none | 1313.8 | 12.81 | 2.655 | 0.922 | – |
| 100x | Parquet snappy | 1051.7 | 14.55 | 2.735 | 0.977 | – |
| 100x | Parquet gzip | 867.0 | 566.16 | 5.361 | 1.621 | – |
| 100x | Parquet zstd | 872.4 | 16.68 | 3.549 | 1.224 | – |

\*Selective read = columns `date, close, name` for one ticker (a typical dashboard query).

**Method:** 10x/100x datasets are built by repeating the original data with renamed tickers (e.g. `AAPL_3`), written in chunks with PyArrow. Full scan streams every row/column and computes mean close, so 100x can be measured without loading 62 M rows into RAM.

### Research notes
- **CSV** is row-based plain text: human-readable and universal, but every value must be parsed from text, there is no schema (dates/types re-inferred on every load), and you must read all columns even when you need two.
- **Parquet** is a columnar, typed binary format (Apache Arrow ecosystem). Columns are stored and compressed separately, so readers fetch only needed columns (*column pruning*) and skip row groups using min/max statistics (*predicate pushdown*) — this is why selective reads stay ~1 s at 100x.
- **Compression codecs:** *snappy* (default) – very fast, moderate ratio; *zstd* – near-gzip ratio at near-snappy speed (best balance for large/archival data); *gzip* – best ratio but slow writes; *none* – largest files, no CPU cost.
- Scaling is roughly linear for both formats, but CSV's absolute costs (23 s scans, 3+ GB files, RAM exhaustion) cross from "annoying" to "impractical" between 10x and 100x.

---

## Part 2 – Pandas vs Polars, technical indicators, and prediction models (1x)

### Technical indicators added (per company)
| Indicator | Definition |
|---|---|
| SMA_20 | 20-day simple moving average of close |
| EMA_20 | 20-day exponential moving average |
| RSI_14 | 14-day Relative Strength Index (Wilder smoothing) — momentum, overbought >70 / oversold <30 |
| MACD / MACD_signal | EMA12 − EMA26, and its 9-day EMA — trend/momentum |

Both libraries produce identical values (max difference ≈ 1e-12, floating-point noise).

### Pandas vs Polars benchmark (seconds, median of 3)

| Step | Pandas | Polars | Polars speed-up |
|---|---:|---:|---:|
| Load CSV | 0.389 | 0.107 | 3.6× |
| Technical indicators | 0.835 | 0.133 | **6.3×** |
| Clean + 80/20 split | 0.178 | 0.036 | 5.0× |
| Group-by analytics | 0.081 | 0.027 | 3.0× |
| Per-ticker model matrices | 0.508 | 0.195 | 2.6× |
| **Total** | **1.991** | **0.497** | **4.0×** |

**Recommendation:** Polars for the data pipeline — 4× faster overall and 6× on the window-heavy indicator step (multi-threaded, Rust engine, `.over("name")` window expressions instead of slow `groupby().transform(lambda)`). Pandas remains convenient at 1x (< 2 s total) and integrates directly with scikit-learn, so it is a reasonable choice for small data or teams already fluent in it.

### Prediction models (next-day close, all 505 companies)
- **Split:** per company, chronological 80% train / 20% test (no shuffling — avoids look-ahead leakage).
- **Features:** close, SMA_20, EMA_20, RSI_14, MACD, MACD_signal, 1-day return, close lags 1–3, volume.
- **Target:** next-day return → converted back to price (`close × (1 + predicted return)`). Predicting returns lets the Random Forest work when test prices exceed the training range (trees cannot extrapolate raw prices).
- **Models:** Linear Regression and Random Forest (100 trees, max depth 8), one per company. A naive "tomorrow = today" baseline is included for honesty.

| Model (avg over 505 companies, test set) | RMSE ($) | MAE ($) | MAPE (%) | Direction accuracy (%) | Train time (s) |
|---|---:|---:|---:|---:|---:|
| Naive (tomorrow = today) | 1.346 | 0.910 | 0.952 | n/a | – |
| Linear Regression | 1.375 | 0.941 | 0.976 | 48.5 | 0.8 |
| Random Forest | 1.444 | 1.016 | 1.071 | 48.9 | 139.2 |

**Interpretation:** Both models track prices closely (MAPE ≈ 1%), but neither beats the naive baseline and direction accuracy is ~49% (coin-flip). This is consistent with the weak-form efficient market hypothesis: indicators built only from past prices carry little exploitable information about tomorrow's move. Linear Regression is preferred — equal accuracy, ~170× faster to train, and interpretable.

---

## Part 3 – Dashboard (Streamlit)

**Library choice:** Streamlit was chosen over Dash and Reflex.
| | Streamlit | Dash (Plotly) | Reflex |
|---|---|---|---|
| Learning curve | Lowest – plain Python script | Medium – callbacks/layout | Higher – React-style state |
| Best for | Data apps & prototypes | Enterprise/complex interactions | Full web apps in Python |
| Deployment | Streamlit Community Cloud (free) | Any WSGI host | Reflex hosting / self-host |

Features: searchable ticker selector (all 505 companies) · actual vs predicted close for both models · SMA/EMA overlays · RSI and MACD panels · train/test split marker · per-ticker metric cards · error distribution & downloadable predictions · all-company leaderboard · Part 1 & 2 benchmark tables.

Run with `streamlit run part3_dashboard/app.py`.
