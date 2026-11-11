# CSIS 4260 – Assignment 1: Stock Data Storage, Analysis and Prediction Dashboard

**Name:** Kanlayakorn (Kandy) Chanthanam
**Course:** CSIS 4260 – Special Topics in Data Analytics, Douglas College, Fall 2026

## About this project

In this assignment I worked with daily stock prices for 505 S&P 500 companies from February 2013 to February 2018 (619,040 rows, about 29 MB) [1]. I did three things:

1. **Part 1:** I tested whether CSV or Parquet is the better file format to store this data, at the normal size and when the data grows 10 times and 100 times bigger.
2. **Part 2:** I compared Pandas and Polars for data analysis, added technical indicators, and trained two models to predict the next day's closing price.
3. **Part 3:** I built a Streamlit dashboard where you can search any company and see its predictions.

## Folder structure

```
├── data/all_stocks_5yr.csv            # the dataset
├── part1_storage/benchmark_storage.py # Part 1: CSV vs Parquet test
├── part2_analysis/
│   ├── indicators.py                  # Part 2: indicators in Pandas and Polars
│   └── run_part2.py                   # Part 2: speed test + prediction models
├── part3_dashboard/app.py             # Part 3: Streamlit dashboard
├── results/                           # all result tables and predictions
└── requirements.txt                   # Python libraries needed
```

## How to run

```bash
pip install -r requirements.txt

python part1_storage/benchmark_storage.py      # Part 1 (about 45 min; add --scales 1 10 for a quick test)
python part2_analysis/run_part2.py             # Part 2 (about 3 min)
python -m streamlit run part3_dashboard/app.py # Part 3 (opens in your browser)
```

I ran all tests on the same computer (2 CPU cores, 7 GB RAM), so the results can be compared fairly.

---

## Part 1 – CSV vs Parquet

### My recommendation

| Data size | Rows | What I recommend | Why |
|---|---|---|---|
| **1x** | 0.6 million | **Parquet (snappy)**, but CSV is still OK | Everything loads in under 0.4 seconds. CSV is easy to open in Excel, but Parquet is already 3× smaller and about 10× faster. |
| **10x** | 6.2 million | **Parquet (snappy or zstd)** | CSV takes 2.3 seconds to read (3.6 s in Pandas) and is 318 MB. Parquet takes 0.3 seconds and is 87–105 MB. |
| **100x** | 62 million | **Parquet (zstd)** | CSV is 3.25 GB and takes 23 seconds to read. Parquet zstd is 0.87 GB (73% smaller) and takes about 3.5 seconds. My computer ran out of memory when I tried to load the whole CSV. |

I do **not** recommend gzip at any size. It made files only 0.5% smaller than zstd, but it was about 35 times slower to save (566 seconds vs 17 seconds at 100x).

### What I learned from research

- **CSV** saves data as plain text, one row per line. It is easy to read and works everywhere. But the computer must change every value from text back into numbers each time, and it has to read all columns even if I only need two [2].
- **Parquet** saves data by **column** instead of by row, and it remembers the data type of each column [2][3]. This means:
  - It can read **only the columns I need** (column pruning).
  - It keeps a min/max summary for each block of rows, so it can **skip blocks** that do not match my filter (predicate pushdown) [2][3].
  - Values in the same column look similar, so they **compress well** [2].
- **Compression options** in PyArrow [3]:
  - **snappy** (default): very fast, medium compression
  - **zstd**: compresses almost as well as gzip but is much faster [4]
  - **gzip**: smallest files but slow to save
  - **none**: no compression, biggest files

My hypothesis was that Parquet would be smaller and faster at every size, and the difference would matter more as data grows.

### How I tested it

- I tested 6 formats: CSV, CSV + gzip, and Parquet with none, snappy, gzip and zstd.
- To make 10x and 100x data, I copied the original data 10 and 100 times and renamed the tickers (for example `AAPL_3`) so each copy is a different company.
- For each format I measured:
  - **File size** (MB)
  - **Write time** (how long it takes to save)
  - **Full scan** (read every row and column and calculate the average close)
  - **Selective read** (read only 3 columns for one company, like a dashboard does)
  - **Pandas read time** (1x and 10x only)
- I used the middle (median) result of 3 runs at 1x and 10x. At 100x I ran it once because each run takes a long time.

**A problem I found:** my first 100x test crashed because loading the 3.25 GB CSV into memory needed more than 7 GB of RAM. I changed my code to read the file in pieces of 1 million rows (streaming). This showed me that with big data, *how* you read the file matters as much as the file format.

### Results

| Size | Format | Size (MB) | Write (s) | Full scan (s) | Selective read (s) | Pandas read (s) |
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
| 100x | Parquet none | 1313.8 | 12.81 | 2.655 | 0.922 | not tested |
| 100x | Parquet snappy | 1051.7 | 14.55 | 2.735 | 0.977 | not tested |
| 100x | Parquet gzip | 867.0 | 566.16 | 5.361 | 1.621 | not tested |
| 100x | Parquet zstd | 872.4 | 16.68 | 3.549 | 1.224 | not tested |

### What the results show

- **File size:** Parquet zstd was about 73% smaller than CSV at every size.
- **Reading speed:** Parquet was about 8–10× faster for a full scan and 10–13× faster for a selective read.
- **Growth:** when the data grew 10×, the time also grew about 10× for both formats. So the gap stays the same, but at 100x the CSV wait time (23 s) becomes a real problem.
- **Compression:** zstd gave almost the same file size as gzip but was much faster to save, so zstd is the best balance for big data.

### Limitations

- My 10x and 100x data are copies of the same data. Real new data may compress a little differently.
- I tested on one computer only. A faster computer would give smaller times, but I expect the ranking to stay the same.
- I ran the 100x test only once.

---

## Part 2 – Pandas vs Polars, and predicting the next day's price

### What I learned from research

- **Pandas** is the most common Python library for tables of data. It mostly uses one CPU core, and per-company calculations with `groupby().transform()` can be slow [5].
- **Polars** is a newer library written in Rust. It uses all CPU cores and is built on Apache Arrow. Its `.over()` window functions can calculate per-company values in one step [6].
- **Technical indicators** I added for each company:
  - **SMA 20** (Simple Moving Average): the average close of the last 20 days. It shows the trend [7].
  - **EMA 20** (Exponential Moving Average): like SMA, but recent days count more [7].
  - **RSI 14** (Relative Strength Index): measures momentum from 0 to 100. Above 70 can mean "overbought" and below 30 "oversold" [8].
  - **MACD**: EMA 12 minus EMA 26, plus a 9-day signal line. It shows changes in trend and momentum [9].

My hypothesis was that Polars would be faster, and that the models would follow the price closely but would find it hard to beat a simple guess. This is because of the efficient market idea: past prices are already reflected in today's price [10].

### Pandas vs Polars speed test

I built the same 5-step pipeline in both libraries and timed each step (median of 3 runs):

| Step | Pandas (s) | Polars (s) | Polars is faster by |
|---|---:|---:|---:|
| 1. Load CSV | 0.389 | 0.107 | 3.6× |
| 2. Add indicators | 0.835 | 0.133 | **6.3×** |
| 3. Clean + 80/20 split | 0.178 | 0.036 | 5.0× |
| 4. Group-by analysis | 0.081 | 0.027 | 3.0× |
| 5. Prepare model data | 0.508 | 0.195 | 2.6× |
| **Total** | **1.991** | **0.497** | **4.0×** |

To check my work, I compared the indicator values from both libraries. They were the same (the biggest difference was about 0.000000000001, which is only rounding).

**My recommendation:** use **Polars** for the data preparation. It was 4× faster overall and 6× faster on indicators. Pandas is still fine for 1x data (under 2 seconds) and works easily with scikit-learn.

### Prediction models

- **Train/test split:** for each company, I used the first 80% of dates to train and the last 20% to test. I did not shuffle the data, so the model never "sees the future."
- **Inputs (features):** close, SMA 20, EMA 20, RSI 14, MACD, MACD signal, daily return, the closing prices of the last 3 days, and volume.
- **Models:** I chose one simple model and one more complex model [11]:
  - **Linear Regression**: simple and easy to explain
  - **Random Forest**: many decision trees combined, which can learn non-linear patterns [12]
- **Problem I found:** Random Forest cannot predict a price higher than any price it saw in training, and many stocks reached new highs in the test period. To fix this, I predicted the **next-day return (% change)** and then changed it back into a price.
- **Baseline:** I also added a simple guess, "tomorrow's price = today's price," to check if my models really add value.

### Model results (average over all 505 companies, test period)

| Model | RMSE ($) | MAE ($) | MAPE (%) | Up/down correct (%) | Training time (s) |
|---|---:|---:|---:|---:|---:|
| Simple guess (tomorrow = today) | 1.346 | 0.910 | 0.952 | n/a | – |
| Linear Regression | 1.375 | 0.941 | 0.976 | 48.5 | 0.8 |
| Random Forest | 1.444 | 1.016 | 1.071 | 48.9 | 139.2 |

- RMSE and MAE show the error in dollars. MAPE shows the error in percent.
- "Up/down correct" means how often the model correctly guessed if the price would go up or down.

### What the results show

- Both models were close to the real price (about 1% error), but **neither beat the simple guess**.
- They guessed up or down correctly only about 49% of the time, which is like flipping a coin.
- This matches the efficient market idea [10]: indicators made only from past prices do not help much to predict tomorrow.
- **My recommendation:** use **Linear Regression**. It was as accurate as Random Forest, about 170× faster to train, and easier to explain.

### Limitations

- I used only price and volume. I did not use news, company earnings or economic data.
- I did not tune the model settings, and I used one fixed 80/20 split instead of rolling (walk-forward) testing.

---

## Part 3 – Dashboard

### Choosing a library

| | Streamlit [13] | Dash [14] | Reflex [15] |
|---|---|---|---|
| How you write it | A normal Python script, top to bottom | Page layout plus "callback" functions | React-style components and state |
| How hard to learn | Easiest | Medium | Hardest |
| Best for | Quick data apps | Complex business dashboards | Full websites in Python |

I chose **Streamlit** because it was the fastest to build, it has a built-in search box for selecting a ticker, it can save loaded data in memory (caching) [13], and it works well with Plotly charts.

### What the dashboard does

- **Search box** at the top right to find any of the 505 companies
- **Candlestick chart** of the real price (mint = up day, yellow = down day) with the model's prediction drawn as a line
- **Time range buttons** (1M, 3M, 6M, Test, All). "All" shades the test period so you can see where testing starts
- **Model switch** (Linear / Random Forest) on the right panel that updates the chart and all numbers
- **Model cards** showing the last prediction, the average daily error compared with the simple guess, and how often the model guessed up or down correctly
- **Indicator cards** for RSI, MACD, SMA 20 and volume, each with a small trend line and a status label (for example "Oversold" or "Bearish")
- Extra pages for **All companies** (ranking of all 505 companies) and **Speed tests** (Part 1 and Part 2 results with a bar chart)
- A button to **download** the predictions as CSV

### My design choices

- I ran the models once in Part 2 and saved the predictions as a **Parquet** file. The dashboard just reads this file, so it opens quickly. This uses what I learned in Part 1.
- I used `@st.cache_data` so the data loads only once, not every time the user clicks [13].
- I showed the simple guess next to the models so users can see honestly how good the models are.
- I used a light, clean style with rounded cards, the Geist font, and only a few colours (mint, yellow, violet and black) so the numbers are easy to read.

### Testing

I tested the dashboard by changing tickers, switching models, and trying every time range and page. It worked with no errors.

### Future improvements

- Get new stock prices automatically and retrain the models on a schedule.
- Put the dashboard online with Streamlit Community Cloud so anyone can open it without installing Python.

---

## References

[1] C. Nugent, "S&P 500 stock data," Kaggle. https://www.kaggle.com/datasets/camnugent/sandp500

[2] The Apache Software Foundation, "Apache Parquet – Overview." https://parquet.apache.org/docs/overview/

[3] Apache Arrow, "Reading and Writing the Apache Parquet Format," PyArrow documentation. https://arrow.apache.org/docs/python/parquet.html

[4] Meta (Facebook), "Zstandard – Real-time data compression algorithm." https://facebook.github.io/zstd/

[5] pandas development team, "Group by: split-apply-combine," pandas user guide. https://pandas.pydata.org/docs/user_guide/groupby.html

[6] Polars, "Window functions," Polars user guide. https://docs.pola.rs/user-guide/expressions/window/

[7] StockCharts, "Moving Averages – Simple and Exponential," ChartSchool. https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/moving-averages-simple-and-exponential

[8] StockCharts, "Relative Strength Index (RSI)," ChartSchool. https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/relative-strength-index-rsi

[9] StockCharts, "MACD (Moving Average Convergence/Divergence Oscillator)," ChartSchool. https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/macd-moving-average-convergence-divergence-oscillator

[10] E. F. Fama, "Efficient Capital Markets: A Review of Theory and Empirical Work," *The Journal of Finance*, vol. 25, no. 2, pp. 383–417, 1970. https://doi.org/10.2307/2325486

[11] scikit-learn developers, "RandomForestRegressor," scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

[12] L. Breiman, "Random Forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001. https://doi.org/10.1023/A:1010933404324

[13] Streamlit, "st.cache_data," Streamlit documentation. https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_data

[14] Plotly, "Basic Callbacks," Dash documentation. https://dash.plotly.com/basic-callbacks

[15] Reflex, "Introduction," Reflex documentation. https://reflex.dev/docs/getting-started/introduction

## AI tools used

I used Claude (Anthropic) to help write and debug the code and to organize this README. I ran all the tests myself, checked the results, and wrote the conclusions in my own words.
