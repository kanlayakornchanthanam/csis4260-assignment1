"""
Part 3 - Streamlit dashboard for the Part 2 price-prediction models.

Run:  streamlit run part3_dashboard/app.py
(Run part2_analysis/run_part2.py first so results/predictions.parquet exists.)
"""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

st.set_page_config(page_title="S&P 500 Price Predictions", page_icon="📈", layout="wide")


@st.cache_data
def load():
    preds = pd.read_parquet(RES / "predictions.parquet")
    preds["date"] = pd.to_datetime(preds["date"])
    mets = pd.read_csv(RES / "metrics_by_ticker.csv")
    return preds, mets


@st.cache_data
def load_optional(name):
    p = RES / name
    return pd.read_csv(p) if p.exists() else None


if not (RES / "predictions.parquet").exists():
    st.error("results/predictions.parquet not found - run `python part2_analysis/run_part2.py` first.")
    st.stop()

preds, mets = load()
tickers = sorted(preds["name"].unique())

# ---------------------------------------------------------------- sidebar
st.sidebar.title("📈 Controls")
ticker = st.sidebar.selectbox("Search / select a ticker", tickers,
                              index=tickers.index("AAPL") if "AAPL" in tickers else 0,
                              help="Type to search all 505 companies")
models = st.sidebar.multiselect("Models to show", ["Linear Regression", "Random Forest"],
                                default=["Linear Regression", "Random Forest"])
period = st.sidebar.radio("Period", ["Test period only (last 20%)", "Full history"])
show_ind = st.sidebar.multiselect("Overlay indicators", ["SMA_20", "EMA_20"], default=["SMA_20"])

# ---------------------------------------------------------------- data for ticker
d = preds[preds["name"] == ticker].sort_values("date").copy()
d["next_date"] = d["date"].shift(-1)          # prediction made on `date` is for the next trading day
d = d.dropna(subset=["next_date"])
if period.startswith("Test"):
    d = d[d["split"] == "test"]
m = mets[mets["name"] == ticker].set_index("model")

st.title(f"{ticker} — next-day closing price prediction")
st.caption("Models trained on the first 80% of each company's history (Feb 2013 → ~Feb 2017) "
           "and back-tested on the last 20%. Metrics below are on the test period.")

# ---------------------------------------------------------------- KPI cards
cols = st.columns(3)
for col, model in zip(cols, ["Naive (today=tomorrow)", "Linear Regression", "Random Forest"]):
    if model in m.index:
        r = m.loc[model]
        col.metric(model, f"RMSE ${r['rmse']:.2f}", f"MAPE {r['mape']:.2f}% · Direction {r['direction_acc']:.1f}%",
                   delta_color="off")

# ---------------------------------------------------------------- main chart
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.04,
                    subplot_titles=("Actual vs predicted close", "RSI (14)", "MACD"))
fig.add_trace(go.Scatter(x=d["next_date"], y=d["target_close"], name="Actual close",
                         line=dict(color="#1f2937", width=2)), 1, 1)
colors = {"Linear Regression": "#2563eb", "Random Forest": "#ea580c"}
for model in models:
    col = "pred_linear" if model == "Linear Regression" else "pred_rf"
    fig.add_trace(go.Scatter(x=d["next_date"], y=d[col], name=f"{model} prediction",
                             line=dict(color=colors[model], width=1.3, dash="dot")), 1, 1)
for ind in show_ind:
    fig.add_trace(go.Scatter(x=d["date"], y=d[ind], name=ind, line=dict(width=1), opacity=0.6), 1, 1)
if period == "Full history":
    split_date = preds[(preds["name"] == ticker) & (preds["split"] == "test")]["date"].min()
    fig.add_vline(x=split_date, line_dash="dash", line_color="gray")
    fig.add_annotation(x=split_date, y=1, yref="paper", text="train | test", showarrow=False)
fig.add_trace(go.Scatter(x=d["date"], y=d["RSI_14"], name="RSI 14", line=dict(color="#7c3aed")), 2, 1)
fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
fig.add_trace(go.Scatter(x=d["date"], y=d["MACD"], name="MACD", line=dict(color="#0891b2")), 3, 1)
fig.add_trace(go.Scatter(x=d["date"], y=d["MACD_signal"], name="Signal", line=dict(color="#f59e0b")), 3, 1)
fig.update_layout(height=780, hovermode="x unified", legend=dict(orientation="h", y=1.06),
                  margin=dict(t=60, b=20))
st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- tabs
tab1, tab2, tab3 = st.tabs(["Prediction errors", "All-company leaderboard", "Benchmarks (Part 1 & 2)"])
with tab1:
    e = d[["next_date", "target_close", "pred_linear", "pred_rf"]].rename(
        columns={"next_date": "date", "target_close": "actual"})
    e["err_linear"] = e["pred_linear"] - e["actual"]
    e["err_rf"] = e["pred_rf"] - e["actual"]
    ef = go.Figure()
    ef.add_trace(go.Histogram(x=e["err_linear"], name="Linear Regression", opacity=0.6, marker_color="#2563eb"))
    ef.add_trace(go.Histogram(x=e["err_rf"], name="Random Forest", opacity=0.6, marker_color="#ea580c"))
    ef.update_layout(barmode="overlay", title="Distribution of prediction errors ($)", height=350)
    st.plotly_chart(ef, width="stretch")
    st.dataframe(e.tail(30).iloc[::-1].round({"actual": 2, "pred_linear": 2, "pred_rf": 2, "err_linear": 2, "err_rf": 2}), width="stretch", hide_index=True)
    st.download_button("Download predictions (CSV)", e.to_csv(index=False), f"{ticker}_predictions.csv")
with tab2:
    st.write("Average test metrics across all 505 companies")
    st.dataframe(mets.groupby("model")[["rmse", "mae", "mape", "direction_acc"]].mean().round(3),
                 width="stretch")
    pick = st.selectbox("Rank companies by MAPE for", ["Linear Regression", "Random Forest"])
    st.dataframe(mets[mets["model"] == pick].sort_values("mape")[["name", "rmse", "mae", "mape", "direction_acc"]]
                 .round(3), width="stretch", hide_index=True, height=400)
with tab3:
    b1 = load_optional("part1_storage_benchmark.csv")
    b2 = load_optional("part2_pandas_vs_polars.csv")
    if b1 is not None:
        st.subheader("Part 1 — CSV vs Parquet")
        st.dataframe(b1, width="stretch", hide_index=True)
    if b2 is not None:
        st.subheader("Part 2 — Pandas vs Polars (seconds)")
        st.dataframe(b2, width="stretch", hide_index=True)
