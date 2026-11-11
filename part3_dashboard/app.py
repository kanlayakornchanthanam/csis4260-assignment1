"""
Part 3 - Streamlit dashboard for the Part 2 price-prediction models.

Run:  python -m streamlit run part3_dashboard/app.py
(Run part2_analysis/run_part2.py first so results/predictions.parquet exists.)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

# ---------------------------------------------------------------- design tokens
INK = "#16233B"        # deep navy - text and the real price
MUTED = "#5B6B82"      # secondary text
LINE = "#D5DCE6"       # hairlines
PAPER = "#FFFFFF"      # panels
LR = "#0F7C80"         # Linear Regression - teal
RF = "#D98A1F"         # Random Forest - amber
UP, DOWN = "#2F7D4F", "#B23A48"
MODEL_COLORS = {"Linear Regression": LR, "Random Forest": RF}
MODEL_COLS = {"Linear Regression": "pred_linear", "Random Forest": "pred_rf"}
NAIVE = "Naive (today=tomorrow)"

st.set_page_config(page_title="Stock Prediction Back-test", page_icon="📈", layout="wide")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@75..100,500..800&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, label, p, li, input, button, textarea {{
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
}}
.block-container {{ padding-top: 2.2rem; max-width: 1280px; }}
[data-testid="stSidebar"] {{ background: {INK}; }}
[data-testid="stSidebar"] * {{ color: #E8EDF5; }}
[data-testid="stSidebar"] [data-baseweb="select"] div, [data-testid="stSidebar"] [data-baseweb="select"] input,
[data-testid="stSidebar"] [data-baseweb="select"] span {{ color: {INK} !important; -webkit-text-fill-color: {INK}; }}
[data-testid="stSidebar"] [data-testid="stSelectbox"] input,
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] *,
[data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-baseweb="select"] > div > div:first-child {{
  color: {INK} !important; -webkit-text-fill-color: {INK} !important; opacity: 1 !important; }}
[data-testid="stSidebar"] .side-note {{ color: #9FB0C8; font-size: .82rem; line-height: 1.45; }}
[data-testid="stSidebar"] [data-baseweb="tag"] {{ background: #2A3A57 !important; }}
[data-testid="stSidebar"] [data-baseweb="tag"] * {{ color: #E8EDF5 !important; }}

.hero {{ display: flex; align-items: flex-end; justify-content: space-between; gap: 2rem;
        border-bottom: 2px solid {INK}; padding-bottom: 1rem; margin-bottom: 1.1rem; flex-wrap: wrap; }}
.ticker {{ font-family: 'Archivo', sans-serif; font-stretch: 75%; font-weight: 800; font-size: 6.2rem;
          line-height: .82; letter-spacing: -.02em; color: {INK}; margin: 0; }}
.hero-sub {{ color: {MUTED}; font-size: .95rem; margin-top: .55rem; }}
.price {{ text-align: right; }}
.price .v {{ font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 2.6rem; color: {INK};
            font-variant-numeric: tabular-nums; line-height: 1; }}
.price .d {{ font-size: .95rem; font-variant-numeric: tabular-nums; margin-top: .3rem; }}
.price .l {{ color: {MUTED}; font-size: .82rem; margin-top: .15rem; }}

.verdict {{ font-size: 1.12rem; line-height: 1.5; color: {INK}; max-width: 68ch; margin: 0 0 1.2rem 0; }}

.board {{ width: 100%; border-collapse: collapse; background: {PAPER}; border: 1px solid {LINE};
         font-variant-numeric: tabular-nums; margin-bottom: 1.4rem; }}
.board th, .board td {{ border-left: none !important; border-right: none !important; }}
.board th {{ text-align: left; font-weight: 500; color: {MUTED}; font-size: .82rem; padding: .6rem 1rem;
            border-bottom: 1px solid {LINE}; }}
.board th.num, .board td.num {{ text-align: right; }}
.board td {{ padding: .75rem 1rem; border-bottom: 1px solid {LINE}; font-size: 1rem; color: {INK}; }}
.board tr:last-child td {{ border-bottom: none; }}
.board tr.best td {{ background: #EEF6F5; }}
.swatch {{ display: inline-block; width: 18px; height: 3px; vertical-align: middle; margin-right: .6rem; }}
.badge {{ font-size: .75rem; color: {LR}; border: 1px solid {LR}; padding: .05rem .45rem; margin-left: .6rem;
         border-radius: 2px; }}

h3 {{ font-family: 'Archivo', sans-serif !important; font-weight: 700 !important; color: {INK} !important; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 500; }}
:focus-visible {{ outline: 2px solid {LR} !important; outline-offset: 2px; }}
@media (max-width: 700px) {{
  .ticker {{ font-size: 4rem; }}
  .price {{ text-align: left; }}
  .board th, .board td {{ padding: .5rem .55rem; font-size: .88rem; }}
}}
</style>
""", unsafe_allow_html=True)


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
    st.error("Prediction file not found. Run `python part2_analysis/run_part2.py` first, then reload this page.")
    st.stop()

preds, mets = load()
tickers = sorted(preds["name"].unique())

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Choose a company")
    ticker = st.selectbox("Ticker", tickers, index=tickers.index("AAPL") if "AAPL" in tickers else 0,
                          help="Type to search all 505 S&P 500 companies")
    models = st.multiselect("Models on the chart", list(MODEL_COLS), default=list(MODEL_COLS))
    period = st.radio("Dates", ["Test period (last 20%)", "Full history"])
    show_ind = st.multiselect("Moving averages", ["SMA_20", "EMA_20"], default=[])
    st.markdown(
        "<p class='side-note'>Each model was trained on the first 80% of a company's dates "
        "and tested on the last 20%. The simple guess assumes tomorrow's price equals today's.</p>",
        unsafe_allow_html=True)

# ---------------------------------------------------------------- data for ticker
full = preds[preds["name"] == ticker].sort_values("date").copy()
full["next_date"] = full["date"].shift(-1)
full = full.dropna(subset=["next_date"])
d = full[full["split"] == "test"] if period.startswith("Test") else full
m = mets[mets["name"] == ticker].set_index("model")

last = full.iloc[-1]
first_test = full[full["split"] == "test"].iloc[0]
chg = last["target_close"] / first_test["close"] - 1

# ---------------------------------------------------------------- hero
st.markdown(f"""
<div class="hero">
  <div>
    <div class="ticker" role="heading" aria-level="1">{ticker}</div>
    <div class="hero-sub">Next-day closing price, predicted vs actual</div>
  </div>
  <div class="price">
    <div class="v">${last['target_close']:,.2f}</div>
    <div class="d" style="color:{UP if chg >= 0 else DOWN}">{'+' if chg >= 0 else '−'}{abs(chg) * 100:.1f}% over the test period</div>
    <div class="l">Last close, {last['next_date']:%b %d, %Y}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- verdict + scoreboard
order = [NAIVE, "Linear Regression", "Random Forest"]
rows = [(k, m.loc[k]) for k in order if k in m.index]
best_model = min((r for r in rows if r[0] != NAIVE), key=lambda r: r[1]["rmse"])
naive_rmse = m.loc[NAIVE, "rmse"]
gap = best_model[1]["rmse"] - naive_rmse
if gap < -0.005:
    verdict = (f"{best_model[0]} was the most accurate here, off by <b>${best_model[1]['mae']:.2f}</b> on a typical day "
               f"and <b>${-gap:.2f}</b> better than simply guessing tomorrow = today.")
else:
    verdict = (f"Neither model beat the simple guess for {ticker}. The best one, {best_model[0]}, "
               f"was off by <b>${best_model[1]['mae']:.2f}</b> on a typical day; guessing tomorrow = today "
               f"was off by <b>${m.loc[NAIVE, 'mae']:.2f}</b>.")
st.markdown(f"<p class='verdict'>{verdict}</p>", unsafe_allow_html=True)

swatch = {NAIVE: MUTED, "Linear Regression": LR, "Random Forest": RF}
label = {NAIVE: "Simple guess (tomorrow = today)"}
lowest = min(rows, key=lambda r: r[1]["rmse"])[0]
body = ""
for k, r in rows:
    dir_txt = "—" if pd.isna(r["direction_acc"]) else f"{r['direction_acc']:.1f}%"
    badge = "<span class='badge'>Lowest error</span>" if k == lowest else ""
    body += (f"<tr class='{'best' if k == lowest else ''}'>"
             f"<td><span class='swatch' style='background:{swatch[k]}'></span>{label.get(k, k)}{badge}</td>"
             f"<td class='num'>${r['rmse']:.2f}</td><td class='num'>${r['mae']:.2f}</td>"
             f"<td class='num'>{r['mape']:.2f}%</td><td class='num'>{dir_txt}</td></tr>")
st.markdown(f"""
<table class="board">
<thead><tr><th>Model</th><th class="num">RMSE</th><th class="num">Avg. error (MAE)</th>
<th class="num">Error %</th><th class="num">Up/down correct</th></tr></thead>
<tbody>{body}</tbody></table>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- main chart
FONT = dict(family="IBM Plex Sans, sans-serif", color=INK, size=12)
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.62, 0.19, 0.19], vertical_spacing=0.05,
                    subplot_titles=("Closing price", "RSI (14 days)", "MACD"))
fig.add_trace(go.Scatter(x=d["next_date"], y=d["target_close"], name="Actual",
                         line=dict(color=INK, width=2.2)), 1, 1)
for model in models:
    fig.add_trace(go.Scatter(x=d["next_date"], y=d[MODEL_COLS[model]], name=model,
                             line=dict(color=MODEL_COLORS[model], width=1.4)), 1, 1)
for ind, dash in zip(show_ind, ["dot", "dash"]):
    fig.add_trace(go.Scatter(x=d["date"], y=d[ind], name=ind.replace("_", " "),
                             line=dict(color=MUTED, width=1, dash=dash)), 1, 1)
if period == "Full history":
    split_date = first_test["date"]
    fig.add_vrect(x0=split_date, x1=d["next_date"].max(), fillcolor=LR, opacity=0.05, line_width=0, row=1, col=1)
    fig.add_annotation(x=split_date, y=1, yref="y domain", text="  Testing starts", showarrow=False,
                       xanchor="left", font=dict(color=LR, size=11), row=1, col=1)
fig.add_hrect(y0=30, y1=70, fillcolor=LINE, opacity=0.35, line_width=0, row=2, col=1)
fig.add_trace(go.Scatter(x=d["date"], y=d["RSI_14"], name="RSI", line=dict(color=INK, width=1.2),
                         showlegend=False), 2, 1)
hist = d["MACD"] - d["MACD_signal"]
fig.add_trace(go.Bar(x=d["date"], y=hist, name="MACD − signal", showlegend=False,
                     marker_color=np.where(hist >= 0, UP, DOWN), opacity=0.55), 3, 1)
fig.add_trace(go.Scatter(x=d["date"], y=d["MACD"], name="MACD", line=dict(color=INK, width=1.1),
                         showlegend=False), 3, 1)
fig.add_trace(go.Scatter(x=d["date"], y=d["MACD_signal"], name="Signal", line=dict(color=RF, width=1.1),
                         showlegend=False), 3, 1)
fig.update_layout(height=760, hovermode="x unified", font=FONT, plot_bgcolor=PAPER, paper_bgcolor=PAPER,
                  legend=dict(orientation="h", y=1.0, yanchor="bottom", x=1, xanchor="right", font=dict(size=12)),
                  margin=dict(t=40, b=30, l=60, r=20), bargap=0)
fig.update_xaxes(showgrid=False, linecolor=LINE, ticks="outside", tickcolor=LINE)
fig.update_yaxes(gridcolor="#EDF0F4", zeroline=False, linecolor=LINE)
fig.update_yaxes(tickprefix="$", row=1, col=1)
fig.update_yaxes(range=[0, 100], tickvals=[30, 70], row=2, col=1)
for a in fig.layout.annotations:
    if a.text in ("Closing price", "RSI (14 days)", "MACD"):
        a.update(x=0, xanchor="left", font=dict(size=13, color=MUTED))
st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

# ---------------------------------------------------------------- tabs
tab1, tab2, tab3 = st.tabs(["Daily errors", "All companies", "Speed tests (Part 1 & 2)"])
with tab1:
    e = d[["next_date", "target_close", "pred_linear", "pred_rf"]].rename(
        columns={"next_date": "Date", "target_close": "Actual", "pred_linear": "Linear Regression",
                 "pred_rf": "Random Forest"})
    ef = go.Figure()
    for model in MODEL_COLS:
        ef.add_trace(go.Histogram(x=e[model] - e["Actual"], name=model, opacity=0.6,
                                  marker_color=MODEL_COLORS[model], nbinsx=50))
    ef.add_vline(x=0, line_color=INK, line_width=1)
    ef.update_layout(barmode="overlay", height=320, font=FONT, plot_bgcolor=PAPER, paper_bgcolor=PAPER,
                     title=dict(text="How far off each prediction was (predicted − actual, $)",
                                font=dict(size=14)), margin=dict(t=50, b=30, l=50, r=20),
                     legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))
    ef.update_yaxes(title="Days", gridcolor="#EDF0F4")
    st.plotly_chart(ef, width="stretch", config={"displaylogo": False})
    show = e.tail(30).iloc[::-1].copy()
    show["Date"] = show["Date"].dt.date
    st.dataframe(show.round(2), width="stretch", hide_index=True)
    st.download_button("Download predictions as CSV", e.to_csv(index=False), f"{ticker}_predictions.csv")
with tab2:
    st.markdown("**Average over all 505 companies (test period)**")
    avg = mets.groupby("model")[["rmse", "mae", "mape", "direction_acc"]].mean().reindex(order).round(3)
    avg.index = [label.get(i, i) for i in avg.index]
    st.dataframe(avg.rename(columns={"rmse": "RMSE ($)", "mae": "MAE ($)", "mape": "Error %",
                                     "direction_acc": "Up/down correct %"}), width="stretch")
    pick = st.selectbox("Rank companies by error % for", list(MODEL_COLS))
    rank = (mets[mets["model"] == pick].sort_values("mape")[["name", "rmse", "mae", "mape", "direction_acc"]]
            .rename(columns={"name": "Ticker", "rmse": "RMSE ($)", "mae": "MAE ($)", "mape": "Error %",
                             "direction_acc": "Up/down correct %"}).round(3))
    st.dataframe(rank, width="stretch", hide_index=True, height=400)
with tab3:
    b1 = load_optional("part1_storage_benchmark.csv")
    b2 = load_optional("part2_pandas_vs_polars.csv")
    if b1 is not None:
        st.markdown("**CSV vs Parquet at 1x, 10x and 100x**")
        st.dataframe(b1, width="stretch", hide_index=True)
    if b2 is not None:
        st.markdown("**Pandas vs Polars, seconds per step**")
        st.dataframe(b2, width="stretch", hide_index=True)
