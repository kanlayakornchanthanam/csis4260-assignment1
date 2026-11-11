"""
Part 3 - Streamlit dashboard for the Part 2 price-prediction models.

Run:  python -m streamlit run part3_dashboard/app.py
(Run part2_analysis/run_part2.py first so results/predictions.parquet exists.)
"""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
DATA = ROOT / "data" / "all_stocks_5yr.csv"

# ---------------------------------------------------------------- design tokens
INK = "#17171C"
MUTED = "#8A8C99"
HAIR = "#ECECF3"
MINT, MINT_SOFT = "#3FC8A0", "#E2F6EF"
BUTTER = "#F6CF5B"
CORAL, CORAL_SOFT = "#EB7A5A", "#FCEAE4"
VIOLET = "#8C7CF0"
MODEL_COLS = {"Linear Regression": "pred_linear", "Random Forest": "pred_rf"}
MODEL_LINE = {"Linear Regression": INK, "Random Forest": VIOLET}
NAIVE = "Naive (today=tomorrow)"
FONT = "Geist, Inter, system-ui, sans-serif"

st.set_page_config(page_title="Stock Prediction Back-test", page_icon="📈", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600&display=swap');
html, body, .stApp, .stMarkdown, p, label, input, button, li, td, th {{ font-family: {FONT}; }}
.stApp {{
  background:
    radial-gradient(1200px 700px at 85% 10%, #F6EAF0 0%, rgba(246,234,240,0) 60%),
    radial-gradient(900px 600px at 10% 90%, #E7ECF8 0%, rgba(231,236,248,0) 60%),
    #ECECF6;
}}
header[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {{ display: none; }}
.block-container, [data-testid="stMainBlockContainer"] {{ padding: 1.6rem 2rem 2rem; max-width: 1360px; }}
[data-testid="stVerticalBlock"] {{ gap: .85rem; }}

/* frosted panels */
.st-key-chart_card, .st-key-side_panel, .st-key-mini_0, .st-key-mini_1, .st-key-mini_2, .st-key-mini_3,
.st-key-table_card, .st-key-bench_card {{
  background: rgba(255,255,255,.62); border: 1px solid rgba(255,255,255,.85);
  border-radius: 22px; padding: 18px 20px; backdrop-filter: blur(18px);
  box-shadow: 0 1px 2px rgba(23,23,28,.04), 0 12px 32px -18px rgba(60,60,100,.25);
}}
.st-key-side_panel {{ padding: 14px; }}
.st-key-mini_0, .st-key-mini_1, .st-key-mini_2, .st-key-mini_3 {{ background: rgba(255,255,255,.92); padding: 16px 18px 8px; }}

/* top bar */
.brand {{ display: flex; align-items: center; gap: .55rem; font-weight: 600; font-size: 1.2rem; color: {INK};
         letter-spacing: -.01em; }}
.brand .ring {{ width: 20px; height: 20px; border-radius: 50%; border: 3.5px solid {MINT}; }}

/* segmented controls (nav, timeframe, model toggle) */
[data-testid="stButtonGroup"] {{ gap: 4px; }}
[data-testid="stButtonGroup"] button {{
  border: none !important; background: transparent !important; color: {INK} !important;
  border-radius: 999px !important; padding: .32rem .9rem !important;
  min-height: 0 !important; box-shadow: none !important; }}
[data-testid="stButtonGroup"] button p {{ font-size: .86rem !important; }}
[data-testid="stButtonGroup"] button[aria-checked="true"] {{ background: #fff !important;
  box-shadow: 0 1px 3px rgba(23,23,28,.10) !important; }}
[data-testid="stButtonGroup"] button[aria-checked="true"] p {{ font-weight: 500; }}
.st-key-nav button[aria-checked="true"] {{
  background: #fff !important; box-shadow: 0 1px 3px rgba(23,23,28,.08) !important; }}
.st-key-nav button[aria-checked="true"] p {{ font-weight: 500; }}
.st-key-nav [role="radiogroup"] {{ justify-content: flex-start; }}
.st-key-tf [role="radiogroup"] {{ justify-content: flex-end; }}
.st-key-tf [data-testid="stButtonGroup"] button {{ background: #EFEFF5 !important; padding: .2rem .7rem !important; }}
.st-key-tf [data-testid="stButtonGroup"] button p {{ font-size: .78rem !important; color: {MUTED}; }}
.st-key-tf button[aria-checked="true"] {{ background: {INK} !important; }}
.st-key-tf button[aria-checked="true"] p {{ color: #fff !important; }}
.st-key-model_toggle [data-testid="stButtonGroup"], .st-key-model_toggle [role="radiogroup"] {{ width: 100% !important; }}
.st-key-model_toggle button {{ flex: 1 1 0 !important; max-width: none !important; width: auto !important; }}
.st-key-model_toggle button * {{ max-width: none !important; overflow: visible !important;
  text-overflow: clip !important; white-space: nowrap !important; }}
.st-key-model_toggle [data-testid="stElementContainer"], .st-key-model_toggle [data-testid="stButtonGroup"] > div {{
  width: 100% !important; max-width: none !important; }}
.st-key-model_toggle [role="radiogroup"] {{ background: #EFEFF5; border-radius: 999px; padding: 3px;
  display: flex; width: 100%; flex-wrap: nowrap; gap: 2px; }}
.st-key-model_toggle [data-testid="stButtonGroup"] button {{ flex: 1 1 0; }}
.st-key-model_toggle button[aria-checked="true"] {{ background: {INK} !important; }}
.st-key-model_toggle button[aria-checked="true"] p {{ color: #fff !important; font-weight: 500; }}

/* ticker picker */
.st-key-picker .react-aria-ComboBox [role="group"] {{ border-radius: 999px; background: rgba(255,255,255,.85);
  border: 1px solid #fff; min-height: 40px; }}

/* chart header */
.ch-head {{ display: flex; align-items: center; gap: .6rem; color: {INK}; font-size: 1.05rem; font-weight: 500; }}
.ch-icon {{ width: 28px; height: 28px; border-radius: 50%; background: {INK}; color: #fff; display: grid;
           place-items: center; font-size: .8rem; font-weight: 600; }}
.ch-price {{ display: flex; align-items: center; gap: .8rem; margin-top: .35rem; }}
.big {{ font-size: 3.1rem; font-weight: 400; letter-spacing: -.035em; color: {INK}; line-height: 1;
       font-variant-numeric: tabular-nums; }}
.pill {{ display: inline-flex; align-items: center; gap: .25rem; font-size: .76rem; font-weight: 500;
        padding: .22rem .6rem; border-radius: 999px; font-variant-numeric: tabular-nums; white-space: nowrap; }}
.pill.up {{ background: {MINT_SOFT}; color: #1E9C78; }}
.pill.down {{ background: {CORAL_SOFT}; color: {CORAL}; }}
.pill.flat {{ background: #EFEFF5; color: {MUTED}; }}
.legend {{ display: flex; gap: 1.1rem; color: {MUTED}; font-size: .8rem; flex-wrap: wrap; }}
.legend i {{ display: inline-block; width: 16px; height: 2px; vertical-align: middle; margin-right: .4rem; }}
.legend b {{ display: inline-block; width: 7px; height: 12px; border-radius: 2px; vertical-align: middle;
            margin-right: 2px; }}

/* side panel cards */
.card {{ background: #fff; border-radius: 16px; padding: 14px 16px; margin-bottom: .35rem; }}
.card .lbl {{ display: flex; justify-content: space-between; align-items: center; gap: .5rem; color: {MUTED};
             font-size: .82rem; }}
.card .num {{ font-size: 2rem; font-weight: 400; letter-spacing: -.03em; color: {INK}; margin: .2rem 0 .55rem;
             font-variant-numeric: tabular-nums; line-height: 1.1; }}
.chips {{ display: flex; gap: 6px; flex-wrap: wrap; }}
.chip {{ background: #F3F3F7; color: #4A4C57; font-size: .74rem; padding: .22rem .6rem; border-radius: 999px;
        font-variant-numeric: tabular-nums; }}
.chip.dark {{ background: {INK}; color: #fff; }}
.track {{ position: relative; height: 12px; background: #F0F0F5; border-radius: 999px; margin: .3rem 0 .35rem; }}
.track .fill {{ position: absolute; top: 0; bottom: 0; left: 0; background: {INK}; border-radius: 999px; }}
.track .knob {{ position: absolute; top: -3px; width: 18px; height: 18px; border-radius: 50%; background: #fff;
               border: 4px solid {INK}; transform: translateX(-50%); box-sizing: border-box; }}
.track .mark {{ position: absolute; top: -4px; bottom: -4px; width: 2px; background: {CORAL}; }}
.ticks {{ display: flex; justify-content: space-between; color: {MUTED}; font-size: .72rem; }}
.kv {{ display: grid; grid-template-columns: 1fr auto; row-gap: .5rem; font-size: .84rem; padding: .2rem .3rem .5rem; }}
.kv span:nth-child(odd) {{ color: {MUTED}; }}
.kv span:nth-child(even) {{ color: {INK}; text-align: right; font-variant-numeric: tabular-nums; }}

.st-key-side_panel [data-testid="stDownloadButton"] button {{
  width: 100%; background: {INK}; border: none; border-radius: 999px; min-height: 44px; }}
.st-key-side_panel [data-testid="stDownloadButton"] button p {{ color: #fff; font-weight: 500; }}
.st-key-side_panel [data-testid="stDownloadButton"] button:hover {{ background: #2B2B33; }}

/* mini cards */
.mini-head {{ display: flex; justify-content: space-between; align-items: center; gap: .4rem;
             font-size: .86rem; color: {INK}; }}
.mini-head .t {{ display: flex; align-items: center; gap: .5rem; white-space: nowrap; }}
.mini-head .dot {{ width: 22px; height: 22px; border-radius: 50%; display: grid; place-items: center;
                  color: #fff; font-size: .62rem; font-weight: 600; }}
.mini-num {{ font-size: 1.9rem; font-weight: 400; letter-spacing: -.03em; color: {INK}; margin-top: .55rem;
            font-variant-numeric: tabular-nums; }}

.verdict {{ font-size: .95rem; color: {INK}; line-height: 1.5; max-width: 70ch; }}
:focus-visible {{ outline: 2px solid {VIOLET} !important; outline-offset: 2px; }}
@media (max-width: 900px) {{ .big {{ font-size: 2.3rem; }} }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data
@st.cache_data
def load():
    preds = pd.read_parquet(RES / "predictions.parquet")
    preds["date"] = pd.to_datetime(preds["date"])
    mets = pd.read_csv(RES / "metrics_by_ticker.csv")
    ohlc = pd.read_csv(DATA, parse_dates=["date"])
    return preds, mets, ohlc


@st.cache_data
def load_optional(name):
    p = RES / name
    return pd.read_csv(p) if p.exists() else None


if not (RES / "predictions.parquet").exists():
    st.error("Prediction file not found. Run `python part2_analysis/run_part2.py` first, then reload this page.")
    st.stop()

preds, mets, ohlc = load()
tickers = sorted(preds["name"].unique())


def pill(value, text=None):
    cls = "flat" if abs(value) < 1e-9 else ("up" if value > 0 else "down")
    arrow = "▲" if value > 0 else ("▼" if value < 0 else "")
    return f"<span class='pill {cls}'>{arrow} {text if text is not None else f'{abs(value):.2f}%'}</span>"


def sparkline(y, color):
    y = pd.Series(y).dropna().reset_index(drop=True)
    f = go.Figure(go.Scatter(y=y.values, mode="lines", line=dict(color=color, width=2, shape="spline")))
    f.add_trace(go.Scatter(x=[len(y) - 1], y=[y.iloc[-1]], mode="markers",
                           marker=dict(size=8, color="#fff", line=dict(color=color, width=2))))
    f.update_layout(height=64, margin=dict(l=0, r=6, t=4, b=4), showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(visible=False), yaxis=dict(visible=False))
    return f


# ---------------------------------------------------------------- top bar
c2, c3 = st.columns([3, 1.2], vertical_alignment="center")
with c2:
    with st.container(key="nav"):
        view = st.segmented_control("View", ["Predictions", "All companies", "Speed tests"],
                                    default="Predictions", label_visibility="collapsed") or "Predictions"
with c3:
    with st.container(key="picker"):
        ticker = st.selectbox("Company", tickers, index=tickers.index("AAPL") if "AAPL" in tickers else 0,
                              label_visibility="collapsed", help="Type to search all 505 companies")

# ================================================================ PREDICTIONS VIEW
if view == "Predictions":
    full = preds[preds["name"] == ticker].sort_values("date").copy()
    full["next_date"] = full["date"].shift(-1)
    full = full.dropna(subset=["next_date"])
    px = ohlc[ohlc["name"] == ticker].set_index("date")
    m = mets[mets["name"] == ticker].set_index("model")
    test = full[full["split"] == "test"]

    left, right = st.columns([2.55, 1], gap="medium")

    # ---------------- right panel (built first so the chart can use the chosen model)
    with right:
        with st.container(key="side_panel"):
            with st.container(key="model_toggle"):
                model = st.segmented_control("Model", list(MODEL_COLS), default="Linear Regression",
                                             format_func=lambda x: "Linear" if x.startswith("Linear") else x,
                                             label_visibility="collapsed") or "Linear Regression"
            r, nv = m.loc[model], m.loc[NAIVE]
            col = MODEL_COLS[model]
            last = test.iloc[-1]
            off = last[col] - last["target_close"]
            beats = r["mae"] < nv["mae"] - 0.005
            verdict_pill = ("<span class='pill up'>Beats simple guess</span>" if beats
                            else "<span class='pill down'>Below simple guess</span>")

            st.markdown(f"""
<div class="card"><div class="lbl">Last prediction<span>{last['next_date']:%b %d, %Y}</span></div>
  <div class="num">${last[col]:,.2f}</div>
  <div class="chips"><span class="chip">Actual ${last['target_close']:,.2f}</span>
  <span class="chip">Off by ${abs(off):,.2f}</span></div></div>
""", unsafe_allow_html=True)
            st.markdown(f"""
<div class="card"><div class="lbl">Average daily error{verdict_pill}</div>
  <div class="num">${r['mae']:,.2f}</div>
  <div class="chips"><span class="chip">RMSE ${r['rmse']:.2f}</span><span class="chip">{r['mape']:.2f}% of price</span>
  <span class="chip dark">Simple guess ${nv['mae']:.2f}</span></div></div>
""", unsafe_allow_html=True)
            acc = float(r["direction_acc"])
            st.markdown(f"""
<div class="card"><div class="lbl">Up or down called correctly
  <span class="pill {'up' if acc > 52 else 'flat'}">{'Better than' if acc > 52 else 'Close to'} a coin flip</span></div>
  <div class="num">{acc:.1f}%</div>
  <div class="track"><div class="fill" style="width:{acc}%"></div><div class="mark" style="left:50%"></div>
  <div class="knob" style="left:{acc}%"></div></div>
  <div class="ticks"><span>0</span><span>25</span><span>50</span><span>75</span><span>100</span></div></div>
""", unsafe_allow_html=True)

            e = test[["next_date", "target_close", "pred_linear", "pred_rf"]].rename(
                columns={"next_date": "date", "target_close": "actual"})
            st.download_button("Download predictions", e.to_csv(index=False), f"{ticker}_predictions.csv",
                               width="stretch")
            st.markdown(f"""
<div class="kv">
  <span>Trained on</span><span>{full['date'].min():%b %Y} to {test['date'].min():%b %Y}</span>
  <span>Tested on</span><span>{test['date'].min():%b %Y} to {test['next_date'].max():%b %Y}</span>
  <span>Test days</span><span>{len(test):,}</span>
  <span>Linear Regression error</span><span>${m.loc['Linear Regression', 'mae']:.2f}</span>
  <span>Random Forest error</span><span>${m.loc['Random Forest', 'mae']:.2f}</span>
  <span>Simple guess error</span><span>${nv['mae']:.2f}</span>
</div>""", unsafe_allow_html=True)

    # ---------------- chart card
    with left:
        with st.container(key="chart_card"):
            h1, h2 = st.columns([1.4, 1], vertical_alignment="top")
            with h2:
                with st.container(key="tf"):
                    tf = st.segmented_control("Range", ["1M", "3M", "6M", "Test", "All"], default="3M",
                                              label_visibility="collapsed") or "3M"
            days = {"1M": 21, "3M": 63, "6M": 126}
            d = full.tail(days[tf]) if tf in days else (test if tf == "Test" else full)
            d = d.merge(px[["open", "high", "low", "close"]].add_prefix("px_"),
                        left_on="next_date", right_index=True, how="left")
            first_c, last_c = d["px_close"].iloc[0], d["px_close"].iloc[-1]
            chg = (last_c / first_c - 1) * 100
            with h1:
                st.markdown(f"""
<div class="ch-head"><span class="ch-icon">{ticker[0]}</span>{ticker}
<span style="color:{MUTED};font-weight:400">S&amp;P 500</span></div>
<div class="ch-price"><span class="big">{last_c:,.2f}</span>{pill(chg)}</div>
""", unsafe_allow_html=True)

            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=d["next_date"], open=d["px_open"], high=d["px_high"], low=d["px_low"], close=d["px_close"],
                name="Actual", increasing=dict(line=dict(color=MINT, width=1), fillcolor=MINT),
                decreasing=dict(line=dict(color=BUTTER, width=1), fillcolor=BUTTER), whiskerwidth=0))
            fig.add_trace(go.Scatter(x=d["next_date"], y=d[col], name="Predicted", mode="lines",
                                     line=dict(color=MODEL_LINE[model], width=1.6),
                                     hovertemplate="$%{y:,.2f}"))
            if tf == "All":
                sd = test["date"].min()
                fig.add_vrect(x0=sd, x1=d["next_date"].max(), fillcolor=VIOLET, opacity=0.07, line_width=0)
                fig.add_annotation(x=sd, y=1, yref="paper", text=" Testing starts", showarrow=False,
                                   xanchor="left", font=dict(color=VIOLET, size=11))
            fig.add_hline(y=last_c, line=dict(color=INK, width=1, dash="dash"))
            fig.add_annotation(x=1, xref="paper", y=last_c, text=f" {last_c:,.2f} ", showarrow=False,
                               xanchor="left", bgcolor=INK, font=dict(color="#fff", size=10), borderpad=3)
            fig.update_layout(
                height=430, margin=dict(l=52, r=58, t=8, b=24), showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family=FONT, color=MUTED, size=11), hovermode="x unified",
                hoverlabel=dict(bgcolor="#fff", bordercolor=HAIR, font=dict(family=FONT, color=INK)),
                xaxis=dict(rangeslider=dict(visible=False), showgrid=False, showline=False,
                           showspikes=True, spikemode="across", spikedash="dot", spikecolor=MUTED,
                           spikethickness=1),
                yaxis=dict(showgrid=False, zeroline=False, tickprefix="$"))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            st.markdown(f"""
<div class="legend"><span><b style="background:{MINT}"></b><b style="background:{BUTTER}"></b>
 Actual price (up day, down day)</span>
<span><i style="background:{MODEL_LINE[model]}"></i>{model} prediction</span></div>
""", unsafe_allow_html=True)

        # ---------------- indicator cards
        cur = full.iloc[-1]
        vol = px["volume"]
        rsi = cur["RSI_14"]
        rsi_pill = ("<span class='pill down'>Overbought</span>" if rsi > 70 else
                    "<span class='pill up'>Oversold</span>" if rsi < 30 else "<span class='pill flat'>Neutral</span>")
        macd_diff = cur["MACD"] - cur["MACD_signal"]
        sma_gap = (cur["close"] / cur["SMA_20"] - 1) * 100
        vol_gap = (vol.iloc[-1] / vol.tail(20).mean() - 1) * 100
        tail = full.tail(60)
        cards = [
            ("RSI 14", "R", VIOLET, f"{rsi:.1f}", rsi_pill, tail["RSI_14"], VIOLET),
            ("MACD", "M", INK, f"{cur['MACD']:+.2f}",
             f"<span class='pill {'up' if macd_diff > 0 else 'down'}'>"
             f"{'Bullish' if macd_diff > 0 else 'Bearish'}</span>",
             tail["MACD"], MINT if macd_diff > 0 else CORAL),
            ("SMA 20", "S", MINT, f"${cur['SMA_20']:,.2f}",
             pill(sma_gap, f"{abs(sma_gap):.1f}% {'below' if sma_gap > 0 else 'above'} price"),
             tail["SMA_20"], MINT),
            ("Volume", "V", BUTTER, f"{vol.iloc[-1] / 1e6:,.1f}M",
             pill(vol_gap, f"{abs(vol_gap):.0f}% vs avg"), vol.tail(60), BUTTER),
        ]
        mc = st.columns(4, gap="small")
        for i, (name, letter, dot, val, badge, series, line) in enumerate(cards):
            with mc[i]:
                with st.container(key=f"mini_{i}"):
                    st.markdown(f"""
<div class="mini-head"><span class="t"><span class="dot" style="background:{dot}">{letter}</span>{name}</span>
{badge}</div><div class="mini-num">{val}</div>""", unsafe_allow_html=True)
                    st.plotly_chart(sparkline(series, line), width="stretch",
                                    config={"staticPlot": True}, key=f"spark_{i}")

# ================================================================ ALL COMPANIES VIEW
elif view == "All companies":
    with st.container(key="table_card"):
        avg = mets.groupby("model")[["rmse", "mae", "mape", "direction_acc"]].mean()
        st.markdown("<div class='ch-head'>Average over all 505 companies, test period</div>",
                    unsafe_allow_html=True)
        cols = st.columns(3)
        for c, (k, lab) in zip(cols, [(NAIVE, "Simple guess"), ("Linear Regression", "Linear Regression"),
                                      ("Random Forest", "Random Forest")]):
            a = avg.loc[k]
            dacc = ("" if pd.isna(a["direction_acc"])
                    else f"<span class='chip'>Up/down {a['direction_acc']:.1f}%</span>")
            c.markdown(f"""<div class="card"><div class="lbl">{lab}</div><div class="num">${a['mae']:.2f}</div>
<div class="chips"><span class="chip">RMSE ${a['rmse']:.2f}</span><span class="chip">{a['mape']:.2f}% of price</span>
{dacc}</div></div>""", unsafe_allow_html=True)
        st.markdown("<p class='verdict'>The big number is the average daily error in dollars. Both models follow "
                    "prices closely, but on average neither beats simply guessing that tomorrow's price equals "
                    "today's.</p>", unsafe_allow_html=True)
        with st.container(key="tf2"):
            pick = st.segmented_control("Rank companies by error for", list(MODEL_COLS),
                                        default="Linear Regression") or "Linear Regression"
        rank = (mets[mets["model"] == pick].sort_values("mape")[["name", "rmse", "mae", "mape", "direction_acc"]]
                .rename(columns={"name": "Ticker", "rmse": "RMSE ($)", "mae": "Avg. error ($)",
                                 "mape": "Error %", "direction_acc": "Up/down correct %"}).round(3))
        st.dataframe(rank, width="stretch", hide_index=True, height=460)

# ================================================================ SPEED TESTS VIEW
else:
    b1 = load_optional("part1_storage_benchmark.csv")
    b2 = load_optional("part2_pandas_vs_polars.csv")
    with st.container(key="bench_card"):
        if b1 is not None:
            st.markdown("<div class='ch-head'>CSV vs Parquet</div>", unsafe_allow_html=True)
            scale = st.segmented_control("Data size", ["1x", "10x", "100x"], default="100x") or "100x"
            s = b1[b1["scale"] == scale]
            bf = go.Figure(go.Bar(
                x=s["format"], y=s["read_full_s"], marker=dict(
                    color=[MINT if f.startswith("parquet") else BUTTER for f in s["format"]],
                    cornerradius=8),
                text=[f"{v:.2f}s" for v in s["read_full_s"]], textposition="outside"))
            bf.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=30), paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT, color=MUTED),
                             title=dict(text="Time to read the whole file (seconds)",
                                        font=dict(size=13, color=INK)),
                             yaxis=dict(showgrid=False, visible=False))
            st.plotly_chart(bf, width="stretch", config={"displayModeBar": False})
            st.dataframe(s, width="stretch", hide_index=True)
        if b2 is not None:
            st.markdown("<div class='ch-head' style='margin-top:1rem'>Pandas vs Polars, seconds per step</div>",
                        unsafe_allow_html=True)
            st.dataframe(b2.round(3), width="stretch", hide_index=True)
