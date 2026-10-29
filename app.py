"""Streamlit front end. Run with:  streamlit run app.py"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from stockanalysis import indicators as ind
from stockanalysis.data import DataError, PriceRequest, load_prices
from stockanalysis.forecast import walk_forward_forecast

st.set_page_config(page_title="Stock Analysis Tool", page_icon="📈", layout="wide")

UP, DOWN, ACCENT, MUTED = "#2e9e6b", "#d1495b", "#3b6fd8", "#8a8f98"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch(ticker: str, start: date, end: date) -> pd.DataFrame:
    return load_prices(PriceRequest(ticker, start, end))


def pct(x: float) -> str:
    return "–" if x is None or np.isnan(x) else f"{x:+.1%}" if abs(x) < 10 else f"{x:+.0%}"


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker", value="AAPL", help="Any Yahoo Finance symbol, e.g. MSFT, SHOP.TO, ^GSPC").strip()
    benchmark = st.text_input("Benchmark", value="SPY").strip()
    today = date.today()
    start = st.date_input("Start", value=today - timedelta(days=365 * 5), max_value=today - timedelta(days=2))
    end = st.date_input("End", value=today, max_value=today)
    st.divider()
    st.subheader("Chart overlays")
    sma_windows = st.multiselect("Moving averages (days)", [10, 20, 50, 100, 200], default=[50, 200])
    show_bb = st.checkbox("Bollinger bands (20, 2σ)")
    st.caption("Data: Yahoo Finance via yfinance, adjusted for splits and dividends.")

st.title(f"📈 {ticker.upper() or 'Stock'} analysis")

try:
    prices = fetch(ticker, start, end)
except DataError as exc:
    st.error(str(exc))
    st.stop()

close = prices["Close"]
bench_close = None
if benchmark:
    try:
        bench_close = fetch(benchmark, start, end)["Close"]
    except DataError as exc:
        st.warning(f"Benchmark unavailable: {exc}")

# ---------------------------------------------------------------- headline metrics
risk = ind.risk_summary(close)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Last close", f"{close.iloc[-1]:,.2f}", pct(close.iloc[-1] / close.iloc[-2] - 1))
c2.metric("Total return", pct(risk["Total return"]))
c3.metric("CAGR", pct(risk["Annualised return (CAGR)"]))
c4.metric("Volatility (ann.)", f"{risk['Annualised volatility']:.1%}")
c5.metric("Max drawdown", pct(risk["Max drawdown"]))

overview, risk_tab, forecast_tab, data_tab = st.tabs(["Chart", "Risk & benchmark", "Forecast", "Data"])

# ---------------------------------------------------------------- chart tab
with overview:
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.55, 0.15, 0.15, 0.15],
        subplot_titles=("Price", "Volume", "RSI (14)", "MACD (12, 26, 9)"),
    )
    fig.add_trace(
        go.Candlestick(
            x=prices.index,
            open=prices["Open"],
            high=prices["High"],
            low=prices["Low"],
            close=close,
            name="Price",
            increasing_line_color=UP,
            decreasing_line_color=DOWN,
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    for w in sorted(sma_windows):
        fig.add_trace(
            go.Scatter(x=close.index, y=ind.simple_moving_average(close, w), name=f"SMA {w}", line=dict(width=1.4)),
            row=1,
            col=1,
        )
    if show_bb:
        bb = ind.bollinger_bands(close)
        fig.add_trace(
            go.Scatter(
                x=bb.index, y=bb["Upper"], line=dict(color=MUTED, width=0.8), name="Bollinger", legendgroup="bb"
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=bb.index,
                y=bb["Lower"],
                line=dict(color=MUTED, width=0.8),
                fill="tonexty",
                fillcolor="rgba(138,143,152,0.12)",
                showlegend=False,
                legendgroup="bb",
            ),
            row=1,
            col=1,
        )
    vol_colors = np.where(close.diff().fillna(0) >= 0, UP, DOWN)
    fig.add_trace(go.Bar(x=prices.index, y=prices["Volume"], marker_color=vol_colors, showlegend=False), row=2, col=1)
    fig.add_trace(
        go.Scatter(x=close.index, y=ind.rsi(close), line=dict(color=ACCENT, width=1.2), showlegend=False), row=3, col=1
    )
    for lvl in (30, 70):
        fig.add_hline(y=lvl, line=dict(color=MUTED, dash="dot", width=1), row=3, col=1)
    m = ind.macd(close)
    fig.add_trace(
        go.Bar(x=m.index, y=m["Histogram"], marker_color=np.where(m["Histogram"] >= 0, UP, DOWN), showlegend=False),
        row=4,
        col=1,
    )
    fig.add_trace(go.Scatter(x=m.index, y=m["MACD"], line=dict(color=ACCENT, width=1.2), name="MACD"), row=4, col=1)
    fig.add_trace(
        go.Scatter(x=m.index, y=m["Signal"], line=dict(color="#e0a030", width=1.2), name="Signal"), row=4, col=1
    )
    fig.update_layout(
        height=820,
        xaxis_rangeslider_visible=False,
        margin=dict(t=40, b=10, l=10, r=10),
        legend=dict(orientation="h", y=1.04, x=0),
        hovermode="x unified",
    )
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- risk tab
with risk_tab:
    left, right = st.columns([2, 1])
    with left:
        growth = pd.DataFrame({ticker.upper(): close / close.iloc[0] * 100})
        if bench_close is not None:
            aligned = bench_close.reindex(close.index).ffill()
            growth[benchmark.upper()] = aligned / aligned.dropna().iloc[0] * 100
        gfig = go.Figure()
        for i, col in enumerate(growth.columns):
            gfig.add_trace(
                go.Scatter(x=growth.index, y=growth[col], name=col, line=dict(color=[ACCENT, MUTED][i % 2], width=1.8))
            )
        gfig.update_layout(
            title="Growth of 100", height=360, margin=dict(t=40, b=10, l=10, r=10), hovermode="x unified"
        )
        st.plotly_chart(gfig, width="stretch")

        dd = ind.drawdown(close)
        dfig = go.Figure(
            go.Scatter(
                x=dd.index,
                y=dd * 100,
                fill="tozeroy",
                line=dict(color=DOWN, width=1),
                fillcolor="rgba(209,73,91,0.18)",
                name="Drawdown",
            )
        )
        dfig.update_layout(title="Drawdown from peak (%)", height=260, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(dfig, width="stretch")
    with right:
        rows = {k: (f"{v:.2f}" if k == "Sharpe ratio" else pct(v)) for k, v in risk.items()}
        if bench_close is not None:
            beta, corr = ind.beta_and_correlation(close, bench_close)
            rows[f"Beta vs {benchmark.upper()}"] = f"{beta:.2f}"
            rows[f"Correlation vs {benchmark.upper()}"] = f"{corr:.2f}"
        st.dataframe(pd.Series(rows, name="Value").to_frame(), width="stretch")
        st.caption("Sharpe ratio assumes a 0% risk-free rate. Annualised with 252 trading days.")

        rets = close.pct_change().dropna() * 100
        hfig = go.Figure(go.Histogram(x=rets, nbinsx=60, marker_color=ACCENT))
        hfig.update_layout(title="Daily returns (%)", height=300, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(hfig, width="stretch")

# ---------------------------------------------------------------- forecast tab
with forecast_tab:
    st.markdown(
        "Forecasts the **log return over the next _h_ trading days** and scores each model with "
        "**walk-forward validation**: every prediction uses only data available at the time, with a gap "
        "between training labels and test rows so overlapping horizons can't leak. "
        "Models must beat a **random walk** (predict no change) to be worth anything."
    )
    horizon = st.select_slider("Horizon (trading days)", options=[1, 5, 10, 21], value=5)
    try:
        with st.spinner("Running walk-forward backtest…"):
            res = walk_forward_forecast(close, horizon=horizon)
    except ValueError as exc:
        st.info(str(exc))
    else:
        score_df = pd.DataFrame(
            [
                {
                    "Model": s.name,
                    "MAE": f"{s.mae:.2%}",
                    "RMSE": f"{s.rmse:.2%}",
                    "Direction hit rate": "–" if np.isnan(s.hit_rate) else f"{s.hit_rate:.0%}",
                    "Skill vs random walk": f"{s.skill_vs_random_walk:+.1%}",
                }
                for s in res.scores
            ]
        ).set_index("Model")
        a, b = st.columns([3, 2])
        with a:
            st.dataframe(score_df, width="stretch")
            st.caption(f"{len(res.predictions):,} out-of-sample predictions. Skill = 1 − MSE / MSE(random walk).")
        with b:
            lo, hi = res.interval_price
            st.metric(
                f"{horizon}-day forecast ({res.best_model})",
                f"{res.forecast_price:,.2f}",
                pct(np.exp(res.forecast_return) - 1),
            )
            st.caption(f"80% band from backtest errors: {lo:,.2f} – {hi:,.2f}")
        if res.beats_baseline:
            st.success(
                f"{res.best_model} had the lowest out-of-sample error. Gains over a random walk are "
                "usually small and unstable; treat this as a statistical exercise, not a trading signal."
            )
        else:
            st.warning(
                "No model beat a random walk on this stock and period, which is what the "
                "efficient-market hypothesis predicts. The forecast shown is therefore 'no change'."
            )

        pfig = go.Figure()
        p = res.predictions
        pfig.add_trace(go.Scatter(x=p.index, y=p["Actual"] * 100, name="Actual", line=dict(color=MUTED, width=1)))
        pfig.add_trace(
            go.Scatter(x=p.index, y=p[res.best_model] * 100, name=res.best_model, line=dict(color=ACCENT, width=1.4))
        )
        pfig.update_layout(
            title=f"Walk-forward predictions vs actual {horizon}-day returns (%)",
            height=340,
            margin=dict(t=40, b=10, l=10, r=10),
            hovermode="x unified",
        )
        st.plotly_chart(pfig, width="stretch")

# ---------------------------------------------------------------- data tab
with data_tab:
    st.dataframe(prices.sort_index(ascending=False), width="stretch")
    st.download_button(
        "Download CSV", prices.to_csv().encode(), file_name=f"{ticker.upper()}_prices.csv", mime="text/csv"
    )

st.caption("For education only. Not investment advice.")
