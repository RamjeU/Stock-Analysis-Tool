"""Streamlit front end. Run with:  streamlit run app.py"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from stockanalysis import indicators as ind
from stockanalysis.analytics import monte_carlo, regression_trend
from stockanalysis.data import DataError, PriceRequest, load_prices
from stockanalysis.forecast import BASELINES, ML_MODELS, strategy_backtest, walk_forward_forecast

st.set_page_config(page_title="Stock Analysis Tool", page_icon="📈", layout="wide")

UP, DOWN, ACCENT, MUTED, AMBER = "#2e9e6b", "#d1495b", "#3b6fd8", "#8a8f98", "#e0a030"
MARGIN = dict(t=40, b=10, l=10, r=10)

POPULAR = [
    # US large caps
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "JPM", "V",
    "MA", "UNH", "XOM", "JNJ", "WMT", "COST", "NFLX", "AMD", "INTC", "DIS",
    # Canadian (TSX)
    "SHOP.TO", "RY.TO", "TD.TO", "ENB.TO", "CNR.TO", "BN.TO", "SU.TO", "BMO.TO",
    # ETFs and indices
    "SPY", "QQQ", "VFV.TO", "XIU.TO", "VTI", "^GSPC", "^IXIC", "^GSPTSE",
]  # fmt: skip
BENCHMARKS = ["SPY", "QQQ", "^GSPC", "XIU.TO", "^GSPTSE", "None"]
PERIODS = {"1Y": 365, "3Y": 365 * 3, "5Y": 365 * 5, "10Y": 365 * 10, "Custom": None}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch(ticker: str, start: date, end: date) -> pd.DataFrame:
    return load_prices(PriceRequest(ticker, start, end))


@st.cache_data(ttl=3600, show_spinner=False)
def run_forecast(ticker: str, start: date, end: date, horizon: int, models: tuple[str, ...]):
    prices = fetch(ticker, start, end)
    return walk_forward_forecast(prices["Close"], horizon=horizon, models=models, volume=prices["Volume"])


def pct(x: float) -> str:
    if x is None or np.isnan(x):
        return "–"
    return f"{x:+.1%}" if abs(x) < 10 else f"{x:+.0%}"


def fmt_row(d: dict[str, float]) -> dict[str, str]:
    return {
        k: (f"{v:.2f}" if "Sharpe" in k or "Beta" in k else f"{v:.0f}" if k == "Trades" else pct(v))
        for k, v in d.items()
    }


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Tickers")
    watchlist = st.multiselect(
        "Watchlist",
        POPULAR,
        default=["AAPL", "MSFT", "NVDA"],
        accept_new_options=True,
        help="Pick from the list or type any Yahoo Finance symbol (e.g. SHOP.TO, ^GSPC) and press Enter.",
    )
    watchlist = list(dict.fromkeys(t.strip().upper() for t in watchlist if t.strip()))
    if not watchlist:
        st.info("Add at least one ticker to the watchlist.")
        st.stop()
    ticker = st.selectbox("Analyse in detail", watchlist)
    benchmark = st.selectbox("Benchmark", BENCHMARKS, index=0)
    benchmark = None if benchmark == "None" else benchmark

    st.header("Period")
    today = date.today()
    period = st.segmented_control("Range", list(PERIODS), default="5Y", label_visibility="collapsed") or "5Y"
    if PERIODS[period] is None:
        start = st.date_input("Start", value=today - timedelta(days=365 * 5), max_value=today - timedelta(days=2))
        end = st.date_input("End", value=today, max_value=today)
    else:
        start, end = today - timedelta(days=PERIODS[period]), today

    st.header("Chart overlays")
    sma_windows = st.multiselect("Moving averages (days)", [10, 20, 50, 100, 200], default=[50, 200])
    show_bb = st.checkbox("Bollinger bands (20, 2σ)")
    show_trend = st.checkbox("Regression trend channel (±2σ)", value=True)
    st.caption("Data: Yahoo Finance via yfinance, adjusted for splits and dividends.")

st.title(f"📈 {ticker} analysis")

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
trend = regression_trend(close)
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Last close", f"{close.iloc[-1]:,.2f}", pct(close.iloc[-1] / close.iloc[-2] - 1))
c2.metric("Total return", pct(risk["Total return"]))
c3.metric("CAGR", pct(risk["Annualised return (CAGR)"]))
c4.metric("Volatility (ann.)", f"{risk['Annualised volatility']:.1%}")
c5.metric("Max drawdown", pct(risk["Max drawdown"]))
c6.metric("Vs. trend", f"{trend.zscore:+.1f}σ", help="Latest close relative to the log-linear regression trend.")

tabs = st.tabs(["Chart", "Compare", "Risk & benchmark", "Machine learning", "Monte Carlo", "Data"])
chart_tab, compare_tab, risk_tab, ml_tab, mc_tab, data_tab = tabs

# ---------------------------------------------------------------- chart
with chart_tab:
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
            x=prices.index, open=prices["Open"], high=prices["High"], low=prices["Low"], close=close,
            name="Price", increasing_line_color=UP, decreasing_line_color=DOWN, showlegend=False,
        ),
        row=1, col=1,
    )  # fmt: skip
    for w in sorted(sma_windows):
        sma = ind.simple_moving_average(close, w)
        fig.add_trace(go.Scatter(x=close.index, y=sma, name=f"SMA {w}", line=dict(width=1.4)), row=1, col=1)
    if show_bb:
        bb = ind.bollinger_bands(close)
        fig.add_trace(
            go.Scatter(
                x=bb.index, y=bb["Upper"], line=dict(color=MUTED, width=0.8), name="Bollinger", legendgroup="bb"
            ),
            row=1, col=1,
        )  # fmt: skip
        fig.add_trace(
            go.Scatter(
                x=bb.index, y=bb["Lower"], line=dict(color=MUTED, width=0.8), fill="tonexty",
                fillcolor="rgba(138,143,152,0.12)", showlegend=False, legendgroup="bb",
            ),
            row=1, col=1,
        )  # fmt: skip
    if show_trend:
        for series, name, dash in (
            (trend.fitted, "Trend", "solid"),
            (trend.upper, "+2σ", "dot"),
            (trend.lower, "−2σ", "dot"),
        ):
            fig.add_trace(
                go.Scatter(
                    x=series.index, y=series, name=name, legendgroup="trend", showlegend=name == "Trend",
                    line=dict(color=AMBER, width=1.2, dash=dash),
                ),
                row=1, col=1,
            )  # fmt: skip
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
    fig.add_trace(go.Scatter(x=m.index, y=m["Signal"], line=dict(color=AMBER, width=1.2), name="Signal"), row=4, col=1)
    fig.update_layout(
        height=820,
        xaxis_rangeslider_visible=False,
        margin=MARGIN,
        legend=dict(orientation="h", y=1.04, x=0),
        hovermode="x unified",
    )
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    st.plotly_chart(fig, width="stretch")
    if show_trend:
        st.caption(
            f"Log-linear regression trend: {pct(trend.annual_growth)} per year, R² = {trend.r_squared:.2f}. "
            f"The latest close is {trend.zscore:+.1f} standard deviations from the trend line."
        )

# ---------------------------------------------------------------- compare
with compare_tab:
    closes, failed = {}, []
    for t in watchlist:
        try:
            closes[t] = close if t == ticker else fetch(t, start, end)["Close"]
        except DataError:
            failed.append(t)
    if failed:
        st.warning(f"Could not load: {', '.join(failed)}")
    if len(closes) < 2:
        st.info("Add a second ticker to the watchlist to compare performance, risk and correlation.")
    else:
        table = pd.DataFrame(closes).dropna(how="all").ffill()
        table = table.loc[table.dropna().index[0] :]  # start where every ticker has data
        growth = table / table.iloc[0] * 100
        gfig = go.Figure([go.Scatter(x=growth.index, y=growth[c], name=c, line=dict(width=1.8)) for c in growth])
        gfig.update_layout(title="Growth of 100 (common start date)", height=420, margin=MARGIN, hovermode="x unified")
        st.plotly_chart(gfig, width="stretch")

        left, right = st.columns([3, 2])
        with left:
            rows = {}
            for t, s in table.items():
                r = ind.risk_summary(s.dropna())
                row = {
                    k: r[k]
                    for k in (
                        "Total return",
                        "Annualised return (CAGR)",
                        "Annualised volatility",
                        "Sharpe ratio",
                        "Max drawdown",
                    )
                }
                if bench_close is not None:
                    row[f"Beta vs {benchmark}"] = ind.beta_and_correlation(s, bench_close)[0]
                rows[t] = fmt_row(row)
            st.dataframe(pd.DataFrame(rows).T, width="stretch")
            st.caption("Compare each return with the volatility and drawdown it took to earn it.")
        with right:
            corr = table.pct_change().dropna().corr()
            hfig = go.Figure(
                go.Heatmap(
                    z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1, colorscale="RdBu",
                    text=np.round(corr.values, 2), texttemplate="%{text}", showscale=False,
                )
            )  # fmt: skip
            hfig.update_layout(title="Correlation of daily returns", height=360, margin=MARGIN)
            st.plotly_chart(hfig, width="stretch")

# ---------------------------------------------------------------- risk
with risk_tab:
    left, right = st.columns([2, 1])
    with left:
        growth = pd.DataFrame({ticker: close / close.iloc[0] * 100})
        if bench_close is not None:
            aligned = bench_close.reindex(close.index).ffill()
            growth[benchmark] = aligned / aligned.dropna().iloc[0] * 100
        gfig = go.Figure()
        for i, col in enumerate(growth.columns):
            gfig.add_trace(
                go.Scatter(x=growth.index, y=growth[col], name=col, line=dict(color=[ACCENT, MUTED][i % 2], width=1.8))
            )
        gfig.update_layout(title="Growth of 100", height=360, margin=MARGIN, hovermode="x unified")
        st.plotly_chart(gfig, width="stretch")

        dd = ind.drawdown(close)
        dfig = go.Figure(
            go.Scatter(
                x=dd.index, y=dd * 100, fill="tozeroy", line=dict(color=DOWN, width=1), fillcolor="rgba(209,73,91,0.18)"
            )
        )
        dfig.update_layout(title="Drawdown from peak (%)", height=260, margin=MARGIN)
        st.plotly_chart(dfig, width="stretch")
    with right:
        rows = {k: (f"{v:.2f}" if k == "Sharpe ratio" else pct(v)) for k, v in risk.items()}
        if bench_close is not None:
            beta, corr = ind.beta_and_correlation(close, bench_close)
            rows[f"Beta vs {benchmark}"] = f"{beta:.2f}"
            rows[f"Correlation vs {benchmark}"] = f"{corr:.2f}"
        st.dataframe(pd.Series(rows, name="Value").to_frame(), width="stretch")
        st.caption("Sharpe ratio assumes a 0% risk-free rate. Annualised with 252 trading days.")
        rets = close.pct_change().dropna() * 100
        hfig = go.Figure(go.Histogram(x=rets, nbinsx=60, marker_color=ACCENT))
        hfig.update_layout(title="Daily returns (%)", height=300, margin=MARGIN)
        st.plotly_chart(hfig, width="stretch")

# ---------------------------------------------------------------- machine learning
with ml_tab:
    st.markdown(
        "Five machine-learning models forecast the **log return over the next _h_ trading days** from 14 "
        "technical features. Every model is scored by **walk-forward validation**: it only ever trains on "
        "the past, with a gap so overlapping horizons can't leak. To be worth anything, a model has to beat "
        "a **random walk**, which simply predicts no change."
    )
    a, b = st.columns([1, 2])
    horizon = a.select_slider("Horizon (trading days)", options=[1, 5, 10, 21], value=5)
    chosen = b.multiselect("Models", ML_MODELS + ("Historical drift",), default=list(ML_MODELS))
    try:
        with st.spinner("Training and backtesting models…"):
            res = run_forecast(ticker, start, end, horizon, tuple(chosen))
    except ValueError as exc:
        st.info(str(exc))
    else:
        scores = pd.DataFrame(
            [
                {
                    "Model": s.name,
                    "Skill vs random walk": s.skill_vs_random_walk,
                    "Direction hit rate": s.hit_rate,
                    "MAE": s.mae,
                    "RMSE": s.rmse,
                }
                for s in res.scores
            ]
        ).set_index("Model")

        left, right = st.columns([3, 2])
        with left:
            st.dataframe(
                scores.style.format(
                    {
                        "Skill vs random walk": "{:+.1%}",
                        "Direction hit rate": "{:.0%}",
                        "MAE": "{:.2%}",
                        "RMSE": "{:.2%}",
                    },
                    na_rep="–",
                ),
                width="stretch",
            )
            st.caption(f"{len(res.predictions):,} out-of-sample predictions. Skill = 1 − MSE / MSE(random walk).")
        with right:
            lo, hi = res.interval_price
            st.metric(
                f"{horizon}-day forecast · {res.best_model}",
                f"{res.forecast_price:,.2f}",
                pct(np.exp(res.forecast_return) - 1),
            )
            st.caption(f"80% band from backtest errors: {lo:,.2f} – {hi:,.2f}")
            if res.beats_baseline:
                st.success(
                    f"{res.best_model} beat the random walk out of sample. "
                    "Edges like this are usually small and unstable."
                )
            else:
                st.warning(
                    "No model beat a random walk here, as the efficient-market hypothesis predicts. "
                    "The forecast shown is 'no change'."
                )

        skill = scores["Skill vs random walk"].drop("Random walk").sort_values()
        sfig = go.Figure(
            go.Bar(
                x=skill.values * 100, y=skill.index, orientation="h", marker_color=np.where(skill.values > 0, UP, DOWN)
            )
        )
        sfig.add_vline(x=0, line=dict(color=MUTED, width=1))
        sfig.update_layout(
            title="Skill vs random walk (%) — right of zero beats the baseline", height=300, margin=MARGIN
        )

        left, right = st.columns(2)
        with left:
            st.plotly_chart(sfig, width="stretch")
        with right:
            if not res.feature_importance.empty:
                imp = res.feature_importance.sort_values()
                ifig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=ACCENT))
                ifig.update_layout(
                    title=f"What drives {res.best_ml_model}: permutation importance", height=300, margin=MARGIN
                )
                st.plotly_chart(ifig, width="stretch")

        st.subheader("Trade the forecast")
        st.markdown(
            "Each day, go long if the model predicts a positive return, otherwise step aside (or go short). "
            "The position decided at today's close earns tomorrow's return, and every trade pays a cost."
        )
        tradable = [m for m in res.predictions.columns if m not in ("Actual", *BASELINES)]
        if tradable:
            c1, c2, c3 = st.columns(3)
            default_idx = tradable.index(res.best_ml_model) if res.best_ml_model in tradable else 0
            strat_model = c1.selectbox("Signal from", tradable, index=default_idx)
            cost = c2.slider("Cost per trade (basis points)", 0, 50, 5)
            shorts = c3.toggle("Allow short positions")
            equity, stats = strategy_backtest(close, res.predictions[strat_model], cost_bps=cost, allow_short=shorts)
            efig = go.Figure(
                [
                    go.Scatter(
                        x=equity.index,
                        y=equity["Strategy"],
                        name=f"{strat_model} strategy",
                        line=dict(color=ACCENT, width=1.8),
                    ),
                    go.Scatter(
                        x=equity.index, y=equity["Buy and hold"], name="Buy and hold", line=dict(color=MUTED, width=1.5)
                    ),
                ]
            )
            efig.update_layout(
                title="Growth of 100 over the out-of-sample period", height=360, margin=MARGIN, hovermode="x unified"
            )
            left, right = st.columns([2, 1])
            left.plotly_chart(efig, width="stretch")
            right.dataframe(pd.Series(fmt_row(stats), name="Value").to_frame(), width="stretch")

        with st.expander("Predicted vs actual returns"):
            p = res.predictions
            pfig = go.Figure()
            pfig.add_trace(go.Scatter(x=p.index, y=p["Actual"] * 100, name="Actual", line=dict(color=MUTED, width=1)))
            show = res.best_model if res.best_model in p else res.best_ml_model
            pfig.add_trace(go.Scatter(x=p.index, y=p[show] * 100, name=show, line=dict(color=ACCENT, width=1.4)))
            pfig.update_layout(
                title=f"Walk-forward {horizon}-day returns (%)", height=340, margin=MARGIN, hovermode="x unified"
            )
            st.plotly_chart(pfig, width="stretch")

# ---------------------------------------------------------------- monte carlo
with mc_tab:
    st.markdown(
        "Simulates thousands of possible price paths. **Bootstrap** resamples real past daily returns, "
        "keeping their fat tails; **GBM** (geometric Brownian motion) assumes normally distributed returns."
    )
    c1, c2, c3 = st.columns(3)
    mc_days = c1.select_slider("Days ahead", options=[21, 63, 126, 252], value=63)
    method = c2.radio(
        "Method",
        ["bootstrap", "gbm"],
        horizontal=True,
        format_func=lambda m: "Bootstrap" if m == "bootstrap" else "GBM",
    )
    n_paths = c3.select_slider("Paths", options=[500, 1000, 2000, 5000], value=2000)
    try:
        sim = monte_carlo(close, days=mc_days, n_paths=n_paths, method=method)
    except ValueError as exc:
        st.info(str(exc))
    else:
        hist = close.iloc[-126:]
        mfig = go.Figure()
        mfig.add_trace(go.Scatter(x=hist.index, y=hist, name="History", line=dict(color=MUTED, width=1.5)))
        for col in sim.attrs["sample_paths"]:
            path = sim.attrs["sample_paths"][col]
            mfig.add_trace(
                go.Scatter(
                    x=path.index,
                    y=path,
                    line=dict(color=ACCENT, width=0.5),
                    opacity=0.25,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        for lo, hi, alpha, label in (("P5", "P95", 0.12, "90% range"), ("P25", "P75", 0.25, "50% range")):
            mfig.add_trace(go.Scatter(x=sim.index, y=sim[hi], line=dict(width=0), showlegend=False, hoverinfo="skip"))
            mfig.add_trace(
                go.Scatter(
                    x=sim.index,
                    y=sim[lo],
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=f"rgba(59,111,216,{alpha})",
                    name=label,
                )
            )
        mfig.add_trace(go.Scatter(x=sim.index, y=sim["P50"], name="Median", line=dict(color=ACCENT, width=2)))
        mfig.update_layout(
            title=f"{n_paths:,} simulated paths, {mc_days} trading days",
            height=460,
            margin=MARGIN,
            hovermode="x unified",
        )
        st.plotly_chart(mfig, width="stretch")
        last = sim.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Median outcome", f"{last['P50']:,.2f}", pct(last["P50"] / close.iloc[-1] - 1))
        c2.metric("5th percentile", f"{last['P5']:,.2f}", pct(last["P5"] / close.iloc[-1] - 1))
        c3.metric("95th percentile", f"{last['P95']:,.2f}", pct(last["P95"] / close.iloc[-1] - 1))
        c4.metric("Chance of ending higher", f"{sim.attrs['prob_up']:.0%}")
        st.caption(
            "Uses the last two years of daily returns. "
            "Simulations assume the future resembles the past, which it often doesn't."
        )

# ---------------------------------------------------------------- data
with data_tab:
    st.dataframe(prices.sort_index(ascending=False), width="stretch")
    st.download_button("Download CSV", prices.to_csv().encode(), file_name=f"{ticker}_prices.csv", mime="text/csv")

st.caption("For education only. Not investment advice.")
