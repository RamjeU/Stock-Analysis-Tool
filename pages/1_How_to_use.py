"""How to use the app, and what every number means."""

import streamlit as st

st.set_page_config(page_title="How to use", page_icon="📖", layout="wide")

st.title("📖 How to use this app")
st.markdown(
    "Pick some stocks, pick one to study, and read the tabs from left to right. "
    "Every number in the app is explained in the glossary below."
)

st.header("Quick start")
st.markdown(
    """
1. **Build a watchlist.** In the sidebar, choose tickers from the list or type any Yahoo Finance
   symbol and press Enter. Canadian stocks end in `.TO` (for example `SHOP.TO`), and indices start with `^` (`^GSPC`).
2. **Choose one to analyse in detail** and a **benchmark** to measure it against. SPY, an S&P 500 fund, is the usual choice.
3. **Pick a period:** 1Y, 3Y, 5Y, 10Y, or Custom. The machine-learning tab needs at least about 3 years.
4. **Work through the tabs:**

| Tab | What it answers |
| --- | --- |
| **Chart** | What has the price done, and what do the technical indicators say? |
| **Compare** | How did my watchlist do against each other, and do they move together? |
| **Risk & benchmark** | How bumpy was the ride, and how much does it move with the market? |
| **Machine learning** | Can any model predict the next few days better than guessing "no change"? |
| **Monte Carlo** | If the future looks like the past, what range of prices is plausible? |
| **Data** | The raw daily prices, which you can download as CSV. |

5. Change any setting and the app reruns by itself. Press **R** to force a rerun.
"""
)

st.header("Glossary")
st.caption("Grouped by the tab where you will see each term.")

with st.expander("Headline numbers (top of the page)", expanded=True):
    st.markdown(
        """
- **Last close:** the most recent closing price. The small number underneath is the change from the day before.
- **Total return:** how much the price grew over the whole period, adjusted for splits and dividends. +50% means $100 became $150.
- **CAGR (compound annual growth rate):** total return expressed as a steady yearly rate.
  It lets you compare periods of different lengths.
- **Volatility (annualised):** how much daily returns swing, scaled to a year.
  About 15% is typical for the S&P 500; 50% and above is a wild ride.
- **Max drawdown:** the worst fall from a peak to a later low. −40% means that at some point you were down 40% from the top.
- **Vs. trend (σ):** how far today's price sits from its long-run regression trend, in standard deviations.
  Around +2σ is unusually stretched above trend; around −2σ is unusually far below.
"""
    )

with st.expander("Chart tab"):
    st.markdown(
        """
- **Candlestick:** each bar shows one day's open, high, low and close. Green means it closed higher than it opened; red means lower.
- **SMA (simple moving average):** the average closing price over the last N days, which smooths out noise.
  Price above its 200-day SMA is commonly read as a long-term uptrend.
- **Bollinger bands:** a 20-day average with bands two standard deviations above and below.
  Bands widen when the market is volatile and tighten when it is calm.
- **Regression trend channel:** a straight line fitted to log price, which means steady percentage growth,
  with ±2σ bands. The caption gives the implied growth per year and **R²**, the share of
  price movement the line explains (1.0 is a perfect fit).
- **Volume:** how many shares traded each day. Spikes often mark news.
- **RSI (Relative Strength Index):** a 0–100 gauge of recent momentum.
  Above 70 is traditionally "overbought" and below 30 "oversold". These are rules of thumb, not laws.
- **MACD:** the gap between a fast (12-day) and slow (26-day) exponential moving average, plus a 9-day **signal** line.
  The bars show MACD minus signal; bars crossing zero mark momentum shifts.
"""
    )

with st.expander("Compare and Risk & benchmark tabs"):
    st.markdown(
        """
- **Growth of 100:** what $100 invested at the start would be worth each day. It puts every stock on the same scale.
- **Sharpe ratio:** return earned per unit of volatility. Higher is better.
  Above 1 is good, and above 2 over many years is rare. This app assumes a 0% risk-free rate.
- **Beta:** how much the stock tends to move when the benchmark moves 1%.
  1.5 means it typically moves 1.5%; 0.5 means half as much.
- **Correlation:** how closely two return series move together, from −1 (opposite) to +1 (in lockstep).
  Low correlation between holdings is what makes diversification work.
- **Best / worst day:** the single biggest daily gain and loss in the period.
- **Daily returns histogram:** how often each size of daily move happened. Long tails mean big surprise days.
"""
    )

with st.expander("Machine learning tab"):
    st.markdown(
        """
- **Horizon:** how many trading days ahead the models predict. 5 days is about a week; 21 is about a month.
- **Features:** the 14 inputs each model sees, all known at the close of the day: past returns, volatility,
  RSI, MACD, Bollinger %B, distance from moving averages, drawdown, and unusual volume.
- **Models:**
  - **Linear regression:** the straight-line formula that best fits features to future returns.
  - **Ridge / Lasso:** linear regression with a penalty that stops it overreacting to noise.
    Lasso can switch useless features off entirely.
  - **Random forest:** averages many decision trees, each trained on a random slice of the data.
  - **Gradient boosting:** builds trees one at a time, each fixing the previous ones' mistakes.
  - **Random walk:** the baseline. It always predicts "no change".
  - **Historical drift:** another baseline. It always predicts the average past return.
- **Walk-forward validation:** the models train only on the past and are tested on the period after it,
  repeated five times, the way they would have worked in real life. There is a gap so the answer can't leak in.
- **Skill vs random walk:** how much smaller a model's squared error is than the random walk's.
  **Positive means it beat the baseline; negative means you would have done better guessing "no change".**
- **Direction hit rate:** how often the model called up or down correctly. 50% is a coin flip.
- **MAE / RMSE:** average size of the prediction errors. RMSE punishes big misses more. Lower is better.
- **80% band:** the range the price landed in 80% of the time, based on the model's own past errors.
- **Permutation importance:** how much worse the model gets when one feature is shuffled into nonsense.
  Bigger bars are features the model actually relies on.
- **Strategy backtest:** trades the model's signal, going long when it predicts a rise and otherwise
  sitting in cash (or shorting, if switched on), then compares the result with simply holding the stock.
- **Basis points (bps):** hundredths of a percent. A 5 bps trading cost is 0.05% per trade.
- **Time in market:** the share of days the strategy held a position.
"""
    )

with st.expander("Monte Carlo tab"):
    st.markdown(
        """
- **Monte Carlo simulation:** generates thousands of random future price paths and looks at the spread of outcomes.
- **Bootstrap:** builds each path by reusing real daily returns from the last two years in random order.
  This keeps crash days and the like.
- **GBM (geometric Brownian motion):** assumes daily returns follow a bell curve, the textbook model.
  It understates extreme days.
- **Percentiles:** P5 means 5% of simulated paths ended below this price; P95 means 5% ended above it.
  The dark band holds the middle 50% of paths and the light band the middle 90%.
- **Chance of ending higher:** the share of simulated paths that finish above today's price.
"""
    )

st.info(
    "Everything here describes the past or simulates from it. Markets change, so no backtest or "
    "simulation guarantees future results. For education only, not investment advice."
)
