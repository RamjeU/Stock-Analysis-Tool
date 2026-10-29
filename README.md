# Stock Analysis Tool

[![CI](https://github.com/RamjeU/Stock-Analysis-Tool/actions/workflows/ci.yml/badge.svg)](https://github.com/RamjeU/Stock-Analysis-Tool/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)

An interactive web app for analysing and comparing stocks on Yahoo Finance. It covers technical indicators, risk and benchmark metrics, a regression trend channel, Monte Carlo simulation, and five machine-learning models. Every model is **backtested without look-ahead bias**, scored against the baseline it has to beat, and traded in a cost-aware strategy backtest.

> For education only. Nothing here is investment advice.

## Features

**Watchlist**
- Pick from about 35 popular US, TSX and ETF symbols, or type any Yahoo Finance ticker.
- Choose one ticker to analyse in depth and a benchmark (SPY, QQQ, S&P 500, XIU, TSX).
- Quick date ranges (1Y, 3Y, 5Y, 10Y) or custom dates.

**Chart**
- Candlesticks with SMA overlays (10–200 days), Bollinger Bands, and a **log-linear regression trend channel** (±2σ) with its implied annual growth, R², and how far the price sits from trend.
- Volume, RSI (14, Wilder's smoothing), and MACD (12, 26, 9) on linked panels.

**Compare**
- Growth of 100 for every ticker from a common start date.
- A risk table (return, CAGR, volatility, Sharpe ratio, max drawdown, beta) and a correlation heatmap of daily returns.

**Risk & benchmark**
- Total return, CAGR, annualised volatility, Sharpe ratio, maximum drawdown, and best and worst days.
- Beta and correlation against the benchmark, a drawdown curve, and a histogram of daily returns.

**Machine learning**
- Forecasts the log return over the next 1, 5, 10, or 21 trading days from 14 technical features, using:
  - **Linear regression** (ordinary least squares)
  - **Ridge** and **Lasso** regression (regularised linear models)
  - **Random forest**
  - **Gradient boosting**
  - plus two baselines: **random walk** (no change) and **historical drift** (average past return)
- Walk-forward validation, reporting skill against the random walk, directional hit rate, MAE, and RMSE.
- **Permutation feature importance**, showing which signals the best model actually relies on.
- **Strategy backtest:** trades each model's signal (long/flat or long/short) with adjustable transaction costs, compared with buy-and-hold on return, Sharpe ratio, and drawdown.
- Says plainly when no model beats the random walk, which is the usual result for liquid stocks.

**Monte Carlo**
- Simulates up to 5,000 price paths up to a year ahead, by **bootstrapping** real past returns (keeping fat tails) or by **geometric Brownian motion**.
- A fan chart of the 50% and 90% ranges, the median outcome, and the probability of ending higher.

**Data**
- Split- and dividend-adjusted prices, with a CSV download.
- Clear errors for unknown or delisted tickers (for example, `FB` is now `META`), rate limits, SSL problems, and outdated yfinance installs.

## How the forecast avoids fooling itself

The question each model answers is: "Given only what was known at today's close, what will the return be over the next *h* days?"

1. **Features look only backwards.** They are lagged returns over 1, 5, 10, 20, and 60 days; 20-day and 5/20-day volatility; RSI; the MACD histogram; Bollinger %B; distance from the 50- and 200-day SMAs; drawdown; and a volume z-score. A unit test changes future prices and checks that past features stay identical.
2. **Training respects time order.** `TimeSeriesSplit` trains on the past and tests on the future. A gap of *h* days separates the last training label from the first test row, because an *h*-day target overlaps the next *h* rows and would otherwise leak into training.
3. **Trades happen the next day.** In the strategy backtest, a position chosen at today's close earns tomorrow's return, and every change of position pays a cost.
4. **Every model faces a baseline.** Skill is defined as `1 − MSE(model) / MSE(random walk)`. If skill is not positive, the app reports the random walk's forecast of no change.

Version 1 of this project skipped all of these steps. It predicted the price 30 days ahead from today's price alone and used a randomly shuffled train/test split, so its error figure was optimistic. The tests now check both that pure noise yields **no** edge and that a genuinely mean-reverting series **is** detected.

## Quick start

```bash
git clone https://github.com/RamjeU/Stock-Analysis-Tool.git
cd Stock-Analysis-Tool
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py
```

A virtual environment matters here. Anaconda and Homebrew installs often carry old or broken copies of yfinance and pyarrow that the app would otherwise pick up.

Then open the address Streamlit prints (usually http://localhost:8501).

### Deploy for free

Push the repo to GitHub. At [share.streamlit.io](https://share.streamlit.io), create an app pointing at `app.py` on `main`. No secrets or API keys are needed.

## Project structure

```
app.py                      Streamlit interface (charts, tabs, controls)
stockanalysis/
├── data.py                 Input validation, Yahoo Finance download, cleaning
├── indicators.py           SMA, EMA, RSI, MACD, Bollinger, drawdown, risk metrics, beta
├── forecast.py             Features, five ML models, walk-forward validation, importance, strategy backtest
└── analytics.py            Log-linear regression trend channel, Monte Carlo simulation
tests/
├── test_analysis.py        Indicator maths, look-ahead checks, models, backtest, simulation
└── test_app.py             Renders the whole app on synthetic data (no network)
```

The analysis code has no dependency on Streamlit or the network, so it can be reused in notebooks or scripts.

## Tests

```bash
pip install -r requirements-dev.txt
ruff check .
pytest -q
```

GitHub Actions runs the linter and all 23 tests on Python 3.10 and 3.12 for every push.

## Tech

Python · Streamlit · Plotly · pandas · NumPy · scikit-learn · yfinance · pytest · Ruff · GitHub Actions
