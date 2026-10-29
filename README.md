# Stock Analysis Tool

[![CI](https://github.com/RamjeU/Stock-Analysis-Tool/actions/workflows/ci.yml/badge.svg)](https://github.com/RamjeU/Stock-Analysis-Tool/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)

An interactive web app for analysing any stock on Yahoo Finance. It covers technical indicators, risk and benchmark metrics, and a return forecast that is **backtested without look-ahead bias** and scored against the baseline it has to beat.

> For education only. Nothing here is investment advice.

## Features

**Chart**
- Candlesticks with selectable SMA overlays (10–200 days) and Bollinger Bands.
- Volume, RSI (14, Wilder's smoothing), and MACD (12, 26, 9) on linked panels.

**Risk & benchmark**
- Total return, CAGR, annualised volatility, Sharpe ratio, maximum drawdown, and best and worst days.
- Beta and correlation against any benchmark (SPY by default).
- A growth-of-100 comparison, a drawdown curve, and a histogram of daily returns.

**Forecast**
- Predicts the log return over the next 1, 5, 10, or 21 trading days using four models:
  - **Random walk** (predict no change)
  - **Historical drift** (the average past return)
  - **Ridge regression**
  - **Gradient boosting**
- Scores each model by walk-forward validation, reporting MAE, RMSE, directional hit rate, and **skill against the random walk**.
- Shows the best model's forecast with an 80% band derived from its own backtest errors.
- Says plainly when no model beats the random walk, which is the usual result for liquid stocks.

**Data**
- Split- and dividend-adjusted prices, with a CSV download.
- Clear error messages for unknown or delisted tickers (for example, `FB` is now `META`) and for empty date ranges.

## How the forecast avoids fooling itself

The question each model answers is: "Given only what was known at today's close, what will the return be over the next *h* days?"

1. **Features look only backwards.** They are lagged returns over 1, 5, 10, and 20 days; 20-day and 5/20-day volatility; RSI; distance from the 50-day SMA; and current drawdown. A unit test changes future prices and checks that past features stay identical.
2. **Training respects time order.** `TimeSeriesSplit` trains on the past and tests on the future. A gap of *h* days separates the last training label from the first test row, because an *h*-day target overlaps the next *h* rows and would otherwise leak into training.
3. **Every model faces a baseline.** Skill is defined as `1 − MSE(model) / MSE(random walk)`. If skill is not positive, the app reports the random walk's forecast of no change.

Version 1 of this project skipped all three steps. It predicted the price 30 days ahead from today's price alone and used a randomly shuffled train/test split, so its error figure was optimistic. The tests now check both that pure noise yields **no** edge and that a genuinely mean-reverting series **is** detected.

## Quick start

```bash
git clone https://github.com/RamjeU/Stock-Analysis-Tool.git
cd Stock-Analysis-Tool
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the address Streamlit prints (usually http://localhost:8501).

### Deploy for free

Push the repo to GitHub. At [share.streamlit.io](https://share.streamlit.io), create an app pointing at `app.py` on `main`. No secrets or API keys are needed.

## Project structure

```
app.py                      Streamlit interface (charts, tabs, controls)
stockanalysis/
├── data.py                 Input validation, Yahoo Finance download, cleaning
├── indicators.py           SMA, EMA, RSI, MACD, Bollinger, drawdown, risk metrics, beta
└── forecast.py             Leak-free features, walk-forward backtest, baselines
tests/
├── test_analysis.py        Indicator maths, look-ahead checks, forecasting behaviour
└── test_app.py             Renders the whole app on synthetic data (no network)
```

The analysis code has no dependency on Streamlit or the network, so it can be reused in notebooks or scripts.

## Tests

```bash
pip install -r requirements-dev.txt
ruff check .
pytest -q
```

GitHub Actions runs the linter and all 14 tests on Python 3.10 and 3.12 for every push.

## Tech

Python · Streamlit · Plotly · pandas · NumPy · scikit-learn · yfinance · pytest · Ruff · GitHub Actions
