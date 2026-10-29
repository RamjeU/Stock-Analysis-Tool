"""Technical indicators and risk metrics. All functions are pure and take pandas objects."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def simple_moving_average(close: pd.Series, window: int) -> pd.Series:
    _check_window(window)
    return close.rolling(window, min_periods=window).mean()


def exponential_moving_average(close: pd.Series, span: int) -> pd.Series:
    _check_window(span)
    return close.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing (0-100)."""
    _check_window(period)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    # No losses in the window means maximal strength; no movement at all is neutral.
    out = out.where(avg_loss != 0, 100.0)
    out = out.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return out.where(avg_gain.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    if fast >= slow:
        raise ValueError("The fast span must be shorter than the slow span.")
    line = exponential_moving_average(close, fast) - exponential_moving_average(close, slow)
    signal_line = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"MACD": line, "Signal": signal_line, "Histogram": line - signal_line})


def bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = simple_moving_average(close, window)
    std = close.rolling(window, min_periods=window).std()
    return pd.DataFrame({"Middle": mid, "Upper": mid + num_std * std, "Lower": mid - num_std * std})


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def drawdown(close: pd.Series) -> pd.Series:
    """Fractional decline from the running peak (0 at a new high, negative below it)."""
    return close / close.cummax() - 1


def risk_summary(close: pd.Series, risk_free_rate: float = 0.0) -> dict[str, float]:
    """Headline performance and risk figures for a price series."""
    rets = close.pct_change().dropna()
    years = max(len(rets) / TRADING_DAYS, 1e-9)
    total = close.iloc[-1] / close.iloc[0] - 1
    cagr = (1 + total) ** (1 / years) - 1
    vol = rets.std(ddof=1) * np.sqrt(TRADING_DAYS) if len(rets) > 1 else float("nan")
    excess = rets - risk_free_rate / TRADING_DAYS
    sharpe = (
        excess.mean() / rets.std(ddof=1) * np.sqrt(TRADING_DAYS)
        if len(rets) > 1 and rets.std(ddof=1) > 0
        else float("nan")
    )
    return {
        "Total return": float(total),
        "Annualised return (CAGR)": float(cagr),
        "Annualised volatility": float(vol),
        "Sharpe ratio": float(sharpe),
        "Max drawdown": float(drawdown(close).min()),
        "Best day": float(rets.max()) if len(rets) else float("nan"),
        "Worst day": float(rets.min()) if len(rets) else float("nan"),
    }


def _check_window(window: int) -> None:
    if not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError("Window lengths must be positive integers.")


def beta_and_correlation(asset: pd.Series, benchmark: pd.Series) -> tuple[float, float]:
    """Beta and correlation of daily returns against a benchmark, on shared trading days."""
    joined = pd.concat([asset.pct_change(), benchmark.pct_change()], axis=1, join="inner").dropna()
    if len(joined) < 20:
        return float("nan"), float("nan")
    a, b = joined.iloc[:, 0], joined.iloc[:, 1]
    beta = a.cov(b) / b.var()
    return float(beta), float(a.corr(b))
