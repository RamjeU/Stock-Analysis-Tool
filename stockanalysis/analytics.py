"""Statistical models of the price path: a log-linear regression trend and Monte Carlo simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .indicators import TRADING_DAYS


@dataclass
class TrendFit:
    fitted: pd.Series  # trend line in price units
    upper: pd.Series  # +2 standard errors of the residuals
    lower: pd.Series  # -2 standard errors
    annual_growth: float  # implied growth rate per year from the slope
    r_squared: float
    zscore: float  # where the latest close sits relative to the channel, in residual std units


def regression_trend(close: pd.Series) -> TrendFit:
    """Ordinary least squares fit of log(price) on time, with a +/- 2 sigma channel.

    A straight line in log space is constant percentage growth, so the slope converts directly
    into an annual growth rate. The residual z-score says how stretched the price is versus trend.
    """
    if len(close) < 10:
        raise ValueError("At least 10 prices are needed to fit a trend.")
    t = np.arange(len(close), dtype=float)
    y = np.log(close.to_numpy(dtype=float))
    slope, intercept = np.polyfit(t, y, 1)
    fit = intercept + slope * t
    resid = y - fit
    sigma = resid.std(ddof=2)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    idx = close.index
    return TrendFit(
        fitted=pd.Series(np.exp(fit), index=idx),
        upper=pd.Series(np.exp(fit + 2 * sigma), index=idx),
        lower=pd.Series(np.exp(fit - 2 * sigma), index=idx),
        annual_growth=float(np.exp(slope * TRADING_DAYS) - 1),
        r_squared=r2,
        zscore=float(resid[-1] / sigma) if sigma > 0 else float("nan"),
    )


def monte_carlo(
    close: pd.Series,
    days: int = 63,
    n_paths: int = 2000,
    method: str = "bootstrap",
    lookback: int = 504,
    seed: int | None = 0,
) -> pd.DataFrame:
    """Simulate future price paths and return percentile bands per future trading day.

    ``method="gbm"`` draws normal log returns with the historical mean and volatility
    (geometric Brownian motion). ``method="bootstrap"`` resamples actual past daily log
    returns, which keeps their fat tails and skew.
    """
    if days < 1 or n_paths < 10:
        raise ValueError("Need at least 1 day and 10 paths.")
    hist = np.log(close).diff().dropna().iloc[-lookback:].to_numpy()
    if len(hist) < 20:
        raise ValueError("At least 20 daily returns are needed for a simulation.")
    rng = np.random.default_rng(seed)
    if method == "gbm":
        draws = rng.normal(hist.mean(), hist.std(ddof=1), size=(n_paths, days))
    elif method == "bootstrap":
        draws = rng.choice(hist, size=(n_paths, days), replace=True)
    else:
        raise ValueError("method must be 'gbm' or 'bootstrap'.")
    paths = float(close.iloc[-1]) * np.exp(np.cumsum(draws, axis=1))
    future = pd.bdate_range(close.index[-1] + pd.offsets.BDay(1), periods=days)
    qs = [5, 25, 50, 75, 95]
    bands = np.percentile(paths, qs, axis=0).T
    out = pd.DataFrame(bands, index=future, columns=[f"P{q}" for q in qs])
    out.attrs["prob_up"] = float((paths[:, -1] > close.iloc[-1]).mean())
    out.attrs["sample_paths"] = pd.DataFrame(paths[:30].T, index=future)
    return out
