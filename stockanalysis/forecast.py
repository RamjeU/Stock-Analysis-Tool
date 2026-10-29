"""Leak-free forecasting of forward returns, evaluated against honest baselines.

The question asked is: "what will the log return be over the next ``horizon`` trading days?"
Every model is scored by walk-forward validation: it is trained only on data that would
have been available at the time, with a gap of ``horizon`` days between the last training
label and the first test row so overlapping labels cannot leak the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import indicators as ind

MODELS = ("Random walk", "Historical drift", "Ridge regression", "Gradient boosting")


def build_features(close: pd.Series) -> pd.DataFrame:
    """Features known at the close of each day. Nothing here looks forward."""
    r = ind.log_returns(close)
    feats = pd.DataFrame(index=close.index)
    for lag in (1, 5, 10, 20):
        feats[f"ret_{lag}d"] = np.log(close / close.shift(lag))
    feats["vol_20d"] = r.rolling(20).std()
    feats["vol_ratio"] = r.rolling(5).std() / feats["vol_20d"]
    feats["rsi_14"] = ind.rsi(close, 14) / 100
    feats["dist_sma_50"] = close / ind.simple_moving_average(close, 50) - 1
    feats["drawdown"] = ind.drawdown(close)
    return feats


def build_target(close: pd.Series, horizon: int) -> pd.Series:
    """Log return from today's close to the close ``horizon`` trading days later."""
    return np.log(close.shift(-horizon) / close).rename("target")


@dataclass
class ModelScore:
    name: str
    mae: float
    rmse: float
    hit_rate: float  # share of test days where the sign of the move was called correctly
    skill_vs_random_walk: float  # 1 - MSE / MSE(random walk); > 0 means it beat the baseline


@dataclass
class ForecastResult:
    horizon: int
    scores: list[ModelScore]
    predictions: pd.DataFrame  # walk-forward predictions vs actuals, one column per model
    best_model: str
    last_close: float
    last_date: pd.Timestamp
    forecast_return: float
    forecast_price: float
    interval_price: tuple[float, float]  # 80% band from walk-forward residuals
    n_train_rows: int = field(default=0)

    @property
    def beats_baseline(self) -> bool:
        return self.best_model != "Random walk"


def _make_model(name: str):
    if name == "Ridge regression":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "Gradient boosting":
        return HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=0
        )
    raise ValueError(name)


def _predict(name: str, X_train, y_train, X_test) -> np.ndarray:
    if name == "Random walk":
        return np.zeros(len(X_test))
    if name == "Historical drift":
        return np.full(len(X_test), float(np.mean(y_train)))
    model = _make_model(name)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def walk_forward_forecast(close: pd.Series, horizon: int = 5, n_splits: int = 5) -> ForecastResult:
    if horizon < 1:
        raise ValueError("The horizon must be at least one trading day.")

    feats = build_features(close)
    target = build_target(close, horizon)
    data = feats.join(target).dropna()
    min_rows = 60 * n_splits + horizon * n_splits
    if len(data) < min_rows:
        raise ValueError(
            f"Not enough history to evaluate a {horizon}-day forecast: need about "
            f"{min_rows + 60} trading days, have {len(close)}. Choose an earlier start date."
        )

    X, y = data.drop(columns="target").to_numpy(), data["target"].to_numpy()
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=horizon)

    preds = {m: np.full(len(y), np.nan) for m in MODELS}
    for train_idx, test_idx in splitter.split(X):
        for m in MODELS:
            preds[m][test_idx] = _predict(m, X[train_idx], y[train_idx], X[test_idx])

    tested = ~np.isnan(preds["Random walk"])
    actual = y[tested]
    rw_mse = float(np.mean(actual**2))
    scores = []
    for m in MODELS:
        p = preds[m][tested]
        err = actual - p
        mse = float(np.mean(err**2))
        hit = float(np.mean(np.sign(p) == np.sign(actual))) if m != "Random walk" else float("nan")
        scores.append(
            ModelScore(
                name=m,
                mae=float(np.mean(np.abs(err))),
                rmse=float(np.sqrt(mse)),
                hit_rate=hit,
                skill_vs_random_walk=1 - mse / rw_mse if rw_mse > 0 else float("nan"),
            )
        )

    best = min(scores, key=lambda s: s.rmse).name

    # Refit the winner on every labelled row, then forecast from the latest features.
    latest_feats = feats.dropna().iloc[[-1]].to_numpy()
    fc_return = float(_predict(best, X, y, latest_feats)[0])
    resid = actual - preds[best][tested]
    lo_q, hi_q = np.quantile(resid, [0.1, 0.9])
    last_close = float(close.iloc[-1])

    pred_frame = pd.DataFrame(
        {"Actual": actual, **{m: preds[m][tested] for m in MODELS}},
        index=data.index[tested],
    )
    return ForecastResult(
        horizon=horizon,
        scores=scores,
        predictions=pred_frame,
        best_model=best,
        last_close=last_close,
        last_date=close.index[-1],
        forecast_return=fc_return,
        forecast_price=last_close * float(np.exp(fc_return)),
        interval_price=(
            last_close * float(np.exp(fc_return + lo_q)),
            last_close * float(np.exp(fc_return + hi_q)),
        ),
        n_train_rows=len(y),
    )
