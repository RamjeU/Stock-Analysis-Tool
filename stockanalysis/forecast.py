"""Leak-free machine-learning forecasts of forward returns, judged against honest baselines.

The question asked is: "what will the log return be over the next ``horizon`` trading days?"
Every model is scored by walk-forward validation: it is trained only on data that would
have been available at the time, with a gap of ``horizon`` days between the last training
label and the first test row so overlapping labels cannot leak the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import indicators as ind

BASELINES = ("Random walk", "Historical drift")
ML_MODELS = ("Linear regression", "Ridge regression", "Lasso regression", "Random forest", "Gradient boosting")
MODELS = BASELINES + ML_MODELS

FEATURE_LABELS = {
    "ret_1d": "1-day return",
    "ret_5d": "5-day return",
    "ret_10d": "10-day return",
    "ret_20d": "20-day return",
    "ret_60d": "60-day return",
    "vol_20d": "20-day volatility",
    "vol_ratio": "Volatility ratio (5d / 20d)",
    "rsi_14": "RSI (14)",
    "macd_hist": "MACD histogram",
    "bb_pct": "Bollinger %B",
    "dist_sma_50": "Distance from SMA 50",
    "dist_sma_200": "Distance from SMA 200",
    "drawdown": "Drawdown from peak",
    "volume_z": "Volume z-score (20d)",
}


def build_features(close: pd.Series, volume: pd.Series | None = None) -> pd.DataFrame:
    """Features known at the close of each day. Nothing here looks forward."""
    r = ind.log_returns(close)
    feats = pd.DataFrame(index=close.index)
    for lag in (1, 5, 10, 20, 60):
        feats[f"ret_{lag}d"] = np.log(close / close.shift(lag))
    feats["vol_20d"] = r.rolling(20).std()
    feats["vol_ratio"] = r.rolling(5).std() / feats["vol_20d"]
    feats["rsi_14"] = ind.rsi(close, 14) / 100
    feats["macd_hist"] = ind.macd(close)["Histogram"] / close
    bb = ind.bollinger_bands(close)
    feats["bb_pct"] = (close - bb["Lower"]) / (bb["Upper"] - bb["Lower"])
    feats["dist_sma_50"] = close / ind.simple_moving_average(close, 50) - 1
    feats["dist_sma_200"] = close / ind.simple_moving_average(close, 200) - 1
    feats["drawdown"] = ind.drawdown(close)
    if volume is not None:
        logv = np.log(volume.replace(0, np.nan))
        feats["volume_z"] = (logv - logv.rolling(20).mean()) / logv.rolling(20).std()
    return feats.replace([np.inf, -np.inf], np.nan)


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
    best_ml_model: str
    last_close: float
    last_date: pd.Timestamp
    forecast_return: float
    forecast_price: float
    interval_price: tuple[float, float]  # 80% band from walk-forward residuals
    feature_importance: pd.Series  # permutation importance of the best ML model, last fold
    n_train_rows: int = field(default=0)

    @property
    def beats_baseline(self) -> bool:
        return self.best_model not in BASELINES


def _make_model(name: str):
    if name == "Linear regression":
        return make_pipeline(StandardScaler(), LinearRegression())
    if name == "Ridge regression":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "Lasso regression":
        return make_pipeline(StandardScaler(), Lasso(alpha=1e-4, max_iter=5000))
    if name == "Random forest":
        return RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=20, n_jobs=-1, random_state=0)
    if name == "Gradient boosting":
        return HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=0
        )
    raise ValueError(f"Unknown model: {name}")


def _fit_predict(name: str, X_train, y_train, X_test):
    if name == "Random walk":
        return np.zeros(len(X_test)), None
    if name == "Historical drift":
        return np.full(len(X_test), float(np.mean(y_train))), None
    model = _make_model(name)
    model.fit(X_train, y_train)
    return model.predict(X_test), model


def walk_forward_forecast(
    close: pd.Series,
    horizon: int = 5,
    n_splits: int = 5,
    models: tuple[str, ...] | list[str] = MODELS,
    volume: pd.Series | None = None,
) -> ForecastResult:
    if horizon < 1:
        raise ValueError("The horizon must be at least one trading day.")
    models = tuple(dict.fromkeys(("Random walk", *models)))  # the baseline is always scored
    unknown = set(models) - set(MODELS)
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(sorted(unknown))}")

    feats = build_features(close, volume)
    target = build_target(close, horizon)
    data = feats.join(target).dropna()
    min_rows = 60 * n_splits + horizon * n_splits
    if len(data) < min_rows:
        raise ValueError(
            f"Not enough history to evaluate a {horizon}-day forecast: need about "
            f"{min_rows + 260} trading days, have {len(close)}. Choose an earlier start date."
        )

    feature_names = list(data.columns.drop("target"))
    X, y = data[feature_names].to_numpy(), data["target"].to_numpy()
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=horizon)

    preds = {m: np.full(len(y), np.nan) for m in models}
    last_fold: dict[str, tuple] = {}
    for train_idx, test_idx in splitter.split(X):
        for m in models:
            p, fitted = _fit_predict(m, X[train_idx], y[train_idx], X[test_idx])
            preds[m][test_idx] = p
            if fitted is not None:
                last_fold[m] = (fitted, test_idx)

    tested = ~np.isnan(preds["Random walk"])
    actual = y[tested]
    rw_mse = float(np.mean(actual**2))
    scores = []
    for m in models:
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
    ml_scores = [s for s in scores if s.name in ML_MODELS]
    best_ml = min(ml_scores, key=lambda s: s.rmse).name if ml_scores else ""

    importance = pd.Series(dtype=float)
    if best_ml:
        model, test_idx = last_fold[best_ml]
        pi = permutation_importance(
            model, X[test_idx], y[test_idx], scoring="neg_mean_squared_error", n_repeats=5, random_state=0
        )
        importance = (
            pd.Series(pi.importances_mean, index=[FEATURE_LABELS.get(f, f) for f in feature_names])
            .sort_values(ascending=False)
            .rename("Increase in MSE when shuffled")
        )

    # Refit the winner on every labelled row, then forecast from the latest features.
    latest_feats = feats[feature_names].dropna().iloc[[-1]].to_numpy()
    fc_return = float(_fit_predict(best, X, y, latest_feats)[0][0])
    resid = actual - preds[best][tested]
    lo_q, hi_q = np.quantile(resid, [0.1, 0.9])
    last_close = float(close.iloc[-1])

    pred_frame = pd.DataFrame(
        {"Actual": actual, **{m: preds[m][tested] for m in models}},
        index=data.index[tested],
    )
    return ForecastResult(
        horizon=horizon,
        scores=scores,
        predictions=pred_frame,
        best_model=best,
        best_ml_model=best_ml,
        last_close=last_close,
        last_date=close.index[-1],
        forecast_return=fc_return,
        forecast_price=last_close * float(np.exp(fc_return)),
        interval_price=(
            last_close * float(np.exp(fc_return + lo_q)),
            last_close * float(np.exp(fc_return + hi_q)),
        ),
        feature_importance=importance,
        n_train_rows=len(y),
    )


def strategy_backtest(
    close: pd.Series, signal: pd.Series, cost_bps: float = 5.0, allow_short: bool = False
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Trade a walk-forward forecast and compare it with buy-and-hold.

    The position decided at day t's close (long if the forecast is positive; flat or short
    otherwise) earns day t+1's return, so no trade uses information it would not have had.
    ``cost_bps`` is charged on every change in position.
    """
    signal = signal.dropna()
    nxt = close.pct_change().shift(-1).reindex(signal.index)
    low = -1.0 if allow_short else 0.0
    position = pd.Series(np.where(signal > 0, 1.0, low), index=signal.index)
    turnover = position.diff().abs().fillna(position.abs())
    strat = (position * nxt - turnover * cost_bps / 10_000).dropna()
    bench = nxt.reindex(strat.index)

    equity = pd.DataFrame({"Strategy": (1 + strat).cumprod() * 100, "Buy and hold": (1 + bench).cumprod() * 100})

    def stats(r: pd.Series) -> tuple[float, float, float, float]:
        total = float((1 + r).prod() - 1)
        years = max(len(r) / ind.TRADING_DAYS, 1e-9)
        cagr = (1 + total) ** (1 / years) - 1
        sharpe = float(r.mean() / r.std() * np.sqrt(ind.TRADING_DAYS)) if r.std() > 0 else float("nan")
        mdd = float(ind.drawdown((1 + r).cumprod()).min())
        return total, cagr, sharpe, mdd

    s_total, s_cagr, s_sharpe, s_mdd = stats(strat)
    b_total, b_cagr, b_sharpe, b_mdd = stats(bench)
    summary = {
        "Strategy total return": s_total,
        "Buy-and-hold total return": b_total,
        "Strategy CAGR": s_cagr,
        "Buy-and-hold CAGR": b_cagr,
        "Strategy Sharpe": s_sharpe,
        "Buy-and-hold Sharpe": b_sharpe,
        "Strategy max drawdown": s_mdd,
        "Buy-and-hold max drawdown": b_mdd,
        "Time in market": float((position.reindex(strat.index) != 0).mean()),
        "Trades": float((turnover.reindex(strat.index) > 0).sum()),
    }
    return equity, summary
