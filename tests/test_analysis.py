"""Offline tests: synthetic prices only, no network."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from stockanalysis import indicators as ind
from stockanalysis.data import DataError, PriceRequest, clean_prices
from stockanalysis.forecast import build_features, build_target, walk_forward_forecast


def make_prices(n=900, drift=0.0003, vol=0.015, seed=1) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, vol, n))), index=idx, name="Close")


# ---------------------------------------------------------------- data
def test_price_request_normalises_and_validates():
    assert PriceRequest(" msft ", date(2020, 1, 1), date(2021, 1, 1)).ticker == "MSFT"
    with pytest.raises(DataError):
        PriceRequest("AAPL", date(2021, 1, 1), date(2020, 1, 1))
    with pytest.raises(DataError):
        PriceRequest("AA PL", date(2020, 1, 1), date(2021, 1, 1))


def test_clean_prices_flattens_multiindex_and_sorts():
    idx = pd.to_datetime(["2024-01-03", "2024-01-02"]).tz_localize("America/New_York")
    cols = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Volume"], ["AAPL"]])
    raw = pd.DataFrame(np.ones((2, 5)) * [1, 2, 0.5, 1.5, 100], index=idx, columns=cols)
    out = clean_prices(raw)
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert out.index.is_monotonic_increasing and out.index.tz is None


def test_clean_prices_rejects_empty():
    with pytest.raises(DataError):
        clean_prices(pd.DataFrame())


# ---------------------------------------------------------------- indicators
def test_sma_matches_manual_mean():
    s = pd.Series([1.0, 2, 3, 4, 5])
    assert ind.simple_moving_average(s, 3).tolist()[2:] == [2.0, 3.0, 4.0]
    with pytest.raises(ValueError):
        ind.simple_moving_average(s, 0)


def test_rsi_bounds_and_extremes():
    rising = pd.Series(np.arange(1.0, 60.0))
    assert ind.rsi(rising).dropna().eq(100).all()
    r = ind.rsi(make_prices()).dropna()
    assert r.between(0, 100).all()


def test_drawdown_and_risk_summary():
    s = pd.Series([100.0, 120, 90, 130])
    assert ind.drawdown(s).min() == pytest.approx(90 / 120 - 1)
    summary = ind.risk_summary(s)
    assert summary["Total return"] == pytest.approx(0.3)
    assert summary["Max drawdown"] == pytest.approx(-0.25)


def test_beta_of_series_against_itself_is_one():
    s = make_prices()
    beta, corr = ind.beta_and_correlation(s, s)
    assert beta == pytest.approx(1.0) and corr == pytest.approx(1.0)


# ---------------------------------------------------------------- forecasting
def test_features_do_not_use_future_prices():
    """Changing tomorrow's price must not change today's features."""
    s = make_prices()
    base = build_features(s)
    shocked = s.copy()
    shocked.iloc[500:] *= 1.5
    after = build_features(shocked)
    pd.testing.assert_frame_equal(base.iloc[:500], after.iloc[:500])


def test_target_is_forward_return():
    s = pd.Series([100.0, 110, 121])
    t = build_target(s, 1)
    assert t.iloc[0] == pytest.approx(np.log(1.1)) and np.isnan(t.iloc[-1])


def test_walk_forward_on_random_walk_finds_no_real_edge():
    res = walk_forward_forecast(make_prices(n=1200, drift=0.0), horizon=5)
    assert {s.name for s in res.scores} >= {"Random walk", "Ridge regression"}
    ridge = next(s for s in res.scores if s.name == "Ridge regression")
    assert ridge.skill_vs_random_walk < 0.05  # pure noise cannot be predicted
    lo, hi = res.interval_price
    assert lo < hi and res.forecast_price > 0


def test_walk_forward_detects_genuine_signal():
    """Mean-reverting prices are predictable; the models should beat the random walk."""
    rng = np.random.default_rng(7)
    n, x = 1500, [0.0]
    for _ in range(n - 1):
        x.append(0.9 * x[-1] + rng.normal(0, 0.02))
    s = pd.Series(100 * np.exp(np.array(x)), index=pd.bdate_range("2018-01-01", periods=n))
    res = walk_forward_forecast(s, horizon=5)
    assert res.beats_baseline
    best = next(sc for sc in res.scores if sc.name == res.best_model)
    assert best.skill_vs_random_walk > 0.1


def test_walk_forward_requires_enough_history():
    with pytest.raises(ValueError, match="Not enough history"):
        walk_forward_forecast(make_prices(n=150), horizon=5)


# ---------------------------------------------------------------- loader error handling
class _FakeTicker:
    def __init__(self, exc=None, frame=None, calls=None):
        self.exc, self.frame, self.calls = exc, frame, calls

    def __call__(self, symbol):
        return self

    def history(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc:
            raise self.exc
        return self.frame


def _fake_yf(monkeypatch, ticker):
    import sys
    import types

    mod = types.SimpleNamespace(Ticker=ticker, __version__="1.7.0")
    monkeypatch.setitem(sys.modules, "yfinance", mod)
    monkeypatch.setattr("stockanalysis.data.time.sleep", lambda s: None)


def test_loader_surfaces_real_error_with_hint(monkeypatch):
    from stockanalysis.data import load_prices

    calls = []
    _fake_yf(monkeypatch, _FakeTicker(exc=RuntimeError("SSL: CERTIFICATE_VERIFY_FAILED"), calls=calls))
    with pytest.raises(DataError, match="Install Certificates"):
        load_prices(PriceRequest("META", date(2022, 10, 1), date(2022, 12, 31)))
    assert len(calls) == 1  # not a transient error, so no retry


def test_loader_retries_rate_limits(monkeypatch):
    from stockanalysis.data import load_prices

    calls = []
    _fake_yf(monkeypatch, _FakeTicker(exc=RuntimeError("Too Many Requests. Rate limited."), calls=calls))
    with pytest.raises(DataError, match="rate-limiting"):
        load_prices(PriceRequest("META", date(2022, 10, 1), date(2022, 12, 31)))
    assert len(calls) == 3


def test_loader_cleans_history_frame(monkeypatch):
    from stockanalysis.data import load_prices

    idx = pd.bdate_range("2022-10-03", periods=5, tz="America/New_York")
    frame = pd.DataFrame(
        {
            "Open": 1.0,
            "High": 2.0,
            "Low": 0.5,
            "Close": [1.0, 1.1, 1.2, 1.3, 1.4],
            "Volume": 10.0,
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        },
        index=idx,
    )
    _fake_yf(monkeypatch, _FakeTicker(frame=frame, calls=[]))
    out = load_prices(PriceRequest("META", date(2022, 10, 1), date(2022, 12, 31)))
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"] and out.index.tz is None


def test_outdated_yfinance_is_rejected_with_upgrade_command():
    from stockanalysis.data import check_yfinance_version

    with pytest.raises(DataError, match="pip install -U yfinance"):
        check_yfinance_version("0.2.40")
    check_yfinance_version("1.7.0")
    check_yfinance_version("1.0.0.dev1")


# ---------------------------------------------------------------- ML extras, trend, Monte Carlo
def test_all_models_scored_and_importance_reported():
    from stockanalysis.forecast import ML_MODELS

    s = make_prices(n=1000)
    vol = pd.Series(np.random.default_rng(3).integers(1e6, 5e6, len(s)).astype(float), index=s.index)
    res = walk_forward_forecast(s, horizon=5, volume=vol)
    assert {sc.name for sc in res.scores} == {"Random walk", "Historical drift", *ML_MODELS}
    assert res.best_ml_model in ML_MODELS
    assert "Volume z-score (20d)" in res.feature_importance.index
    assert res.feature_importance.index.is_unique


def test_model_subset_always_includes_random_walk():
    res = walk_forward_forecast(make_prices(n=900), horizon=5, models=("Linear regression",))
    assert [sc.name for sc in res.scores] == ["Random walk", "Linear regression"]
    with pytest.raises(ValueError, match="Unknown model"):
        walk_forward_forecast(make_prices(n=900), models=("Crystal ball",))


def test_strategy_backtest_uses_next_day_return():
    from stockanalysis.forecast import strategy_backtest

    s = pd.Series([100.0, 110, 99, 108.9], index=pd.bdate_range("2024-01-01", periods=4))
    perfect = pd.Series([1.0, -1.0, 1.0, 1.0], index=s.index)  # knows tomorrow: up, down, up
    equity, stats = strategy_backtest(s, perfect, cost_bps=0)
    # long day 1 (+10%), flat day 2 (avoids -10%), long day 3 (+10%)
    assert stats["Strategy total return"] == pytest.approx(1.1 * 1.1 - 1)
    assert stats["Buy-and-hold total return"] == pytest.approx(108.9 / 100 - 1)
    _, costly = strategy_backtest(s, perfect, cost_bps=100)
    assert costly["Strategy total return"] < stats["Strategy total return"]


def test_regression_trend_recovers_known_growth():
    from stockanalysis.analytics import regression_trend

    idx = pd.bdate_range("2015-01-01", periods=1260)
    s = pd.Series(100 * np.exp(np.log(1.10) / 252 * np.arange(1260)), index=idx)
    fit = regression_trend(s)
    assert fit.annual_growth == pytest.approx(0.10, rel=1e-6)
    assert fit.r_squared == pytest.approx(1.0)


def test_monte_carlo_bands_are_ordered():
    from stockanalysis.analytics import monte_carlo

    s = make_prices()
    for method in ("bootstrap", "gbm"):
        sim = monte_carlo(s, days=30, n_paths=500, method=method)
        assert len(sim) == 30 and sim.index[0] > s.index[-1]
        assert (sim["P5"] <= sim["P50"]).all() and (sim["P50"] <= sim["P95"]).all()
        assert 0 <= sim.attrs["prob_up"] <= 1
    with pytest.raises(ValueError):
        monte_carlo(s, method="magic")
