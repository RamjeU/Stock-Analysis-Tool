"""Smoke test: the Streamlit app renders every tab without errors, using synthetic data."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def fake_load_prices(request):
    seed = sum(map(ord, request.ticker))
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(request.start, request.end)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(idx))))
    return pd.DataFrame(
        {
            "Open": close * 0.995,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, len(idx)).astype(float),
        },
        index=idx,
    )


@pytest.fixture(autouse=True)
def clear_cache():
    import streamlit as st

    st.cache_data.clear()


def test_app_renders(monkeypatch):
    import stockanalysis.data as data

    monkeypatch.setattr(data, "load_prices", fake_load_prices)
    at = st_testing.AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    assert len(at.metric) >= 11  # six headline, forecast, four Monte Carlo
    assert len(at.tabs) == 6
    assert not at.error


def test_app_shows_error_for_bad_ticker(monkeypatch):
    import stockanalysis.data as data

    def boom(request):
        raise data.DataError("No data found for 'ZZZZ'.")

    monkeypatch.setattr(data, "load_prices", boom)
    at = st_testing.AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.error and "No data found" in at.error[0].value
