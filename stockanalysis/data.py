"""Loading and validating daily price history."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date

import pandas as pd

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


class DataError(ValueError):
    """Raised when price data cannot be loaded or fails validation."""


@dataclass(frozen=True)
class PriceRequest:
    ticker: str
    start: date
    end: date

    def __post_init__(self) -> None:
        ticker = self.ticker.strip().upper()
        if not ticker or not all(c.isalnum() or c in ".-^=" for c in ticker):
            raise DataError(f"'{self.ticker}' is not a valid ticker symbol.")
        if self.start >= self.end:
            raise DataError("The start date must be before the end date.")
        object.__setattr__(self, "ticker", ticker)


def clean_prices(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise a raw OHLCV frame: flat columns, tz-naive sorted index, no gaps in Close."""
    if raw is None or raw.empty:
        raise DataError("No price data was returned for that ticker and date range.")

    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"Price data is missing columns: {', '.join(missing)}.")

    df = df[REQUIRED_COLUMNS].astype(float)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["Close"])

    if (df["Close"] <= 0).any():
        raise DataError("Price data contains non-positive closing prices.")
    if len(df) < 2:
        raise DataError("At least two trading days are needed for analysis.")
    return df


def load_prices(request: PriceRequest) -> pd.DataFrame:
    """Download daily, split- and dividend-adjusted prices from Yahoo Finance."""
    import yfinance as yf  # imported lazily so the analysis code and tests need no network

    check_yfinance_version(yf.__version__)

    # Ticker.history with raise_errors=True surfaces Yahoo's real error (rate limit, SSL,
    # delisted symbol...). yf.download swallows those and returns an empty frame instead.
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            raw = yf.Ticker(request.ticker).history(
                start=request.start,
                end=request.end,
                interval="1d",
                auto_adjust=True,
                raise_errors=True,
            )
            break
        except Exception as exc:  # yfinance raises many exception types
            last_exc = exc
            if not _is_transient(exc) or attempt == 2:
                raise DataError(_explain(exc, request)) from exc
            time.sleep(1.5 * (attempt + 1))
    else:  # pragma: no cover - loop always breaks or raises
        raise DataError(_explain(last_exc, request))

    if raw is None or raw.empty:
        raise DataError(
            f"Yahoo Finance returned no rows for '{request.ticker}' between {request.start} and "
            f"{request.end}. Check the symbol (e.g. META, not FB) and the date range."
        )
    return clean_prices(raw)


MIN_YFINANCE = (1, 0)


def check_yfinance_version(version: str) -> None:
    """Old yfinance releases break whenever Yahoo changes its API; fail early with the exact fix."""
    import sys

    try:
        parts = tuple(int(p) for p in version.split(".")[:2])
    except ValueError:
        return  # unusual version string (dev build); let the request decide
    if parts < MIN_YFINANCE:
        raise DataError(
            f"This app needs yfinance 1.0 or newer, but the Python running it has yfinance {version}, "
            "which can no longer read Yahoo Finance data. Stop the app, then run:\n\n"
            f"{sys.executable} -m pip install -U yfinance\n\n"
            f"and restart it with: {sys.executable} -m streamlit run app.py"
        )


def _is_transient(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(k in text for k in ("ratelimit", "too many requests", "timeout", "timed out", "connection"))


def _explain(exc: Exception, request: PriceRequest) -> str:
    """Turn a yfinance exception into a message a user can act on."""
    import yfinance as yf

    text = f"{type(exc).__name__}: {exc}"
    low = text.lower()
    if "ratelimit" in low or "too many requests" in low:
        hint = "Yahoo is rate-limiting this connection. Wait a minute and try again."
    elif "certificate" in low or "ssl" in low:
        hint = (
            "Python could not verify Yahoo's SSL certificate. On macOS with the python.org installer, "
            "run 'Install Certificates.command' from your Python folder in Applications."
        )
    elif "delisted" in low or "no timezone" in low or "not found" in low or "no data" in low:
        hint = f"'{request.ticker}' may be misspelled or delisted, or had no trades in that range."
    elif "connect" in low or "resolve" in low or "timed out" in low:
        hint = "Could not connect to Yahoo Finance. Check your internet connection or VPN."
    else:
        hint = "Try upgrading yfinance: pip install -U yfinance"
    return f"{hint}\n\nDetails ({text}; yfinance {yf.__version__})"
