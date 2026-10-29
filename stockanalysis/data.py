"""Loading and validating daily price history."""

from __future__ import annotations

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

    try:
        raw = yf.download(
            request.ticker,
            start=request.start,
            end=request.end,
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:  # yfinance raises a wide variety of network errors
        raise DataError(f"Could not reach Yahoo Finance: {exc}") from exc

    if raw is None or raw.empty:
        raise DataError(
            f"No data found for '{request.ticker}' between {request.start} and {request.end}. "
            "Check the symbol (e.g. META, not FB) and the date range."
        )
    return clean_prices(raw)
