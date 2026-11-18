"""Market data: yfinance download with parquet cache, plus an offline synthetic generator."""
from __future__ import annotations

import logging
import os
import time
import zlib
from datetime import datetime

import numpy as np
import pandas as pd

from src import config

log = logging.getLogger(__name__)
REQUIRED = ["open", "high", "low", "close", "volume"]


def safe_symbol(symbol: str) -> str:
    return symbol.replace("^", "IDX_").replace("/", "_").replace("=", "_")


def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(columns=str.lower).rename(columns={"adj close": "adj_close"})
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise RuntimeError(f"Market data missing required columns: {missing}")
    df = df[REQUIRED].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    # repair impossible bars so range-based estimators stay valid
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)
    df["volume"] = df["volume"].fillna(0.0)
    return df


def fetch_market_data(symbol: str, refresh: bool = False, max_age_hours: float | None = None) -> pd.DataFrame:
    """Daily OHLCV with lowercase columns and a DatetimeIndex.

    Prices are split/dividend ADJUSTED (auto_adjust=True) so stock splits never
    appear as fake volatility spikes. The parquet cache is reused unless
    ``refresh`` is set or it is older than ``max_age_hours``. If a refresh
    fails and a cache exists, the stale cache is returned with a warning.
    """
    if os.environ.get("VOL_DATA_SOURCE", config.DATA_SOURCE) == "synthetic":
        return synthetic_prices(seed=zlib.crc32(symbol.encode()) % 10_000)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = config.DATA_DIR / f"{safe_symbol(symbol)}.parquet"
    if cache.exists() and not refresh:
        age_h = (time.time() - cache.stat().st_mtime) / 3600
        if max_age_hours is None or age_h < max_age_hours:
            return pd.read_parquet(cache)

    try:
        import yfinance as yf

        raw = yf.download(symbol, start=config.START_DATE, end=config.END_DATE, auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            raise RuntimeError(f"No market data returned for symbol={symbol}")
        df = _normalize(raw)
    except Exception as exc:
        if cache.exists():
            log.warning("Refresh failed for %s (%s); using stale cache", symbol, exc)
            return pd.read_parquet(cache)
        raise RuntimeError(f"Could not fetch market data for {symbol}: {exc}") from exc

    df.to_parquet(cache)
    return df


def drop_incomplete_bar(df: pd.DataFrame) -> pd.DataFrame:
    """Drop today's bar while the US market is still open (its high/low/close are partial)."""
    try:
        from zoneinfo import ZoneInfo

        now = datetime.now(ZoneInfo("America/New_York"))
    except Exception:  # tzdata missing: keep the data as-is
        return df
    if df.empty or df.index[-1].date() != now.date():
        return df
    market_closed = now.weekday() >= 5 or now.hour >= 16
    return df if market_closed else df.iloc[:-1]


def synthetic_prices(n: int = 2800, seed: int = 0, s0: float = 100.0, end: str = "2025-06-30") -> pd.DataFrame:
    """GARCH(1,1) returns with fat tails; range and volume scale with volatility.

    For smoke tests and demos only. Never report results from this data.
    """
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 2e-6, 0.09, 0.89
    var = omega / (1 - alpha - beta)
    rets, vols = np.empty(n), np.empty(n)
    for i in range(n):
        vols[i] = np.sqrt(var)
        rets[i] = vols[i] * rng.standard_t(6) / np.sqrt(6 / 4)
        var = omega + alpha * rets[i] ** 2 + beta * var
    close = s0 * np.exp(np.cumsum(rets))
    prev = np.r_[s0, close[:-1]]
    open_ = prev * np.exp(rng.normal(0, vols * 0.3))
    span = np.abs(rng.normal(1.6, 0.4, n)) * vols
    high = np.maximum(open_, close) * np.exp(span * rng.uniform(0.2, 1.0, n))
    low = np.minimum(open_, close) * np.exp(-span * rng.uniform(0.2, 1.0, n))
    volume = 1e7 * np.exp(rng.normal(0, 0.25, n)) * (1 + 40 * vols)
    idx = pd.bdate_range(end=pd.Timestamp(end), periods=n, name="date")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)
