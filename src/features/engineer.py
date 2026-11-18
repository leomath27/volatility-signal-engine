from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.realized_vol import (TRADING_DAYS, add_returns, garman_klass_vol, hist_vol,
                                       parkinson_var, parkinson_vol)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).rename(f"rsi_{window}")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """All features use only information available at the close of each bar."""
    out = add_returns(df)
    r, c = out["log_ret"], out["close"]

    # Return / momentum
    out["ret_1"] = r
    out["ret_5"] = r.rolling(5).sum()
    out["ret_10"] = r.rolling(10).sum()
    out["ret_20"] = r.rolling(20).sum()

    # Price structure / range
    out["hl_range"] = (out["high"] - out["low"]) / c
    out["oc_range"] = (c - out["open"]) / out["open"]
    out["co_gap"] = (out["open"] - c.shift(1)) / c.shift(1)
    out["close_loc"] = (c - out["low"]) / (out["high"] - out["low"]).replace(0, np.nan)

    # Rolling volatility (close-to-close and range-based) at several horizons
    for w in (5, 10, 20, 60):
        out[f"hv_{w}"] = hist_vol(out, w)
        out[f"pkv_{w}"] = parkinson_vol(out, w)
    out["gkv_20"] = garman_klass_vol(out, 20)
    out["pkv_ewm_10"] = np.sqrt(parkinson_var(out).ewm(span=10, min_periods=10).mean() * TRADING_DAYS)

    # Vol term structure and vol-of-vol
    out["vol_ratio_5_20"] = out["pkv_5"] / out["pkv_20"]
    out["vol_ratio_20_60"] = out["pkv_20"] / out["pkv_60"]
    out["hv_ratio_5_20"] = out["hv_5"] / out["hv_20"]
    out["vol_of_vol_20"] = out["hv_5"].rolling(20).std()
    out["pkv_20_pctile_252"] = out["pkv_20"].rolling(252, min_periods=126).rank(pct=True)

    # Downside / drawdown (leverage effect)
    out["downside_vol_20"] = np.sqrt((r.clip(upper=0) ** 2).rolling(20, min_periods=20).mean() * TRADING_DAYS)
    out["drawdown_60"] = c / c.rolling(60).max() - 1

    # Trend / mean reversion (ratios only: raw SMA price levels are non-stationary)
    for w in (10, 20, 50):
        out[f"price_to_sma_{w}"] = c / c.rolling(w).mean()
    out["rsi_14"] = _rsi(c, 14)

    # Volume (skipped for instruments with no volume, e.g. some indices)
    if "volume" in out.columns and (out["volume"] > 0).mean() > 0.9:
        out["vol_chg"] = out["volume"].pct_change()
        vol_mean_20 = out["volume"].rolling(20).mean()
        vol_std_20 = out["volume"].rolling(20).std().replace(0, np.nan)
        out["vol_z20"] = (out["volume"] - vol_mean_20) / vol_std_20

    out["dow"] = out.index.dayofweek
    return out.replace([np.inf, -np.inf], np.nan)


# Raw prices/levels and duplicates are never model inputs
DEFAULT_DROP_COLS = {"open", "high", "low", "close", "adj_close", "volume", "log_ret", "ret"}


def make_dataset(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Training frame: rows with a complete feature vector AND a known forward target."""
    out = df.copy().sort_index()
    out = out.dropna(subset=[target_col])
    return out.dropna()


def latest_feature_row(df: pd.DataFrame, features: list[str]) -> tuple[pd.Series, pd.Timestamp]:
    """Feature vector for the most recent bar. It has no forward target yet, which is the point."""
    feats = engineer_features(df)
    row = feats.iloc[-1]
    missing = row[features].isna()
    if missing.any():
        raise ValueError(
            f"Not enough history to compute features: {list(missing[missing].index)[:5]} (need ~260+ trading days)"
        )
    return row, feats.index[-1]
