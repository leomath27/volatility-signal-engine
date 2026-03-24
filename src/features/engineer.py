from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.realized_vol import add_returns, hist_vol, parkinson_vol


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).rename(f"rsi_{window}")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_returns(df)

    # Return / momentum features
    out["ret_1"] = out["log_ret"]
    out["ret_5"] = out["log_ret"].rolling(5).sum()
    out["ret_10"] = out["log_ret"].rolling(10).sum()
    out["ret_20"] = out["log_ret"].rolling(20).sum()

    # Price structure / range features
    out["hl_range"] = (out["high"] - out["low"]) / out["close"]
    out["oc_range"] = (out["close"] - out["open"]) / out["open"]
    out["co_gap"] = (out["open"] - out["close"].shift(1)) / out["close"].shift(1)

    # Rolling volatility features
    out["hv_5"] = hist_vol(out, 5)
    out["hv_10"] = hist_vol(out, 10)
    out["hv_20"] = hist_vol(out, 20)
    out["hv_60"] = hist_vol(out, 60)
    out["pkv_10"] = parkinson_vol(out, 10)
    out["pkv_20"] = parkinson_vol(out, 20)
    out["vol_of_vol_20"] = out["hv_5"].rolling(20).std()

    # Trend / mean reversion
    out["sma_5"] = out["close"].rolling(5).mean()
    out["sma_10"] = out["close"].rolling(10).mean()
    out["sma_20"] = out["close"].rolling(20).mean()
    out["sma_50"] = out["close"].rolling(50).mean()
    out["price_to_sma_10"] = out["close"] / out["sma_10"]
    out["price_to_sma_20"] = out["close"] / out["sma_20"]
    out["price_to_sma_50"] = out["close"] / out["sma_50"]
    out["rsi_14"] = _rsi(out["close"], 14)

    # Volume features
    if "volume" in out.columns:
        out["vol_chg"] = out["volume"].pct_change()
        vol_mean_20 = out["volume"].rolling(20).mean()
        vol_std_20 = out["volume"].rolling(20).std().replace(0, np.nan)
        out["vol_z20"] = (out["volume"] - vol_mean_20) / vol_std_20

    return out


DEFAULT_DROP_COLS = {
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
}


def make_dataset(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = df.copy().sort_index()
    out = out.dropna(subset=[target_col])
    out = out.dropna()
    return out
