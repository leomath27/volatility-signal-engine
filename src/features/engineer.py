from __future__ import annotations
import numpy as np
import pandas as pd
from src.features.realized_vol import add_returns, hist_vol

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_returns(df)

    # Price features
    out["ret_1"] = out["log_ret"]
    out["ret_5"] = out["log_ret"].rolling(5).sum()
    out["ret_10"] = out["log_ret"].rolling(10).sum()

    # Range / volatility-ish features
    out["hl_range"] = (out["high"] - out["low"]) / out["close"]
    out["oc_range"] = (out["close"] - out["open"]) / out["open"]

    out["hv_10"] = hist_vol(out, 10)
    out["hv_20"] = hist_vol(out, 20)
    out["hv_60"] = hist_vol(out, 60)

    # Volume features (if volume exists)
    if "volume" in out.columns:
        out["vol_chg"] = out["volume"].pct_change()
        out["vol_z20"] = (out["volume"] - out["volume"].rolling(20).mean()) / out["volume"].rolling(20).std()

    return out

def make_dataset(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(subset=[target_col])
    out = out.dropna()
    return out
