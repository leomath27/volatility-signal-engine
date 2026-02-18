from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252

def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_ret"] = np.log(out["close"]).diff()
    return out

def forward_realized_vol(df: pd.DataFrame, horizon_days: int = 1) -> pd.Series:
    """
    Forward realized vol (annualized) using mean of squared log returns over the NEXT horizon_days.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")

    r2 = df["log_ret"] ** 2
    fwd_var = r2.rolling(horizon_days).mean().shift(-(horizon_days - 1))
    rv = np.sqrt(fwd_var) * np.sqrt(TRADING_DAYS)
    return rv.rename(f"rv_fwd_{horizon_days}d")

def hist_vol(df: pd.DataFrame, window_days: int = 20) -> pd.Series:
    hv = df["log_ret"].rolling(window_days).std() * np.sqrt(TRADING_DAYS)
    return hv.rename(f"hv_{window_days}d")
