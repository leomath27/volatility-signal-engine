from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_ret"] = np.log(out["close"]).diff()
    out["ret"] = out["close"].pct_change()
    return out


def forward_realized_vol(df: pd.DataFrame, horizon_days: int = 1) -> pd.Series:
    """
    Annualized realized volatility over the NEXT `horizon_days` sessions.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")

    r2 = df["log_ret"] ** 2
    fwd_var = r2.rolling(horizon_days, min_periods=horizon_days).mean().shift(-(horizon_days - 1))
    return (np.sqrt(fwd_var) * np.sqrt(TRADING_DAYS)).rename(f"rv_fwd_{horizon_days}d")


def hist_vol(df: pd.DataFrame, window_days: int = 20) -> pd.Series:
    return (df["log_ret"].rolling(window_days, min_periods=window_days).std() * np.sqrt(TRADING_DAYS)).rename(
        f"hv_{window_days}d"
    )


def parkinson_vol(df: pd.DataFrame, window_days: int = 20) -> pd.Series:
    hl = np.log(df["high"] / df["low"]) ** 2
    coef = 1.0 / (4.0 * np.log(2.0))
    pk = np.sqrt(coef * hl.rolling(window_days, min_periods=window_days).mean() * TRADING_DAYS)
    return pk.rename(f"pkv_{window_days}d")
