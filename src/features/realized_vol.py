from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
_PK_COEF = 1.0 / (4.0 * np.log(2.0))


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_ret"] = np.log(out["close"]).diff()
    out["ret"] = out["close"].pct_change()
    return out


def parkinson_var(df: pd.DataFrame) -> pd.Series:
    """Per-day Parkinson variance (daily units, not annualized). Uses the high-low range."""
    return _PK_COEF * np.log(df["high"] / df["low"]) ** 2


def garman_klass_var(df: pd.DataFrame) -> pd.Series:
    hl = np.log(df["high"] / df["low"]) ** 2
    co = np.log(df["close"] / df["open"]) ** 2
    return (0.5 * hl - (2 * np.log(2) - 1) * co).clip(lower=0)


def forward_realized_vol(df: pd.DataFrame, horizon_days: int = 1, estimator: str = "parkinson") -> pd.Series:
    """Annualized realized volatility over the NEXT ``horizon_days`` sessions (t+1 .. t+h).

    The window is strictly in the future: nothing from bar ``t`` (the bar the
    features are computed on) enters the target.

    ``estimator``:
      - "parkinson": range-based (high/low). Far less noisy than |return|, which matters
        most at the 1-day horizon where a close-to-close target is a single squared return.
      - "close": classic close-to-close realized variance.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")
    if estimator == "parkinson":
        var = parkinson_var(df)
    elif estimator == "close":
        var = (df["log_ret"] if "log_ret" in df else np.log(df["close"]).diff()) ** 2
    else:
        raise ValueError("estimator must be 'parkinson' or 'close'")
    fwd_var = var.rolling(horizon_days, min_periods=horizon_days).mean().shift(-horizon_days)
    return (np.sqrt(fwd_var * TRADING_DAYS)).rename(f"rv_fwd_{horizon_days}d")


def hist_vol(df: pd.DataFrame, window_days: int = 20) -> pd.Series:
    """Trailing close-to-close vol (annualized)."""
    return (df["log_ret"].rolling(window_days, min_periods=window_days).std() * np.sqrt(TRADING_DAYS)).rename(
        f"hv_{window_days}d"
    )


def parkinson_vol(df: pd.DataFrame, window_days: int = 20) -> pd.Series:
    """Trailing Parkinson vol (annualized). Same estimator as the forecast target."""
    pk = np.sqrt(parkinson_var(df).rolling(window_days, min_periods=window_days).mean() * TRADING_DAYS)
    return pk.rename(f"pkv_{window_days}d")


def garman_klass_vol(df: pd.DataFrame, window_days: int = 20) -> pd.Series:
    gk = np.sqrt(garman_klass_var(df).rolling(window_days, min_periods=window_days).mean() * TRADING_DAYS)
    return gk.rename(f"gkv_{window_days}d")
