"""Proxy options backtest driven by OUT-OF-SAMPLE walk-forward forecasts.

yfinance has no historical option chains, so each trade is priced with Black-Scholes
(r=q=0, flat vol) using trailing 20d close-to-close vol (x ``iv_markup``) as an
implied-vol proxy, then settled on the actual close ``horizon_days`` later. This tests
whether the forecasts carry information beyond trailing vol. It is NOT tradable P&L:
real IV usually exceeds realized vol (variance risk premium), has skew, and spreads are
wider. Stress it with ``iv_markup`` > 1 and a higher ``cost_pct``.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from src.config import SYMBOL, TRADING_DAYS
from src.features.engineer import engineer_features
from src.marketdata.market import fetch_market_data
from src.models.train_xgb import DEFAULT_HORIZONS, _model_dir
from src.signals.generator import (LONG_VOL_STRADDLE, LONG_VOL_STRANGLE, NO_TRADE, SHORT_PREMIUM_IRON_CONDOR,
                                   SHORT_STRANGLE, generate_signal)

LONG_VOL = {LONG_VOL_STRADDLE, LONG_VOL_STRANGLE}
SHORT_VOL = {SHORT_PREMIUM_IRON_CONDOR, SHORT_STRANGLE}


def _bs(S, K, sigma, T, call: bool) -> float:
    sd = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + 0.5 * sd**2) / sd
    d2 = d1 - sd
    return float(S * norm.cdf(d1) - K * norm.cdf(d2)) if call else float(K * norm.cdf(-d2) - S * norm.cdf(-d1))


def legs(strategy: str, S0: float, sigma: float, h: int) -> list[tuple[bool, float, int]]:
    """(is_call, strike, qty) per leg. Strikes are set in standard deviations of the h-day move."""
    sd = sigma * np.sqrt(h / TRADING_DAYS)

    def K(m: float) -> float:
        return S0 * float(np.exp(m * sd))

    if strategy == LONG_VOL_STRADDLE:
        return [(True, S0, 1), (False, S0, 1)]
    if strategy == LONG_VOL_STRANGLE:
        return [(True, K(0.5), 1), (False, K(-0.5), 1)]
    if strategy == SHORT_STRANGLE:
        return [(True, K(1.0), -1), (False, K(-1.0), -1)]
    if strategy == SHORT_PREMIUM_IRON_CONDOR:  # short 1sd strangle, long 2sd wings (defined risk)
        return [(True, K(1.0), -1), (False, K(-1.0), -1), (True, K(2.0), 1), (False, K(-2.0), 1)]
    raise ValueError(f"Unknown strategy {strategy}")


def trade_pnl(strategy: str, S0: float, S1: float, sigma: float, h: int, cost_pct: float = 0.02) -> float:
    """P&L as a fraction of spot. ``cost_pct`` is round-trip cost as a fraction of gross premium traded."""
    T = h / TRADING_DAYS
    price = payoff = gross = 0.0
    for is_call, K, q in legs(strategy, S0, sigma, h):
        p = _bs(S0, K, sigma, T, is_call)
        intrinsic = max(S1 - K, 0.0) if is_call else max(K - S1, 0.0)
        price += q * p
        payoff += q * intrinsic
        gross += abs(q) * p
    return (payoff - price - cost_pct * gross) / S0


def simulate(close: pd.Series, signals: pd.Series, pricing_vol: pd.Series, h: int,
             iv_markup: float = 1.0, cost_pct: float = 0.02) -> pd.DataFrame:
    """Non-overlapping trades: a new trade opens only after the previous one expires."""
    pos = {d: i for i, d in enumerate(close.index)}
    rows, next_ok = [], -1
    for d, strat in signals.items():
        i = pos.get(d)
        if strat == NO_TRADE or i is None or i < next_ok or i + h >= len(close):
            continue
        sigma = float(pricing_vol.get(d, np.nan)) * iv_markup
        if not np.isfinite(sigma) or sigma <= 0:
            continue
        S0, S1 = float(close.iloc[i]), float(close.iloc[i + h])
        rows.append({"date": d, "strategy": strat, "pnl": trade_pnl(strat, S0, S1, sigma, h, cost_pct)})
        next_ok = i + h
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(columns=["strategy", "pnl"])


def summarize(trades: pd.DataFrame, h: int) -> dict:
    if trades.empty:
        return {"trade_count": 0}
    p = trades["pnl"]
    sd = p.std(ddof=1)
    eq = p.cumsum()
    return {
        "trade_count": int(len(p)),
        "hit_rate": float((p > 0).mean()),
        "avg_pnl_pct_spot": float(p.mean() * 100),
        "total_pnl_pct_spot": float(p.sum() * 100),
        "sharpe_ann": float(p.mean() / sd * np.sqrt(TRADING_DAYS / h)) if sd and sd > 0 else None,
        "max_drawdown_pct_spot": float((eq - eq.cummax()).min() * 100),
        "worst_trade_pct_spot": float(p.min() * 100),
        "by_strategy": {k: {"n": int(len(g)), "avg_pnl_pct_spot": float(g["pnl"].mean() * 100),
                            "hit_rate": float((g["pnl"] > 0).mean())} for k, g in trades.groupby("strategy")},
    }


def run_backtest(symbol: str = SYMBOL, horizon_days: int = 1, iv_markup: float = 1.0, cost_pct: float = 0.02) -> dict:
    if horizon_days not in DEFAULT_HORIZONS:
        raise ValueError(f"Supported horizons: {DEFAULT_HORIZONS}")
    oos_path = _model_dir(symbol, horizon_days) / "oos_predictions.parquet"
    if not oos_path.exists():
        raise FileNotFoundError(f"No out-of-sample predictions for {symbol} h{horizon_days}. Train first.")
    oos = pd.read_parquet(oos_path)
    prices = fetch_market_data(symbol)
    feats = engineer_features(prices)
    close, pricing_vol = prices["close"], feats["hv_20"]

    strategies = pd.Series(
        [generate_signal(symbol=symbol, horizon_days=horizon_days, predicted_vol=float(r.pred),
                         baseline_vol=float(r.baseline), recent_return_5d=float(feats["ret_5"].get(d, 0.0))).strategy
         for d, r in oos.iterrows()], index=oos.index)

    kw = dict(h=horizon_days, iv_markup=iv_markup, cost_pct=cost_pct)
    trades = simulate(close, strategies, pricing_vol, **kw)
    bench = {
        "always_long_straddle": summarize(simulate(close, pd.Series(LONG_VOL_STRADDLE, index=oos.index), pricing_vol, **kw), horizon_days),
        "always_short_strangle": summarize(simulate(close, pd.Series(SHORT_STRANGLE, index=oos.index), pricing_vol, **kw), horizon_days),
    }
    ic = spearmanr(np.log(oos["pred"] / oos["baseline"]), np.log(oos["actual"] / oos["baseline"])).correlation
    summary = summarize(trades, horizon_days)
    return {
        "symbol": symbol, "horizon_days": horizon_days, "test_rows": int(len(oos)),
        "oos_period": [str(oos.index[0].date()), str(oos.index[-1].date())],
        "settings": {"iv_markup": iv_markup, "cost_pct_of_premium": cost_pct},
        "edge_ic": float(ic),  # forecast skill independent of option pricing
        "strategy_mix": dict(Counter(strategies)),
        **summary,
        "benchmarks": bench,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Proxy options backtest on out-of-sample forecasts")
    ap.add_argument("--symbol", default=SYMBOL)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--iv-markup", type=float, default=1.0)
    ap.add_argument("--cost", type=float, default=0.02)
    a = ap.parse_args()
    print(json.dumps(run_backtest(a.symbol, a.horizon, a.iv_markup, a.cost), indent=2))
