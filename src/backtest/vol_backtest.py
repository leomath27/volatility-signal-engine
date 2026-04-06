from __future__ import annotations

from collections import Counter

from src.config import SYMBOL
from src.models.inference import load_bundle
from src.models.train_xgb import DEFAULT_HORIZONS, build_dataset, time_split
from src.signals.generator import generate_signal


LONG_VOL = {"LONG_VOL_STRADDLE", "LONG_VOL_STRANGLE"}
SHORT_VOL = {"SHORT_PREMIUM_IRON_CONDOR", "SHORT_STRANGLE"}


def run_backtest(symbol: str = SYMBOL, horizon_days: int = 1) -> dict:
    if horizon_days not in DEFAULT_HORIZONS:
        raise ValueError(f"Supported horizons: {DEFAULT_HORIZONS}")

    bundle = load_bundle(symbol=symbol, horizon_days=horizon_days)
    data, target_col = build_dataset(symbol=symbol, horizon_days=horizon_days, refresh=False)
    _, _, test_df = time_split(data)

    features = bundle["features"]
    preds = bundle["model"].predict(test_df[features])

    decisions = []
    pnl_proxy = []
    hit_flags = []

    for i, (_, row) in enumerate(test_df.iterrows()):
        pred = float(preds[i])
        baseline = float(row["hv_20"])
        actual = float(row[target_col])
        signal = generate_signal(
            symbol=symbol,
            horizon_days=horizon_days,
            predicted_vol=pred,
            baseline_vol=baseline,
            recent_return_5d=float(row["ret_5"]),
        )
        decisions.append(signal.strategy)

        if signal.strategy in LONG_VOL:
            edge = actual - baseline
        elif signal.strategy in SHORT_VOL:
            edge = baseline - actual
        else:
            edge = 0.0

        edge -= 0.01
        pnl_proxy.append(edge)
        hit_flags.append(edge > 0)

    trades = [p for p, s in zip(pnl_proxy, decisions) if s != "NO_TRADE"]
    trade_hits = [h for h, s in zip(hit_flags, decisions) if s != "NO_TRADE"]
    strategy_counts = Counter(decisions)

    return {
        "symbol": symbol,
        "horizon_days": horizon_days,
        "test_rows": int(len(test_df)),
        "trade_count": int(len(trades)),
        "hit_rate": float(sum(trade_hits) / len(trade_hits)) if trade_hits else 0.0,
        "avg_edge": float(sum(trades) / len(trades)) if trades else 0.0,
        "total_edge": float(sum(trades)) if trades else 0.0,
        "strategy_mix": dict(strategy_counts),
    }
