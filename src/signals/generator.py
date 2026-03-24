from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class SignalDecision:
    symbol: str
    horizon_days: int
    predicted_vol: float
    baseline_vol: float
    edge: float
    edge_pct: float
    strategy: str
    regime: str
    directional_bias: str
    confidence: float
    rationale: str
    suggested_dte: int

    def to_dict(self) -> dict:
        return asdict(self)


def generate_signal(
    *,
    symbol: str,
    horizon_days: int,
    predicted_vol: float,
    baseline_vol: float,
    recent_return_5d: float,
) -> SignalDecision:
    baseline = max(float(baseline_vol), 1e-8)
    predicted = float(predicted_vol)
    edge = predicted - baseline
    edge_pct = edge / baseline

    if recent_return_5d > 0.015:
        bias = "bullish"
    elif recent_return_5d < -0.015:
        bias = "bearish"
    else:
        bias = "neutral"

    if edge_pct >= 0.20:
        strategy = "LONG_VOL_STRADDLE"
        regime = "vol_expansion"
        rationale = "Forecasted volatility is materially above the recent realized baseline."
    elif edge_pct >= 0.08:
        strategy = "LONG_VOL_STRANGLE"
        regime = "vol_expansion"
        rationale = "Forecasted volatility is moderately above the recent realized baseline."
    elif edge_pct <= -0.15:
        strategy = "SHORT_PREMIUM_IRON_CONDOR"
        regime = "vol_compression"
        rationale = "Forecasted volatility is far below the recent realized baseline."
    elif edge_pct <= -0.08:
        strategy = "SHORT_STRANGLE"
        regime = "vol_compression"
        rationale = "Forecasted volatility is below the recent realized baseline."
    else:
        strategy = "NO_TRADE"
        regime = "range_bound"
        rationale = "Forecast edge is too small versus the recent realized baseline."

    confidence = min(abs(edge_pct) / 0.25, 1.0)
    suggested_dte = max(7, horizon_days * 5)

    return SignalDecision(
        symbol=symbol,
        horizon_days=horizon_days,
        predicted_vol=predicted,
        baseline_vol=baseline,
        edge=edge,
        edge_pct=edge_pct,
        strategy=strategy,
        regime=regime,
        directional_bias=bias,
        confidence=round(confidence, 4),
        rationale=rationale,
        suggested_dte=suggested_dte,
    )
