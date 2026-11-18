from __future__ import annotations

import math
from dataclasses import asdict, dataclass

LONG_VOL_STRADDLE = "LONG_VOL_STRADDLE"
LONG_VOL_STRANGLE = "LONG_VOL_STRANGLE"
SHORT_PREMIUM_IRON_CONDOR = "SHORT_PREMIUM_IRON_CONDOR"
SHORT_STRANGLE = "SHORT_STRANGLE"
NO_TRADE = "NO_TRADE"

# edge_pct = predicted_vol / baseline_vol - 1
STRADDLE_EDGE = 0.20
STRANGLE_EDGE = 0.08
CONDOR_EDGE = -0.15
SHORT_STRANGLE_EDGE = -0.08
BIAS_THRESHOLD = 0.015  # |5d return| for a directional tilt


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
    if not (math.isfinite(predicted_vol) and math.isfinite(baseline_vol)) or baseline_vol <= 0:
        raise ValueError("predicted_vol and baseline_vol must be finite, with baseline_vol > 0")
    baseline = float(baseline_vol)
    predicted = float(predicted_vol)
    edge = predicted - baseline
    edge_pct = edge / baseline

    if recent_return_5d > BIAS_THRESHOLD:
        bias = "bullish"
    elif recent_return_5d < -BIAS_THRESHOLD:
        bias = "bearish"
    else:
        bias = "neutral"

    if edge_pct >= STRADDLE_EDGE:
        strategy = LONG_VOL_STRADDLE
        regime = "vol_expansion"
        rationale = "Forecasted volatility is materially above the recent baseline (trailing 20d range-based vol)."
    elif edge_pct >= STRANGLE_EDGE:
        strategy = LONG_VOL_STRANGLE
        regime = "vol_expansion"
        rationale = "Forecasted volatility is moderately above the recent baseline (trailing 20d range-based vol)."
    elif edge_pct <= CONDOR_EDGE:
        strategy = SHORT_PREMIUM_IRON_CONDOR
        regime = "vol_compression"
        rationale = "Forecasted volatility is far below the recent baseline (trailing 20d range-based vol)."
    elif edge_pct <= SHORT_STRANGLE_EDGE:
        strategy = SHORT_STRANGLE
        regime = "vol_compression"
        rationale = "Forecasted volatility is below the recent baseline (trailing 20d range-based vol)."
    else:
        strategy = NO_TRADE
        regime = "range_bound"
        rationale = "Forecast edge is too small versus the recent baseline (trailing 20d range-based vol)."

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
