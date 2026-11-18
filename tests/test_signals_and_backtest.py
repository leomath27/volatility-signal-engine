import numpy as np
import pandas as pd
import pytest

from src.backtest.vol_backtest import legs, simulate, trade_pnl
from src.signals.generator import (LONG_VOL_STRADDLE, LONG_VOL_STRANGLE, NO_TRADE, SHORT_PREMIUM_IRON_CONDOR,
                                   SHORT_STRANGLE, generate_signal)


def sig(pred, base=0.20, ret=0.0):
    return generate_signal(symbol="X", horizon_days=5, predicted_vol=pred, baseline_vol=base, recent_return_5d=ret)


@pytest.mark.parametrize("pred,expected", [
    (0.26, LONG_VOL_STRADDLE), (0.23, LONG_VOL_STRANGLE), (0.20, NO_TRADE),
    (0.18, SHORT_STRANGLE), (0.16, SHORT_PREMIUM_IRON_CONDOR)])
def test_signal_regimes(pred, expected):
    assert sig(pred).strategy == expected


def test_signal_bias_and_validation():
    assert sig(0.2, ret=0.03).directional_bias == "bullish"
    assert sig(0.2, ret=-0.03).directional_bias == "bearish"
    with pytest.raises(ValueError):
        sig(float("nan"))
    with pytest.raises(ValueError):
        sig(0.2, base=0.0)


def test_atm_straddle_price_matches_rule_of_thumb():
    # ATM straddle ~= 0.8 * sigma * sqrt(T) * S; payoff of a 10% move = 10% of spot
    pnl = trade_pnl(LONG_VOL_STRADDLE, 100, 110, 0.2, 21, cost_pct=0.0)
    expected_premium = 0.8 * 0.2 * np.sqrt(21 / 252)
    assert pnl == pytest.approx(0.10 - expected_premium, abs=0.002)


def test_short_vol_profit_when_nothing_happens_and_condor_caps_loss():
    assert trade_pnl(SHORT_STRANGLE, 100, 100, 0.2, 5, 0.0) > 0
    crash = trade_pnl(SHORT_STRANGLE, 100, 70, 0.2, 5, 0.0)
    condor = trade_pnl(SHORT_PREMIUM_IRON_CONDOR, 100, 70, 0.2, 5, 0.0)
    assert condor > crash  # wings cap the loss
    assert len(legs(SHORT_PREMIUM_IRON_CONDOR, 100, 0.2, 5)) == 4


def test_costs_reduce_pnl():
    assert trade_pnl(LONG_VOL_STRANGLE, 100, 103, 0.2, 5, 0.05) < trade_pnl(LONG_VOL_STRANGLE, 100, 103, 0.2, 5, 0.0)


def test_simulate_non_overlapping():
    idx = pd.bdate_range("2024-01-01", periods=60)
    close = pd.Series(100 + np.random.default_rng(0).normal(0, 1, 60).cumsum(), index=idx)
    vol = pd.Series(0.2, index=idx)
    trades = simulate(close, pd.Series(LONG_VOL_STRADDLE, index=idx), vol, h=5)
    pos = [list(idx).index(d) for d in trades.index]
    assert all(b - a >= 5 for a, b in zip(pos, pos[1:])) and len(trades) > 5
    assert simulate(close, pd.Series(NO_TRADE, index=idx), vol, h=5).empty
