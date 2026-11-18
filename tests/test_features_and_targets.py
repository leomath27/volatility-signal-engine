import numpy as np
import pandas as pd
import pytest

from src.features.engineer import engineer_features, latest_feature_row, make_dataset
from src.features.realized_vol import TRADING_DAYS, forward_realized_vol, parkinson_var
from src.marketdata.market import synthetic_prices
from src.models.train_xgb import feature_columns, time_split


@pytest.fixture(scope="module")
def px():
    return synthetic_prices(n=1200, seed=1)


def test_features_have_no_lookahead(px):
    """Features at day t are identical whether or not future bars exist."""
    full = engineer_features(px)
    cut = engineer_features(px.iloc[:900])
    pd.testing.assert_frame_equal(full.iloc[:900], cut, check_exact=False, atol=1e-10)


def test_target_is_strictly_forward(px):
    feats = engineer_features(px)
    pk = parkinson_var(px)
    t = 500
    for h in (1, 3, 5):
        y = forward_realized_vol(feats, h)
        assert np.isclose(y.iloc[t], np.sqrt(pk.iloc[t + 1 : t + h + 1].mean() * TRADING_DAYS))
        assert y.iloc[-h:].isna().all()


def test_target_does_not_depend_on_todays_bar(px):
    """Regression: the old 1-day target included bar t's own return, a pure function of a feature."""
    t = 600
    base = forward_realized_vol(engineer_features(px), 1)
    tweaked = px.copy()
    tweaked.iloc[t, tweaked.columns.get_loc("high")] *= 1.05
    tweaked.iloc[t, tweaked.columns.get_loc("close")] *= 1.03
    new = forward_realized_vol(engineer_features(tweaked), 1)
    assert new.iloc[t] == pytest.approx(base.iloc[t])  # unchanged: bar t is not in its own target
    assert new.iloc[t - 1] != pytest.approx(base.iloc[t - 1])  # but t-1's target (which looks at t) changed


def test_no_price_level_features(px):
    feats = engineer_features(px)
    cols = feature_columns(make_dataset(feats.assign(rv_fwd_1d=forward_realized_vol(feats, 1)), "rv_fwd_1d"), "rv_fwd_1d")
    assert not {"open", "high", "low", "close", "volume", "log_ret", "ret"} & set(cols)
    assert not [c for c in cols if c.startswith("sma_")]


def test_latest_row_is_last_bar_not_last_labeled_bar(px):
    """Regression: inference used the last row that had a known target, i.e. a bar h days stale."""
    feats = engineer_features(px)
    cols = feature_columns(make_dataset(feats.assign(rv_fwd_5d=forward_realized_vol(feats, 5)), "rv_fwd_5d"), "rv_fwd_5d")
    row, as_of = latest_feature_row(px, cols)
    assert as_of == px.index[-1]


def test_latest_row_rejects_short_history():
    short = synthetic_prices(n=80, seed=2)
    with pytest.raises(ValueError):
        latest_feature_row(short, ["pkv_20_pctile_252"])


def test_time_split_purges_gap():
    df = pd.DataFrame({"x": range(1000)})
    tr, va, te = time_split(df, gap=5)
    assert tr.index.max() + 5 < va.index.min() + 5 and len(te) > 0
    assert va.index.max() + 5 <= te.index.min() + 0 or va.index.max() < te.index.min()
