import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.backtest.vol_backtest import run_backtest
from src.marketdata.market import fetch_market_data
from src.models import train_xgb
from src.models.inference import predict_latest

FAST = dict(train_xgb.XGB_PARAMS, n_estimators=150, learning_rate=0.1, early_stopping_rounds=15)


@pytest.fixture(scope="module")
def trained():
    for sym in ("TEST", "^GSPC"):
        train_xgb.train_many(sym, horizons=(1, 5), n_splits=3, params=FAST)
    return "TEST"


def test_training_reports_baselines(trained):
    r = train_xgb.train_one("TEST", 3, n_splits=3, params=FAST)
    for k in ("xgb_qlike", "har_qlike", "naive_qlike", "qlike_gain_vs_naive_pct", "xgb_edge_ic"):
        assert k in r.metrics
    assert r.metrics["xgb_r2_log"] > 0  # beats a constant on synthetic GARCH data


def test_predict_latest_uses_most_recent_bar(trained):
    p = predict_latest("TEST", 5)
    assert p["as_of"] == str(fetch_market_data("TEST").index[-1].date())
    assert 0.0 < p["predicted_vol"] < 3.0


def test_backtest_runs_on_oos_predictions(trained):
    b = run_backtest("TEST", 5)
    assert b["trade_count"] > 0 and "always_long_straddle" in b["benchmarks"]
    assert -1.0 <= b["edge_ic"] <= 1.0


def test_backtest_requires_training():
    with pytest.raises(FileNotFoundError):
        run_backtest("NOPE", 5)


def test_api(trained):
    c = TestClient(app)
    assert c.get("/health").json() == {"status": "ok"}
    r = c.post("/signal", json={"symbol": "TEST", "horizon_days": 5})
    assert r.status_code == 200
    d = r.json()
    assert d["signal"]["strategy"] and d["forecast"]["predicted_vol"] > 0 and d["latency_ms"] >= 0
    assert c.get("/signal/TEST", params={"horizon_days": 1}).status_code == 200
    assert c.post("/signal", json={"symbol": "TEST", "horizon_days": 2}).status_code == 400
    assert c.post("/signal", json={"symbol": "UNTRAINED", "horizon_days": 1}).status_code == 404
    assert c.post("/signal", json={"symbol": "$$$", "horizon_days": 1}).status_code == 422
    batch = c.get("/signals", params={"symbols": "TEST,UNTRAINED", "horizon_days": 5}).json()
    assert "strategy" in batch["TEST"] and "error" in batch["UNTRAINED"]
    assert {m["symbol"] for m in c.get("/models").json()["models"]} >= {"TEST"}
