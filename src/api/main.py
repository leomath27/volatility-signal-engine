from __future__ import annotations

import json
import re
import time

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src import config
from src.config import SYMBOL
from src.models.inference import predict_latest
from src.models.train_xgb import DEFAULT_HORIZONS
from src.signals.generator import generate_signal

SYMBOL_RE = re.compile(r"^[A-Za-z0-9.\-^=]{1,12}$")


class SignalRequest(BaseModel):
    symbol: str = Field(default=SYMBOL, description="Ticker symbol supported by yfinance")
    horizon_days: int = Field(default=1, description="Forecast horizon in trading days (1, 3 or 5)")


app = FastAPI(title="Volatility Forecasting & Options Signal Engine", version="2.0.0")


def _build_signal(symbol: str, horizon_days: int) -> dict:
    if not SYMBOL_RE.match(symbol):
        raise HTTPException(status_code=422, detail="Invalid symbol")
    if horizon_days not in DEFAULT_HORIZONS:
        raise HTTPException(status_code=400, detail=f"horizon_days must be one of {DEFAULT_HORIZONS}")
    t0 = time.perf_counter()
    try:
        pred = predict_latest(symbol=symbol, horizon_days=horizon_days)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:  # not enough history
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:  # market data unavailable
        raise HTTPException(status_code=502, detail=f"Market data unavailable: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to produce signal: {exc}") from exc

    decision = generate_signal(
        symbol=symbol,
        horizon_days=horizon_days,
        predicted_vol=pred["predicted_vol"],
        baseline_vol=pred["baseline_vol"],
        recent_return_5d=pred["recent_return_5d"],
    )
    return {
        "as_of": pred["as_of"],
        "last_close": pred["last_close"],
        "forecast": {
            "predicted_vol": pred["predicted_vol"],
            "baseline_vol": pred["baseline_vol"],
            "recent_return_5d": pred["recent_return_5d"],
        },
        "signal": decision.to_dict(),
        "model": pred["model"],
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/models")
def models() -> dict:
    """Trained symbol/horizon models with their out-of-sample metrics."""
    out = []
    for meta_path in sorted(config.MODELS_DIR.glob("*/h*/metadata.json")):
        m = json.loads(meta_path.read_text())
        out.append({k: m.get(k) for k in ("symbol", "horizon_days", "trained_at", "last_training_date",
                                          "data_source", "metrics")})
    return {"models": out}


@app.post("/signal")
def signal(req: SignalRequest) -> dict:
    return _build_signal(req.symbol, req.horizon_days)


@app.get("/signal/{symbol}")
def signal_get(symbol: str, horizon_days: int = Query(1)) -> dict:
    return _build_signal(symbol, horizon_days)


@app.get("/signals")
def signals(symbols: str = Query(..., description="Comma-separated, max 20"), horizon_days: int = Query(5)) -> dict:
    """Batch endpoint: one bad symbol never fails the whole request."""
    out = {}
    for s in [x.strip() for x in symbols.split(",") if x.strip()][:20]:
        try:
            r = _build_signal(s, horizon_days)
            out[s] = {"as_of": r["as_of"], "strategy": r["signal"]["strategy"], "edge_pct": r["signal"]["edge_pct"],
                      "confidence": r["signal"]["confidence"], "predicted_vol": r["forecast"]["predicted_vol"]}
        except HTTPException as e:
            out[s] = {"error": e.detail}
    return out
