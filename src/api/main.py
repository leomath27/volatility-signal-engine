from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import SYMBOL
from src.models.inference import predict_latest
from src.models.train_xgb import DEFAULT_HORIZONS
from src.signals.generator import generate_signal


class SignalRequest(BaseModel):
    symbol: str = Field(default=SYMBOL, description="Ticker symbol supported by yfinance")
    horizon_days: int = Field(default=1, description="Forecast horizon in trading days")


app = FastAPI(title="Volatility Forecasting & Options Signal Engine", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/signal")
def signal(req: SignalRequest) -> dict:
    if req.horizon_days not in DEFAULT_HORIZONS:
        raise HTTPException(status_code=400, detail=f"horizon_days must be one of {DEFAULT_HORIZONS}")

    try:
        pred = predict_latest(symbol=req.symbol, horizon_days=req.horizon_days)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to produce signal: {exc}") from exc

    decision = generate_signal(
        symbol=req.symbol,
        horizon_days=req.horizon_days,
        predicted_vol=pred["predicted_vol"],
        baseline_vol=pred["baseline_vol"],
        recent_return_5d=pred["recent_return_5d"],
    )

    return {
        "as_of": pred["as_of"],
        "forecast": {
            "predicted_vol": pred["predicted_vol"],
            "baseline_vol": pred["baseline_vol"],
            "recent_return_5d": pred["recent_return_5d"],
        },
        "signal": decision.to_dict(),
    }
