from __future__ import annotations

import json

import joblib

from src.models.train_xgb import build_dataset, _model_dir


def load_bundle(symbol: str, horizon_days: int) -> dict:
    d = _model_dir(symbol, horizon_days)
    model_path = d / "model.joblib"
    meta_path = d / "metadata.json"

    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Model artifacts not found for {symbol} horizon={horizon_days}. Train first."
        )

    model = joblib.load(model_path)
    metadata = json.loads(meta_path.read_text())
    return {
        "model": model,
        "features": metadata["features"],
        "target_col": metadata["target_col"],
        "metadata": metadata,
    }


def predict_latest(symbol: str, horizon_days: int) -> dict:
    bundle = load_bundle(symbol=symbol, horizon_days=horizon_days)
    data, _ = build_dataset(symbol=symbol, horizon_days=horizon_days, refresh=False)
    latest = data.iloc[-1]

    x = latest[bundle["features"]].to_frame().T
    pred = float(bundle["model"].predict(x)[0])

    return {
        "predicted_vol": pred,
        "baseline_vol": float(latest.get("hv_20", pred)),
        "recent_return_5d": float(latest.get("ret_5", 0.0)),
        "as_of": str(data.index[-1].date()),
        "raw_features": {k: float(latest[k]) for k in bundle["features"] if k in latest.index},
    }
