from __future__ import annotations

import json
from functools import lru_cache

import joblib
import pandas as pd

from src import config
from src.features.engineer import latest_feature_row
from src.marketdata.market import drop_incomplete_bar, fetch_market_data
from src.models.train_xgb import BASELINE_COL, _model_dir, predict_vol


@lru_cache(maxsize=64)
def _load(symbol: str, horizon_days: int, mtime: float) -> dict:
    d = _model_dir(symbol, horizon_days)
    metadata = json.loads((d / "metadata.json").read_text())
    return {
        "model": joblib.load(d / "model.joblib"),
        "features": metadata["features"],
        "target_col": metadata["target_col"],
        "smearing": metadata.get("smearing", 1.0),
        "metadata": metadata,
    }


def load_bundle(symbol: str, horizon_days: int) -> dict:
    """Load (and cache) a trained model. The cache refreshes automatically after retraining."""
    d = _model_dir(symbol, horizon_days)
    model_path, meta_path = d / "model.joblib", d / "metadata.json"
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Model artifacts not found for {symbol} horizon={horizon_days}. Train first.")
    return _load(symbol, horizon_days, meta_path.stat().st_mtime)


def predict_latest(symbol: str, horizon_days: int) -> dict:
    """Forecast vol over the next ``horizon_days`` sessions from the MOST RECENT completed bar."""
    bundle = load_bundle(symbol=symbol, horizon_days=horizon_days)
    prices = drop_incomplete_bar(fetch_market_data(symbol, max_age_hours=config.LIVE_MAX_AGE_HOURS))
    row, as_of = latest_feature_row(prices, bundle["features"])

    x = row[bundle["features"]].astype(float).to_frame().T
    pred = float(predict_vol(bundle["model"], bundle["smearing"], x)[0])
    meta = bundle["metadata"]
    return {
        "predicted_vol": pred,
        "baseline_vol": float(row[BASELINE_COL]),
        "hist_vol_20d": float(row["hv_20"]),
        "recent_return_5d": float(row["ret_5"]),
        "as_of": str(pd.Timestamp(as_of).date()),
        "last_close": float(row["close"]),
        "model": {
            "trained_at": meta.get("trained_at"),
            "data_source": meta.get("data_source"),
            "oos_qlike_gain_vs_naive_pct": meta["metrics"].get("qlike_gain_vs_naive_pct"),
        },
        "raw_features": {k: float(row[k]) for k in bundle["features"]},
    }
