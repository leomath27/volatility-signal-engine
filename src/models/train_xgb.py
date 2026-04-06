from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import mlflow
import mlflow.xgboost
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.config import ARTIFACTS_DIR, DATA_DIR, MODELS_DIR, RANDOM_STATE
from src.features.engineer import DEFAULT_DROP_COLS, engineer_features, make_dataset
from src.features.realized_vol import forward_realized_vol

DEFAULT_HORIZONS = (1, 3, 5)


@dataclass
class TrainResult:
    symbol: str
    horizon_days: int
    rows: int
    n_features: int
    train_rows: int
    valid_rows: int
    test_rows: int
    metrics: dict
    model_path: str


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("^", "IDX_").replace("/", "_")


def _model_dir(symbol: str, horizon_days: int) -> Path:
    return MODELS_DIR / _safe_symbol(symbol) / f"h{horizon_days}"


def fetch_market_data(symbol: str, refresh: bool = False) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / f"{_safe_symbol(symbol)}.parquet"

    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    raw = yf.download(symbol, auto_adjust=False, progress=False)
    if raw.empty:
        raise RuntimeError(f"No market data returned for symbol={symbol}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.rename(columns=str.lower).rename(columns={"adj close": "adj_close"})
    expected = ["open", "high", "low", "close", "volume"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise RuntimeError(f"Market data missing required columns: {missing}")

    df = df.sort_index()
    df.to_parquet(cache)
    return df


def build_dataset(symbol: str, horizon_days: int, refresh: bool = False) -> tuple[pd.DataFrame, str]:
    df = fetch_market_data(symbol=symbol, refresh=refresh)
    feats = engineer_features(df)

    target_col = f"rv_fwd_{horizon_days}d"
    feats[target_col] = forward_realized_vol(feats, horizon_days=horizon_days)

    ds = make_dataset(feats, target_col=target_col)
    if ds.empty:
        raise RuntimeError("Dataset is empty after feature engineering/dropna")
    return ds, target_col


def time_split(df: pd.DataFrame, train_ratio: float = 0.7, valid_ratio: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    i1 = int(n * train_ratio)
    i2 = int(n * (train_ratio + valid_ratio))
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]


def _feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    drop_cols = set(DEFAULT_DROP_COLS) | {target_col}
    return [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]


def train_one(symbol: str, horizon_days: int, refresh_data: bool = False) -> TrainResult:
    dataset, target_col = build_dataset(symbol=symbol, horizon_days=horizon_days, refresh=refresh_data)
    train_df, valid_df, test_df = time_split(dataset)

    if len(test_df) < 10:
        raise RuntimeError("Test split is too small. Need more historical data.")

    features = _feature_columns(dataset, target_col)
    X_train, y_train = train_df[features], train_df[target_col]
    X_valid, y_valid = valid_df[features], valid_df[target_col]
    X_test, y_test = test_df[features], test_df[target_col]

    model = XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    pred_test = model.predict(X_test)
    metrics = {
        "rmse": float(mean_squared_error(y_test, pred_test) ** 0.5),
        "mae": float(mean_absolute_error(y_test, pred_test)),
        "r2": float(r2_score(y_test, pred_test)),
    }

    out_dir = _model_dir(symbol, horizon_days)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.joblib"
    meta_path = out_dir / "metadata.json"

    joblib.dump(model, model_path)
    metadata = {
        "symbol": symbol,
        "horizon_days": horizon_days,
        "target_col": target_col,
        "features": features,
        "metrics": metrics,
        "rows": len(dataset),
    }
    meta_path.write_text(json.dumps(metadata, indent=2))

    mlflow.set_tracking_uri(f"file://{ARTIFACTS_DIR / 'mlruns'}")
    mlflow.set_experiment("volatility_forecasting")
    with mlflow.start_run(run_name=f"{_safe_symbol(symbol)}_h{horizon_days}"):
        mlflow.log_params(
            {
                "symbol": symbol,
                "horizon_days": horizon_days,
                "n_features": len(features),
                "rows": len(dataset),
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model.get_booster(), artifact_path="model")
        mlflow.log_artifact(str(meta_path), artifact_path="metadata")

    return TrainResult(
        symbol=symbol,
        horizon_days=horizon_days,
        rows=len(dataset),
        n_features=len(features),
        train_rows=len(train_df),
        valid_rows=len(valid_df),
        test_rows=len(test_df),
        metrics=metrics,
        model_path=str(model_path),
    )


def train_many(symbol: str, horizons: Iterable[int] = DEFAULT_HORIZONS, refresh_data: bool = False) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    results = [asdict(train_one(symbol=symbol, horizon_days=h, refresh_data=refresh_data)) for h in horizons]
    summary = {"symbol": symbol, "horizons": list(horizons), "results": results}

    summary_path = MODELS_DIR / _safe_symbol(symbol) / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary
