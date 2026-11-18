from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from src import config
from src.config import HORIZONS, RANDOM_STATE
from src.features.engineer import DEFAULT_DROP_COLS, engineer_features, make_dataset
from src.features.realized_vol import forward_realized_vol
from src.marketdata.market import fetch_market_data, safe_symbol

log = logging.getLogger(__name__)

DEFAULT_HORIZONS = HORIZONS
EPS = 1e-6
N_SPLITS = 5
HAR_COLS = ["pkv_5", "pkv_20", "pkv_60"]  # HAR-RV style baseline inputs
BASELINE_COL = "pkv_20"  # naive baseline = trailing vol of the same estimator as the target

XGB_PARAMS = dict(
    n_estimators=800,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    reg_lambda=3.0,
    early_stopping_rounds=50,
    objective="reg:squarederror",
    random_state=RANDOM_STATE,
)


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


_safe_symbol = safe_symbol


def _model_dir(symbol: str, horizon_days: int) -> Path:
    return config.MODELS_DIR / safe_symbol(symbol) / f"h{horizon_days}"


# ------------------------------------------------------------------ dataset
def build_dataset(symbol: str, horizon_days: int, refresh: bool = False) -> tuple[pd.DataFrame, str]:
    df = fetch_market_data(symbol=symbol, refresh=refresh)
    feats = engineer_features(df)
    target_col = f"rv_fwd_{horizon_days}d"
    feats[target_col] = forward_realized_vol(feats, horizon_days=horizon_days)
    ds = make_dataset(feats, target_col=target_col)
    if ds.empty:
        raise RuntimeError("Dataset is empty after feature engineering/dropna")
    return ds, target_col


def feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    drop = set(DEFAULT_DROP_COLS) | {target_col}
    return [c for c in df.columns if c not in drop and not c.startswith("rv_fwd_") and pd.api.types.is_numeric_dtype(df[c])]


def time_split(df: pd.DataFrame, train_ratio: float = 0.7, valid_ratio: float = 0.15, gap: int = 0):
    """Chronological train/valid/test split. ``gap`` rows are purged between blocks so
    overlapping forward-vol labels cannot leak across a boundary."""
    n = len(df)
    i1, i2 = int(n * train_ratio), int(n * (train_ratio + valid_ratio))
    return df.iloc[: max(i1 - gap, 0)], df.iloc[i1:max(i2 - gap, i1)], df.iloc[i2:]


def _train_valid(df: pd.DataFrame, gap: int, val_frac: float = 0.15):
    k = int(len(df) * (1 - val_frac))
    return df.iloc[: max(k - gap, 1)], df.iloc[k:]


# ------------------------------------------------------------------ metrics
def qlike(y, f) -> float:
    """QLIKE loss on variances: the standard robust loss for volatility forecasts (lower is better)."""
    r = (np.asarray(y) / np.asarray(f)) ** 2
    return float(np.mean(r - np.log(r) - 1))


def score(y, f, base) -> dict:
    y, f, base = (np.asarray(a, dtype=float) for a in (y, f, base))
    ly, lf = np.log(y + EPS), np.log(f + EPS)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    if np.allclose(f, base):  # the baseline itself has no "edge" by construction
        ic = float("nan")
    else:
        ic = spearmanr(np.log((f + EPS) / (base + EPS)), np.log((y + EPS) / (base + EPS))).correlation
    return {
        "rmse": float(np.sqrt(np.mean((y - f) ** 2))),
        "mae": float(np.mean(np.abs(y - f))),
        "r2_log": float(1 - np.sum((ly - lf) ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
        "qlike": qlike(y + EPS, f + EPS),
        "edge_ic": float(ic),  # rank corr of predicted vs realized vol change relative to trailing vol
    }


# ------------------------------------------------------------------ fitting
def fit_model(X_tr, y_tr, X_va, y_va, params: dict | None = None):
    """XGBoost on the log RATIO of forward vol to trailing vol: log(y / pkv_20).

    Why a ratio target: tree models cannot extrapolate. When vol moves outside the range seen
    in training (e.g. a 2020-style spike), a model on raw vol levels is stuck at its training
    maximum. The ratio is stationary and is exactly the quantity the signals act on
    (edge_pct = predicted / baseline - 1). Returns (model, smearing); Duan's smearing factor
    corrects the bias of exp(E[log ratio]).
    """
    model = XGBRegressor(**(params or XGB_PARAMS))
    z_tr = np.log((y_tr + EPS) / (X_tr[BASELINE_COL] + EPS))
    z_va = np.log((y_va + EPS) / (X_va[BASELINE_COL] + EPS))
    model.fit(X_tr, z_tr, eval_set=[(X_va, z_va)], verbose=False)
    smear = float(np.mean(np.exp(z_va - model.predict(X_va))))
    return model, smear


def predict_vol(model, smear: float, X) -> np.ndarray:
    return (X[BASELINE_COL].to_numpy() + EPS) * np.exp(model.predict(X)) * smear


def evaluate_walk_forward(dataset, features, target_col, horizon, n_splits=N_SPLITS, params=None):
    """Expanding-window walk-forward CV. Returns (per-fold metrics, out-of-sample predictions)."""
    n = len(dataset)
    start = int(n * 0.4)
    size = (n - start) // n_splits
    folds, oos = [], []
    for i in range(n_splits):
        ts = start + i * size
        te = n if i == n_splits - 1 else ts + size
        train = dataset.iloc[: ts - horizon]  # purge: last label window ends before the test starts
        test = dataset.iloc[ts:te]
        tr, va = _train_valid(train, gap=horizon)
        if len(tr) < 200 or len(va) < 30 or test.empty:
            continue
        y_te = test[target_col].to_numpy()
        model, smear = fit_model(tr[features], tr[target_col], va[features], va[target_col], params)
        pred = predict_vol(model, smear, test[features])
        har = LinearRegression().fit(np.log(tr[HAR_COLS] + EPS), np.log(tr[target_col] + EPS))
        har_pred = np.exp(har.predict(np.log(test[HAR_COLS] + EPS)))
        naive = test[BASELINE_COL].to_numpy()

        row = {"fold": i, "n_test": len(test), "best_iteration": int(model.best_iteration)}
        for name, p in (("xgb", pred), ("har", har_pred), ("naive", naive)):
            for k, v in score(y_te, p, naive).items():
                row[f"{name}_{k}"] = v
        folds.append(row)
        oos.append(pd.DataFrame(
            {"pred": pred, "har": har_pred, "baseline": naive, "actual": y_te, "fold": i}, index=test.index))
    if not folds:
        raise RuntimeError("Not enough data for walk-forward evaluation. Need more history.")
    return pd.DataFrame(folds), pd.concat(oos)


# ------------------------------------------------------------------ MLflow
def _setup_mlflow():
    import mlflow

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{config.ARTIFACTS_DIR / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(config.MLFLOW_EXPERIMENT,
                                 artifact_location=(config.ARTIFACTS_DIR / "mlartifacts").resolve().as_uri())
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    return mlflow


def _log_run(symbol, horizon, params, features, rows, summary, folds, model, meta_path, importance, source):
    mlflow = _setup_mlflow()
    import mlflow.xgboost

    with mlflow.start_run(run_name=f"{safe_symbol(symbol)}_h{horizon}"):
        mlflow.set_tags({"symbol": symbol, "horizon_days": str(horizon), "data_source": source})
        mlflow.log_params({"symbol": symbol, "horizon_days": horizon, "n_features": len(features), "rows": rows,
                           "n_splits": len(folds), **{f"xgb_{k}": v for k, v in params.items()}})
        mlflow.log_metrics({k: float(v) for k, v in summary.items()})
        for _, r in folds.iterrows():
            mlflow.log_metric("fold_xgb_qlike", float(r["xgb_qlike"]), step=int(r["fold"]))
            mlflow.log_metric("fold_naive_qlike", float(r["naive_qlike"]), step=int(r["fold"]))
        try:
            mlflow.xgboost.log_model(model.get_booster(), name="model")  # MLflow >= 3
        except TypeError:  # MLflow 2.x
            mlflow.xgboost.log_model(model.get_booster(), artifact_path="model")
        mlflow.log_artifact(str(meta_path), artifact_path="metadata")
        imp_path = meta_path.parent / "feature_importance.csv"
        importance.to_csv(imp_path, header=["importance"])
        mlflow.log_artifact(str(imp_path), artifact_path="metadata")


# ------------------------------------------------------------------ training
def train_one(symbol: str, horizon_days: int, refresh_data: bool = False,
              n_splits: int = N_SPLITS, params: dict | None = None) -> TrainResult:
    params = params or XGB_PARAMS
    dataset, target_col = build_dataset(symbol=symbol, horizon_days=horizon_days, refresh=refresh_data)
    features = feature_columns(dataset, target_col)

    # 1) honest evaluation: walk-forward, purged, versus naive + HAR baselines
    folds, oos = evaluate_walk_forward(dataset, features, target_col, horizon_days, n_splits, params)
    summary = {}
    for col in folds.columns:
        if col not in ("fold", "n_test", "best_iteration"):
            summary[col] = float(folds[col].mean())
    summary["qlike_gain_vs_naive_pct"] = 100 * (1 - summary["xgb_qlike"] / summary["naive_qlike"])
    summary["qlike_gain_vs_har_pct"] = 100 * (1 - summary["xgb_qlike"] / summary["har_qlike"])

    # 2) final model on all data (trailing block for early stopping)
    tr, va = _train_valid(dataset, gap=horizon_days)
    model, smear = fit_model(tr[features], tr[target_col], va[features], va[target_col], params)
    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

    out_dir = _model_dir(symbol, horizon_days)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path, meta_path = out_dir / "model.joblib", out_dir / "metadata.json"
    joblib.dump(model, model_path)
    oos.to_parquet(out_dir / "oos_predictions.parquet")
    source = config.DATA_SOURCE if config.DATA_SOURCE != "yfinance" else "yfinance"
    metadata = {
        "symbol": symbol, "horizon_days": horizon_days, "target_col": target_col, "features": features,
        "smearing": smear, "metrics": summary, "rows": len(dataset), "n_splits": len(folds),
        "last_training_date": str(dataset.index[-1].date()), "data_source": source,
        "trained_at": datetime.now(timezone.utc).isoformat(), "params": params,
        "top_features": importance.head(8).round(4).to_dict(),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))

    try:
        _log_run(symbol, horizon_days, params, features, len(dataset), summary, folds, model, meta_path,
                 importance, source)
    except Exception as exc:  # tracking problems must never lose a trained model
        log.warning("MLflow logging failed for %s h%s: %s", symbol, horizon_days, exc)

    return TrainResult(symbol=symbol, horizon_days=horizon_days, rows=len(dataset), n_features=len(features),
                       train_rows=len(tr), valid_rows=len(va), test_rows=len(oos), metrics=summary,
                       model_path=str(model_path))


def train_many(symbol: str, horizons: Iterable[int] = DEFAULT_HORIZONS, refresh_data: bool = False,
               n_splits: int = N_SPLITS, params: dict | None = None) -> dict:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for i, h in enumerate(horizons):
        results.append(asdict(train_one(symbol, h, refresh_data=refresh_data and i == 0,
                                        n_splits=n_splits, params=params)))
    summary = {"symbol": symbol, "horizons": list(horizons), "results": results}
    path = config.MODELS_DIR / safe_symbol(symbol) / "training_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
    return summary
