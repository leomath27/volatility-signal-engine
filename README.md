# Volatility Forecasting & Options Signal Engine

An end-to-end quant research project: forecast **1/3/5-day forward realized volatility** with XGBoost, convert the
forecasts into **options regime signals** (long straddle/strangle, short strangle, iron condor), validate them with an
**out-of-sample proxy options backtest**, track experiments in **MLflow**, and serve live signals through **FastAPI**.

**Stack:** Python 3.10+, pandas, NumPy, SciPy, scikit-learn, XGBoost, yfinance, MLflow, FastAPI, Uvicorn, pytest

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.experiments.run_pipeline                    # train 1/3/5d models for the default symbols, evaluate, backtest
python -m src.experiments.run_pipeline --symbols SPY NVDA --horizons 1 5
python -m src.backtest.vol_backtest --symbol "^GSPC" --horizon 5 --iv-markup 1.10 --cost 0.05
uvicorn src.api.main:app --port 8000                      # docs at /docs
mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db
pytest                                                    # offline: uses synthetic GARCH data
```

```bash
curl -X POST localhost:8000/signal -H "Content-Type: application/json" -d '{"symbol":"^GSPC","horizon_days":5}'
curl "localhost:8000/signal/NVDA?horizon_days=1"
curl "localhost:8000/signals?symbols=^GSPC,QQQ,AAPL&horizon_days=5"     # batch, one bad symbol never fails the rest
curl localhost:8000/models                                              # trained models + out-of-sample metrics
```

Train a symbol before requesting it (the API returns 404 with a hint otherwise).

## How it works

| Stage | Where | Notes |
|---|---|---|
| Data | `src/marketdata/market.py` | yfinance daily OHLCV, **split/dividend adjusted**, parquet cache with TTL, stale-cache fallback, incomplete intraday bar dropped |
| Target | `src/features/realized_vol.py` | Forward h-day **Parkinson (range-based)** vol over days t+1..t+h. Far less noisy than \|return\| at the 1-day horizon |
| Features | `src/features/engineer.py` | Close-to-close, Parkinson and Garman-Klass vol at 5/10/20/60d, vol term structure, vol-of-vol, percentile rank, downside vol, drawdown, momentum, RSI, volume z-score, gap and range stats |
| Model | `src/models/train_xgb.py` | XGBoost on `log(forward vol / trailing vol)`, early stopping, Duan smearing for the back-transform |
| Validation | `src/models/train_xgb.py` | **Expanding-window walk-forward**, purge gap = horizon, compared with a naive baseline and a **HAR-RV** linear model. Metrics: RMSE, MAE, **QLIKE**, log R², rank IC |
| Signals | `src/signals/generator.py` | `edge_pct = predicted / baseline - 1` maps to LONG_VOL_STRADDLE, LONG_VOL_STRANGLE, SHORT_STRANGLE, SHORT_PREMIUM_IRON_CONDOR or NO_TRADE |
| Backtest | `src/backtest/vol_backtest.py` | Black-Scholes priced structures on **out-of-sample** forecasts, non-overlapping trades, costs, always-on benchmarks |
| Tracking | MLflow (SQLite under `artifacts/`) | Params, per-fold metrics, baselines, model, feature importance per symbol/horizon |
| Serving | `src/api/main.py` | Cached model loading (auto-reloads after retraining), fresh data per request, input validation, latency reporting |

## Design decisions worth knowing

- **Ratio target.** Trees cannot extrapolate. A model on raw vol levels is stuck at its training maximum when a volatility spike
  arrives; predicting `log(forward / trailing)` is stationary and is exactly the quantity the signals trade on.
- **Strictly forward target and purged validation.** The target never includes the bar the features are computed on, and a gap
  equal to the horizon separates train from test so overlapping label windows cannot leak.
- **Honest baselines.** The model has to beat trailing vol and a HAR-RV regression on out-of-sample QLIKE. The pipeline prints the gain.
- **Backtest uses only out-of-sample predictions**, never a final model that has seen the test period.

## Read before trusting the backtest

yfinance has no historical option chains. The backtest prices each trade with Black-Scholes using trailing 20-day
close-to-close vol (times `--iv-markup`) as an implied-vol proxy and settles on the actual close `h` days later.
It tests whether the forecasts add information beyond trailing vol, **not** whether the strategies are tradable.
Real implied vol usually sits above realized vol (the variance risk premium), has skew, and spreads are wider than the
default 2% of premium. Short-vol structures have fat tail losses (see `worst_trade_pct_spot`). Stress it with
`--iv-markup 1.10 --cost 0.05`; if the edge disappears, it was not an edge. A real test needs historical implied vol
(e.g. CBOE DataShop, OptionMetrics, Polygon).

The API's live bar: if the US market is still open, today's partial bar is dropped so features always come from a completed session.

## Tests

`pytest` runs offline on synthetic GARCH data (`VOL_DATA_SOURCE=synthetic`) in an isolated temp directory. It covers:
look-ahead checks on features, target alignment, a regression test that today's bar is not in its own target, a regression test that
inference uses the latest bar, walk-forward purging, option pricing sanity, signal regimes, and the training-to-API round trip.
Never report model or backtest results produced from synthetic data.

## Layout

```
src/config.py            paths, symbols, horizons (override root with VOL_HOME)
src/marketdata/          yfinance loader + synthetic generator
src/features/            realized-vol estimators, feature engineering
src/models/              training (walk-forward, MLflow) and inference
src/signals/             regime -> options strategy logic
src/backtest/            proxy options P&L backtest
src/api/                 FastAPI service
src/experiments/         end-to-end pipeline entrypoint
tests/
```

## Ideas to extend

- Replace the IV proxy with real implied vol and trade on the IV-vs-forecast spread
- Earnings-date and VIX term-structure features
- Quantile XGBoost for forecast intervals; Optuna search logged as MLflow child runs
- MLflow Model Registry with alias-based loading in the API
