# Volatility Forecasting & Options Signal Engine

An end-to-end quantitative research project that forecasts short-term realized volatility and converts forecasts into options trading signals.

## What this project does
- Forecasts forward **1/3/5-day realized volatility** with XGBoost.
- Builds predictive features from Yahoo Finance OHLCV data (returns, realized/historical vol, Parkinson vol, momentum, RSI, volume factors).
- Tracks training runs and metrics in **MLflow**.
- Produces options-style regime signals (`LONG_VOL_STRADDLE`, `LONG_VOL_STRANGLE`, `SHORT_STRANGLE`, etc.).
- Serves live forecasts/signals through a lightweight **FastAPI** API.

## Tech stack
- Python 3.11
- pandas, NumPy, scikit-learn, XGBoost
- yfinance for market data
- MLflow for experiment tracking
- FastAPI + Uvicorn for API serving

## Project layout
- `src/features/`: feature engineering + realized volatility targets
- `src/models/train_xgb.py`: dataset building, splitting, training, MLflow logging
- `src/models/inference.py`: model loading and latest prediction generation
- `src/signals/generator.py`: trading signal decision logic
- `src/backtest/vol_backtest.py`: simple proxy backtest
- `src/api/main.py`: FastAPI service
- `src/experiments/run_pipeline.py`: end-to-end training entrypoint

## Quickstart

### 1) Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Train models (1/3/5 day horizons)
```bash
python -m src.experiments.run_pipeline
```

Artifacts are saved to:
- `data/` market cache
- `models/` trained model + metadata
- `artifacts/mlruns/` MLflow tracking files

### 3) Run a quick backtest
```bash
python -c "from src.backtest.vol_backtest import run_backtest; print(run_backtest())"
```

### 4) Start the API
```bash
uvicorn src.api.main:app --reload --port 8000
```

### 5) Request a signal
```bash
curl -X POST "http://127.0.0.1:8000/signal" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"^GSPC", "horizon_days":1}'
```

## Notes
- You must train at least one model before calling `/signal`.
- Yahoo Finance data availability may vary by symbol and date.
- This repository is for research and educational use, **not investment advice**.
