from __future__ import annotations

import os
from pathlib import Path

# Project root = repo root (override with VOL_HOME, e.g. for tests or containers)
PROJECT_ROOT = Path(os.environ.get("VOL_HOME", Path(__file__).resolve().parents[1]))

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

SYMBOL = "^GSPC"  # S&P 500 index: default symbol for the API and CLI
DEFAULT_SYMBOLS = ("^GSPC", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA")
HORIZONS = (1, 3, 5)  # forecast horizons in trading days
START_DATE = "2012-01-01"
END_DATE = None

# "yfinance" (live) or "synthetic" (offline GARCH data for tests/demos only)
DATA_SOURCE = os.environ.get("VOL_DATA_SOURCE", "yfinance")

# Inference refetches market data when the cache is older than this
LIVE_MAX_AGE_HOURS = 0.25

MLFLOW_EXPERIMENT = "volatility_forecasting"

RANDOM_STATE = 42
TRADING_DAYS = 252
