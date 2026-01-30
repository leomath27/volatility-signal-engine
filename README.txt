# Volatility Forecasting & Options Signal Engine

An end-to-end quantitative research project that forecasts short-term realized volatility and converts forecasts into options trading signals.

## Project Overview
- Forecasts 1–5 day realized volatility using ML models (XGBoost / LSTM)
- Engineers technical and volatility-based features
- Converts forecasts into options strategies (straddles / strangles)
- Backtests signals with risk constraints
- Tracks experiments with MLflow
- Exposes signals via a lightweight FastAPI endpoint

## Tech Stack
- Python 3.11
- pandas, numpy, scikit-learn, xgboost
- yfinance (market data)
- MLflow (experiment tracking)
- FastAPI (signal API)

## Project Structure
