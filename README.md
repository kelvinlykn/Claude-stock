# StockPredict AI

Multi-model ensemble stock prediction web application using 10 different machine learning and statistical methods for maximum accuracy.

## Prediction Models

| # | Model | Type |
|---|-------|------|
| 1 | LSTM Neural Network (Bidirectional) | Deep Learning |
| 2 | XGBoost | Gradient Boosting |
| 3 | LightGBM | Gradient Boosting |
| 4 | Random Forest | Ensemble |
| 5 | Gradient Boosting | Ensemble |
| 6 | Support Vector Regression (SVR) | Kernel |
| 7 | Ridge Regression | Linear |
| 8 | ARIMA/SARIMAX | Time Series |
| 9 | Facebook Prophet | Time Series |
| 10 | **Ensemble** (Inverse-MAE Weighted) | Meta-model |

## Features

- **Multi-Model Predictions** with confidence intervals
- **Monte Carlo Simulation** (500 paths) with probability distributions
- **Walk-Forward Validation** for realistic accuracy metrics
- **Technical Analysis** with 6+ indicators (RSI, MACD, Bollinger, Stochastic, etc.)
- **Risk Analytics** - VaR, CVaR, Sharpe, Sortino, Calmar, Max Drawdown, Beta, Alpha
- **Stock Comparison** - Normalized price charts and metrics
- **Correlation Matrix** - Return correlation heatmap
- **Watchlist** - Track multiple stocks
- **Price Alerts** - Set target price notifications
- **Export to CSV** - Download prediction data
- **Keyboard Shortcuts** - Quick navigation

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict` | POST | Run all prediction models |
| `/api/backtest` | POST | Backtest models on historical data |
| `/api/technical` | POST | Technical analysis signals |
| `/api/risk` | POST | Risk metrics (VaR, Sharpe, etc.) |
| `/api/compare` | POST | Compare multiple stocks |
| `/api/correlation` | POST | Correlation matrix |
| `/api/watchlist` | GET/POST/DELETE | Manage watchlist |

## Tech Stack

- **Backend**: Flask, scikit-learn, XGBoost, LightGBM, TensorFlow, statsmodels, Prophet
- **Frontend**: Vanilla JS, Plotly.js
- **Data**: Yahoo Finance (yfinance)

## Testing

```bash
python -m pytest tests/ -v
```
