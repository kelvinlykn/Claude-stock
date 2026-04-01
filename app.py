"""
Stock Prediction Web App v6 - Full-featured application
Models: LSTM, XGBoost, LightGBM, Random Forest, ARIMA/SARIMAX,
        Prophet, SVR, Ridge Regression, Gradient Boosting, Ensemble
Features: Predictions, Backtesting, Technical Analysis, Risk Analytics,
          Stock Comparison, Correlation Matrix, Watchlist
"""

import os
import json
import traceback
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, render_template, request, jsonify

from prediction_engine import PredictionEngine
from risk_analytics import compute_risk_metrics, compute_correlation, compute_beta_alpha

app = Flask(__name__)
engine = PredictionEngine()

WATCHLIST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "watchlist.json")


def _load_watchlist():
    """Load watchlist from JSON file, returning empty list if file doesn't exist."""
    try:
        with open(WATCHLIST_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_watchlist(watchlist):
    """Save watchlist to JSON file."""
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(watchlist, f, indent=2)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        ticker = data.get("ticker", "AAPL").upper().strip()
        days = int(data.get("days", 30))
        days = max(1, min(days, 365))
        result = engine.predict(ticker, days)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/api/backtest", methods=["POST"])
def backtest():
    try:
        data = request.get_json()
        ticker = data.get("ticker", "AAPL").upper().strip()
        result = engine.backtest(ticker)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/api/technical", methods=["POST"])
def technical_analysis():
    try:
        data = request.get_json()
        ticker = data.get("ticker", "AAPL").upper().strip()
        result = engine.get_technical_analysis(ticker)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/api/risk", methods=["POST"])
def risk_analysis():
    try:
        data = request.get_json()
        ticker = data.get("ticker", "AAPL").upper().strip()

        df = engine._fetch_data(ticker)
        risk = compute_risk_metrics(df)
        beta_alpha = compute_beta_alpha(ticker)
        risk.update(beta_alpha)
        risk["ticker"] = ticker
        risk["current_price"] = round(float(df["Close"].iloc[-1]), 2)

        return jsonify(risk)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/api/correlation", methods=["POST"])
def correlation():
    try:
        data = request.get_json()
        tickers = [t.strip().upper() for t in data.get("tickers", ["AAPL", "MSFT", "GOOGL"])]
        if len(tickers) < 2:
            return jsonify({"error": "Need at least 2 tickers"}), 400

        result = compute_correlation(tickers)
        if result is None:
            return jsonify({"error": "Could not fetch data for tickers"}), 400
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/api/compare", methods=["POST"])
def compare():
    try:
        data = request.get_json()
        tickers = [t.strip().upper() for t in data.get("tickers", [])]
        if len(tickers) < 2:
            return jsonify({"error": "Need at least 2 tickers"}), 400

        results = {}
        for ticker in tickers[:5]:  # Max 5 tickers
            try:
                df = engine._fetch_data(ticker, period="1y")
                risk = compute_risk_metrics(df)
                current_price = round(float(df["Close"].iloc[-1]), 2)
                change_1y = round(float((df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100), 2)
                results[ticker] = {
                    "current_price": current_price,
                    "change_1y": change_1y,
                    "sharpe": risk["sharpe_ratio"],
                    "volatility": risk["annual_volatility"],
                    "max_drawdown": risk["max_drawdown"],
                    "win_rate": risk["win_rate"],
                    "prices": df["Close"].tail(252).tolist(),
                    "dates": [d.strftime("%Y-%m-%d") for d in df.tail(252).index]
                }
            except Exception:
                results[ticker] = {"error": "Could not fetch data"}

        return jsonify(results)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/api/check_alerts", methods=["POST"])
def check_alerts():
    """Check price alerts against current market prices."""
    try:
        data = request.get_json()
        alerts_list = data.get("alerts", [])
        if not alerts_list:
            return jsonify({"triggered": []})

        # Gather unique tickers
        tickers = list(set(a.get("ticker", "").upper().strip() for a in alerts_list if a.get("ticker")))
        if not tickers:
            return jsonify({"triggered": []})

        # Fetch current prices for all unique tickers
        current_prices = {}
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="5d")
                if not hist.empty:
                    current_prices[ticker] = round(float(hist["Close"].iloc[-1]), 2)
            except Exception:
                pass

        # Check each alert
        triggered = []
        for i, alert in enumerate(alerts_list):
            ticker = alert.get("ticker", "").upper().strip()
            target_price = float(alert.get("price", 0))
            direction = alert.get("direction", "above")

            if ticker not in current_prices:
                continue

            price = current_prices[ticker]
            if direction == "above" and price >= target_price:
                triggered.append({
                    "index": i,
                    "ticker": ticker,
                    "target_price": target_price,
                    "current_price": price,
                    "direction": direction
                })
            elif direction == "below" and price <= target_price:
                triggered.append({
                    "index": i,
                    "ticker": ticker,
                    "target_price": target_price,
                    "current_price": price,
                    "direction": direction
                })

        return jsonify({"triggered": triggered, "prices": current_prices})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/api/watchlist", methods=["GET"])
def get_watchlist():
    results = []
    for ticker in _load_watchlist():
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5d")
            if not hist.empty:
                price = round(float(hist["Close"].iloc[-1]), 2)
                change = round(float(hist["Close"].pct_change().iloc[-1] * 100), 2)
                results.append({"ticker": ticker, "price": price, "change": change})
        except Exception:
            results.append({"ticker": ticker, "price": 0, "change": 0})
    return jsonify({"watchlist": results})


@app.route("/api/watchlist", methods=["POST"])
def add_to_watchlist():
    data = request.get_json()
    ticker = data.get("ticker", "").upper().strip()
    watchlist = _load_watchlist()
    if ticker and ticker not in watchlist:
        watchlist.append(ticker)
        _save_watchlist(watchlist)
    return jsonify({"watchlist": watchlist})


@app.route("/api/watchlist", methods=["DELETE"])
def remove_from_watchlist():
    data = request.get_json()
    ticker = data.get("ticker", "").upper().strip()
    watchlist = _load_watchlist()
    if ticker in watchlist:
        watchlist.remove(ticker)
        _save_watchlist(watchlist)
    return jsonify({"watchlist": watchlist})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
