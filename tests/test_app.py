"""Comprehensive tests for Stock Prediction App v8."""
import pytest
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestIndexPage:
    def test_loads(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert b"StockPredict AI" in response.data

    def test_has_tabs(self, client):
        response = client.get("/")
        assert b"Predict" in response.data
        assert b"Risk Analysis" in response.data
        assert b"Compare" in response.data
        assert b"Watchlist" in response.data


class TestPredictAPI:
    def test_exists(self, client):
        response = client.post("/api/predict",
                              data=json.dumps({"ticker": "AAPL", "days": 5}),
                              content_type="application/json")
        assert response.status_code in [200, 400]

    def test_invalid_ticker(self, client):
        response = client.post("/api/predict",
                              data=json.dumps({"ticker": "XXXINVALID123", "days": 5}),
                              content_type="application/json")
        assert response.status_code == 400

    def test_days_clamping(self, client):
        response = client.post("/api/predict",
                              data=json.dumps({"ticker": "AAPL", "days": 1000}),
                              content_type="application/json")
        assert response.status_code in [200, 400]


class TestBacktestAPI:
    def test_exists(self, client):
        response = client.post("/api/backtest",
                              data=json.dumps({"ticker": "AAPL"}),
                              content_type="application/json")
        assert response.status_code in [200, 400]


class TestTechnicalAPI:
    def test_exists(self, client):
        response = client.post("/api/technical",
                              data=json.dumps({"ticker": "AAPL"}),
                              content_type="application/json")
        assert response.status_code in [200, 400]


class TestRiskAPI:
    def test_exists(self, client):
        response = client.post("/api/risk",
                              data=json.dumps({"ticker": "AAPL"}),
                              content_type="application/json")
        assert response.status_code in [200, 400]

    def test_invalid_ticker(self, client):
        response = client.post("/api/risk",
                              data=json.dumps({"ticker": "XXXINVALID123"}),
                              content_type="application/json")
        assert response.status_code == 400


class TestCorrelationAPI:
    def test_exists(self, client):
        response = client.post("/api/correlation",
                              data=json.dumps({"tickers": ["AAPL", "MSFT"]}),
                              content_type="application/json")
        assert response.status_code in [200, 400]

    def test_minimum_tickers(self, client):
        response = client.post("/api/correlation",
                              data=json.dumps({"tickers": ["AAPL"]}),
                              content_type="application/json")
        assert response.status_code == 400


class TestCompareAPI:
    def test_exists(self, client):
        response = client.post("/api/compare",
                              data=json.dumps({"tickers": ["AAPL", "MSFT"]}),
                              content_type="application/json")
        assert response.status_code in [200, 400]

    def test_minimum_tickers(self, client):
        response = client.post("/api/compare",
                              data=json.dumps({"tickers": ["AAPL"]}),
                              content_type="application/json")
        assert response.status_code == 400


class TestWatchlistAPI:
    def test_get_empty(self, client):
        response = client.get("/api/watchlist")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "watchlist" in data

    def test_add_and_get(self, client):
        client.post("/api/watchlist",
                    data=json.dumps({"ticker": "AAPL"}),
                    content_type="application/json")
        response = client.get("/api/watchlist")
        assert response.status_code == 200

    def test_delete(self, client):
        client.post("/api/watchlist",
                    data=json.dumps({"ticker": "TEST"}),
                    content_type="application/json")
        response = client.delete("/api/watchlist",
                                data=json.dumps({"ticker": "TEST"}),
                                content_type="application/json")
        assert response.status_code == 200


class TestRiskAnalytics:
    def test_imports(self):
        from risk_analytics import compute_risk_metrics, compute_correlation, compute_beta_alpha

    def test_risk_metrics_computation(self):
        import pandas as pd
        import numpy as np
        from risk_analytics import compute_risk_metrics

        dates = pd.date_range("2023-01-01", periods=500)
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(500) * 2)
        prices = np.maximum(prices, 10)  # Ensure positive
        df = pd.DataFrame({
            "Close": prices,
            "High": prices * 1.01,
            "Low": prices * 0.99,
            "Volume": np.random.randint(1000000, 10000000, 500)
        }, index=dates)

        result = compute_risk_metrics(df)
        assert "sharpe_ratio" in result
        assert "max_drawdown" in result
        assert "var_95_historical" in result
        assert "sortino_ratio" in result
        assert isinstance(result["sharpe_ratio"], float)


class TestPredictionEngine:
    def test_imports(self):
        from prediction_engine import PredictionEngine, compute_technical_indicators

    def test_technical_indicators(self):
        import pandas as pd
        import numpy as np
        from prediction_engine import compute_technical_indicators

        dates = pd.date_range("2023-01-01", periods=300)
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(300))
        df = pd.DataFrame({
            "Open": prices * 0.99,
            "High": prices * 1.02,
            "Low": prices * 0.98,
            "Close": prices,
            "Volume": np.random.randint(1000000, 10000000, 300)
        }, index=dates)

        result = compute_technical_indicators(df)
        assert "RSI_14" in result.columns
        assert "MACD" in result.columns
        assert "SMA_50" in result.columns
        assert "BB_upper_20" in result.columns
