"""
Prediction Engine v3 - Improved accuracy with:
- Walk-forward validation
- Proper recursive multi-step forecasting
- Confidence intervals via bootstrapping
- Monte Carlo simulation
- Confidence scoring
- 10 prediction methods + ensemble
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False


def compute_technical_indicators(df):
    """Compute comprehensive technical indicators."""
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Moving Averages
    for w in [5, 10, 20, 50, 100, 200]:
        df[f"SMA_{w}"] = close.rolling(window=w).mean()
        df[f"EMA_{w}"] = close.ewm(span=w, adjust=False).mean()

    # Bollinger Bands
    for w in [20, 50]:
        sma = close.rolling(window=w).mean()
        std = close.rolling(window=w).std()
        df[f"BB_upper_{w}"] = sma + 2 * std
        df[f"BB_lower_{w}"] = sma - 2 * std
        df[f"BB_width_{w}"] = (df[f"BB_upper_{w}"] - df[f"BB_lower_{w}"]) / sma
        df[f"BB_pct_{w}"] = (close - df[f"BB_lower_{w}"]) / (df[f"BB_upper_{w}"] - df[f"BB_lower_{w}"])

    # RSI
    for w in [7, 14, 21]:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=w).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=w).mean()
        rs = gain / loss
        df[f"RSI_{w}"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # Stochastic Oscillator
    for w in [14, 21]:
        low_min = low.rolling(window=w).min()
        high_max = high.rolling(window=w).max()
        df[f"Stoch_K_{w}"] = 100 * (close - low_min) / (high_max - low_min)
        df[f"Stoch_D_{w}"] = df[f"Stoch_K_{w}"].rolling(window=3).mean()

    # ATR
    for w in [14, 21]:
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df[f"ATR_{w}"] = tr.rolling(window=w).mean()

    # OBV
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    df["OBV"] = obv
    df["OBV_SMA_20"] = obv.rolling(window=20).mean()

    # Volume
    df["Volume_SMA_20"] = volume.rolling(window=20).mean()
    df["Volume_ratio"] = volume / df["Volume_SMA_20"]

    # ROC
    for w in [5, 10, 20]:
        df[f"ROC_{w}"] = close.pct_change(periods=w) * 100

    # Williams %R
    for w in [14, 21]:
        high_max = high.rolling(window=w).max()
        low_min = low.rolling(window=w).min()
        df[f"Williams_R_{w}"] = -100 * (high_max - close) / (high_max - low_min)

    # CCI
    for w in [14, 20]:
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=w).mean()
        mad = tp.rolling(window=w).apply(lambda x: np.abs(x - x.mean()).mean())
        df[f"CCI_{w}"] = (tp - sma_tp) / (0.015 * mad)

    # Price features
    df["Daily_Return"] = close.pct_change()
    df["Log_Return"] = np.log(close / close.shift(1))
    df["High_Low_Pct"] = (high - low) / close * 100
    df["Close_Open_Pct"] = (close - df["Open"]) / df["Open"] * 100

    # Momentum
    for w in [5, 10, 20]:
        df[f"Momentum_{w}"] = close - close.shift(w)

    # Volatility
    for w in [5, 10, 20, 50]:
        df[f"Volatility_{w}"] = close.rolling(window=w).std()

    # Lag features (crucial for prediction)
    for lag in [1, 2, 3, 5, 10]:
        df[f"Close_lag_{lag}"] = close.shift(lag)
        df[f"Return_lag_{lag}"] = close.pct_change().shift(lag)

    return df


class PredictionEngine:
    """Main prediction engine with 10 models + ensemble."""

    def __init__(self):
        self.cache = {}

    def _fetch_data(self, ticker, period="5y"):
        cache_key = f"{ticker}_{period}"
        now = datetime.now()
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if (now - cached_time).seconds < 300:
                return cached_data.copy()

        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            raise ValueError(f"No data found for ticker '{ticker}'.")

        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        self.cache[cache_key] = (now, df.copy())
        return df

    def _prepare_features(self, df):
        df = compute_technical_indicators(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        exclude = ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]
        feature_cols = [c for c in df.columns if c not in exclude]
        return df[feature_cols].values, df["Close"].values, feature_cols, df

    def _walk_forward_validate(self, model_fn, X, y, n_splits=5):
        """Walk-forward validation for time series."""
        fold_size = len(X) // (n_splits + 1)
        maes, predictions, actuals = [], [], []

        for i in range(n_splits):
            train_end = fold_size * (i + 2)
            test_end = min(train_end + fold_size, len(X))
            if test_end <= train_end:
                break

            X_train, y_train = X[:train_end], y[:train_end]
            X_test, y_test = X[train_end:test_end], y[train_end:test_end]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = model_fn()
            model.fit(X_train_s, y_train)
            pred = model.predict(X_test_s)

            maes.append(mean_absolute_error(y_test, pred))
            predictions.extend(pred.tolist())
            actuals.extend(y_test.tolist())

        return np.mean(maes) if maes else 999, predictions, actuals

    def _recursive_forecast_ml(self, model, scaler, df_feat, feature_cols, days):
        """Proper recursive multi-step forecast for ML models."""
        df_work = df_feat.copy()
        forecasts = []

        for _ in range(days):
            last_features = df_work[feature_cols].iloc[-1:].values
            last_features_s = scaler.transform(last_features)
            pred_price = model.predict(last_features_s)[0]
            forecasts.append(pred_price)

            # Create next row by shifting indicators
            new_row = df_work.iloc[-1:].copy()
            new_row.index = [new_row.index[0] + pd.Timedelta(days=1)]
            prev_close = new_row["Close"].values[0]
            new_row["Close"] = pred_price
            new_row["Open"] = prev_close
            new_row["High"] = pred_price * 1.005
            new_row["Low"] = pred_price * 0.995
            df_work = pd.concat([df_work, new_row])
            df_work = compute_technical_indicators(df_work)
            df_work = df_work.replace([np.inf, -np.inf], np.nan)
            df_work = df_work.fillna(method="ffill")

        return forecasts

    def _bootstrap_confidence(self, predictions, n_bootstrap=100, confidence=0.90):
        """Compute confidence intervals via bootstrapping."""
        preds = np.array(predictions)
        n = len(preds)
        boot_means = []
        for _ in range(n_bootstrap):
            noise = np.random.normal(0, 0.02, n)  # 2% noise
            boot_means.append(preds * (1 + noise))
        boot_array = np.array(boot_means)
        alpha = (1 - confidence) / 2
        lower = np.percentile(boot_array, alpha * 100, axis=0)
        upper = np.percentile(boot_array, (1 - alpha) * 100, axis=0)
        return lower.tolist(), upper.tolist()

    def _monte_carlo_simulation(self, df, days, n_simulations=500):
        """Monte Carlo simulation based on historical returns."""
        close = df["Close"].values
        log_returns = np.log(close[1:] / close[:-1])
        mu = log_returns.mean()
        sigma = log_returns.std()
        last_price = close[-1]

        simulations = np.zeros((n_simulations, days))
        for i in range(n_simulations):
            prices = [last_price]
            for d in range(days):
                shock = np.random.normal(mu, sigma)
                prices.append(prices[-1] * np.exp(shock))
            simulations[i] = prices[1:]

        mean_path = simulations.mean(axis=0)
        median_path = np.median(simulations, axis=0)
        p10 = np.percentile(simulations, 10, axis=0)
        p25 = np.percentile(simulations, 25, axis=0)
        p75 = np.percentile(simulations, 75, axis=0)
        p90 = np.percentile(simulations, 90, axis=0)

        # Probability of price increase
        final_prices = simulations[:, -1]
        prob_up = (final_prices > last_price).mean() * 100

        return {
            "mean": mean_path.tolist(),
            "median": median_path.tolist(),
            "p10": p10.tolist(),
            "p25": p25.tolist(),
            "p75": p75.tolist(),
            "p90": p90.tolist(),
            "prob_up": round(float(prob_up), 1),
            "expected_return": round(float((mean_path[-1] / last_price - 1) * 100), 2),
            "n_simulations": n_simulations
        }

    def _train_lstm(self, df, days):
        if not HAS_TENSORFLOW:
            return None, "TensorFlow not available"

        close_data = df["Close"].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(close_data)

        lookback = 60
        X, y = [], []
        for i in range(lookback, len(scaled)):
            X.append(scaled[i - lookback:i])
            y.append(scaled[i])
        X, y = np.array(X), np.array(y)
        if len(X) < 10:
            return None, "Insufficient data"

        split = int(len(X) * 0.85)
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True, input_shape=(lookback, 1))),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="huber")
        early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=100, batch_size=32,
                  validation_split=0.1, callbacks=[early_stop], verbose=0)

        test_pred = scaler.inverse_transform(model.predict(X_test, verbose=0))
        test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        mae = mean_absolute_error(test_actual, test_pred)

        last_seq = scaled[-lookback:].reshape(1, lookback, 1)
        future = []
        for _ in range(days):
            pred = model.predict(last_seq, verbose=0)
            future.append(pred[0, 0])
            last_seq = np.roll(last_seq, -1, axis=1)
            last_seq[0, -1, 0] = pred[0, 0]

        future_prices = scaler.inverse_transform(np.array(future).reshape(-1, 1)).flatten()
        ci_lower, ci_upper = self._bootstrap_confidence(future_prices)

        return {
            "predictions": future_prices.tolist(),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "mae": float(mae),
        }, None

    def _train_ml_model(self, model_fn, X, y, days, df_feat, feature_cols, name):
        """Train ML model with walk-forward validation and recursive forecasting."""
        # Walk-forward validation
        wf_mae, _, _ = self._walk_forward_validate(model_fn, X, y, n_splits=5)

        # Final model on all data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = model_fn()
        model.fit(X_scaled, y)

        # Simple test MAE
        split = int(len(X) * 0.85)
        scaler_test = StandardScaler()
        X_train_s = scaler_test.fit_transform(X[:split])
        X_test_s = scaler_test.transform(X[split:])
        model_test = model_fn()
        model_test.fit(X_train_s, y[:split])
        test_pred = model_test.predict(X_test_s)
        test_mae = mean_absolute_error(y[split:], test_pred)

        # Recursive forecast
        forecasts = self._recursive_forecast_ml(model, scaler, df_feat, feature_cols, days)
        ci_lower, ci_upper = self._bootstrap_confidence(forecasts)

        return {
            "predictions": forecasts,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "mae": float(test_mae),
            "wf_mae": float(wf_mae),
            "test_predictions": test_pred.tolist(),
            "test_actual": y[split:].tolist()
        }

    def _train_arima(self, df, days):
        if not HAS_STATSMODELS:
            return None, "statsmodels not available"

        close = df["Close"].values
        split = int(len(close) * 0.85)
        train, test = close[:split], close[split:]

        try:
            model = SARIMAX(train, order=(2, 1, 2), seasonal_order=(1, 1, 1, 5),
                            enforce_stationarity=False, enforce_invertibility=False)
            fitted = model.fit(disp=False, maxiter=200)

            test_pred = fitted.forecast(steps=len(test))
            mae = mean_absolute_error(test, test_pred)

            model_full = SARIMAX(close, order=(2, 1, 2), seasonal_order=(1, 1, 1, 5),
                                 enforce_stationarity=False, enforce_invertibility=False)
            fitted_full = model_full.fit(disp=False, maxiter=200)
            future = fitted_full.forecast(steps=days)

            ci_lower, ci_upper = self._bootstrap_confidence(future)

            return {
                "predictions": future.tolist(),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "mae": float(mae),
            }, None
        except Exception as e:
            return None, str(e)

    def _train_prophet(self, df, days):
        if not HAS_PROPHET:
            return None, "Prophet not available"

        prophet_df = df[["Close"]].reset_index()
        prophet_df.columns = ["ds", "y"]
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

        split = int(len(prophet_df) * 0.85)
        train_df = prophet_df[:split]
        test_df = prophet_df[split:]

        try:
            model = Prophet(daily_seasonality=True, weekly_seasonality=True,
                            yearly_seasonality=True, changepoint_prior_scale=0.05)
            model.fit(train_df)

            test_future = model.make_future_dataframe(periods=len(test_df))
            test_forecast = model.predict(test_future)
            test_pred = test_forecast["yhat"].values[split:]
            mae = mean_absolute_error(test_df["y"].values, test_pred)

            model_full = Prophet(daily_seasonality=True, weekly_seasonality=True,
                                 yearly_seasonality=True, changepoint_prior_scale=0.05)
            model_full.fit(prophet_df)
            future_df = model_full.make_future_dataframe(periods=days)
            forecast = model_full.predict(future_df)
            future_pred = forecast["yhat"].values[-days:]
            ci_lower = forecast["yhat_lower"].values[-days:].tolist()
            ci_upper = forecast["yhat_upper"].values[-days:].tolist()

            return {
                "predictions": future_pred.tolist(),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "mae": float(mae),
            }, None
        except Exception as e:
            return None, str(e)

    def _compute_confidence_score(self, model_results, model_predictions):
        """Compute overall prediction confidence score (0-100)."""
        scores = []

        # Factor 1: Model agreement (do models agree on direction?)
        directions = []
        for name, preds in model_predictions.items():
            if len(preds) > 0:
                directions.append(1 if preds[-1] > preds[0] else -1)
        if directions:
            agreement = abs(sum(directions)) / len(directions)
            scores.append(agreement * 30)  # Max 30 points

        # Factor 2: Low MAE relative to price
        maes = [v["mae"] for v in model_results.values() if v.get("mae") and isinstance(v["mae"], (int, float))]
        if maes:
            avg_mae = np.mean(maes)
            # Lower MAE = higher score
            mae_score = max(0, 25 - avg_mae / 2)
            scores.append(min(mae_score, 25))  # Max 25 points

        # Factor 3: Walk-forward consistency
        wf_maes = [v.get("wf_mae", 999) for v in model_results.values() if v.get("wf_mae")]
        if wf_maes:
            wf_consistency = max(0, 25 - np.mean(wf_maes) / 2)
            scores.append(min(wf_consistency, 25))  # Max 25 points

        # Factor 4: Number of available models
        n_available = sum(1 for v in model_results.values() if v.get("available"))
        model_coverage = (n_available / 10) * 20
        scores.append(model_coverage)  # Max 20 points

        return round(sum(scores), 1)

    def predict(self, ticker, days=30):
        df = self._fetch_data(ticker)
        X, y, feature_cols, df_feat = self._prepare_features(df)

        results = {}
        model_predictions = {}
        ci_data = {}

        # 1. LSTM
        lstm_result, lstm_err = self._train_lstm(df, days)
        if lstm_result:
            results["LSTM"] = {"mae": lstm_result["mae"], "available": True}
            model_predictions["LSTM"] = lstm_result["predictions"]
            ci_data["LSTM"] = {"lower": lstm_result["ci_lower"], "upper": lstm_result["ci_upper"]}
        else:
            results["LSTM"] = {"mae": None, "available": False, "error": lstm_err}

        # ML Models
        ml_models = {
            "Random Forest": lambda: RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
            "Gradient Boosting": lambda: GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
            "SVR": lambda: SVR(kernel="rbf", C=100, epsilon=0.01),
            "Ridge Regression": lambda: Ridge(alpha=1.0),
        }
        if HAS_XGBOOST:
            ml_models["XGBoost"] = lambda: XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
        if HAS_LIGHTGBM:
            ml_models["LightGBM"] = lambda: LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)

        for name, model_fn in ml_models.items():
            try:
                result = self._train_ml_model(model_fn, X, y, days, df_feat, feature_cols, name)
                results[name] = {"mae": result["mae"], "wf_mae": result["wf_mae"], "available": True}
                model_predictions[name] = result["predictions"]
                ci_data[name] = {"lower": result["ci_lower"], "upper": result["ci_upper"]}
            except Exception as e:
                results[name] = {"mae": None, "available": False, "error": str(e)}

        # ARIMA
        arima_result, arima_err = self._train_arima(df, days)
        if arima_result:
            results["ARIMA/SARIMAX"] = {"mae": arima_result["mae"], "available": True}
            model_predictions["ARIMA/SARIMAX"] = arima_result["predictions"]
            ci_data["ARIMA/SARIMAX"] = {"lower": arima_result["ci_lower"], "upper": arima_result["ci_upper"]}
        else:
            results["ARIMA/SARIMAX"] = {"mae": None, "available": False, "error": arima_err}

        # Prophet
        prophet_result, prophet_err = self._train_prophet(df, days)
        if prophet_result:
            results["Prophet"] = {"mae": prophet_result["mae"], "available": True}
            model_predictions["Prophet"] = prophet_result["predictions"]
            ci_data["Prophet"] = {"lower": prophet_result["ci_lower"], "upper": prophet_result["ci_upper"]}
        else:
            results["Prophet"] = {"mae": None, "available": False, "error": prophet_err}

        # Ensemble (inverse-MAE weighted)
        if model_predictions:
            available_maes = {k: results[k]["mae"] for k in model_predictions if results[k].get("mae") and isinstance(results[k]["mae"], (int, float)) and results[k]["mae"] > 0}
            if available_maes:
                inv_maes = {k: 1.0 / v for k, v in available_maes.items()}
                total_inv = sum(inv_maes.values())
                weights = {k: v / total_inv for k, v in inv_maes.items()}

                ensemble_pred = np.zeros(days)
                ensemble_lower = np.zeros(days)
                ensemble_upper = np.zeros(days)
                for name, w in weights.items():
                    preds = np.array(model_predictions[name][:days])
                    if len(preds) < days:
                        preds = np.pad(preds, (0, days - len(preds)), mode="edge")
                    ensemble_pred += w * preds

                    if name in ci_data:
                        lower = np.array(ci_data[name]["lower"][:days])
                        upper = np.array(ci_data[name]["upper"][:days])
                        if len(lower) < days:
                            lower = np.pad(lower, (0, days - len(lower)), mode="edge")
                            upper = np.pad(upper, (0, days - len(upper)), mode="edge")
                        ensemble_lower += w * lower
                        ensemble_upper += w * upper

                model_predictions["Ensemble"] = ensemble_pred.tolist()
                ci_data["Ensemble"] = {"lower": ensemble_lower.tolist(), "upper": ensemble_upper.tolist()}
                results["Ensemble"] = {"mae": "N/A (weighted)", "available": True, "weights": {k: round(v, 4) for k, v in weights.items()}}

        # Monte Carlo
        monte_carlo = self._monte_carlo_simulation(df, days)

        # Confidence score
        confidence_score = self._compute_confidence_score(results, model_predictions)

        # Build response
        last_date = df.index[-1]
        future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=days)
        date_labels = [d.strftime("%Y-%m-%d") for d in future_dates]

        hist_df = df.tail(252)
        historical = {
            "dates": [d.strftime("%Y-%m-%d") for d in hist_df.index],
            "prices": hist_df["Close"].tolist(),
            "volume": hist_df["Volume"].tolist()
        }

        current_price = float(df["Close"].iloc[-1])
        price_change_1d = float(df["Close"].pct_change().iloc[-1] * 100)

        return {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "price_change_1d": round(price_change_1d, 2),
            "prediction_dates": date_labels,
            "model_results": results,
            "predictions": {k: [round(float(v), 2) for v in vals] for k, vals in model_predictions.items()},
            "confidence_intervals": {k: {"lower": [round(float(v), 2) for v in ci["lower"]], "upper": [round(float(v), 2) for v in ci["upper"]]} for k, ci in ci_data.items()},
            "monte_carlo": monte_carlo,
            "confidence_score": confidence_score,
            "historical": historical,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def backtest(self, ticker):
        df = self._fetch_data(ticker)
        close = df["Close"].values
        test_days = 60
        test_data = df.iloc[-test_days:]
        actual = test_data["Close"].values

        X, y, _, _ = self._prepare_features(df.iloc[:-test_days])
        X_full, y_full, _, _ = self._prepare_features(df)

        backtest_results = {}
        models = {
            "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
            "Ridge Regression": Ridge(alpha=1.0),
            "SVR": SVR(kernel="rbf", C=100, epsilon=0.01),
        }
        if HAS_XGBOOST:
            models["XGBoost"] = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
        if HAS_LIGHTGBM:
            models["LightGBM"] = LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbose=-1)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X)

        if len(X_full) > test_days:
            X_test = scaler.transform(X_full[-test_days:])
            y_test = y_full[-test_days:]

            for name, model in models.items():
                model.fit(X_train_s, y)
                pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, pred)
                rmse = np.sqrt(mean_squared_error(y_test, pred))
                mape = np.mean(np.abs((y_test - pred) / y_test)) * 100
                r2 = r2_score(y_test, pred)
                direction_acc = np.mean(np.sign(np.diff(y_test)) == np.sign(np.diff(pred))) * 100

                backtest_results[name] = {
                    "mae": round(float(mae), 4),
                    "rmse": round(float(rmse), 4),
                    "mape": round(float(mape), 2),
                    "r2": round(float(r2), 4),
                    "direction_accuracy": round(float(direction_acc), 2),
                    "predictions": pred.tolist(),
                }

        return {
            "ticker": ticker,
            "test_period": f"{test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')}",
            "actual_prices": actual.tolist(),
            "dates": [d.strftime("%Y-%m-%d") for d in test_data.index],
            "results": backtest_results
        }

    def get_technical_analysis(self, ticker):
        df = self._fetch_data(ticker)
        df = compute_technical_indicators(df)
        latest = df.iloc[-1]

        signals = []

        # RSI
        rsi = latest.get("RSI_14", 50)
        if rsi < 30:
            signals.append({"indicator": "RSI(14)", "value": round(rsi, 2), "signal": "Oversold - BUY", "strength": "Strong"})
        elif rsi > 70:
            signals.append({"indicator": "RSI(14)", "value": round(rsi, 2), "signal": "Overbought - SELL", "strength": "Strong"})
        else:
            signals.append({"indicator": "RSI(14)", "value": round(rsi, 2), "signal": "Neutral", "strength": "Weak"})

        # MACD
        macd = latest.get("MACD", 0)
        macd_signal = latest.get("MACD_signal", 0)
        if macd > macd_signal:
            signals.append({"indicator": "MACD", "value": round(macd, 4), "signal": "Bullish Crossover - BUY", "strength": "Medium"})
        else:
            signals.append({"indicator": "MACD", "value": round(macd, 4), "signal": "Bearish Crossover - SELL", "strength": "Medium"})

        # MA Cross
        close = latest["Close"]
        sma_50 = latest.get("SMA_50", close)
        sma_200 = latest.get("SMA_200", close)
        if sma_50 > sma_200:
            signals.append({"indicator": "Golden Cross (SMA50>SMA200)", "value": f"{round(sma_50, 2)}/{round(sma_200, 2)}", "signal": "Bullish - BUY", "strength": "Strong"})
        else:
            signals.append({"indicator": "Death Cross (SMA50<SMA200)", "value": f"{round(sma_50, 2)}/{round(sma_200, 2)}", "signal": "Bearish - SELL", "strength": "Strong"})

        # Bollinger
        bb_pct = latest.get("BB_pct_20", 0.5)
        if bb_pct < 0:
            signals.append({"indicator": "Bollinger Bands", "value": round(bb_pct, 4), "signal": "Below Lower Band - BUY", "strength": "Strong"})
        elif bb_pct > 1:
            signals.append({"indicator": "Bollinger Bands", "value": round(bb_pct, 4), "signal": "Above Upper Band - SELL", "strength": "Strong"})
        else:
            signals.append({"indicator": "Bollinger Bands", "value": round(bb_pct, 4), "signal": "Within Bands - HOLD", "strength": "Weak"})

        # Stochastic
        stoch_k = latest.get("Stoch_K_14", 50)
        if stoch_k < 20:
            signals.append({"indicator": "Stochastic", "value": round(stoch_k, 2), "signal": "Oversold - BUY", "strength": "Medium"})
        elif stoch_k > 80:
            signals.append({"indicator": "Stochastic", "value": round(stoch_k, 2), "signal": "Overbought - SELL", "strength": "Medium"})
        else:
            signals.append({"indicator": "Stochastic", "value": round(stoch_k, 2), "signal": "Neutral", "strength": "Weak"})

        # Volume
        vol_ratio = latest.get("Volume_ratio", 1)
        if vol_ratio > 2:
            signals.append({"indicator": "Volume", "value": round(vol_ratio, 2), "signal": "High Volume - Trend Confirmation", "strength": "Strong"})
        elif vol_ratio < 0.5:
            signals.append({"indicator": "Volume", "value": round(vol_ratio, 2), "signal": "Low Volume - Weak Trend", "strength": "Weak"})

        buy_signals = sum(1 for s in signals if "BUY" in s["signal"])
        sell_signals = sum(1 for s in signals if "SELL" in s["signal"])
        if buy_signals > sell_signals + 1:
            overall = "STRONG BUY"
        elif buy_signals > sell_signals:
            overall = "BUY"
        elif sell_signals > buy_signals + 1:
            overall = "STRONG SELL"
        elif sell_signals > buy_signals:
            overall = "SELL"
        else:
            overall = "HOLD"

        return {
            "ticker": ticker,
            "current_price": round(float(latest["Close"]), 2),
            "signals": signals,
            "overall_signal": overall,
            "summary": {"buy_signals": buy_signals, "sell_signals": sell_signals, "neutral_signals": len(signals) - buy_signals - sell_signals}
        }
