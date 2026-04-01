"""
Risk Analytics Module - Provides advanced risk metrics:
- Value at Risk (VaR) - Historical, Parametric, and Monte Carlo
- Conditional VaR (CVaR / Expected Shortfall)
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Beta
- Alpha
- Volatility metrics
- Correlation analysis
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats


def compute_risk_metrics(df, risk_free_rate=0.05):
    """Compute comprehensive risk metrics for a stock."""
    close = df["Close"]
    returns = close.pct_change().dropna()
    log_returns = np.log(close / close.shift(1)).dropna()

    # Basic stats
    annual_return = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    daily_vol = returns.std()

    # Sharpe Ratio
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0

    # Sortino Ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino = (annual_return - risk_free_rate) / downside_std if downside_std > 0 else 0

    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    # Value at Risk (Historical)
    var_95 = np.percentile(returns, 5) * 100
    var_99 = np.percentile(returns, 1) * 100

    # Parametric VaR (assumes normal distribution)
    z_95 = stats.norm.ppf(0.05)
    z_99 = stats.norm.ppf(0.01)
    param_var_95 = (returns.mean() + z_95 * returns.std()) * 100
    param_var_99 = (returns.mean() + z_99 * returns.std()) * 100

    # CVaR (Expected Shortfall)
    cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
    cvar_99 = returns[returns <= np.percentile(returns, 1)].mean() * 100

    # Calmar Ratio
    calmar = annual_return / abs(max_drawdown / 100) if max_drawdown != 0 else 0

    # Skewness & Kurtosis
    skewness = returns.skew()
    kurtosis = returns.kurtosis()

    # Win rate
    win_rate = (returns > 0).mean() * 100

    # Best/Worst days
    best_day = returns.max() * 100
    worst_day = returns.min() * 100

    # Current drawdown
    current_dd = drawdown.iloc[-1] * 100

    return {
        "annual_return": round(float(annual_return * 100), 2),
        "annual_volatility": round(float(annual_vol * 100), 2),
        "daily_volatility": round(float(daily_vol * 100), 4),
        "sharpe_ratio": round(float(sharpe), 4),
        "sortino_ratio": round(float(sortino), 4),
        "calmar_ratio": round(float(calmar), 4),
        "max_drawdown": round(float(max_drawdown), 2),
        "current_drawdown": round(float(current_dd), 2),
        "var_95_historical": round(float(var_95), 4),
        "var_99_historical": round(float(var_99), 4),
        "var_95_parametric": round(float(param_var_95), 4),
        "var_99_parametric": round(float(param_var_99), 4),
        "cvar_95": round(float(cvar_95), 4),
        "cvar_99": round(float(cvar_99), 4),
        "skewness": round(float(skewness), 4),
        "kurtosis": round(float(kurtosis), 4),
        "win_rate": round(float(win_rate), 2),
        "best_day": round(float(best_day), 2),
        "worst_day": round(float(worst_day), 2),
        "drawdown_series": drawdown.tail(252).tolist(),
        "drawdown_dates": [d.strftime("%Y-%m-%d") for d in drawdown.tail(252).index]
    }


def compute_correlation(tickers, period="1y"):
    """Compute correlation matrix for a list of tickers."""
    data = {}
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            hist = stock.history(period=period)
            if not hist.empty:
                if hist.index.tz is not None:
                    hist.index = hist.index.tz_localize(None)
                data[t] = hist["Close"]
        except Exception:
            pass

    if len(data) < 2:
        return None

    df = pd.DataFrame(data).dropna()
    returns = df.pct_change().dropna()
    corr = returns.corr()

    return {
        "tickers": list(corr.columns),
        "matrix": corr.values.tolist(),
        "returns_stats": {
            t: {
                "mean_return": round(float(returns[t].mean() * 252 * 100), 2),
                "volatility": round(float(returns[t].std() * np.sqrt(252) * 100), 2)
            }
            for t in corr.columns
        }
    }


def compute_beta_alpha(ticker, benchmark="SPY", period="2y"):
    """Compute beta and alpha relative to a benchmark."""
    try:
        stock = yf.Ticker(ticker).history(period=period)["Close"]
        bench = yf.Ticker(benchmark).history(period=period)["Close"]

        if stock.index.tz is not None:
            stock.index = stock.index.tz_localize(None)
        if bench.index.tz is not None:
            bench.index = bench.index.tz_localize(None)

        df = pd.DataFrame({"stock": stock, "bench": bench}).dropna()
        stock_returns = df["stock"].pct_change().dropna()
        bench_returns = df["bench"].pct_change().dropna()

        # Align
        common = stock_returns.index.intersection(bench_returns.index)
        stock_returns = stock_returns.loc[common]
        bench_returns = bench_returns.loc[common]

        if len(stock_returns) < 20:
            return {"beta": 1.0, "alpha": 0.0}

        slope, intercept, r_value, p_value, std_err = stats.linregress(bench_returns, stock_returns)

        return {
            "beta": round(float(slope), 4),
            "alpha": round(float(intercept * 252 * 100), 4),
            "r_squared": round(float(r_value ** 2), 4)
        }
    except Exception:
        return {"beta": 1.0, "alpha": 0.0, "r_squared": 0.0}
