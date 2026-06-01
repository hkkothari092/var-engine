"""Parametric (Variance-Covariance) Value-at-Risk and CVaR."""
import numpy as np
import pandas as pd
from scipy import stats


def parametric_var(returns, confidence=0.95, horizon=1):
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    mu = returns.mean()
    sigma = returns.std()
    z = stats.norm.ppf(1 - confidence)
    var_1day = -(mu + z * sigma)
    return var_1day * np.sqrt(horizon)


def parametric_cvar(returns, confidence=0.95, horizon=1):
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    mu = returns.mean()
    sigma = returns.std()
    alpha = 1 - confidence
    z = stats.norm.ppf(alpha)
    cvar_1day = -(mu - sigma * stats.norm.pdf(z) / alpha)
    return cvar_1day * np.sqrt(horizon)


def parametric_summary(returns, confidences=(0.95, 0.99), horizons=(1, 10)):
    rows = []
    for conf in confidences:
        row = {"Confidence": f"{int(conf*100)}%"}
        for h in horizons:
            row[f"VaR_{h}d"] = parametric_var(returns, conf, h)
            row[f"CVaR_{h}d"] = parametric_cvar(returns, conf, h)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from src.data_loader import get_returns
    data = get_returns("NVDA", "2020-01-01")
    print(parametric_summary(data["log_return"]).to_string(index=False))
