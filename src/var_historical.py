"""Historical Value-at-Risk and Conditional VaR (Expected Shortfall).

Historical VaR makes NO distributional assumption — it reads the loss
directly from the empirical return distribution. The trade-off: it's
slow to react to regime changes and limited by the worst observed loss.
"""
import numpy as np
import pandas as pd


def historical_var(returns: pd.Series, confidence: float = 0.95,
                   horizon: int = 1) -> float:
    """
    Compute Historical VaR as a positive loss number.

    Args:
        returns: Series of daily log returns.
        confidence: e.g. 0.95 for 95% VaR.
        horizon: holding period in days. Uses sqrt-of-time scaling
                 (flagged as flawed in README — assumes i.i.d.).

    Returns:
        VaR as a POSITIVE decimal (e.g. 0.045 means 4.5% loss).
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    alpha = 1 - confidence                       # tail probability (e.g. 0.05)
    var_1day = -np.percentile(returns, alpha * 100)
    return var_1day * np.sqrt(horizon)


def historical_cvar(returns: pd.Series, confidence: float = 0.95,
                    horizon: int = 1) -> float:
    """
    Conditional VaR (Expected Shortfall): the AVERAGE loss in the
    worst (1 - confidence) of cases. Coherent risk measure — strictly
    superior to VaR for tail risk.

    Returns:
        CVaR as a POSITIVE decimal.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    alpha = 1 - confidence
    threshold = np.percentile(returns, alpha * 100)
    tail_losses = returns[returns <= threshold]
    cvar_1day = -tail_losses.mean()
    return cvar_1day * np.sqrt(horizon)


def var_summary(returns: pd.Series,
                confidences=(0.95, 0.99),
                horizons=(1, 10)) -> pd.DataFrame:
    """
    Build a clean VaR/CVaR results table.
    Rows: confidence levels. Columns: VaR and CVaR at each horizon.
    """
    rows = []
    for conf in confidences:
        row = {'Confidence': f"{int(conf*100)}%"}
        for h in horizons:
            row[f'VaR_{h}d']  = historical_var(returns, conf, h)
            row[f'CVaR_{h}d'] = historical_cvar(returns, conf, h)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Smoke test
    from src.data_loader import get_returns
    data = get_returns("NVDA", "2020-01-01")
    print(var_summary(data['log_return']).to_string(index=False))