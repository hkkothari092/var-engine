"""Monte Carlo Value-at-Risk under Normal and Student-t innovations.

Monte Carlo VaR simulates many possible next-day returns from a fitted
distribution, then reads VaR/CVaR off the simulated tail. Unlike historical
VaR (bounded by observed data) it can generate losses worse than anything
seen. Unlike parametric VaR (closed-form normal) it can use fat-tailed
distributions like Student-t that better match real returns.
"""
import numpy as np
import pandas as pd
from scipy import stats


def mc_var_normal(returns, confidence=0.95, horizon=1,
                  n_sims=100_000, seed=42):
    """Monte Carlo VaR/CVaR assuming returns ~ Normal(mu, sigma)."""
    rng = np.random.default_rng(seed)
    mu, sigma = returns.mean(), returns.std()

    sims = rng.normal(mu, sigma, n_sims)
    if horizon > 1:
        # sum of `horizon` i.i.d. daily draws per path
        sims = rng.normal(mu, sigma, (n_sims, horizon)).sum(axis=1)

    alpha = 1 - confidence
    var = -np.percentile(sims, alpha * 100)
    cvar = -sims[sims <= np.percentile(sims, alpha * 100)].mean()
    return var, cvar


def mc_var_student_t(returns, confidence=0.95, horizon=1,
                     n_sims=100_000, seed=42):
    """Monte Carlo VaR/CVaR assuming returns follow a fitted Student-t.

    Student-t has fatter tails than normal, controlled by degrees of
    freedom (df). Low df = fat tails. We FIT df to the data via MLE.
    """
    rng = np.random.default_rng(seed)

    # Fit Student-t to the returns (returns df, loc, scale)
    df, loc, scale = stats.t.fit(returns)

    sims = stats.t.rvs(df, loc=loc, scale=scale, size=n_sims,
                       random_state=rng)
    if horizon > 1:
        sims = stats.t.rvs(df, loc=loc, scale=scale,
                           size=(n_sims, horizon),
                           random_state=rng).sum(axis=1)

    alpha = 1 - confidence
    var = -np.percentile(sims, alpha * 100)
    cvar = -sims[sims <= np.percentile(sims, alpha * 100)].mean()
    return var, cvar, df


def mc_summary(returns, confidences=(0.95, 0.99), horizon=1,
               n_sims=100_000, seed=42):
    """Side-by-side MC Normal vs MC Student-t summary."""
    rows = []
    for conf in confidences:
        nv, ncv = mc_var_normal(returns, conf, horizon, n_sims, seed)
        tv, tcv, df = mc_var_student_t(returns, conf, horizon, n_sims, seed)
        rows.append({
            "Confidence": f"{int(conf*100)}%",
            "MC_Normal_VaR": nv,
            "MC_Normal_CVaR": ncv,
            "MC_t_VaR": tv,
            "MC_t_CVaR": tcv,
            "fitted_df": df,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from src.data_loader import get_returns
    data = get_returns("NVDA", "2020-01-01")
    print(mc_summary(data["log_return"]).to_string(index=False))