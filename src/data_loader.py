"""Data loader for VaR engine. Pulls prices and computes log returns."""
import yfinance as yf
import numpy as np
import pandas as pd


def get_returns(ticker: str, start: str, end: str = None) -> pd.DataFrame:
    """
    Fetch daily prices and compute log returns.

    Args:
        ticker: e.g. 'NVDA'
        start: 'YYYY-MM-DD'
        end: 'YYYY-MM-DD' or None (defaults to today)

    Returns:
        DataFrame with columns: close, log_return
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Close']].rename(columns={'Close': 'close'})
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df = df.dropna()
    return df


if __name__ == "__main__":
    data = get_returns("NVDA", "2020-01-01")
    print(data.head())
    print(f"\nRows: {len(data)}")
    print(f"Date range: {data.index.min().date()} to {data.index.max().date()}")
