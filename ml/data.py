from __future__ import annotations
import pandas as pd
import yfinance as yf

def fetch_ohlcv(symbol: str, start: str = "2015-01-01", end: str | None = None) -> pd.DataFrame:
    """
    symbol ör: 'ASELS.IS', 'THYAO.IS', 'GARAN.IS', 'AAPL'
    """
    df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"Veri gelmedi. Symbol yanlış olabilir: {symbol}")

    # Kolonları standardize
    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    })
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # YFinance bazen multiindex döndürebiliyor
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]

    return df

def save_raw(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=True)

if __name__ == "__main__":
    symbol = "ASELS.IS"
    df = fetch_ohlcv(symbol, start="2018-01-01")
    save_raw(df, f"data/{symbol.replace('.','_')}_raw.csv")
    print(df.tail())
