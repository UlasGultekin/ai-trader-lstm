from __future__ import annotations
import numpy as np
import pandas as pd

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Girdi df: open, high, low, close, volume (+ adj_close olabilir)
    Çıktı: feature kolonları + target (y)
    """
    out = df.copy()

    # Eksik değerleri doldurma (forward fill + gerekiyorsa back fill)
    out = out.ffill().bfill()

    # Temel dönüşümler
    out["log_return"] = np.log(out["close"]).diff()

    # RSI / EMA / MACD
    out["rsi14"] = rsi(out["close"], 14)
    out["ema20"] = ema(out["close"], 20)
    out["ema50"] = ema(out["close"], 50)

    macd_line, signal_line, hist = macd(out["close"])
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist

    # Volatilite (rolling std)
    out["volatility20"] = out["log_return"].rolling(20).std()

    # Hedef: yarınki close
    out["y"] = out["close"].shift(-1)

    # İlk satırlarda NaN oluşur → temizle
    out = out.dropna().copy()

    return out

def feature_columns() -> list[str]:
    return [
        "open", "high", "low", "close", "volume",
        "log_return", "rsi14", "ema20", "ema50",
        "macd", "macd_signal", "macd_hist",
        "volatility20",
    ]
