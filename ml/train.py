from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
from joblib import dump

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

from ml.data import fetch_ohlcv, save_raw
from ml.features import build_features, feature_columns


# 10 Türkiye hissesi (symbol -> display name)
DEFAULT_SYMBOLS: dict[str, str] = {
    "ASELS.IS": "Aselsan",
    "THYAO.IS": "Türk Hava Yolları",
    "GARAN.IS": "Garanti BBVA",
    "AKBNK.IS": "Akbank",
    "KCHOL.IS": "Koç Holding",
    "BIMAS.IS": "BİM",
    "EREGL.IS": "Ereğli Demir Çelik",
    "TUPRS.IS": "Tüpraş",
    "SISE.IS": "Şişecam",
    "FROTO.IS": "Ford Otosan",
}


def make_windows(X: np.ndarray, y: np.ndarray, window: int):
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i - window:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def train_one(
    symbol: str,
    start: str = "2018-01-01",
    window: int = 60,
    test_ratio: float = 0.2,
    epochs: int = 25,
    batch_size: int = 32,
) -> dict:
    """
    Tek sembol için:
    - yfinance veri çek
    - feature çıkar
    - train/test split (time series)
    - scaler fit (train) / transform
    - LSTM train
    - evaluate (RMSE/MAE)
    - model + scaler kaydet
    """
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 1) veri çek
    raw = fetch_ohlcv(symbol, start=start)
    save_raw(raw, f"data/{symbol.replace('.','_')}_raw.csv")

    # 2) feature
    df = build_features(raw)
    cols = feature_columns()

    X_df = df[cols].copy()
    y_df = df[["y"]].copy()

    # 3) split (shuffle yok)
    n = len(df)
    test_size = int(n * test_ratio)
    train_size = n - test_size

    if train_size <= window + 10:
        raise ValueError(f"{symbol}: Veri yetersiz. window={window} için daha eski start seç.")

    X_train_df, X_test_df = X_df.iloc[:train_size], X_df.iloc[train_size:]
    y_train_df, y_test_df = y_df.iloc[:train_size], y_df.iloc[train_size:]

    # 4) scaler (sadece train'e fit!)
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_train = x_scaler.fit_transform(X_train_df.values)
    X_test = x_scaler.transform(X_test_df.values)

    y_train = y_scaler.fit_transform(y_train_df.values)
    y_test = y_scaler.transform(y_test_df.values)

    # 5) window
    X_train_w, y_train_w = make_windows(X_train, y_train, window)
    X_test_w, y_test_w = make_windows(X_test, y_test, window)

    # 6) model
    n_features = X_train_w.shape[-1]
    model = models.Sequential([
        layers.Input(shape=(window, n_features)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"]
    )

    cb = [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
    ]

    history = model.fit(
        X_train_w, y_train_w,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb,
        verbose=1
    )

    # 7) evaluate (inverse scale)
    y_pred_scaled = model.predict(X_test_w, verbose=0)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_true = y_scaler.inverse_transform(y_test_w)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    # 8) kaydet
    base = symbol.replace(".", "_")
    model_path = f"models/{base}_lstm.keras"
    model.save(model_path)
    dump(x_scaler, f"models/{base}_x_scaler.joblib")
    dump(y_scaler, f"models/{base}_y_scaler.joblib")

    # report csv
    report = pd.DataFrame({
        "date": df.index[-len(y_true):],
        "y_true": y_true.flatten(),
        "y_pred": y_pred.flatten()
    })
    report_path = f"data/{base}_pred_report.csv"
    report.to_csv(report_path, index=False)

    return {
        "symbol": symbol,
        "rmse": rmse,
        "mae": mae,
        "model_path": model_path,
        "report_path": report_path,
        "epochs_ran": len(history.history.get("loss", [])),
        "n_rows": int(len(df)),
        "window": window,
        "start": start,
    }


def train_many(
    symbols: list[str] | None = None,
    start: str = "2018-01-01",
    window: int = 60,
    test_ratio: float = 0.2,
    epochs: int = 20,
    batch_size: int = 32,
    sleep_sec: float = 0.0,
) -> dict:
    """
    Çoklu sembol eğitimi:
    - Her sembol için train_one çağırır
    - Hata olanları ayrı raporlar
    """
    if symbols is None:
        symbols = list(DEFAULT_SYMBOLS.keys())

    results = []
    errors = []

    t0 = time.time()
    for sym in symbols:
        try:
            print(f"\n===== TRAIN START: {sym} =====")
            r = train_one(
                symbol=sym,
                start=start,
                window=window,
                test_ratio=test_ratio,
                epochs=epochs,
                batch_size=batch_size,
            )
            results.append(r)
            print(f"✅ DONE: {sym} | RMSE={r['rmse']:.4f} MAE={r['mae']:.4f}")
        except Exception as e:
            errors.append({"symbol": sym, "error": str(e)})
            print(f"❌ FAIL: {sym} | {e}")

        if sleep_sec > 0:
            time.sleep(sleep_sec)

    total_sec = float(time.time() - t0)

    # Özet dataframe (opsiyonel kaydedelim)
    os.makedirs("data", exist_ok=True)
    summary_path = "data/train_summary.csv"
    if results:
        pd.DataFrame(results).sort_values("rmse").to_csv(summary_path, index=False)
    else:
        # boşsa yine de dosya yaratmayalım
        summary_path = None

    return {
        "count_ok": len(results),
        "count_err": len(errors),
        "ok": results,
        "errors": errors,
        "total_seconds": total_sec,
        "summary_csv": summary_path,
    }


if __name__ == "__main__":
    # Tek tek yerine: 10 hisselik batch eğitim
    report = train_many(
        symbols=list(DEFAULT_SYMBOLS.keys()),
        start="2018-01-01",
        window=60,
        test_ratio=0.2,
        epochs=15,
        batch_size=32,
    )
    print("\nBATCH REPORT:", report["count_ok"], "OK,", report["count_err"], "ERR")
