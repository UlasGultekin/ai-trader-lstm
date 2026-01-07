from __future__ import annotations

import numpy as np
from joblib import load
import tensorflow as tf

from ml.data import fetch_ohlcv
from ml.features import build_features, feature_columns

def predict_next_close(symbol: str, start: str = "2018-01-01", window: int = 60):
    model_path = f"models/{symbol.replace('.','_')}_lstm.keras"
    x_scaler_path = f"models/{symbol.replace('.','_')}_x_scaler.joblib"
    y_scaler_path = f"models/{symbol.replace('.','_')}_y_scaler.joblib"

    model = tf.keras.models.load_model(model_path)
    x_scaler = load(x_scaler_path)
    y_scaler = load(y_scaler_path)

    raw = fetch_ohlcv(symbol, start=start)
    feat_df = build_features(raw)

    cols = feature_columns()
    X_df = feat_df[cols].copy()

    # son window satırı al
    X_last = X_df.iloc[-window:].values
    X_last_scaled = x_scaler.transform(X_last)

    X_input = np.expand_dims(X_last_scaled, axis=0)  # (1, window, features)

    y_pred_scaled = model.predict(X_input, verbose=0)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)[0, 0]

    last_close = float(feat_df["close"].iloc[-1])
    return {
        "symbol": symbol,
        "last_close": last_close,
        "pred_next_close": float(y_pred),
    }

if __name__ == "__main__":
    print(predict_next_close("ASELS.IS"))
