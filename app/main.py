from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ml.predict import predict_next_close
from ml.train import train_one, train_many, DEFAULT_SYMBOLS

app = FastAPI(title="AI Trader LSTM API", version="1.1.0")


class PredictRequest(BaseModel):
    symbol: str = "ASELS.IS"
    start: str = "2018-01-01"
    window: int = 60


class TrainRequest(BaseModel):
    symbols: list[str] | None = None  # None ise default 10 hisse
    start: str = "2018-01-01"
    window: int = 60
    test_ratio: float = 0.2
    epochs: int = 20
    batch_size: int = 32


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/symbols")
def symbols():
    """
    Frontend dropdown için 10 hisse listesi.
    """
    return {"symbols": [{"symbol": s, "name": n} for s, n in DEFAULT_SYMBOLS.items()]}


@app.get("/models/status")
def models_status():
    """
    Hangi hisselerin modeli var? Basit kontrol.
    """
    import os
    statuses = []
    for sym, name in DEFAULT_SYMBOLS.items():
        base = sym.replace(".", "_")
        model_path = f"models/{base}_lstm.keras"
        xsc_path = f"models/{base}_x_scaler.joblib"
        ysc_path = f"models/{base}_y_scaler.joblib"
        ok = os.path.exists(model_path) and os.path.exists(xsc_path) and os.path.exists(ysc_path)
        statuses.append({
            "symbol": sym,
            "name": name,
            "ready": ok,
            "model_path": model_path if ok else None
        })
    return {"models": statuses}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        result = predict_next_close(req.symbol, start=req.start, window=req.window)
        diff = result["pred_next_close"] - result["last_close"]
        pct = (diff / result["last_close"]) * 100 if result["last_close"] else 0.0
        result["diff"] = float(diff)
        result["pct"] = float(pct)
        return result
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model bulunamadı. Önce train çalıştır.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/train")
def train(req: TrainRequest):
    """
    Çoklu sembol eğitimi API'den tetiklemek için.
    """
    try:
        symbols = req.symbols if req.symbols else list(DEFAULT_SYMBOLS.keys())
        report = train_many(
            symbols=symbols,
            start=req.start,
            window=req.window,
            test_ratio=req.test_ratio,
            epochs=req.epochs,
            batch_size=req.batch_size,
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
