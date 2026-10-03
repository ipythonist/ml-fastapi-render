"""
FastAPI app serving a California house-price regressor (scikit-learn
HistGradientBoostingRegressor) stored in model.pkl.

Run locally:
    uvicorn main:app --reload

Test it:
    curl -X POST http://127.0.0.1:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"MedInc": 8.3, "HouseAge": 41, "AveRooms": 6.98, "AveBedrms": 1.02,
              "Population": 322, "AveOccup": 2.56, "Latitude": 37.88, "Longitude": -122.23}'

Endpoints:
    GET  /         info + model metrics
    GET  /health   liveness check (Render pings this)
    POST /predict  predict the median house value for one census block
    GET  /metrics  in-process request counters for monitoring

Every request is written as one JSON line to stdout. Render captures stdout,
so these lines show up in the service's Logs tab.
"""
import json
import logging
import os
import sys
import time
from threading import Lock

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

# ---------- logging ----------
# One JSON object per line so humans and log tools can both grep it.
logger = logging.getLogger("house-price-api")
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)
logger.propagate = False


def log_event(level: str, event: str, **fields) -> None:
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "level": level, "event": event, **fields}
    getattr(logger, level.lower())(json.dumps(record))


# ---------- model ----------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

try:
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    FEATURE_NAMES = artifact["feature_names"]
    MODEL_INFO = {"sklearn_version": artifact.get("sklearn_version"), **artifact.get("metrics", {})}
    log_event("INFO", "model_loaded", path=MODEL_PATH, **MODEL_INFO)
except FileNotFoundError:
    model, FEATURE_NAMES, MODEL_INFO = None, [], {}
    log_event("ERROR", "model_missing", path=MODEL_PATH)

# ---------- metrics ----------
# Plain counters behind a lock. They reset on restart, which is fine for a
# first look at monitoring without adding Prometheus.
_lock = Lock()
METRICS = {
    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "requests_total": 0,
    "errors_total": 0,
    "predictions_total": 0,
    "latency_ms_sum": 0.0,
    "predicted_price_usd_sum": 0.0,
}

# ---------- app ----------
app = FastAPI(
    title="California House Price API",
    description="Getting Started with ML in Production - assignment. "
                "HistGradientBoostingRegressor on the California housing dataset, served with FastAPI.",
    version="1.0.0",
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Time every request, update counters, emit one JSON log line."""
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        with _lock:
            METRICS["requests_total"] += 1
            METRICS["errors_total"] += 1
        log_event("ERROR", "request_failed", method=request.method, path=request.url.path, error=str(exc))
        raise
    elapsed_ms = (time.perf_counter() - start) * 1000
    with _lock:
        METRICS["requests_total"] += 1
        METRICS["latency_ms_sum"] += elapsed_ms
        if response.status_code >= 400:
            METRICS["errors_total"] += 1
    log_event("INFO", "request", method=request.method, path=request.url.path,
              status=response.status_code, latency_ms=round(elapsed_ms, 2))
    return response


class HouseFeatures(BaseModel):
    """One census block group, same fields as the California housing dataset."""
    MedInc: float = Field(..., ge=0, le=20, description="Median income of the block, in $10,000s", examples=[8.3252])
    HouseAge: float = Field(..., ge=0, le=100, description="Median house age in years", examples=[41.0])
    AveRooms: float = Field(..., gt=0, le=50, description="Average rooms per household", examples=[6.98])
    AveBedrms: float = Field(..., gt=0, le=10, description="Average bedrooms per household", examples=[1.02])
    Population: float = Field(..., ge=0, le=50000, description="Block population", examples=[322.0])
    AveOccup: float = Field(..., gt=0, le=50, description="Average household members", examples=[2.56])
    Latitude: float = Field(..., ge=32, le=42.5, description="Block latitude (California)", examples=[37.88])
    Longitude: float = Field(..., ge=-125, le=-114, description="Block longitude (California)", examples=[-122.23])


class PredictionResponse(BaseModel):
    predicted_price_usd: float
    predicted_value_100k: float


@app.get("/")
def root():
    return {
        "message": "California House Price API is running. See /docs for usage.",
        "model": "HistGradientBoostingRegressor (California housing)",
        "model_info": MODEL_INFO,
        "endpoints": ["/health", "/predict", "/metrics", "/docs"],
    }


@app.get("/health")
def health():
    """Health check used by Render and load balancers."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Did you run train.py?")

    # Build the row in the exact column order the model was trained on.
    row = np.array([[getattr(features, name) for name in FEATURE_NAMES]])
    try:
        value_100k = float(model.predict(row)[0])
    except Exception as exc:
        log_event("ERROR", "prediction_failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Model inference failed")

    value_100k = max(value_100k, 0.0)  # a regressor can extrapolate below zero; prices cannot
    price_usd = round(value_100k * 100_000, 2)

    with _lock:
        METRICS["predictions_total"] += 1
        METRICS["predicted_price_usd_sum"] += price_usd
    log_event("INFO", "prediction", input=features.model_dump(), predicted_price_usd=price_usd)
    return PredictionResponse(predicted_price_usd=price_usd, predicted_value_100k=round(value_100k, 4))


@app.get("/metrics")
def metrics():
    with _lock:
        total, preds = METRICS["requests_total"], METRICS["predictions_total"]
        return {
            **METRICS,
            "avg_latency_ms": round(METRICS["latency_ms_sum"] / total, 2) if total else 0.0,
            "avg_predicted_price_usd": round(METRICS["predicted_price_usd_sum"] / preds, 2) if preds else 0.0,
        }
