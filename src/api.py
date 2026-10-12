"""FastAPI inference service with structured logs and Prometheus metrics."""
from __future__ import annotations
import logging
import os
import time
from pathlib import Path
from typing import Annotated
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, make_asgi_app
from src.data import FEATURES

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "artifacts/model.joblib"))
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("heart_disease_api")
REQUESTS = Counter("prediction_requests_total", "Prediction requests", ["endpoint", "status"])
LATENCY = Histogram("prediction_request_duration_seconds", "Prediction request duration")
app = FastAPI(title="Heart Disease Risk Classification API", version="1.0.0", description="Educational demonstration only; not for clinical use.")
app.mount("/metrics", make_asgi_app())
model = None

class PatientFeatures(BaseModel):
    age: Annotated[float, Field(ge=1, le=120)]
    sex: Annotated[int, Field(ge=0, le=1)]
    cp: Annotated[int, Field(ge=1, le=4)]
    trestbps: Annotated[float, Field(ge=40, le=300)]
    chol: Annotated[float, Field(ge=50, le=800)]
    fbs: Annotated[int, Field(ge=0, le=1)]
    restecg: Annotated[int, Field(ge=0, le=2)]
    thalach: Annotated[float, Field(ge=40, le=250)]
    exang: Annotated[int, Field(ge=0, le=1)]
    oldpeak: Annotated[float, Field(ge=-5, le=15)]
    slope: Annotated[int, Field(ge=1, le=3)]
    ca: Annotated[int, Field(ge=0, le=4)]
    thal: Annotated[int, Field(ge=0, le=7)]

@app.on_event("startup")
def load_model() -> None:
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        logger.info("model_loaded path=%s", MODEL_PATH)
    else:
        logger.error("model_not_found path=%s", MODEL_PATH)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
        REQUESTS.labels(request.url.path, str(response.status_code)).inc()
        return response
    except Exception:
        REQUESTS.labels(request.url.path, "500").inc()
        logger.exception("request_failed method=%s path=%s", request.method, request.url.path)
        raise
    finally:
        LATENCY.observe(time.perf_counter() - start)
        logger.info("request method=%s path=%s duration_ms=%.2f", request.method, request.url.path, (time.perf_counter() - start) * 1000)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(patient: PatientFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact not loaded. Run training first.")
    row = pd.DataFrame([patient.model_dump()], columns=FEATURES)
    try:
        probability = float(model.predict_proba(row)[0, 1])
        label = int(probability >= 0.5)
        logger.info("prediction_completed label=%s probability=%.4f", label, probability)
        return {"prediction": label, "label": "disease_present" if label else "no_disease",
                "confidence": round(probability if label else 1 - probability, 4),
                "disease_probability": round(probability, 4), "threshold": 0.5,
                "disclaimer": "Educational prototype only; not a medical diagnosis."}
    except Exception as exc:
        logger.exception("prediction_error")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
