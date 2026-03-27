"""
api/main.py
────────────
FastAPI application for the IoT Intrusion Detection model.

Endpoints:
  GET  /health          — liveness + model status
  GET  /info            — model metadata, feature list, thresholds
  POST /predict         — single flow prediction
  POST /predict/batch   — batch of up to 1000 flows

Run locally:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Production:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
"""
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.predictor import Predictor
from api.schema import (
    BatchFlowRequest,
    BatchPredictionResponse,
    FlowFeatures,
    HealthResponse,
    ModelInfoResponse,
    PredictionResult,
    SinglePredictionResponse,
)

# ── Shared state ──────────────────────────────────────────────────────────
_predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup; release on shutdown."""
    global _predictor
    print("Loading inference engine…")
    _predictor = Predictor()
    print(
        f"  ✓  Model ready — {_predictor.info['n_features']} features, "
        f"{len(_predictor.info['classes'])} classes"
    )
    yield
    _predictor = None
    print("Inference engine unloaded.")


# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="IoT Intrusion Detection API",
    description=(
        "Classifies network flows as Benign or one of six attack categories "
        "(DDoS, DoS, Mirai, Recon, Spoofing, Web_BruteForce) using a "
        "LightGBM model trained on CICIoT2023."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Middleware: request timing ────────────────────────────────────────────
@app.middleware("http")
async def add_process_time(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Process-Time-Ms"] = f"{ms:.2f}"
    return response


# ── Dependency ────────────────────────────────────────────────────────────
def get_predictor() -> Predictor:
    if _predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Try again in a moment.",
        )
    return _predictor


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Liveness probe. Returns 200 when model is loaded."""
    p = get_predictor()
    info = p.info
    return HealthResponse(
        status="ok",
        model_loaded=True,
        n_features=info["n_features"],
        classes=info["classes"],
        thresholds=info["thresholds"],
    )


@app.get("/info", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    """Full model metadata — feature names, class list, per-class thresholds."""
    p = get_predictor()
    return ModelInfoResponse(**p.info)


@app.post(
    "/predict",
    response_model=SinglePredictionResponse,
    tags=["Inference"],
    summary="Classify a single network flow",
)
def predict_single(flow: FlowFeatures):
    """
    Accepts a single flow feature vector.
    Returns the predicted class, confidence, per-class probabilities,
    an is_attack flag, and a low_confidence warning when appropriate.
    """
    p = get_predictor()
    try:
        flow_dict = flow.model_dump(by_alias=True, exclude_none=True)
        result = p.predict_single(flow_dict)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return SinglePredictionResponse(result=PredictionResult(**result))


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Inference"],
    summary="Classify up to 1000 flows in one request",
)
def predict_batch(request: BatchFlowRequest):
    """
    Batch endpoint — significantly more efficient than calling /predict
    in a loop. Recommended for streaming pipeline integrations.
    """
    p = get_predictor()
    try:
        flows_dicts = [
            f.model_dump(by_alias=True, exclude_none=True)
            for f in request.flows
        ]
        results = p.predict_batch(flows_dicts)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return BatchPredictionResponse(
        count=len(results),
        results=[PredictionResult(**r) for r in results],
    )


# ── Global exception handler ──────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"status": "error", "detail": str(exc)},
    )
