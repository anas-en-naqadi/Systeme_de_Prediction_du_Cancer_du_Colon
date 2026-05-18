"""FastAPI application for the colon cancer inference service."""

from __future__ import annotations

import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.predictor import PredictorError, get_predictor
from backend.schemas import ErrorResponse, HealthResponse, MetadataResponse, MetricsResponse, PredictionRequest, PredictionResponse
from backend.utils.logger import logger


PROJECT_ROOT = Path(__file__).resolve().parents[2]
START_TIME = time.time()
FRONTEND_DIST_DIR = PROJECT_ROOT / "app" / "frontend" / "dist"


def _new_telemetry_store(started_at: float) -> dict[str, Any]:
    """Create a small in-memory telemetry store for runtime metrics."""

    return {
        "started_at": started_at,
        "total_requests": 0,
        "total_latency_ms": 0.0,
        "latency_count": 0,
        "recent_request_timestamps": deque(maxlen=4096),
    }


def _record_request_metrics(request: Request, duration_ms: float) -> None:
    """Update rolling latency and throughput counters."""

    telemetry = getattr(request.app.state, "telemetry", None)
    if telemetry is None:
        return

    path = request.url.path
    if path.startswith("/static") or path in {"/docs", "/redoc", "/openapi.json", "/metrics"}:
        return

    now = time.time()
    telemetry["total_requests"] += 1
    telemetry["total_latency_ms"] += max(duration_ms, 0.0)
    telemetry["latency_count"] += 1

    timestamps = telemetry["recent_request_timestamps"]
    timestamps.append(now)
    cutoff = now - 60.0
    while timestamps and timestamps[0] < cutoff:
        timestamps.popleft()


def _compute_telemetry_snapshot(telemetry: dict[str, Any]) -> dict[str, Any]:
    """Compute stable telemetry values from in-memory counters."""

    now = time.time()
    started_at = float(telemetry.get("started_at", START_TIME))
    uptime_seconds = max(now - started_at, 0.0)

    latency_count = int(telemetry.get("latency_count", 0))
    total_latency_ms = float(telemetry.get("total_latency_ms", 0.0))
    average_latency_ms = (total_latency_ms / latency_count) if latency_count > 0 else 0.0

    timestamps = telemetry.get("recent_request_timestamps")
    if isinstance(timestamps, deque):
        cutoff = now - 60.0
        while timestamps and timestamps[0] < cutoff:
            timestamps.popleft()
        throughput_rps_60s = len(timestamps) / 60.0
    else:
        throughput_rps_60s = 0.0

    return {
        "uptime_seconds": round(uptime_seconds, 3),
        "average_latency_ms": round(max(average_latency_ms, 0.0), 3),
        "throughput_rps_60s": round(max(throughput_rps_60s, 0.0), 4),
        "total_requests": int(max(int(telemetry.get("total_requests", 0)), 0)),
    }


def _maybe_mount_static(app: FastAPI) -> None:
    """Mount static assets when the frontend build becomes available."""

    if FRONTEND_DIST_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(FRONTEND_DIST_DIR)), name="static")
        logger.info(f"Static assets mounted from {FRONTEND_DIST_DIR}")
    else:
        logger.warning(f"Frontend build directory not found yet: {FRONTEND_DIST_DIR}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML artifacts once at startup and release resources on shutdown."""

    logger.startup("Starting Colon Cancer ML API")
    try:
        predictor = get_predictor()
        app.state.predictor = predictor
        app.state.started_at = START_TIME
        app.state.telemetry = _new_telemetry_store(START_TIME)
        logger.success(
            f"Predictor ready: model={predictor.model_name}, genes={len(predictor.selected_genes)}"
        )
    except Exception as exc:  # pragma: no cover - startup failure path
        logger.error(f"Failed to start backend: {exc}")
        raise

    yield

    logger.info("Shutting down Colon Cancer ML API")


app = FastAPI(
    title="Colon Cancer ML API",
    description="Production-style FastAPI backend for colon cancer inference.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Status", "description": "Service health and runtime status endpoints."},
        {"name": "Inference", "description": "Model prediction endpoints."},
        {"name": "Metadata", "description": "Model and training metadata endpoints."},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_maybe_mount_static(app)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Log request latency and response status for observability."""

    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        duration_ms = (time.perf_counter() - started) * 1000.0
        _record_request_metrics(request, duration_ms)
        client_host = request.client.host if request.client else None
        logger.request(request.method, request.url.path, 500, duration_ms, client_host)
        logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}")
        raise

    duration_ms = (time.perf_counter() - started) * 1000.0
    _record_request_metrics(request, duration_ms)
    client_host = request.client.host if request.client else None
    logger.request(request.method, request.url.path, response.status_code, duration_ms, client_host)
    return response


def _error_response(status_code: int, error: str, details: Any | None = None) -> JSONResponse:
    """Return a normalized JSON error payload."""

    payload = ErrorResponse(error=error, status_code=status_code, details=details)
    if hasattr(payload, "model_dump"):
        content = payload.model_dump()
    else:
        content = payload.dict()
    return JSONResponse(status_code=status_code, content=content)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Normalize validation errors."""

    return _error_response(status.HTTP_422_UNPROCESSABLE_ENTITY, "Validation error", exc.errors())


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    """Normalize HTTP errors."""

    return _error_response(exc.status_code, str(exc.detail), None)


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(_: Request, exc: FileNotFoundError) -> JSONResponse:
    """Return a clean 500 response when artifacts are missing."""

    return _error_response(status.HTTP_500_INTERNAL_SERVER_ERROR, "Required artifact not found", str(exc))


@app.exception_handler(PredictorError)
async def predictor_error_handler(_: Request, exc: PredictorError) -> JSONResponse:
    """Return a clean 422 response for prediction and payload issues."""

    details = getattr(exc, "details", None)
    return _error_response(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc), details)


@app.exception_handler(Exception)
async def generic_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler to avoid leaking stack traces to clients."""

    logger.error(f"Unhandled backend exception: {exc}")
    return _error_response(status.HTTP_500_INTERNAL_SERVER_ERROR, "Internal server error", None)


@app.get("/", tags=["Status"], summary="Health landing route")
async def root() -> dict[str, str]:
    """Return a lightweight healthy signal for the API root."""

    return {"message": "Colon Cancer ML API running", "status": "healthy"}


@app.get("/health", response_model=HealthResponse, tags=["Status"], summary="Backend health check")
async def health(request: Request) -> HealthResponse:
    """Expose service status, model availability, and uptime."""

    predictor = getattr(request.app.state, "predictor", None)
    uptime_seconds = time.time() - getattr(request.app.state, "started_at", START_TIME)
    if predictor is None:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            selected_algorithm=None,
            selected_genes_count=0,
            uptime_seconds=round(uptime_seconds, 3),
        )

    payload = predictor.health_payload()
    return HealthResponse(
        status=payload["status"],
        model_loaded=bool(payload["model_loaded"]),
        selected_algorithm=payload["selected_algorithm"],
        selected_genes_count=int(payload["selected_genes_count"]),
        uptime_seconds=round(uptime_seconds, 3),
    )


@app.get("/genes", tags=["Metadata"], summary="Return selected genes")
async def genes(request: Request) -> dict[str, Any]:
    """Return the ordered list of selected genes and their count."""

    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not loaded yet.")
    selected_genes = predictor.selected_genes
    return {"genes": selected_genes, "count": len(selected_genes)}


@app.get("/metadata", response_model=MetadataResponse, tags=["Metadata"], summary="Return model metadata")
async def metadata(request: Request) -> MetadataResponse:
    """Expose the model performance metadata stored on disk."""

    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not loaded yet.")

    payload = predictor.metadata_payload()
    return MetadataResponse(
        best_model=str(payload["best_model"]),
        selected_genes=list(payload["selected_genes"]),
        selected_genes_count=int(payload["selected_genes_count"]),
        roc_auc=float(payload["roc_auc"]),
        f1_score=float(payload["f1_score"]),
        recall=float(payload["recall"]),
        precision=float(payload["precision"]),
        accuracy=float(payload["accuracy"]),
        training_date=str(payload["training_date"]),
        dataset_shape=payload["dataset_shape"],
        validation_metrics=payload.get("validation_metrics", {}),
        evaluation_metrics=payload.get("evaluation_metrics", {}),
        best_model_params=payload.get("best_model_params", {}),
        evaluation_method=payload.get("evaluation_method") or None,
        training_set=payload.get("training_set") or None,
        target_column=payload.get("target_column") or None,
        target_mapping=payload.get("target_mapping", {}),
        ranking_progression=payload.get("ranking_progression", []),
        cleaning_summary=payload.get("cleaning_summary", {}),
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Status"], summary="Runtime telemetry")
async def metrics(request: Request) -> MetricsResponse:
    """Expose lightweight runtime telemetry for monitoring panels."""

    telemetry = getattr(request.app.state, "telemetry", _new_telemetry_store(START_TIME))
    snapshot = _compute_telemetry_snapshot(telemetry)
    return MetricsResponse(**snapshot)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="Run colon cancer inference",
    status_code=status.HTTP_200_OK,
)
async def predict(request: Request, body: PredictionRequest) -> PredictionResponse:
    """Validate gene values, run inference, and return a structured prediction."""

    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not loaded yet.")

    try:
        result = predictor.predict(body.gene_values)
    except PredictorError as exc:
        details = getattr(exc, "details", None)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail={"error": str(exc), "details": details}) from exc

    return PredictionResponse(**result)
