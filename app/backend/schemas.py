"""Pydantic schemas for the colon cancer FastAPI backend."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, StrictFloat


class PredictionRequest(BaseModel):
    """Request body for model inference."""

    gene_values: Dict[str, StrictFloat] = Field(
        ...,
        description="Mapping of gene name to measured expression value.",
        example={"T86473": 0.82, "T51261": -1.12},
    )

    class Config:
        extra = "forbid"


class ProbabilityResponse(BaseModel):
    """Class probability breakdown for a binary prediction."""

    normal: float = Field(..., ge=0.0, le=1.0)
    abnormal: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    """Structured response for inference requests."""

    prediction: str
    class_: int = Field(..., alias="class")
    confidence: float = Field(..., ge=0.0, le=100.0)
    probabilities: ProbabilityResponse
    model: str

    class Config:
        allow_population_by_field_name = True
        extra = "forbid"


class DatasetShape(BaseModel):
    """Dataset size summary from the training metadata."""

    rows: int
    columns: int


class HealthResponse(BaseModel):
    """Operational health payload for the service."""

    status: str
    model_loaded: bool
    selected_algorithm: Optional[str] = None
    selected_genes_count: int = 0
    uptime_seconds: float


class MetadataResponse(BaseModel):
    """Model and training metadata exposed through the API."""

    best_model: str
    selected_genes: List[str]
    selected_genes_count: int
    roc_auc: float
    f1_score: float
    recall: float
    precision: float
    accuracy: float
    training_date: str
    dataset_shape: DatasetShape
    validation_metrics: Dict[str, float] = Field(default_factory=dict)
    evaluation_metrics: Dict[str, Any] = Field(default_factory=dict)
    best_model_params: Dict[str, Any] = Field(default_factory=dict)
    evaluation_method: Optional[str] = None
    training_set: Optional[str] = None
    target_column: Optional[str] = None
    target_mapping: Dict[str, Any] = Field(default_factory=dict)
    ranking_progression: List[Dict[str, Any]] = Field(default_factory=list)
    cleaning_summary: Dict[str, Any] = Field(default_factory=dict)


class MetricsResponse(BaseModel):
    """Lightweight runtime telemetry exposed by the backend."""

    uptime_seconds: float = Field(..., ge=0.0)
    average_latency_ms: float = Field(..., ge=0.0)
    throughput_rps_60s: float = Field(..., ge=0.0)
    total_requests: int = Field(..., ge=0)


class ErrorResponse(BaseModel):
    """Normalized API error payload."""

    error: str
    status_code: int
    details: Any | None = None
