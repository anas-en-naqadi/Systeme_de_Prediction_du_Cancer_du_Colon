"""Singleton-style model loading and inference utilities."""

from __future__ import annotations

import json
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from backend.utils.logger import logger


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "model"


class PredictorError(RuntimeError):
    """Raised when model loading or inference fails."""

    def __init__(self, message: str, details: Any | None = None) -> None:
        super().__init__(message)
        self.details = details


@dataclass(slots=True)
class ModelArtifacts:
    """In-memory bundle of the trained inference artifacts."""

    model: Any
    scaler: Any
    selected_genes: list[str]
    metadata: dict[str, Any]


class ColonCancerPredictor:
    """Load once, predict many times."""

    _instance: ColonCancerPredictor | None = None
    _lock = threading.Lock()

    def __init__(self, artifacts: ModelArtifacts) -> None:
        self._artifacts = artifacts
        self._selected_gene_index = {gene: position for position, gene in enumerate(artifacts.selected_genes)}

    @classmethod
    def get_instance(cls) -> ColonCancerPredictor:
        """Return the process-wide predictor singleton."""

        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(cls._load_artifacts())
        return cls._instance

    @staticmethod
    def _artifact_path(name: str) -> Path:
        return MODEL_DIR / name

    @classmethod
    def _read_selected_genes(cls) -> list[str]:
        path = cls._artifact_path("selected_genes.json")
        if not path.exists():
            raise FileNotFoundError(f"Missing artifact: {path}")

        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            genes = payload.get("selected_genes")
        else:
            genes = payload

        if not isinstance(genes, list) or not all(isinstance(gene, str) for gene in genes):
            raise PredictorError("selected_genes.json does not contain a valid gene list.")

        if len(genes) != 6:
            raise PredictorError(f"Expected 6 selected genes, found {len(genes)}.")

        return genes

    @classmethod
    def _read_metadata(cls) -> dict[str, Any]:
        path = cls._artifact_path("model_metadata.json")
        if not path.exists():
            raise FileNotFoundError(f"Missing artifact: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise PredictorError("model_metadata.json must contain a JSON object.")
        return payload

    @classmethod
    def _load_artifacts(cls) -> ModelArtifacts:
        """Load model, scaler, selected genes, and metadata once."""

        model_path = cls._artifact_path("model.pkl")
        scaler_path = cls._artifact_path("scaler.pkl")

        missing_files = [str(path) for path in (model_path, scaler_path) if not path.exists()]
        if missing_files:
            raise FileNotFoundError(f"Missing model artifacts: {', '.join(missing_files)}")

        logger.startup(f"Loading model from {model_path}")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Trying to unpickle estimator*")
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
        selected_genes = cls._read_selected_genes()
        metadata = cls._read_metadata()

        metadata_selected = metadata.get("selected_genes")
        if isinstance(metadata_selected, list) and metadata_selected and metadata_selected != selected_genes:
            logger.warning("selected_genes.json and model_metadata.json are not identical; using selected_genes.json order.")

        if not hasattr(model, "predict"):
            raise PredictorError("Loaded model does not expose predict().")
        if not hasattr(scaler, "transform"):
            raise PredictorError("Loaded scaler does not expose transform().")

        logger.success(
            f"Model artifacts loaded successfully: model={metadata.get('best_model_name', model.__class__.__name__)}, genes={len(selected_genes)}"
        )

        return ModelArtifacts(model=model, scaler=scaler, selected_genes=selected_genes, metadata=metadata)

    @property
    def selected_genes(self) -> list[str]:
        return list(self._artifacts.selected_genes)

    @property
    def model_name(self) -> str:
        return str(self._artifacts.metadata.get("best_model_name", self._artifacts.model.__class__.__name__))

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._artifacts.metadata)

    @property
    def is_loaded(self) -> bool:
        return True

    def validate_gene_payload(self, gene_values: dict[str, float]) -> None:
        """Ensure the request contains exactly the expected genes."""

        expected = self.selected_genes
        provided = list(gene_values.keys())

        missing = [gene for gene in expected if gene not in gene_values]
        extra = [gene for gene in provided if gene not in self._selected_gene_index]

        if missing or extra:
            details = {}
            if missing:
                details["missing_genes"] = missing
            if extra:
                details["extra_genes"] = extra
            raise PredictorError(f"Gene payload mismatch: expected exactly {len(expected)} genes.", details)

    def _ordered_feature_vector(self, gene_values: dict[str, float]) -> np.ndarray:
        ordered_values: list[float] = []
        for gene in self.selected_genes:
            if gene not in gene_values:
                raise PredictorError(f"Missing required gene: {gene}")

            raw_value = gene_values[gene]
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float, np.integer, np.floating)):
                raise PredictorError(f"Gene '{gene}' must be numeric.")

            value = float(raw_value)
            if not np.isfinite(value):
                raise PredictorError(f"Gene '{gene}' must be a finite numeric value.")
            ordered_values.append(value)

        return np.asarray(ordered_values, dtype=float).reshape(1, -1)

    def _prediction_probabilities(self, features: np.ndarray) -> np.ndarray:
        model = self._artifacts.model

        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(features), dtype=float)
            if probabilities.ndim != 2 or probabilities.shape[1] < 2:
                raise PredictorError("predict_proba() did not return binary probabilities.")
            return probabilities[0]

        if hasattr(model, "decision_function"):
            scores = np.asarray(model.decision_function(features), dtype=float).reshape(-1)
            score = float(scores[0])
            abnormal = 1.0 / (1.0 + np.exp(-score))
            return np.asarray([1.0 - abnormal, abnormal], dtype=float)

        raise PredictorError("Model does not support probability or decision scores.")

    def predict(self, gene_values: dict[str, float]) -> dict[str, Any]:
        """Run the full preprocessing and inference pipeline."""

        self.validate_gene_payload(gene_values)
        ordered_vector = self._ordered_feature_vector(gene_values)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names*")
            scaled_vector = self._artifacts.scaler.transform(ordered_vector)

            model = self._artifacts.model
            predicted_class = int(np.asarray(model.predict(scaled_vector)).reshape(-1)[0])
            probabilities = self._prediction_probabilities(scaled_vector)

        classes = list(getattr(model, "classes_", [0, 1]))
        if len(classes) < 2:
            classes = [0, 1]

        class_to_probability = {int(classes[index]): float(probabilities[index]) for index in range(min(len(classes), 2))}
        normal_probability = class_to_probability.get(0, float(probabilities[0]))
        abnormal_probability = class_to_probability.get(1, float(probabilities[min(1, len(probabilities) - 1)]))

        confidence = max(normal_probability, abnormal_probability) * 100.0
        prediction_label = "Abnormal" if predicted_class == 1 else "Normal"

        return {
            "prediction": prediction_label,
            "class": predicted_class,
            "confidence": round(confidence, 1),
            "probabilities": {
                "normal": round(normal_probability, 6),
                "abnormal": round(abnormal_probability, 6),
            },
            "model": self.model_name,
        }

    def health_payload(self) -> dict[str, Any]:
        """Return the operational health payload."""

        return {
            "status": "healthy",
            "model_loaded": True,
            "selected_algorithm": self.model_name,
            "selected_genes_count": len(self.selected_genes),
        }

    def metadata_payload(self) -> dict[str, Any]:
        """Return a flattened metadata summary for the API."""

        validation_metrics = self._artifacts.metadata.get("validation_metrics", {})
        evaluation_metrics = self._artifacts.metadata.get("evaluation_metrics", {})
        # evaluation_metrics can be either a flat metrics dict (accuracy, roc_auc, ...)
        # or a nested dict containing 'holdout_eval' / 'nested_cv_loocv'. Prefer a
        # flat metrics dict when present (this matches exported model_metadata.json),
        # otherwise fall back to holdout_eval, nested_cv_loocv, or validation_metrics.
        metrics_source: dict[str, Any] = {}
        if isinstance(evaluation_metrics, dict):
            # flat metrics check
            if any(k in evaluation_metrics for k in ("roc_auc", "accuracy", "f1_score")):
                metrics_source = evaluation_metrics
            elif "holdout_eval" in evaluation_metrics and isinstance(evaluation_metrics["holdout_eval"], dict):
                metrics_source = evaluation_metrics["holdout_eval"]
            elif "nested_cv_loocv" in evaluation_metrics and isinstance(evaluation_metrics["nested_cv_loocv"], dict):
                metrics_source = evaluation_metrics["nested_cv_loocv"]
        if not metrics_source:
            metrics_source = validation_metrics or {}

        dataset_shape = self._artifacts.metadata.get("dataset_shape", {})
        if not isinstance(dataset_shape, dict):
            dataset_shape = {"rows": 0, "columns": 0}

        best_model_params = self._artifacts.metadata.get("best_model_params", {})
        if not isinstance(best_model_params, dict):
            best_model_params = {}

        target_mapping = self._artifacts.metadata.get("target_mapping", {})
        if not isinstance(target_mapping, dict):
            target_mapping = {}

        ranking_progression = self._artifacts.metadata.get("ranking_progression", [])
        if not isinstance(ranking_progression, list):
            ranking_progression = []

        cleaning_summary_raw = self._artifacts.metadata.get("cleaning_summary", {})
        if not isinstance(cleaning_summary_raw, dict):
            cleaning_summary_raw = {}
        cleaning_summary = {
            "initial_shape": cleaning_summary_raw.get("initial_shape", []),
            "post_dedup_shape": cleaning_summary_raw.get("post_dedup_shape", []),
            "dropped_id_columns": cleaning_summary_raw.get("dropped_id_columns", []),
            "dropped_constant_columns": cleaning_summary_raw.get("dropped_constant_columns", []),
            "missing_values_before_imputation": cleaning_summary_raw.get("missing_values_before_imputation", 0),
            "feature_count": cleaning_summary_raw.get("feature_count", 0),
        }

        return {
            "best_model": self.model_name,
            "selected_genes": self.selected_genes,
            "selected_genes_count": len(self.selected_genes),
            "roc_auc": float(metrics_source.get("roc_auc", 0.0)),
            "f1_score": float(metrics_source.get("f1_score", 0.0)),
            "recall": float(metrics_source.get("recall", 0.0)),
            "precision": float(metrics_source.get("precision", 0.0)),
            "accuracy": float(metrics_source.get("accuracy", 0.0)),
            "training_date": str(self._artifacts.metadata.get("training_date", "")),
            "dataset_shape": {
                "rows": int(dataset_shape.get("rows", 0)),
                "columns": int(dataset_shape.get("columns", 0)),
            },
            "validation_metrics": validation_metrics,
            "evaluation_metrics": evaluation_metrics,
            "best_model_params": best_model_params,
            "evaluation_method": str(self._artifacts.metadata.get("evaluation_method", "")),
            "training_set": str(self._artifacts.metadata.get("training_set", "")),
            "target_column": str(self._artifacts.metadata.get("target_column", "")),
            "target_mapping": target_mapping,
            "ranking_progression": ranking_progression,
            "cleaning_summary": cleaning_summary,
        }


def get_predictor() -> ColonCancerPredictor:
    """Return the process-wide singleton predictor."""

    return ColonCancerPredictor.get_instance()
