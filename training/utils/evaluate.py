"""Reusable evaluation utilities for the colon cancer training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def calculate_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Sequence[float],
) -> dict[str, float]:
    """Return a consistent set of classification metrics."""

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    except ValueError:
        metrics["roc_auc"] = float("nan")

    return metrics


def generate_classification_report_text(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> str:
    """Create a human-readable classification report."""

    return classification_report(
        y_true,
        y_pred,
        target_names=["Normal", "Colon Cancer"],
        zero_division=0,
    )


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    output_path: Path,
    title: str = "Confusion Matrix",
) -> Path:
    """Plot and save a confusion matrix heatmap."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Colon Cancer"],
        yticklabels=["Normal", "Colon Cancer"],
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def plot_roc_curve(
    y_true: Sequence[int],
    y_score: Sequence[float],
    output_path: Path,
    title: str = "ROC Curve",
) -> Path:
    """Plot and save a ROC curve."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_value = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#0ea5e9", linewidth=2, label=f"AUC = {auc_value:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#64748b", linewidth=1)
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def save_text_report(report_text: str, output_path: Path) -> Path:
    """Persist a text report to disk."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")
    return output_path


def safe_probability_scores(model: object, features: object) -> np.ndarray:
    """Return a score array suitable for ROC-AUC and ROC plotting."""

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        return np.asarray(probabilities)[:, 1]

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features))
        return scores.reshape(-1)

    raise AttributeError("Model must expose predict_proba or decision_function.")
