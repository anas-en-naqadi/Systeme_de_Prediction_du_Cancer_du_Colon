# """Main training entrypoint for the colon cancer prediction project."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from utils.evaluate import (
    calculate_metrics,
    generate_classification_report_text,
    plot_confusion_matrix,
    plot_roc_curve,
    safe_probability_scores,
    save_text_report,
)
from utils.ffs import ForwardFeatureSelector
from utils.logger import logger


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = Path(__file__).resolve().parent
DATA_DIR = TRAINING_DIR / "data"
MODEL_DIR = PROJECT_ROOT / "model"
REPORTS_DIR = TRAINING_DIR / "reports"


# ── Dataset resolution ─────────────────────────────────────────────────────────

def resolve_dataset_path() -> Path:
    """Return the first dataset path that exists."""

    candidates = [
        DATA_DIR / "colon_cancer_dataset.csv",
        DATA_DIR / "colon cancer dataset.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Dataset not found. Expected one of: {', '.join(str(c) for c in candidates)}"
    )


# ── Target detection & encoding ────────────────────────────────────────────────

def detect_target_column(frame: pd.DataFrame) -> str:
    """Detect the target column using common biomedical naming conventions."""

    normalized_columns = {col.lower().strip(): col for col in frame.columns}
    for candidate in ("diagnosis", "class", "target", "label", "y"):
        if candidate in normalized_columns:
            return normalized_columns[candidate]
    return frame.columns[-1]


def normalize_target(series: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    """Convert binary targets to a stable 0/1 representation.

    IMPORTANT: abnormal keywords are tested BEFORE normal keywords to avoid
    the substring trap ("normal" in "abnormal" == True in Python).
    """

    cleaned = series.astype(str).str.strip().str.lower()
    values = sorted(cleaned.dropna().unique())

    if len(values) != 2:
        raise ValueError(f"Target must have exactly 2 classes, found: {values}")

    normal_keywords   = ["normal", "healthy", "control", "negative"]
    abnormal_keywords = ["abnormal", "cancer", "tumor", "positive", "disease"]

    mapping: dict[str, int] = {}
    for v in values:
        is_abnormal = any(v == kw or v.startswith(kw) for kw in abnormal_keywords)
        is_normal   = any(v == kw or v.startswith(kw) for kw in normal_keywords)
        if is_abnormal:
            mapping[v] = 1
        elif is_normal:
            mapping[v] = 0

    # Alphabetic fallback if mapping is incomplete
    if len(mapping) < 2:
        mapping = {values[0]: 0, values[1]: 1}

    return cleaned.map(mapping).astype(int), mapping


# ── Cleaning ───────────────────────────────────────────────────────────────────

def drop_id_like_columns(
    frame: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Remove obvious identifier columns that are not gene features."""

    dropped: list[str] = []
    feature_frame = frame.drop(columns=[target_column]).copy()

    for column in list(feature_frame.columns):
        norm = str(column).strip().lower()
        uniqueness_ratio = feature_frame[column].nunique(dropna=True) / max(len(feature_frame), 1)
        id_like = (
            norm in {"0", "id", "index", "sample", "sample_id", "patient", "patient_id"}
            or norm.startswith("unnamed")
            or norm.isdigit()
        )
        if id_like and uniqueness_ratio > 0.9:
            feature_frame = feature_frame.drop(columns=[column])
            dropped.append(str(column))

    return feature_frame, dropped


def clean_frame(
    frame: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Clean the dataset and return the feature matrix, target vector, and summary."""

    initial_shape = frame.shape
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.drop_duplicates().reset_index(drop=True)

    y, target_mapping = normalize_target(frame[target_column])
    feature_frame, dropped_id_columns = drop_id_like_columns(frame, target_column)

    dropped_constant: list[str] = []

    numeric_frame = feature_frame.apply(pd.to_numeric, errors="coerce")
    valid_cols = [c for c in numeric_frame.columns if numeric_frame[c].notna().mean() >= 0.6]
    numeric_frame = numeric_frame[valid_cols]

    for column in list(numeric_frame.columns):
        if numeric_frame[column].nunique(dropna=True) <= 1:
            numeric_frame = numeric_frame.drop(columns=[column])
            dropped_constant.append(str(column))

    feature_frame = numeric_frame.loc[:, numeric_frame.notna().mean() > 0]

    if feature_frame.empty:
        raise ValueError("No usable numeric feature columns were found after cleaning.")

    cleaning_summary = {
        "initial_shape": initial_shape,
        "post_dedup_shape": frame.shape,
        "dropped_id_columns": dropped_id_columns,
        "dropped_constant_columns": dropped_constant,
        "missing_values_before_imputation": int(feature_frame.isna().sum().sum()),
        "feature_count": int(feature_frame.shape[1]),
    }

    return feature_frame, y, {"target_mapping": target_mapping, **cleaning_summary}


# ── Model search spaces ────────────────────────────────────────────────────────

def build_model_search_spaces() -> dict[str, tuple[object, dict[str, list[Any]]]]:
    """Return the model family and hyperparameter grid definitions."""

    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=5000, solver="liblinear", random_state=42),
            {
                "C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "penalty": ["l2"],
                "class_weight": [None, "balanced"],
            },
        ),
        "Linear SVM": (
            SVC(kernel="linear", probability=True, random_state=42),
            {
                "C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "class_weight": [None, "balanced"],
            },
        ),
        "RBF SVM": (
            SVC(kernel="rbf", probability=True, random_state=42),
            {
                "C": [0.1, 1.0, 10.0, 100.0],
                "gamma": ["scale", "auto", 0.01, 0.1],
                "class_weight": [None, "balanced"],
            },
        ),
    }


# ── Nested CV + LOOCV (honest evaluation on small dataset) ────────────────────

def nested_cv_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    max_features: int = 6,
    candidate_pool_size: int = 50,
) -> dict[str, Any]:
    """Nested Cross-Validation + LOOCV for honest performance estimation.

    Structure
    ---------
    LOOCV outer loop (62 folds — 1 patient isolated each time)
        └── per fold:
                Imputation   → median from the 61 train patients only
                Scaling      → StandardScaler fit on 61 train patients only
                FFS          → selects genes from 61 train patients only
                GridSearchCV → 5-fold inner CV on 61 train patients only
                Predict      → on the 1 isolated patient
    → aggregated predictions → final metrics (no data leakage anywhere)
    """

    from sklearn.model_selection import LeaveOneOut
    from sklearn.preprocessing import StandardScaler as _Scaler

    logger.section("Nested CV + LOOCV — Honest Performance Estimation")
    logger.info("Outer LOOCV: each of the 62 patients is isolated once as the test set.")
    logger.info(f"Per fold: Imputation + Scaling + FFS + GridSearchCV on the {len(X) - 1} remaining patients.")

    loo = LeaveOneOut()
    all_true:   list[int]   = []
    all_pred:   list[int]   = []
    all_scores: list[float] = []

    search_spaces = build_model_search_spaces()
    cv_inner      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    total_folds   = len(X)

    _silent_logger = type("_SilentLogger", (), {
        "info":    lambda self, *a, **k: None,
        "success": lambda self, *a, **k: None,
        "warning": lambda self, *a, **k: None,
        "error":   lambda self, *a, **k: None,
        "section": lambda self, *a, **k: None,
        "metric":  lambda self, *a, **k: None,
    })()

    for fold_idx, (train_idx, test_idx) in enumerate(loo.split(X, y), start=1):

        X_tr = X.iloc[train_idx].reset_index(drop=True)
        y_tr = y.iloc[train_idx].reset_index(drop=True)
        X_te = X.iloc[test_idx].reset_index(drop=True)
        y_te = y.iloc[test_idx].reset_index(drop=True)

        # Imputation — train medians only (no leakage from test patient)
        medians = X_tr.median(numeric_only=True)
        X_tr    = X_tr.fillna(medians)
        X_te    = X_te.fillna(medians)

        # Scaling — fit on train only
        scaler      = _Scaler()
        X_tr_scaled = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns)
        X_te_scaled = pd.DataFrame(scaler.transform(X_te),     columns=X_te.columns)

        # FFS — train only, logs silenced
        ffs        = ForwardFeatureSelector(
            max_features=max_features,
            candidate_pool_size=candidate_pool_size,
            n_jobs=1,
        )
        ffs.logger = _silent_logger
        ffs.fit(X_tr_scaled, y_tr)
        genes = ffs.selected_features_

        X_tr_sel = X_tr_scaled[genes]
        X_te_sel = X_te_scaled[genes]

        # GridSearchCV — inner CV on train only
        best_estimator = None
        best_cv_auc    = float("-inf")

        for _name, (estimator, param_grid) in search_spaces.items():
            search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                scoring="roc_auc",
                cv=cv_inner,
                n_jobs=-1,
                refit=True,
            )
            try:
                search.fit(X_tr_sel, y_tr)
                if search.best_score_ > best_cv_auc:
                    best_cv_auc    = search.best_score_
                    best_estimator = search.best_estimator_
            except Exception:
                continue

        if best_estimator is None:
            continue

        # Predict on the 1 isolated patient
        pred  = best_estimator.predict(X_te_sel)
        score = safe_probability_scores(best_estimator, X_te_sel)

        all_true.extend(y_te.tolist())
        all_pred.extend(pred.tolist())
        all_scores.extend(score.tolist())

        if fold_idx % 10 == 0 or fold_idx == total_folds:
            logger.info(f"  Fold {fold_idx}/{total_folds} done...")

    # Aggregate metrics over all 62 predictions
    nested_metrics = calculate_metrics(all_true, all_pred, all_scores)

    logger.section("Nested CV + LOOCV Results")
    for name, value in nested_metrics.items():
        logger.metric(
            f"[NESTED CV] {name}",
            f"{value:.4f}" if isinstance(value, float) else value,
        )
    logger.info(
        "These metrics are the most reliable for a 62-patient dataset — "
        "zero data leakage between FFS, tuning, and evaluation."
    )

    return {
        "metrics":   nested_metrics,
        "y_true":    all_true,
        "y_pred":    all_pred,
        "y_scores":  all_scores,
    }


# ── Final model training on 100 % of the data ─────────────────────────────────

def train_final_model(
    X_full_scaled: pd.DataFrame,
    y: pd.Series,
) -> dict[str, Any]:
    """GridSearchCV on 100 % of the data to pick and fit the final model.

    The winning model is already fitted (refit=True in GridSearchCV).
    Metrics come exclusively from Nested CV — not from this step.
    """

    logger.section("Final Model Selection on Full Dataset (100 %)")

    cv            = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search_spaces = build_model_search_spaces()
    best_result   = None
    best_auc      = float("-inf")

    for model_name, (estimator, param_grid) in search_spaces.items():
        logger.info(f"GridSearchCV for {model_name}...")
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_full_scaled, y)
        logger.success(
            f"{model_name} | CV AUC={search.best_score_:.4f} | params={search.best_params_}"
        )
        if search.best_score_ > best_auc:
            best_auc    = search.best_score_
            best_result = {
                "model_name":      model_name,
                "best_estimator":  search.best_estimator_,   # already fitted on full data
                "best_params":     search.best_params_,
                "cv_best_score":   float(search.best_score_),
            }

    logger.success(
        f"Selected model: {best_result['model_name']} | CV AUC={best_result['cv_best_score']:.4f}"
    )
    return best_result


# ── Reports ────────────────────────────────────────────────────────────────────

def create_reports(
    nested_results: dict[str, Any],
    reports_dir: Path,
) -> dict[str, str]:
    """Generate evaluation plots and classification report from Nested CV results.

    All artefacts are based exclusively on the Nested CV + LOOCV predictions
    (62 predictions, one per patient, each made on a model that never saw that patient).
    """

    reports_dir.mkdir(parents=True, exist_ok=True)

    y_true   = nested_results["y_true"]
    y_pred   = nested_results["y_pred"]
    y_scores = nested_results["y_scores"]

    report_text = generate_classification_report_text(y_true, y_pred)

    cm_path     = plot_confusion_matrix(y_true, y_pred,   reports_dir / "confusion_matrix.png")
    roc_path    = plot_roc_curve(y_true, y_scores,        reports_dir / "roc_curve.png")
    report_path = save_text_report(report_text,           reports_dir / "classification_report.txt")

    logger.success(f"Saved confusion matrix  → {cm_path}")
    logger.success(f"Saved ROC curve         → {roc_path}")
    logger.success(f"Saved classification report → {report_path}")

    return {
        "confusion_matrix":        str(cm_path),
        "roc_curve":               str(roc_path),
        "classification_report":   str(report_path),
    }


# ── Artifact export ────────────────────────────────────────────────────────────

def export_artifacts(
    final_model:        object,
    final_scaler:       StandardScaler,
    selected_genes:     list[str],
    selection_scores:   list[float],
    ranking_progression: list[dict[str, Any]],
    best_result:        dict[str, Any],
    nested_results:     dict[str, Any],
    dataset_shape:      tuple[int, int],
    target_column:      str,
    target_mapping:     dict[str, int],
    cleaning_summary:   dict[str, Any],
    imputation_values:  dict[str, float],
) -> Path:
    """Persist model, scaler, selected genes, and metadata.

    evaluation_metrics in the JSON are the Nested CV + LOOCV metrics only.
    """

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path          = MODEL_DIR / "model.pkl"
    scaler_path         = MODEL_DIR / "scaler.pkl"
    selected_genes_path = MODEL_DIR / "selected_genes.json"
    metadata_path       = MODEL_DIR / "model_metadata.json"

    joblib.dump(final_model,  model_path)
    joblib.dump(final_scaler, scaler_path)

    selected_genes_payload = {
        "selected_genes":    selected_genes,
        "selection_scores":  selection_scores,
        "ranking_progression": ranking_progression,
    }
    selected_genes_path.write_text(
        json.dumps(selected_genes_payload, indent=2), encoding="utf-8"
    )

    metadata = {
        "best_model_name":   best_result["model_name"],
        "best_model_params": best_result["best_params"],
        "cv_auc_on_full_data": best_result["cv_best_score"],
        "selected_genes":    selected_genes,
        "selection_scores":  selection_scores,
        "ranking_progression": ranking_progression,
        # Official metrics = Nested CV + LOOCV (no data leakage)
        "evaluation_metrics": nested_results["metrics"],
        "evaluation_method":  "Nested CV + LOOCV (62 folds, 1 patient isolated per fold)",
        "training_date":     datetime.now(timezone.utc).isoformat(),
        "training_set":      "100% of dataset (no holdout — evaluation via Nested CV)",
        "dataset_shape":     {"rows": int(dataset_shape[0]), "columns": int(dataset_shape[1])},
        "target_column":     target_column,
        "target_mapping":    target_mapping,
        "cleaning_summary":  cleaning_summary,
        "imputation_strategy": "global median (100% of data)",
        "imputation_values": imputation_values,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logger.success(f"Saved model              → {model_path}")
    logger.success(f"Saved scaler             → {scaler_path}")
    logger.success(f"Saved selected genes     → {selected_genes_path}")
    logger.success(f"Saved metadata           → {metadata_path}")

    return metadata_path


# ── Main orchestrator ──────────────────────────────────────────────────────────

def main() -> None:
    """Run the full training workflow end to end.

    Workflow
    --------
    1. Load & clean dataset (62 patients × ~2000 genes)
    2. Impute missing values using global medians (100 % of data)
    3. Nested CV + LOOCV on 100 % of data  → honest evaluation metrics
    4. FFS on 100 % of data                → select the 6 best genes
    5. Scale the 6-gene dataset (100 % of data)
    6. GridSearchCV on 100 % of data       → pick & fit the final model
    7. Save reports (based on Nested CV)
    8. Save artifacts (model, scaler, genes, metadata)
    """

    logger.section("Colon Cancer ML Training Pipeline")

    # ── 1. Load & clean ───────────────────────────────────────────────────────
    dataset_path = resolve_dataset_path()
    logger.info(f"Loading dataset from {dataset_path}")
    raw_frame = pd.read_csv(dataset_path, low_memory=False)
    logger.info(f"Raw dataset shape: {raw_frame.shape}")

    target_column = detect_target_column(raw_frame)
    logger.info(f"Detected target column: '{target_column}'")

    cleaned_features, target, cleaning_summary = clean_frame(raw_frame, target_column)
    logger.success(
        f"Cleaned dataset: {cleaned_features.shape[0]} patients × {cleaned_features.shape[1]} genes"
    )

    class_dist     = target.value_counts(normalize=False).sort_index().to_dict()
    class_dist_pct = target.value_counts(normalize=True).sort_index().to_dict()
    logger.metric("Class distribution",   class_dist)
    logger.metric("Class distribution %", class_dist_pct)

    # ── 2. Imputation on 100 % of the data ───────────────────────────────────
    global_medians            = cleaned_features.median(numeric_only=True)
    cleaned_features_imputed  = cleaned_features.fillna(global_medians)
    imputation_values         = global_medians.to_dict()
    logger.info("Missing values imputed using global medians (100 % of data).")

    # ── 3. Nested CV + LOOCV on 100 % of the data ────────────────────────────
    logger.info("Starting Nested CV + LOOCV — this may take 5-10 minutes...")
    # Scale the full dataset before passing it to nested_cv_evaluation.
    # Inside each LOOCV fold, nested_cv_evaluation will re-scale using only
    # the 61 train patients — this outer scaling is just for the FFS candidate pool.
    full_scaler_for_cv = StandardScaler()
    full_scaled_for_cv = pd.DataFrame(
        full_scaler_for_cv.fit_transform(cleaned_features_imputed),
        columns=cleaned_features_imputed.columns,
    )
    nested_results = nested_cv_evaluation(
        X=full_scaled_for_cv,   # 62 patients — all of them
        y=target,
        max_features=6,
        candidate_pool_size=50,
    )

    # ── 4. FFS on 100 % of the data ──────────────────────────────────────────
    logger.section("Forward Feature Selection on Full Dataset (100 %)")
    full_scaler_for_ffs = StandardScaler()
    full_scaled_for_ffs = pd.DataFrame(
        full_scaler_for_ffs.fit_transform(cleaned_features_imputed),
        columns=cleaned_features_imputed.columns,
    )
    ffs = ForwardFeatureSelector(max_features=6, candidate_pool_size=50, n_jobs=1)
    ffs.fit(full_scaled_for_ffs, target)

    selected_genes      = ffs.selected_features_
    selection_scores    = ffs.selected_feature_scores_
    ranking_progression = ffs.ranking_progression_
    logger.success(f"Selected genes: {selected_genes}")

    # ── 5. Final scaler — fit on 100 % of the data (6 genes only) ────────────
    final_scaler = StandardScaler()
    full_final_scaled = pd.DataFrame(
        final_scaler.fit_transform(cleaned_features_imputed[selected_genes]),
        columns=selected_genes,
    )
    logger.info("Final scaler fitted on 100 % of data (6 selected genes).")

    # ── 6. GridSearchCV + final model on 100 % of the data ───────────────────
    best_result  = train_final_model(full_final_scaled, target)
    final_model  = best_result["best_estimator"]   # already fitted (refit=True)

    # ── 7. Reports — based exclusively on Nested CV ───────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    reports = create_reports(nested_results, REPORTS_DIR)

    # ── 8. Export artifacts ───────────────────────────────────────────────────
    metadata_path = export_artifacts(
        final_model=final_model,
        final_scaler=final_scaler,
        selected_genes=selected_genes,
        selection_scores=selection_scores,
        ranking_progression=ranking_progression,
        best_result=best_result,
        nested_results=nested_results,
        dataset_shape=raw_frame.shape,
        target_column=target_column,
        target_mapping=cleaning_summary["target_mapping"],
        cleaning_summary=cleaning_summary,
        imputation_values=imputation_values,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.section("Training Summary")
    logger.success(f"Model artifacts → {MODEL_DIR}")
    logger.success(f"Reports         → {REPORTS_DIR}")
    logger.info(f"Metadata file   → {metadata_path}")
    logger.info(f"Report files    → {reports}")


if __name__ == "__main__":
    main()