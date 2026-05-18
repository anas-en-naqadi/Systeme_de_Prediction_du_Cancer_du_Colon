"""Custom forward feature selection for the colon cancer pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from utils.logger import logger as default_logger


@dataclass(slots=True)
class ForwardFeatureSelector:
    """Greedy forward feature selection with ROC-AUC and stratified CV."""

    estimator: object | None = None
    cv_splits: int = 5
    scoring: str = "roc_auc"
    max_features: int = 6
    candidate_pool_size: int = 50
    min_improvement: float = 0.0
    random_state: int = 42
    n_jobs: int = 1
    logger: object = field(default_factory=lambda: default_logger)
    selected_features_: list[str] = field(default_factory=list, init=False)
    selected_feature_scores_: list[float] = field(default_factory=list, init=False)
    ranking_progression_: list[dict[str, float | int | str]] = field(default_factory=list, init=False)
    candidate_ranking_: list[dict[str, float | str]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.estimator is None:
            self.estimator = LogisticRegression(max_iter=5000, solver="liblinear", random_state=self.random_state)

    def _build_cv(self) -> StratifiedKFold:
        return StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)

    def _score_subset(self, X: pd.DataFrame, y: Sequence[int], columns: Sequence[str]) -> tuple[float, float]:
        subset = X.loc[:, list(columns)]
        scores = cross_val_score(
            clone(self.estimator),
            subset,
            y,
            cv=self._build_cv(),
            scoring=self.scoring,
            n_jobs=self.n_jobs,
        )
        return float(np.mean(scores)), float(np.std(scores))

    def _build_candidate_pool(self, X: pd.DataFrame, y: Sequence[int]) -> list[str]:
        feature_names = list(X.columns)
        if len(feature_names) <= self.candidate_pool_size:
            return feature_names

        self.logger.info(
            f"Pre-ranking {len(feature_names)} features and keeping the top {self.candidate_pool_size} candidates."
        )
        scored_candidates: list[tuple[str, float]] = []
        for feature_name in feature_names:
            mean_score, std_score = self._score_subset(X, y, [feature_name])
            scored_candidates.append((feature_name, mean_score))
            self.candidate_ranking_.append(
                {"feature": feature_name, "cv_mean_auc": mean_score, "cv_std_auc": std_score}
            )

        scored_candidates.sort(key=lambda item: (item[1], item[0]), reverse=True)
        return [feature_name for feature_name, _ in scored_candidates[: self.candidate_pool_size]]

    def fit(self, X: pd.DataFrame, y: Sequence[int]) -> "ForwardFeatureSelector":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ForwardFeatureSelector expects a pandas DataFrame.")

        if len(X) != len(y):
            raise ValueError("Feature matrix and target vector must have the same length.")

        candidates = self._build_candidate_pool(X, y)
        selected: list[str] = []
        remaining = [feature for feature in candidates if feature not in selected]
        best_previous_score = float("-inf")

        self.logger.section("Forward Feature Selection")
        self.logger.info(f"Candidate pool size: {len(candidates)}")

        for step in range(1, min(self.max_features, len(candidates)) + 1):
            best_feature = None
            best_score = float("-inf")
            best_std = 0.0

            for feature_name in remaining:
                mean_score, std_score = self._score_subset(X, y, selected + [feature_name])
                if mean_score > best_score:
                    best_feature = feature_name
                    best_score = mean_score
                    best_std = std_score

            if best_feature is None:
                break

            improvement = best_score - best_previous_score if np.isfinite(best_previous_score) else best_score
            selected.append(best_feature)
            remaining.remove(best_feature)
            best_previous_score = best_score

            self.selected_features_.append(best_feature)
            self.selected_feature_scores_.append(best_score)
            self.ranking_progression_.append(
                {
                    "step": step,
                    "feature": best_feature,
                    "cv_mean_auc": best_score,
                    "cv_std_auc": best_std,
                    "improvement": improvement,
                    "subset_size": len(selected),
                }
            )

            self.logger.success(
                f"Step {step}: selected {best_feature} | CV ROC-AUC={best_score:.4f} | delta={improvement:.4f}"
            )

            if len(selected) >= self.max_features:
                break

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_features_:
            raise RuntimeError("ForwardFeatureSelector must be fitted before transform().")
        return X.loc[:, self.selected_features_].copy()

    def fit_transform(self, X: pd.DataFrame, y: Sequence[int]) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
