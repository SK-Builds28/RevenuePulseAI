"""Ranking, classification, calibration and segment metrics.

Precision@K is the primary family of metrics because the product is a
prioritisation tool: a manager can review only a small number of accounts, so
what matters is how many true future decliners appear near the top of the queue.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _as_binary(y_true: Iterable[Any]) -> np.ndarray:
    y = np.asarray(list(y_true))
    if y.ndim != 1:
        raise ValueError("y_true must be one-dimensional")
    if len(y) == 0:
        raise ValueError("y_true must not be empty")
    return y.astype(int)


def _as_scores(scores: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(scores), dtype=float)
    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    return values


def top_k_indices(scores: Sequence[float], k: int) -> np.ndarray:
    """Return indices of the top ``k`` scores with stable tie handling."""

    values = np.asarray(scores, dtype=float)
    if len(values) == 0:
        return np.array([], dtype=int)
    effective_k = min(max(int(k), 0), len(values))
    if effective_k == 0:
        return np.array([], dtype=int)
    # Stable sort preserves input order for ties, which is deterministic when
    # the caller supplies a pre-ordered frame.
    order = np.argsort(-values, kind="stable")
    return order[:effective_k]


def ranking_metrics(y_true: Sequence[Any], scores: Sequence[float], k: int) -> dict[str, Any]:
    """Calculate Precision@K, Recall@K, Lift@K and false alarms at K."""

    y = _as_binary(y_true)
    values = _as_scores(scores)
    if len(y) != len(values):
        raise ValueError("y_true and scores must have the same length")
    if k <= 0:
        raise ValueError("k must be positive")

    n = len(y)
    effective_k = min(int(k), n)
    selected = top_k_indices(values, effective_k)
    positives = int(y.sum())
    prevalence = float(positives / n) if n else float("nan")
    true_positives = int(y[selected].sum())
    false_positives = int(effective_k - true_positives)
    false_negatives = int(positives - true_positives)
    precision = float(true_positives / effective_k) if effective_k else float("nan")
    recall = float(true_positives / positives) if positives else float("nan")
    lift = float(precision / prevalence) if prevalence > 0 else float("nan")
    false_alarm_rate = (
        float(false_positives / effective_k) if effective_k else float("nan")
    )

    return {
        "k": int(k),
        "k_effective": int(effective_k),
        "selected": int(effective_k),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "positives": positives,
        "prevalence": prevalence,
        "precision": precision,
        "recall": recall,
        "lift": lift,
        "false_alarm_rate": false_alarm_rate,
    }


def fraction_metrics(
    y_true: Sequence[Any],
    scores: Sequence[float],
    fraction: float,
) -> dict[str, Any]:
    """Calculate the same ranking metrics for a top-fraction capacity rule."""

    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    y = _as_binary(y_true)
    k = max(1, int(np.ceil(fraction * len(y))))
    result = ranking_metrics(y, scores, k)
    result["fraction"] = float(fraction)
    return result


def binary_classification_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
) -> dict[str, Any]:
    """Return threshold-based precision, recall, F1 and confusion counts."""

    y = _as_binary(y_true)
    predictions = _as_binary(y_pred)
    if len(y) != len(predictions):
        raise ValueError("y_true and y_pred must have the same length")

    matrix = confusion_matrix(y, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    selected = int(tp + fp)
    actual_positives = int(tp + fn)

    return {
        "threshold_predictions": selected,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "accuracy": float((tp + tn) / len(y)),
        "false_alarm_rate": float(fp / selected) if selected else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "actual_positives": actual_positives,
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": matrix.tolist(),
        },
    }


def pr_auc(y_true: Sequence[Any], scores: Sequence[float]) -> float:
    y = _as_binary(y_true)
    values = _as_scores(scores)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, values))


def roc_auc(y_true: Sequence[Any], scores: Sequence[float]) -> float:
    y = _as_binary(y_true)
    values = _as_scores(scores)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, values))


def brier(y_true: Sequence[Any], probabilities: Sequence[float]) -> float:
    y = _as_binary(y_true)
    values = _as_scores(probabilities)
    return float(brier_score_loss(y, values))


def calibration_table(
    y_true: Sequence[Any],
    scores: Sequence[float],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Return an equal-count calibration table.

    Bins are formed on the rank of the predicted score so that every bin has
    roughly the same number of observations. This is a table, not a plot, and
    avoids adding a plotting dependency.
    """

    y = _as_binary(y_true)
    values = _as_scores(scores)
    if len(y) != len(values):
        raise ValueError("y_true and scores must have the same length")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")

    bins = min(int(n_bins), len(y))
    ranks = pd.Series(values).rank(method="first")
    bin_ids = pd.qcut(ranks, q=bins, labels=False, duplicates="drop")
    frame = pd.DataFrame(
        {
            "bin": np.asarray(bin_ids, dtype=int),
            "score": values,
            "target": y,
        }
    )
    table = (
        frame.groupby("bin", as_index=False)
        .agg(
            count=("target", "size"),
            mean_predicted=("score", "mean"),
            observed_positive_rate=("target", "mean"),
        )
        .sort_values("bin", kind="mergesort")
        .reset_index(drop=True)
    )
    table["absolute_gap"] = (
        table["mean_predicted"] - table["observed_positive_rate"]
    ).abs()
    return table


def evaluate_predictions(
    frame: pd.DataFrame,
    *,
    target_column: str,
    scores: Sequence[float],
    k_values: Sequence[int],
    fractions: Sequence[float],
    primary_k: int,
    binary_predictions: Sequence[Any] | None = None,
    binary_threshold: float | None = None,
    probability_like: bool = True,
) -> dict[str, Any]:
    """Evaluate one model's scores on a labelled snapshot frame."""

    if target_column not in frame.columns:
        raise ValueError(f"Missing target column: {target_column}")
    y = frame[target_column].to_numpy()
    if pd.isna(y).any():
        raise ValueError("Evaluation frame contains unlabelled target rows")
    values = _as_scores(scores)
    if len(frame) != len(values):
        raise ValueError("Score length does not match evaluation frame")

    if binary_predictions is None:
        if binary_threshold is None:
            raise ValueError(
                "Provide binary_predictions or binary_threshold for threshold metrics"
            )
        predictions = (values >= float(binary_threshold)).astype(int)
    else:
        predictions = _as_binary(binary_predictions)
        if len(predictions) != len(frame):
            raise ValueError("binary_predictions length does not match frame")

    ranking = {
        str(int(k)): ranking_metrics(y, values, int(k))
        for k in k_values
    }
    fraction_results = {
        f"{float(fraction):.2f}": fraction_metrics(y, values, float(fraction))
        for fraction in fractions
    }

    result: dict[str, Any] = {
        "n_rows": int(len(frame)),
        "n_accounts": int(frame["account_id"].nunique()) if "account_id" in frame else None,
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "primary_k": int(primary_k),
        "primary_ranking": ranking_metrics(y, values, int(primary_k)),
        "ranking": ranking,
        "fractions": fraction_results,
        "pr_auc": pr_auc(y, values),
        "roc_auc": roc_auc(y, values),
        "brier": brier(y, values) if probability_like else None,
        "binary": binary_classification_metrics(y, predictions),
    }
    if binary_threshold is not None:
        result["binary"]["threshold"] = float(binary_threshold)
    result["binary"]["probability_like_scores"] = bool(probability_like)
    return result


def segment_analysis(
    frame: pd.DataFrame,
    *,
    target_column: str,
    scores: Sequence[float],
    binary_predictions: Sequence[Any],
    fraction: float = 0.20,
    min_rows: int = 30,
    min_positives: int = 5,
) -> pd.DataFrame:
    """Evaluate ranking and false-alarm behaviour by synthetic segment.

    Segment results are descriptive only. Synthetic segment performance does not
    establish fairness in a real population.
    """

    if "segment" not in frame.columns:
        raise ValueError("Segment analysis requires a 'segment' column")
    values = _as_scores(scores)
    predictions = _as_binary(binary_predictions)
    if not (len(frame) == len(values) == len(predictions)):
        raise ValueError("frame, scores and predictions must have equal length")

    rows: list[dict[str, Any]] = []
    for segment, group in frame.groupby("segment", sort=True):
        indices = np.asarray(group.index)
        # Use positional lookups so the function also works on filtered frames
        # with a non-default index.
        positions = frame.index.get_indexer(indices)
        group_scores = values[positions]
        group_predictions = predictions[positions]
        y = group[target_column].to_numpy()
        n = len(group)
        positives = int(y.sum())
        ranking = fraction_metrics(y, group_scores, fraction)
        binary = binary_classification_metrics(y, group_predictions)
        sufficient = n >= min_rows and positives >= min_positives
        rows.append(
            {
                "segment": str(segment),
                "n_rows": n,
                "n_accounts": int(group["account_id"].nunique()),
                "positives": positives,
                "prevalence": float(y.mean()),
                "precision_at_fraction": ranking["precision"],
                "recall_at_fraction": ranking["recall"],
                "lift_at_fraction": ranking["lift"],
                "pr_auc": pr_auc(y, group_scores),
                "threshold_precision": binary["precision"],
                "threshold_recall": binary["recall"],
                "false_alarm_rate": binary["false_alarm_rate"],
                "predicted_positive": binary["threshold_predictions"],
                "fraction": float(fraction),
                "status": "ok" if sufficient else "insufficient_observations",
            }
        )

    return pd.DataFrame(rows).sort_values("segment", kind="mergesort").reset_index(drop=True)


def ranking_comparison_table(
    results: Mapping[str, Mapping[str, Any]],
    k_values: Sequence[int],
) -> pd.DataFrame:
    """Flatten Precision@K / Recall@K / Lift@K for multiple models."""

    rows: list[dict[str, Any]] = []
    for model_name, metrics in results.items():
        rankings = metrics["ranking"]
        for k in k_values:
            entry = rankings[str(int(k))]
            rows.append(
                {
                    "model": model_name,
                    "k": int(k),
                    "k_effective": entry["k_effective"],
                    "selected": entry["selected"],
                    "true_positives": entry["true_positives"],
                    "false_positives": entry["false_positives"],
                    "precision": entry["precision"],
                    "recall": entry["recall"],
                    "lift": entry["lift"],
                    "false_alarm_rate": entry["false_alarm_rate"],
                }
            )
    return pd.DataFrame(rows)
