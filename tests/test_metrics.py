"""Tests for ranking metrics, calibration and account-level bootstrap intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.bootstrap import account_cluster_bootstrap
from src.evaluation.metrics import (
    brier,
    calibration_table,
    binary_classification_metrics,
    fraction_metrics,
    pr_auc,
    ranking_metrics,
    roc_auc,
)


Y = [1, 0, 1, 0, 0]
SCORES = [0.9, 0.8, 0.7, 0.6, 0.5]


def test_precision_at_k_hand_computed():
    result = ranking_metrics(Y, SCORES, 2)
    assert result["k_effective"] == 2
    assert result["selected"] == 2
    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["prevalence"] == pytest.approx(0.4)
    assert result["lift"] == pytest.approx(1.25)
    assert result["false_alarm_rate"] == pytest.approx(0.5)


def test_false_alarm_rate_is_false_positives_over_selected():
    result = ranking_metrics(Y, SCORES, 3)
    assert result["selected"] == 3
    assert result["true_positives"] == 2
    assert result["false_positives"] == 1
    assert result["false_alarm_rate"] == pytest.approx(1 / 3)


def test_k_larger_than_population_is_handled():
    result = ranking_metrics(Y, SCORES, 50)
    assert result["k"] == 50
    assert result["k_effective"] == len(Y)
    assert result["selected"] == len(Y)
    assert result["precision"] == pytest.approx(0.4)
    assert result["recall"] == pytest.approx(1.0)
    assert result["lift"] == pytest.approx(1.0)


def test_ranking_metrics_require_positive_k():
    with pytest.raises(ValueError, match="k must be positive"):
        ranking_metrics(Y, SCORES, 0)


def test_ties_are_broken_stably_by_input_order():
    scores = [0.5, 0.5, 0.5, 0.5]
    labels = [1, 0, 0, 0]
    first = ranking_metrics(labels, scores, 1)
    second = ranking_metrics(labels, scores, 1)
    assert first == second
    assert first["true_positives"] == 1  # first row wins the tie


def test_fraction_metrics_uses_ceiling_of_population_share():
    result = fraction_metrics(Y, SCORES, 0.5)
    assert result["k_effective"] == 3
    assert result["fraction"] == pytest.approx(0.5)


def test_binary_classification_confusion_matrix():
    result = binary_classification_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["true_negatives"] == 1
    assert result["false_negatives"] == 1
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)
    assert result["confusion_matrix"]["matrix"] == [[1, 1], [1, 1]]


def test_auc_and_brier_are_bounded():
    assert 0.0 <= pr_auc(Y, SCORES) <= 1.0
    assert 0.0 <= roc_auc(Y, SCORES) <= 1.0
    assert 0.0 <= brier(Y, [0.8, 0.1, 0.7, 0.2, 0.3]) <= 1.0


def test_calibration_table_covers_every_row():
    probabilities = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    labels = [0, 0, 0, 1, 0, 0, 1, 1, 1, 1]
    table = calibration_table(labels, probabilities, n_bins=5)
    assert table["count"].sum() == len(labels)
    assert set(table.columns) >= {
        "count",
        "mean_predicted",
        "observed_positive_rate",
        "absolute_gap",
    }
    assert (table["absolute_gap"] >= 0).all()


def bootstrap_frame() -> pd.DataFrame:
    """Repeated snapshots for a few accounts so correlation is visible."""

    rows = []
    for account in range(6):
        for snapshot in range(3):
            positive = int((account + snapshot) % 4 == 0)
            rows.append(
                {
                    "account_id": f"SYN-{account:02d}",
                    "snapshot_id": f"SYN-{account:02d}_{snapshot}",
                    "future_revenue_decline": positive,
                    "baseline": 0.4 + 0.1 * snapshot - 0.05 * account,
                    "model": 0.6 + 0.05 * snapshot - 0.04 * account,
                }
            )
    return pd.DataFrame(rows)


def test_bootstrap_is_deterministic_and_reports_accounts():
    frame = bootstrap_frame()
    kwargs = dict(
        target_column="future_revenue_decline",
        model_scores={
            "baseline": frame["baseline"].to_numpy(),
            "model": frame["model"].to_numpy(),
        },
        primary_k=4,
        iterations=25,
        seed=11,
    )
    first = account_cluster_bootstrap(frame, **kwargs)
    second = account_cluster_bootstrap(frame, **kwargs)

    assert first["n_accounts"] == frame["account_id"].nunique()
    assert first["resampling_unit"].startswith("account_id")
    assert first["models"]["model"]["precision_at_k"] == second["models"]["model"]["precision_at_k"]

    precision = first["models"]["model"]["precision_at_k"]
    assert precision["point"] == pytest.approx(
        ranking_metrics(frame["future_revenue_decline"], frame["model"], 4)["precision"]
    )
    assert precision["lower"] <= precision["upper"]
    assert precision["valid_replicates"] > 0
    assert set(first["models"]["model"]) >= {
        "precision_at_k",
        "pr_auc",
        "precision_difference_vs_baseline_at_k",
    }


def test_bootstrap_requires_account_identifiers():
    frame = bootstrap_frame().drop(columns=["account_id"])
    with pytest.raises(ValueError, match="account_id"):
        account_cluster_bootstrap(
            frame,
            target_column="future_revenue_decline",
            model_scores={
                "baseline": frame["baseline"].to_numpy(),
                "model": frame["model"].to_numpy(),
            },
            primary_k=4,
            iterations=5,
            seed=1,
        )


def test_bootstrap_requires_baseline_scores():
    frame = bootstrap_frame()
    with pytest.raises(ValueError, match="Baseline model"):
        account_cluster_bootstrap(
            frame,
            target_column="future_revenue_decline",
            model_scores={"model": frame["model"].to_numpy()},
            primary_k=4,
            iterations=5,
            seed=1,
        )
