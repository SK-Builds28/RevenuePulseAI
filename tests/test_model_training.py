"""Tests for model training, validation-only selection and the saved artifact."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import ranking_metrics
from src.features.snapshots import FEATURE_COLUMNS, build_snapshots
from src.models.preprocessing import (
    HIST_GRADIENT_BOOSTING,
    LOGISTIC_REGRESSION,
    model_feature_frame,
    predict_risk_probability,
)
from src.models import train as train_module
from src.models.train import (
    ARTIFACT_FILENAME,
    BASELINE_NAME,
    METADATA_FILENAME,
    fit_models,
    load_snapshots,
    run_training,
    score_all_models,
    select_primary_model,
    split_frames,
)


@pytest.fixture(scope="module")
def snapshots(config, small_monthly) -> pd.DataFrame:
    return build_snapshots(small_monthly, config)


@pytest.fixture(scope="module")
def snapshot_file(tmp_path_factory, snapshots) -> Path:
    path = tmp_path_factory.mktemp("snapshots") / "snapshots.csv"
    snapshots.to_csv(path, index=False, date_format="%Y-%m-%d")
    return path


# -- model selection -------------------------------------------------------
def test_selection_prefers_clearly_better_model(config):
    validation = {
        BASELINE_NAME: {"precision_at_k": 0.20, "pr_auc": 0.20},
        LOGISTIC_REGRESSION: {"precision_at_k": 0.30, "pr_auc": 0.25},
        HIST_GRADIENT_BOOSTING: {"precision_at_k": 0.40, "pr_auc": 0.26},
    }
    selection = select_primary_model(validation, config)
    assert selection["selected_model"] == HIST_GRADIENT_BOOSTING
    assert selection["margin_applied"] is False
    assert selection["beats_baseline"] is True


def test_selection_prefers_logistic_regression_inside_margin(config):
    validation = {
        BASELINE_NAME: {"precision_at_k": 0.20, "pr_auc": 0.20},
        LOGISTIC_REGRESSION: {"precision_at_k": 0.32, "pr_auc": 0.25},
        HIST_GRADIENT_BOOSTING: {"precision_at_k": 0.33, "pr_auc": 0.26},
    }
    selection = select_primary_model(validation, config)
    assert selection["best_ml_model"] == HIST_GRADIENT_BOOSTING
    assert selection["selected_model"] == LOGISTIC_REGRESSION
    assert selection["margin_applied"] is True
    assert "interpretability" in selection["reason"]


def test_selection_prefers_logistic_regression_on_exact_tie(config):
    validation = {
        BASELINE_NAME: {"precision_at_k": 0.20, "pr_auc": 0.20},
        LOGISTIC_REGRESSION: {"precision_at_k": 0.30, "pr_auc": 0.25},
        HIST_GRADIENT_BOOSTING: {"precision_at_k": 0.30, "pr_auc": 0.25},
    }
    selection = select_primary_model(validation, config)
    assert selection["selected_model"] == LOGISTIC_REGRESSION


def test_selection_records_when_baseline_wins(config):
    validation = {
        BASELINE_NAME: {"precision_at_k": 0.50, "pr_auc": 0.40},
        LOGISTIC_REGRESSION: {"precision_at_k": 0.30, "pr_auc": 0.25},
        HIST_GRADIENT_BOOSTING: {"precision_at_k": 0.40, "pr_auc": 0.26},
    }
    selection = select_primary_model(validation, config)
    assert selection["selected_model"] == HIST_GRADIENT_BOOSTING
    assert selection["beats_baseline"] is False


# -- fitting, scoring and reproducibility ----------------------------------
def test_scores_cover_baseline_and_both_models(config, snapshots):
    splits = split_frames(load_snapshots_from_frame(snapshots))
    pipelines = fit_models(splits["train"], config)
    scores = score_all_models(splits["validation"], pipelines, config)
    assert set(scores) == {BASELINE_NAME, LOGISTIC_REGRESSION, HIST_GRADIENT_BOOSTING}
    for values in scores.values():
        assert len(values) == len(splits["validation"])
        assert values.min() >= 0.0 and values.max() <= 1.0


def test_training_is_reproducible(config, snapshots):
    splits = split_frames(load_snapshots_from_frame(snapshots))
    first = fit_models(splits["train"], config)
    second = fit_models(splits["train"], config)
    for name in first:
        assert np.allclose(
            predict_risk_probability(first[name], splits["validation"]),
            predict_risk_probability(second[name], splits["validation"]),
        )


def test_repeated_prediction_is_stable(config, snapshots):
    splits = split_frames(load_snapshots_from_frame(snapshots))
    pipeline = fit_models(splits["train"], config)[LOGISTIC_REGRESSION]
    first = predict_risk_probability(pipeline, splits["test"])
    second = predict_risk_probability(pipeline, splits["test"])
    assert np.array_equal(first, second)


def test_missing_values_at_prediction_time_are_tolerated(config, snapshots):
    splits = split_frames(load_snapshots_from_frame(snapshots))
    pipelines = fit_models(splits["train"], config)
    frame = splits["test"].copy()
    frame.loc[frame.index[:5], "revenue_trend_6m_pct"] = np.nan
    frame.loc[frame.index[:5], "support_trend_3m_pct"] = np.nan
    for pipeline in pipelines.values():
        probabilities = predict_risk_probability(pipeline, frame)
        assert np.isfinite(probabilities).all()


def test_model_matrix_rows_are_not_dropped_for_short_history(config, snapshots):
    splits = split_frames(load_snapshots_from_frame(snapshots))
    features = model_feature_frame(splits["train"])
    assert len(features) == len(splits["train"])
    # The allow-list intentionally contains nullable short-history features.
    assert features["revenue_trend_6m_pct"].isna().any()


# -- artifact and metadata -------------------------------------------------
def test_training_writes_reloadable_artifact(config, snapshot_file, tmp_path):
    output_dir = tmp_path / "model"
    summary = run_training(config, snapshot_file, output_dir)

    artifact_path = output_dir / ARTIFACT_FILENAME
    metadata_path = output_dir / METADATA_FILENAME
    assert artifact_path.exists()
    assert metadata_path.exists()

    bundle = joblib.load(artifact_path)
    assert bundle["selected_model"] == summary["selection"]["selected_model"]
    assert bundle["feature_columns"] == list(FEATURE_COLUMNS)
    assert "account_id" not in bundle["feature_columns"]
    assert set(bundle["pipelines"]) == {LOGISTIC_REGRESSION, HIST_GRADIENT_BOOSTING}

    reloaded = joblib.load(artifact_path)["pipelines"][bundle["selected_model"]]
    snapshots = load_snapshots(snapshot_file, config)
    test_frame = snapshots[snapshots["split"] == "test"].reset_index(drop=True)
    assert np.array_equal(
        predict_risk_probability(reloaded, test_frame),
        predict_risk_probability(bundle["pipelines"][bundle["selected_model"]], test_frame),
    )



def test_metadata_records_required_fields(config, snapshot_file, tmp_path):
    import json

    output_dir = tmp_path / "model"
    run_training(config, snapshot_file, output_dir)
    metadata = json.loads((output_dir / METADATA_FILENAME).read_text(encoding="utf-8"))

    for key in (
        "model_name",
        "model_type",
        "trained_at_utc",
        "feature_columns",
        "label_definition",
        "training_period",
        "validation_period",
        "test_period",
        "random_seed",
        "model_parameters",
        "selected_metrics",
        "synthetic_data",
        "version",
    ):
        assert key in metadata, f"Missing metadata key: {key}"

    assert metadata["synthetic_data"] is True
    assert metadata["feature_columns"] == list(FEATURE_COLUMNS)
    assert "account_id" not in metadata["feature_columns"]
    assert metadata["training_period"]["end"] <= str(config.train_end.date())
    assert metadata["validation_period"]["start"] > metadata["training_period"]["end"]
    assert metadata["test_period"]["start"] > metadata["validation_period"]["end"]
    assert "test" not in metadata["selected_metrics"]


def test_training_never_scores_the_test_split(config, snapshot_file, tmp_path, monkeypatch):
    seen: list[set[str]] = []
    original = train_module.evaluate_frame

    def spy(frame, scores, cfg):
        seen.append(set(frame["split"].unique()))
        return original(frame, scores, cfg)

    monkeypatch.setattr(train_module, "evaluate_frame", spy)
    run_training(config, snapshot_file, tmp_path / "model")

    assert seen, "Training did not evaluate any split"
    assert all(splits == {"train"} or splits == {"validation"} for splits in seen)


def test_saved_artifact_ranks_scores_consistently(config, snapshot_file, tmp_path):
    output_dir = tmp_path / "model"
    run_training(config, snapshot_file, output_dir)
    bundle = joblib.load(output_dir / ARTIFACT_FILENAME)

    snapshots = load_snapshots(snapshot_file, config)
    test_frame = snapshots[snapshots["split"] == "test"].reset_index(drop=True)
    probabilities = predict_risk_probability(
        bundle["pipelines"][bundle["selected_model"]], test_frame
    )
    top = np.argsort(-probabilities, kind="stable")[: config.primary_k]
    metrics = ranking_metrics(
        test_frame[config.target_column].to_numpy(), probabilities, config.primary_k
    )
    assert metrics["selected"] == min(config.primary_k, len(test_frame))
    assert len(top) == metrics["k_effective"]


def load_snapshots_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise a freshly built snapshot frame the same way the CLI does."""

    normalised = frame.dropna(subset=["future_revenue_decline"]).copy()
    normalised["future_revenue_decline"] = normalised["future_revenue_decline"].astype(int)
    return normalised.sort_values(
        ["snapshot_date", "account_id"], kind="mergesort"
    ).reset_index(drop=True)
