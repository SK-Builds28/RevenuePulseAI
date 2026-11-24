"""Tests for train-fitted preprocessing and the model feature contract."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from src.features.snapshots import FEATURE_COLUMNS, TARGET_COLUMNS
from src.models.preprocessing import (
    CATEGORICAL_FEATURES,
    HIST_GRADIENT_BOOSTING,
    LOGISTIC_REGRESSION,
    NUMERIC_FEATURES,
    build_model_pipeline,
    build_preprocessor,
    model_feature_frame,
    predict_risk_probability,
)


def feature_rows(n: int = 30) -> pd.DataFrame:
    """A labelled frame with deliberate missing values and mixed segments."""

    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {
            "account_id": [f"SYN-{index:04d}" for index in range(n)],
            "segment": ["SMB"] * (n - 6) + ["Mid-Market"] * 6,
            "future_revenue_decline": [1 if index % 5 == 0 else 0 for index in range(n)],
        }
    )
    for column in FEATURE_COLUMNS:
        if column == "segment":
            continue
        frame[column] = rng.normal(size=n)
    frame.loc[0, "revenue_trend_3m_pct"] = np.nan
    frame.loc[1, "days_since_activity_last"] = np.nan
    frame.loc[2, "segment"] = None
    return frame


def numeric_imputer(pipeline):
    """Accept either a full model pipeline or a bare preprocessor."""

    transformer = (
        pipeline.named_steps["preprocessor"]
        if hasattr(pipeline, "named_steps")
        else pipeline
    )
    return transformer.named_transformers_["numeric"].named_steps["imputer"]


def test_imputation_statistics_come_from_training_rows_only():
    train = feature_rows()
    validation = feature_rows()
    # Validation values are wildly different; if they leaked into the imputer the
    # learned median would change.
    validation["revenue_trend_3m_pct"] = validation["revenue_trend_3m_pct"] + 1000.0

    pipeline = build_model_pipeline(LOGISTIC_REGRESSION, _config())
    pipeline.fit(model_feature_frame(train), train["future_revenue_decline"])

    statistics = numeric_imputer(pipeline).statistics_
    position = list(NUMERIC_FEATURES).index("revenue_trend_3m_pct")
    features = model_feature_frame(train)
    expected_median = float(
        features["revenue_trend_3m_pct"].astype(float).median()
    )
    assert statistics[position] == pytest.approx(expected_median)

    transformer = build_preprocessor(scale_numeric=True)
    transformer.fit(model_feature_frame(train))
    transformed = transformer.transform(model_feature_frame(validation))
    assert np.isfinite(transformed).all()


def test_preprocessor_fit_changes_when_validation_rows_are_added():
    train = feature_rows()
    validation = feature_rows()
    validation["revenue_mean_3m"] = validation["revenue_mean_3m"] + 500.0

    train_only = build_preprocessor(scale_numeric=False)
    train_only.fit(model_feature_frame(train))
    combined = build_preprocessor(scale_numeric=False)
    combined.fit(model_feature_frame(pd.concat([train, validation], ignore_index=True)))

    first = numeric_imputer(train_only).statistics_
    second = numeric_imputer(combined).statistics_
    assert not np.allclose(first, second), "Sanity check: different rows imply different statistics"


def test_unknown_categorical_values_are_ignored_not_crashing():
    train = feature_rows()
    train["segment"] = "SMB"
    validation = feature_rows()
    validation["segment"] = "Enterprise"  # never seen during fitting

    pipeline = build_model_pipeline(LOGISTIC_REGRESSION, _config())
    pipeline.fit(model_feature_frame(train), train["future_revenue_decline"])
    probabilities = predict_risk_probability(pipeline, validation)

    assert probabilities.shape == (len(validation),)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_categorical_imputation_handles_missing_segment():
    train = feature_rows()
    pipeline = build_model_pipeline(HIST_GRADIENT_BOOSTING, _config())
    pipeline.fit(model_feature_frame(train), train["future_revenue_decline"])
    probabilities = predict_risk_probability(pipeline, feature_rows())
    assert np.isfinite(probabilities).all()


def test_model_matrix_excludes_identifiers_and_target():
    frame = feature_rows()
    features = model_feature_frame(frame)
    assert list(features.columns) == list(FEATURE_COLUMNS)
    assert "account_id" not in features.columns
    for column in TARGET_COLUMNS:
        assert column not in features.columns
    assert set(CATEGORICAL_FEATURES) == {"segment"}


def test_model_matrix_requires_allow_listed_columns():
    frame = feature_rows().drop(columns=["support_trend_3m_pct"])
    with pytest.raises(ValueError, match="Missing expected feature columns"):
        model_feature_frame(frame)


def test_account_id_does_not_change_predictions():
    frame = feature_rows()
    pipeline = build_model_pipeline(LOGISTIC_REGRESSION, _config())
    pipeline.fit(model_feature_frame(frame), frame["future_revenue_decline"])

    renamed = frame.copy()
    renamed["account_id"] = [f"OTHER-{index}" for index in range(len(frame))]
    assert np.allclose(
        predict_risk_probability(pipeline, frame),
        predict_risk_probability(pipeline, renamed),
    )


def test_missing_values_do_not_reach_the_classifier():
    frame = feature_rows()
    pipeline = build_model_pipeline(HIST_GRADIENT_BOOSTING, _config())
    pipeline.fit(model_feature_frame(frame), frame["future_revenue_decline"])
    transformed = pipeline.named_steps["preprocessor"].transform(
        model_feature_frame(frame)
    )
    assert np.isfinite(transformed).all()


def test_probabilities_are_bounded_for_both_models():
    frame = feature_rows()
    for model_name in (LOGISTIC_REGRESSION, HIST_GRADIENT_BOOSTING):
        pipeline = build_model_pipeline(model_name, _config())
        pipeline.fit(model_feature_frame(frame), frame["future_revenue_decline"])
        probabilities = predict_risk_probability(pipeline, frame)
        assert probabilities.min() >= 0.0
        assert probabilities.max() <= 1.0


@lru_cache(maxsize=1)
def _config():
    from pathlib import Path

    from src.utils.config import load_config

    return load_config(Path(__file__).resolve().parents[1] / "config.yaml")
