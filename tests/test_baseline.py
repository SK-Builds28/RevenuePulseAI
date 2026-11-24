"""Tests for the transparent scorecard baseline."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.features.snapshots import FEATURE_COLUMNS
from src.models.baseline import (
    SIGNAL_NAMES,
    BaselineScorecard,
    baseline_feature_columns,
    rank_baseline,
    score_baseline,
)
from src.utils.config import Config


def scorecard_frame() -> pd.DataFrame:
    """Small hand-built frame with clear, independent signals."""

    return pd.DataFrame(
        {
            "account_id": ["SYN-A", "SYN-B", "SYN-C", "SYN-D"],
            "snapshot_id": ["SYN-A_1", "SYN-B_1", "SYN-C_1", "SYN-D_1"],
            "segment": ["SMB", "SMB", "Mid-Market", "Enterprise"],
            "revenue_trend_3m_pct": [-0.20, 0.05, -0.01, float("nan")],
            "transaction_trend_3m_pct": [-0.30, 0.10, -0.02, -0.50],
            "engagement_trend_3m_pct": [-0.40, 0.00, 0.01, -0.60],
            "days_since_activity_last": [4.0, 5.0, 10.0, 90.0],
            "future_revenue_decline": [1, 0, 0, 1],
        }
    )


def test_baseline_score_is_deterministic(config):
    frame = scorecard_frame()
    first = score_baseline(frame, config)
    second = score_baseline(frame, config)
    pd.testing.assert_series_equal(first["baseline_score"], second["baseline_score"])
    pd.testing.assert_series_equal(
        first["baseline_probability"], second["baseline_probability"]
    )


def test_baseline_score_totals_configured_weights(config):
    frame = scorecard_frame()
    scored = score_baseline(frame, config)

    # SYN-A triggers revenue, transaction and engagement decline, not inactivity.
    weights = config.baseline_weights
    expected = (
        weights["revenue_decline"]
        + weights["transaction_decline"]
        + weights["engagement_decline"]
    )
    assert scored.loc[0, "baseline_score"] == pytest.approx(expected)

    # SYN-B triggers nothing.
    assert scored.loc[1, "baseline_score"] == pytest.approx(0.0)
    # SYN-D triggers transaction, engagement and inactivity; revenue is missing.
    expected_d = (
        weights["transaction_decline"]
        + weights["engagement_decline"]
        + weights["inactivity"]
    )
    assert scored.loc[3, "baseline_score"] == pytest.approx(expected_d)


def test_missing_signal_never_adds_points(config):
    frame = scorecard_frame()
    scorecard = BaselineScorecard.from_config(config)
    signals = scorecard.signal_frame(frame)
    # Missing revenue trend must not be treated as a decline.
    assert bool(signals.loc[3, "revenue_decline"]) is False


def test_baseline_probability_is_bounded(config):
    frame = scorecard_frame()
    scored = score_baseline(frame, config)
    assert scored["baseline_probability"].between(0.0, 1.0).all()
    scorecard = BaselineScorecard.from_config(config)
    assert scorecard.max_score == pytest.approx(sum(config.baseline_weights.values()))
    assert scorecard.max_score > 0


def test_baseline_high_risk_flag_follows_configured_score(config):
    frame = scorecard_frame()
    scored = score_baseline(frame, config)
    expected = (scored["baseline_score"] >= config.baseline_high_risk_score).astype(int)
    pd.testing.assert_series_equal(
        scored["baseline_high_risk"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_baseline_ranking_is_deterministic_and_ordered(config):
    frame = scorecard_frame()
    first = rank_baseline(frame, config)
    second = rank_baseline(frame, config)
    pd.testing.assert_frame_equal(first, second)
    assert first["baseline_score"].is_monotonic_decreasing
    assert first["baseline_rank"].tolist() == [1, 2, 3, 4]


def test_baseline_thresholds_are_configurable(config):
    frame = scorecard_frame()
    raw = copy.deepcopy(dict(config.raw))
    raw["baseline"]["thresholds"]["inactivity_days"] = 100.0
    stricter = Config(raw=raw, path=config.path)

    before = score_baseline(frame, config)["baseline_score"].tolist()
    after = score_baseline(frame, stricter)["baseline_score"].tolist()
    # SYN-D is inactive at 90 days; the stricter 100-day rule removes its points.
    assert before[3] > after[3]


def test_baseline_only_uses_allow_listed_features(config):
    columns = set(baseline_feature_columns())
    assert columns.issubset(set(FEATURE_COLUMNS))
    assert "account_id" not in columns
    assert "future_revenue_decline" not in columns

    scorecard = BaselineScorecard.from_config(config)
    metadata = scorecard.metadata()
    assert set(metadata["signals"]) == set(SIGNAL_NAMES)
    assert metadata["model_type"] == "transparent_scorecard"


def test_baseline_rejects_missing_feature_columns(config):
    frame = scorecard_frame().drop(columns=["engagement_trend_3m_pct"])
    with pytest.raises(ValueError, match="missing feature columns"):
        score_baseline(frame, config)


def test_baseline_scores_ignore_target_values(config):
    frame = scorecard_frame()
    flipped = frame.copy()
    flipped["future_revenue_decline"] = 1 - flipped["future_revenue_decline"]
    assert np.allclose(
        score_baseline(frame, config)["baseline_score"].to_numpy(),
        score_baseline(flipped, config)["baseline_score"].to_numpy(),
    )
