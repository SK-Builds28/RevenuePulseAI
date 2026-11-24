"""Tests for feature-grounded account reason codes."""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from src.models.reasons import (
    REASON_RULES,
    RULE_NAMES,
    reason_code_frame,
    reason_limit,
    reason_thresholds,
    signal_prevalence,
)
from src.utils.config import Config


def reason_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "account_id": ["SYN-A", "SYN-B", "SYN-C"],
            "revenue_trend_3m_pct": [-0.20, 0.02, float("nan")],
            "revenue_trend_6m_pct": [0.20, -0.10, float("nan")],
            "transaction_trend_3m_pct": [-0.10, 0.05, 0.01],
            "engagement_trend_3m_pct": [-0.20, 0.01, 0.02],
            "engagement_mean_3m": [25.0, 70.0, 65.0],
            "days_since_activity_last": [45.0, 2.0, 7.0],
        }
    )


def test_reason_codes_are_derived_from_features(config):
    frame = reason_frame()
    codes = reason_code_frame(frame, config, limit=len(REASON_RULES))

    first = codes.loc[0, "reason_codes"]
    assert "recent revenue decline" in first
    assert "revenue well above its prior six-month average" in first
    assert "reduced transaction frequency" in first
    assert "increased inactivity" in first
    assert "weakening engagement" in first
    assert "low engagement level" in first

    # SYN-B has no decline-style signal at all.
    second = reason_code_frame(frame, config, limit=len(REASON_RULES)).loc[1, "reason_codes"]
    assert "recent revenue decline" not in second
    assert "increased inactivity" not in second
    assert "revenue decline over the last six months" in second


def test_missing_features_do_not_fire_codes(config):
    codes = reason_code_frame(reason_frame(), config)
    row = codes.loc[2]
    assert not bool(row["signal_revenue_decline"])
    assert not bool(row["signal_revenue_decline_6m"])
    assert not bool(row["signal_revenue_growth_6m"])
    assert row["reason_count"] == 0


def test_reason_codes_are_deterministic(config):
    frame = reason_frame()
    pd.testing.assert_frame_equal(
        reason_code_frame(frame, config), reason_code_frame(frame, config)
    )


def test_reason_limit_truncates_but_keeps_flags(config):
    codes = reason_code_frame(reason_frame(), config, limit=2)
    assert int(codes.loc[0, "reason_count"]) == 2
    assert codes.loc[0, "reason_codes"].count(";") == 1
    # Underlying flags stay available for auditing.
    assert bool(codes.loc[0, "signal_inactivity"])


def test_reason_thresholds_are_configurable(config):
    raw = copy.deepcopy(dict(config.raw))
    raw["reasons"]["revenue_decline_3m_pct"] = -0.50
    stricter = Config(raw=raw, path=config.path)

    before = reason_code_frame(reason_frame(), config)
    after = reason_code_frame(reason_frame(), stricter)
    assert bool(before.loc[0, "signal_revenue_decline"])
    assert not bool(after.loc[0, "signal_revenue_decline"])


def test_reason_rules_reference_allow_listed_features(config):
    from src.features.snapshots import FEATURE_COLUMNS

    for rule in REASON_RULES:
        assert rule.feature in FEATURE_COLUMNS
        assert rule.direction in {"below", "above"}
    assert len(set(RULE_NAMES)) == len(REASON_RULES)
    assert set(reason_thresholds(config)) == {rule.threshold_key for rule in REASON_RULES}
    assert reason_limit(config) >= 1


def test_no_reason_claims_causality(config):
    codes = reason_code_frame(reason_frame(), config)
    assert "associated with" in codes["disclaimer"].iloc[0]
    assert "caused" in codes["disclaimer"].iloc[0]


def test_reason_input_requires_features(config):
    broken = reason_frame().drop(columns=["engagement_mean_3m"])
    with pytest.raises(ValueError, match="missing features"):
        reason_code_frame(broken, config)


def test_signal_prevalence_counts_flags(config):
    codes = reason_code_frame(reason_frame(), config)
    table = signal_prevalence(codes)
    assert len(table) == len(REASON_RULES)
    assert (table["rows"] <= len(codes)).all()
    assert table["share_of_rows"].between(0, 1).all()
