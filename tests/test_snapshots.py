"""Snapshot, label and leakage tests.

The strongest leakage test changes future revenue and asserts that features at
an earlier snapshot are byte-for-byte unchanged.
"""

from __future__ import annotations

import copy
from unittest.mock import patch

import pandas as pd
import pytest

from src.data.quality import DataValidationError
from src.features.snapshots import (
    FEATURE_COLUMNS,
    FORBIDDEN_FEATURE_COLUMNS,
    TARGET_COLUMNS,
    assert_time_ordering,
    build_snapshots,
    get_feature_columns,
    get_target_columns,
    label_prevalence,
)
from src.utils.config import Config


def make_account_frame(account_id: str, revenues: list[float], start: str = "2023-01") -> pd.DataFrame:
    months = pd.date_range(start, periods=len(revenues), freq="MS")
    return pd.DataFrame(
        {
            "account_id": [account_id] * len(revenues),
            "month": months,
            "segment": ["SMB"] * len(revenues),
            "monthly_revenue": revenues,
            "transaction_count": [10] * len(revenues),
            "engagement_score": [70.0] * len(revenues),
            "days_since_last_activity": [4.0] * len(revenues),
            "product_count": [2] * len(revenues),
            "support_interactions": [1] * len(revenues),
            "contract_age_months": list(range(1, len(revenues) + 1)),
            "is_synthetic": [True] * len(revenues),
        }
    )


def test_label_is_positive_for_material_decline(config):
    declining = make_account_frame("SYN-0001", [10000.0] * 9 + [5000.0] * 3)
    stable = make_account_frame("SYN-0002", [10000.0] * 12)
    frame = pd.concat([declining, stable], ignore_index=True)

    snapshots = build_snapshots(frame, config)
    at_september = snapshots[snapshots["snapshot_date"] == pd.Timestamp("2023-09-30")]

    declining_row = at_september[at_september["account_id"] == "SYN-0001"].iloc[0]
    stable_row = at_september[at_september["account_id"] == "SYN-0002"].iloc[0]

    assert declining_row["future_revenue_decline"] == 1
    assert declining_row["future_revenue_change_pct"] == pytest.approx(-50.0)
    assert stable_row["future_revenue_decline"] == 0
    assert stable_row["future_revenue_change_pct"] == pytest.approx(0.0)


def test_label_threshold_is_configurable(config):
    frame = pd.concat(
        [
            make_account_frame("SYN-0001", [10000.0] * 9 + [5000.0] * 3),
            make_account_frame("SYN-0002", [10000.0] * 12),
        ],
        ignore_index=True,
    )
    raw = copy.deepcopy(dict(config.raw))
    raw["label"]["decline_threshold_pct"] = 60.0
    stricter = Config(raw=raw, path=config.path)

    snapshots = build_snapshots(frame, stricter)
    row = snapshots[
        (snapshots["account_id"] == "SYN-0001")
        & (snapshots["snapshot_date"] == pd.Timestamp("2023-09-30"))
    ].iloc[0]
    assert row["future_revenue_decline"] == 0


def test_future_revenue_does_not_change_earlier_features(config, small_monthly):
    baseline = build_snapshots(small_monthly, config)
    account_id = baseline["account_id"].iloc[0]
    snapshot_date = pd.Timestamp("2024-06-30")

    modified = small_monthly.copy()
    future_mask = (modified["account_id"] == account_id) & (
        modified["month"] > snapshot_date
    )
    modified.loc[future_mask, "monthly_revenue"] = (
        modified.loc[future_mask, "monthly_revenue"] * 0.05
    )

    changed = build_snapshots(modified, config)

    def row(frame):
        return frame[
            (frame["account_id"] == account_id)
            & (frame["snapshot_date"] == snapshot_date)
        ][list(FEATURE_COLUMNS)].reset_index(drop=True)

    pd.testing.assert_frame_equal(row(baseline), row(changed))


def test_past_revenue_changes_earlier_features(config, small_monthly):
    baseline = build_snapshots(small_monthly, config)
    account_id = baseline["account_id"].iloc[0]
    snapshot_date = pd.Timestamp("2024-06-30")

    modified = small_monthly.copy()
    past_mask = (modified["account_id"] == account_id) & (
        modified["month"] == pd.Timestamp("2024-05-01")
    )
    modified.loc[past_mask, "monthly_revenue"] = (
        modified.loc[past_mask, "monthly_revenue"] * 3.0
    )

    changed = build_snapshots(modified, config)

    def row(frame):
        return frame[
            (frame["account_id"] == account_id)
            & (frame["snapshot_date"] == snapshot_date)
        ][list(FEATURE_COLUMNS)].reset_index(drop=True)

    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(row(baseline), row(changed))


def test_feature_and_target_lists_are_disjoint():
    assert not set(FEATURE_COLUMNS).intersection(TARGET_COLUMNS)
    assert "account_id" not in FEATURE_COLUMNS
    assert "snapshot_date" not in FEATURE_COLUMNS
    assert not set(FEATURE_COLUMNS).intersection(FORBIDDEN_FEATURE_COLUMNS)


def test_get_feature_columns_rejects_missing_or_forbidden(config, small_monthly):
    snapshots = build_snapshots(small_monthly, config)
    columns = get_feature_columns(snapshots)
    assert columns == list(FEATURE_COLUMNS)
    assert get_target_columns() == list(TARGET_COLUMNS)

    broken = snapshots.drop(columns=["revenue_mean_3m"])
    with pytest.raises(ValueError, match="Missing expected feature columns"):
        get_feature_columns(broken)


def test_snapshot_dates_precede_outcomes(config, small_monthly):
    snapshots = build_snapshots(small_monthly, config)
    assert not snapshots.empty
    assert (snapshots["outcome_start_month"] > snapshots["snapshot_date"]).all()
    assert (snapshots["outcome_end_month"] >= snapshots["outcome_start_month"]).all()
    assert_time_ordering(snapshots)


def test_split_dates_preserve_time_order(config, small_monthly):
    snapshots = build_snapshots(small_monthly, config)
    train = snapshots[snapshots["split"] == "train"]
    validation = snapshots[snapshots["split"] == "validation"]
    test = snapshots[snapshots["split"] == "test"]

    assert not train.empty
    assert not validation.empty
    assert not test.empty
    assert train["snapshot_date"].max() <= config.train_end
    assert validation["snapshot_date"].min() > config.train_end
    assert validation["snapshot_date"].max() <= config.validation_end
    assert test["snapshot_date"].min() > config.validation_end
    assert test["snapshot_date"].max() <= config.test_end


def test_snapshot_output_has_no_duplicate_account_snapshot(config, small_monthly):
    snapshots = build_snapshots(small_monthly, config)
    assert not snapshots.duplicated(subset=["account_id", "snapshot_date"]).any()


def test_duplicate_snapshot_rows_are_detected(config):
    frame = make_account_frame("SYN-0001", [10000.0] * 12)
    duplicate_date = pd.Timestamp("2023-09-30")
    with patch(
        "src.features.snapshots._quarter_end_dates",
        return_value=[duplicate_date, duplicate_date],
    ):
        with pytest.raises(ValueError, match="duplicate account/snapshot"):
            build_snapshots(frame, config, validate=False)


def test_duplicate_input_rows_are_rejected(config):
    frame = pd.concat(
        [
            make_account_frame("SYN-0001", [10000.0] * 12),
            make_account_frame("SYN-0001", [10000.0] * 12).iloc[[0]],
        ],
        ignore_index=True,
    )
    with pytest.raises(DataValidationError, match="duplicate_account_month"):
        build_snapshots(frame, config)

    with pytest.raises(ValueError, match="duplicate account-month"):
        build_snapshots(frame, config, validate=False)


def test_insufficient_history_produces_no_snapshots(config):
    frame = make_account_frame("SYN-0001", [10000.0] * 3)
    snapshots = build_snapshots(frame, config)
    assert snapshots.empty


def test_label_has_usable_prevalence_and_all_splits(config, full_monthly):
    snapshots = build_snapshots(full_monthly, config)
    labelled = snapshots.dropna(subset=["future_revenue_decline"])
    assert len(labelled) > 1000
    prevalence = label_prevalence(labelled, config.target_column)
    assert 0.08 <= prevalence <= 0.45, f"Unexpected label prevalence: {prevalence}"
    assert set(labelled["split"].unique()) == {"train", "validation", "test"}


def test_all_feature_columns_are_populated(config, small_monthly):
    snapshots = build_snapshots(small_monthly, config)
    # Some trailing features are legitimately NaN for short histories; the core
    # current-window features must still be present.
    core = [
        "revenue_mean_3m",
        "revenue_mean_6m",
        "transaction_mean_3m",
        "engagement_mean_3m",
        "contract_age_months",
    ]
    assert not snapshots[core].isna().all().any()
