"""Tests for blocking and non-blocking data quality checks."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.quality import (
    DataValidationError,
    load_monthly_data,
    normalise_monthly_data,
    validate_monthly_data,
)
from src.data.schema import RAW_COLUMNS


def make_frame(accounts: int = 3, months: int = 12, revenue: float = 1000.0) -> pd.DataFrame:
    rows = []
    month_starts = pd.date_range("2023-01-01", periods=months, freq="MS")
    for index in range(accounts):
        for month_index, month in enumerate(month_starts):
            rows.append(
                {
                    "account_id": f"SYN-{index + 1:04d}",
                    "month": month,
                    "segment": "SMB",
                    "monthly_revenue": revenue,
                    "transaction_count": 5,
                    "engagement_score": 70.0,
                    "days_since_last_activity": 4.0,
                    "product_count": 2,
                    "support_interactions": 1,
                    "contract_age_months": month_index + 1,
                    "is_synthetic": True,
                }
            )
    return pd.DataFrame(rows, columns=list(RAW_COLUMNS))


def codes(report) -> set[str]:
    return {issue.code for issue in report.issues}


def test_valid_frame_passes(config):
    report = validate_monthly_data(make_frame(), config)
    assert report.passed
    assert not report.errors


def test_missing_required_column_is_an_error(config):
    frame = make_frame().drop(columns=["monthly_revenue"])
    report = validate_monthly_data(frame, config)
    assert not report.passed
    assert "missing_required_columns" in codes(report)


def test_duplicate_account_month_is_an_error(config):
    frame = make_frame()
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    report = validate_monthly_data(duplicated, config)
    assert not report.passed
    assert "duplicate_account_month" in codes(report)


def test_negative_revenue_is_an_error(config):
    frame = make_frame()
    frame.loc[0, "monthly_revenue"] = -10
    report = validate_monthly_data(frame, config)
    assert not report.passed
    assert "negative_monthly_revenue" in codes(report)


def test_invalid_date_is_an_error(config):
    frame = make_frame()
    frame["month"] = frame["month"].astype(object)
    frame.loc[0, "month"] = "not-a-date"
    report = validate_monthly_data(frame, config)
    assert not report.passed
    assert "invalid_month" in codes(report)


def test_unexpected_segment_is_an_error(config):
    frame = make_frame()
    frame.loc[0, "segment"] = "Unknown Segment"
    report = validate_monthly_data(frame, config)
    assert not report.passed
    assert "unexpected_segment" in codes(report)


def test_insufficient_history_is_a_warning(config):
    frame = make_frame(accounts=2, months=3)
    report = validate_monthly_data(frame, config)
    assert report.passed
    assert "insufficient_history" in codes(report)
    warning = next(issue for issue in report.warnings if issue.code == "insufficient_history")
    assert warning.count == 2


def test_missing_engagement_is_a_warning(config):
    frame = make_frame()
    frame.loc[0, "engagement_score"] = None
    report = validate_monthly_data(frame, config)
    assert report.passed
    assert "missing_optional_engagement_score" in codes(report)


def test_forbidden_personal_column_is_an_error(config):
    frame = make_frame()
    frame["email"] = "not-allowed@example.com"
    report = validate_monthly_data(frame, config)
    assert not report.passed
    assert "forbidden_personal_columns" in codes(report)


def test_load_monthly_data_raises_with_readable_report(config, tmp_path):
    frame = make_frame()
    frame.loc[0, "monthly_revenue"] = -5
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="negative_monthly_revenue"):
        load_monthly_data(path, config)


def test_normalisation_converts_to_month_start(config):
    frame = make_frame()
    frame["month"] = frame["month"] + pd.Timedelta(days=14)
    normalised = normalise_monthly_data(frame)
    assert (normalised["month"].dt.day == 1).all()
    ordered_within_account = normalised.groupby("account_id")["month"].apply(
        lambda values: values.is_monotonic_increasing
    )
    assert ordered_within_account.all()


def test_missing_file_raises(config, tmp_path):
    with pytest.raises(FileNotFoundError):
        load_monthly_data(tmp_path / "missing.csv", config)
