"""Point-in-time snapshots, features and the revenue-decline label.

A snapshot at 2024-09-30 may use data through September 2024 only. Its label is
computed from October-December 2024. The feature and target column lists are
explicit so account identifiers or future revenue cannot silently become model
inputs.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.data.quality import validate_monthly_data
from src.data.schema import RAW_REQUIRED_COLUMNS
from src.utils.config import Config

SNAPSHOT_METADATA_COLUMNS: tuple[str, ...] = (
    "snapshot_id",
    "account_id",
    "snapshot_date",
    "split",
    "outcome_start_month",
    "outcome_end_month",
    "label_eligible",
)

TARGET_COLUMNS: tuple[str, ...] = (
    "reference_revenue_mean",
    "future_revenue_mean",
    "future_revenue_change_pct",
    "future_revenue_decline",
)

# Explicit allow-list. Anything not listed here is not a model feature.
FEATURE_COLUMNS: tuple[str, ...] = (
    "segment",
    "snapshot_month",
    "history_months",
    "revenue_mean_3m",
    "revenue_mean_6m",
    "revenue_mean_12m",
    "revenue_trend_3m_pct",
    "revenue_trend_6m_pct",
    "revenue_volatility_6m",
    "transaction_mean_3m",
    "transaction_trend_3m_pct",
    "revenue_per_transaction_3m",
    "engagement_mean_3m",
    "engagement_trend_3m_pct",
    "engagement_missing_ratio_3m",
    "days_since_activity_last",
    "days_since_activity_mean_3m",
    "product_count_last",
    "product_count_change_6m",
    "support_interactions_sum_3m",
    "support_trend_3m_pct",
    "contract_age_months",
)

FORBIDDEN_FEATURE_COLUMNS: tuple[str, ...] = (
    "account_id",
    "snapshot_id",
    "snapshot_date",
    "is_synthetic",
    "split",
    "future_revenue_mean",
    "future_revenue_change_pct",
    "future_revenue_decline",
    "reference_revenue_mean",
    "outcome_start_month",
    "outcome_end_month",
    "label_eligible",
)


def _pct_change(current: float, previous: float) -> float:
    if pd.isna(current) or pd.isna(previous) or previous <= 0:
        return float("nan")
    return float((current - previous) / previous)


def _window_mean(history: pd.DataFrame, column: str, window: int) -> float:
    return float(history[column].tail(window).mean())


def _window_sum(history: pd.DataFrame, column: str, window: int) -> float:
    return float(history[column].tail(window).sum())


def _last_non_null(history: pd.DataFrame, column: str) -> float:
    values = history[column].dropna()
    if values.empty:
        return float("nan")
    return float(values.iloc[-1])


def _quarter_end_dates(frame: pd.DataFrame) -> list[pd.Timestamp]:
    first = frame["month"].min()
    last = frame["month"].max()
    months = pd.date_range(first, last, freq="MS")
    return [
        month + pd.offsets.MonthEnd(0)
        for month in months
        if month.month in (3, 6, 9, 12)
    ]


def _split_for_date(snapshot_date: pd.Timestamp, config: Config) -> str | None:
    if snapshot_date <= config.train_end:
        return "train"
    if snapshot_date <= config.validation_end:
        return "validation"
    if snapshot_date <= config.test_end:
        return "test"
    return None


def _feature_row(
    history: pd.DataFrame,
    snapshot_date: pd.Timestamp,
) -> dict[str, object]:
    last = history.iloc[-1]

    revenue_3m = _window_mean(history, "monthly_revenue", 3)
    revenue_6m = _window_mean(history, "monthly_revenue", 6)
    revenue_12m = _window_mean(history, "monthly_revenue", 12)

    revenue_prior_3m = float(history["monthly_revenue"].iloc[-6:-3].mean()) if len(history) >= 6 else float("nan")
    revenue_prior_6m = float(history["monthly_revenue"].iloc[-12:-6].mean()) if len(history) >= 12 else float("nan")

    transaction_3m = _window_mean(history, "transaction_count", 3)
    transaction_prior_3m = float(history["transaction_count"].iloc[-6:-3].mean()) if len(history) >= 6 else float("nan")
    transaction_sum_3m = _window_sum(history, "transaction_count", 3)

    engagement_3m = _window_mean(history, "engagement_score", 3)
    engagement_prior_3m = float(history["engagement_score"].iloc[-6:-3].mean()) if len(history) >= 6 else float("nan")
    engagement_missing_ratio = float(history["engagement_score"].tail(3).isna().mean())

    product_reference = (
        float(history["product_count"].iloc[-7]) if len(history) >= 7 else float("nan")
    )

    support_3m = _window_sum(history, "support_interactions", 3)
    support_prior_3m = float(history["support_interactions"].iloc[-6:-3].sum()) if len(history) >= 6 else float("nan")

    revenue_6m_std = float(history["monthly_revenue"].tail(6).std(ddof=0))
    revenue_volatility = revenue_6m_std / revenue_6m if revenue_6m > 0 else float("nan")

    revenue_per_transaction = (
        float(history["monthly_revenue"].tail(3).sum()) / transaction_sum_3m
        if transaction_sum_3m > 0
        else float("nan")
    )

    return {
        "segment": str(last["segment"]),
        "snapshot_month": int(snapshot_date.month),
        "history_months": int(len(history)),
        "revenue_mean_3m": revenue_3m,
        "revenue_mean_6m": revenue_6m,
        "revenue_mean_12m": revenue_12m,
        "revenue_trend_3m_pct": _pct_change(revenue_3m, revenue_prior_3m),
        "revenue_trend_6m_pct": _pct_change(revenue_6m, revenue_prior_6m),
        "revenue_volatility_6m": revenue_volatility,
        "transaction_mean_3m": transaction_3m,
        "transaction_trend_3m_pct": _pct_change(transaction_3m, transaction_prior_3m),
        "revenue_per_transaction_3m": revenue_per_transaction,
        "engagement_mean_3m": engagement_3m,
        "engagement_trend_3m_pct": _pct_change(engagement_3m, engagement_prior_3m),
        "engagement_missing_ratio_3m": engagement_missing_ratio,
        "days_since_activity_last": _last_non_null(history, "days_since_last_activity"),
        "days_since_activity_mean_3m": _window_mean(
            history, "days_since_last_activity", 3
        ),
        "product_count_last": float(last["product_count"]),
        "product_count_change_6m": (
            float(last["product_count"]) - product_reference
            if pd.notna(product_reference)
            else float("nan")
        ),
        "support_interactions_sum_3m": support_3m,
        "support_trend_3m_pct": _pct_change(support_3m, support_prior_3m),
        "contract_age_months": float(last["contract_age_months"]),
    }


def build_snapshots(
    monthly_data: pd.DataFrame,
    config: Config,
    validate: bool = True,
) -> pd.DataFrame:
    """Build one row per account per quarter-end with leakage-safe features.

    The function raises ``DataValidationError`` when the raw data has blocking
    quality issues. It returns rows only for snapshots that belong to the
    configured train/validation/test windows and have enough history.
    """

    if validate:
        report = validate_monthly_data(monthly_data, config)
        report.raise_if_errors(source="monthly_data")

    frame = monthly_data.copy()
    frame["month"] = (
        pd.to_datetime(frame["month"], errors="coerce", format="mixed")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    frame["account_id"] = frame["account_id"].astype("string").str.strip()
    frame["segment"] = frame["segment"].astype("string").str.strip()

    missing = [column for column in RAW_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Cannot build snapshots; missing columns: {missing}")

    if frame["month"].isna().any():
        raise ValueError("Cannot build snapshots with invalid month values")

    if frame.duplicated(subset=["account_id", "month"]).any():
        raise ValueError(
            "Cannot build snapshots from duplicate account-month rows; resolve duplicates first"
        )

    snapshot_dates = _quarter_end_dates(frame)
    records: list[dict[str, object]] = []

    for account_id, account_frame in frame.groupby("account_id", sort=True):
        account_frame = account_frame.sort_values("month", kind="mergesort")
        for snapshot_date in snapshot_dates:
            split = _split_for_date(snapshot_date, config)
            if split is None:
                continue

            history = account_frame[account_frame["month"] <= snapshot_date]
            if len(history) < config.minimum_history_months:
                continue

            future = account_frame[account_frame["month"] > snapshot_date].sort_values(
                "month", kind="mergesort"
            )
            if len(future) < config.horizon_months and config.drop_incomplete_horizon:
                continue

            outcome_window = future.head(config.horizon_months)
            reference_history = history.tail(config.reference_window_months)

            reference_mean = float(reference_history["monthly_revenue"].mean())
            future_mean = (
                float(outcome_window["monthly_revenue"].mean())
                if not outcome_window.empty
                else float("nan")
            )

            eligible = (
                len(outcome_window) == config.horizon_months
                and pd.notna(reference_mean)
                and reference_mean >= config.minimum_reference_revenue
            )
            if not eligible and config.drop_rows_below_minimum_revenue:
                continue

            if pd.notna(reference_mean) and reference_mean > 0 and pd.notna(future_mean):
                change_pct = float(
                    (future_mean - reference_mean) / reference_mean * 100.0
                )
            else:
                change_pct = float("nan")

            if eligible and pd.notna(change_pct):
                decline = int(
                    future_mean
                    <= reference_mean * (1.0 - config.decline_threshold_pct / 100.0)
                )
            else:
                decline = pd.NA

            features = _feature_row(history, snapshot_date)
            records.append(
                {
                    "snapshot_id": f"{account_id}_{snapshot_date.date().isoformat()}",
                    "account_id": str(account_id),
                    "snapshot_date": snapshot_date,
                    "split": split,
                    **features,
                    "reference_revenue_mean": reference_mean,
                    "future_revenue_mean": future_mean,
                    "future_revenue_change_pct": change_pct,
                    "future_revenue_decline": decline,
                    "outcome_start_month": (
                        outcome_window["month"].min() if not outcome_window.empty else pd.NaT
                    ),
                    "outcome_end_month": (
                        outcome_window["month"].max() if not outcome_window.empty else pd.NaT
                    ),
                    "label_eligible": bool(eligible),
                }
            )

    columns = (
        list(SNAPSHOT_METADATA_COLUMNS)
        + list(FEATURE_COLUMNS)
        + list(TARGET_COLUMNS)
    )
    snapshots = pd.DataFrame(records, columns=columns)

    if snapshots.duplicated(subset=["account_id", "snapshot_date"]).any():
        raise ValueError("Snapshot build produced duplicate account/snapshot rows")

    if not snapshots.empty:
        snapshots["snapshot_date"] = pd.to_datetime(snapshots["snapshot_date"])
        snapshots["outcome_start_month"] = pd.to_datetime(
            snapshots["outcome_start_month"]
        )
        snapshots["outcome_end_month"] = pd.to_datetime(snapshots["outcome_end_month"])
        snapshots = snapshots.sort_values(
            ["snapshot_date", "account_id"], kind="mergesort"
        ).reset_index(drop=True)

    return snapshots


def get_feature_columns(frame: pd.DataFrame | None = None) -> list[str]:
    """Return the explicit model-feature allow-list.

    If a frame is supplied, missing or forbidden columns raise immediately.
    """

    columns = list(FEATURE_COLUMNS)
    if frame is not None:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing expected feature columns: {missing}")
        forbidden_present = [column for column in FORBIDDEN_FEATURE_COLUMNS if column in columns]
        if forbidden_present:
            raise AssertionError(
                f"Feature allow-list contains forbidden columns: {forbidden_present}"
            )
    return columns


def get_target_columns() -> list[str]:
    return list(TARGET_COLUMNS)


def get_metadata_columns() -> list[str]:
    return list(SNAPSHOT_METADATA_COLUMNS)


def label_prevalence(snapshots: pd.DataFrame, target_column: str | None = None) -> float:
    """Return the positive rate among labelled rows, or NaN when none exist."""

    column = target_column or "future_revenue_decline"
    if snapshots.empty or column not in snapshots.columns:
        return float("nan")
    labelled = snapshots[column].dropna()
    if labelled.empty:
        return float("nan")
    return float(labelled.astype(int).mean())


def assert_time_ordering(snapshots: pd.DataFrame) -> None:
    """Fail if any label window starts before its snapshot or split order breaks."""

    if snapshots.empty:
        return
    if (snapshots["outcome_start_month"] <= snapshots["snapshot_date"]).any():
        raise AssertionError("Outcome window must start after the snapshot date")

    split_order = {"train": 0, "validation": 1, "test": 2}
    ordered = snapshots.assign(_order=snapshots["split"].map(split_order))
    if ordered["_order"].isna().any():
        raise AssertionError("Unexpected split value")

    train_max = ordered.loc[ordered["split"] == "train", "snapshot_date"].max()
    validation_min = ordered.loc[ordered["split"] == "validation", "snapshot_date"].min()
    validation_max = ordered.loc[ordered["split"] == "validation", "snapshot_date"].max()
    test_min = ordered.loc[ordered["split"] == "test", "snapshot_date"].min()

    if pd.notna(train_max) and pd.notna(validation_min) and not train_max < validation_min:
        raise AssertionError("Train snapshots must precede validation snapshots")
    if pd.notna(validation_max) and pd.notna(test_min) and not validation_max < test_min:
        raise AssertionError("Validation snapshots must precede test snapshots")
