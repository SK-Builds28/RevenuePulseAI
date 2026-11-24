"""Data quality checks for the raw account-month dataset.

The checks are intentionally strict about identifiers, dates, duplicates and
revenue because those errors silently corrupt point-in-time features and labels.
Missing optional signals are warnings rather than hard failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.data.schema import (
    FORBIDDEN_IDENTIFIER_COLUMNS,
    MISSING_VALUE_TOLERATED_COLUMNS,
    NON_NEGATIVE_COLUMNS,
    NUMERIC_COLUMNS,
    RAW_REQUIRED_COLUMNS,
)
from src.utils.config import Config


@dataclass(frozen=True)
class QualityIssue:
    code: str
    severity: str
    message: str
    count: int = 0
    examples: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "count": self.count,
            "examples": list(self.examples),
        }


@dataclass
class DataQualityReport:
    n_rows: int
    n_accounts: int
    issues: list[QualityIssue] = field(default_factory=list)

    def add(
        self,
        code: str,
        severity: str,
        message: str,
        count: int = 0,
        examples: Iterable[str] = (),
    ) -> None:
        self.issues.append(
            QualityIssue(
                code=code,
                severity=severity,
                message=message,
                count=int(count),
                examples=tuple(str(item) for item in list(examples)[:5]),
            )
        )

    @property
    def errors(self) -> list[QualityIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[QualityIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    @property
    def passed(self) -> bool:
        return not self.has_errors

    def summary(self) -> str:
        lines = [
            f"Rows: {self.n_rows:,} | Accounts: {self.n_accounts:,} | "
            f"Errors: {len(self.errors)} | Warnings: {len(self.warnings)}"
        ]
        for issue in self.issues:
            lines.append(
                f"[{issue.severity.upper()}] {issue.code}: {issue.message} "
                f"(count={issue.count})"
            )
            if issue.examples:
                lines.append(f"  examples: {', '.join(issue.examples)}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_accounts": self.n_accounts,
            "passed": self.passed,
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
        }

    def raise_if_errors(self, source: str = "dataset") -> None:
        if self.has_errors:
            raise DataValidationError(source=source, report=self)


class DataValidationError(ValueError):
    """Raised when a dataset fails blocking quality checks."""

    def __init__(self, source: str, report: DataQualityReport):
        self.source = source
        self.report = report
        super().__init__(
            f"Data validation failed for {source}.\n{report.summary()}"
        )


def _examples(values: pd.Series, limit: int = 5) -> tuple[str, ...]:
    unique = values.dropna().astype(str).drop_duplicates().head(limit)
    return tuple(unique.tolist())


def _account_count(frame: pd.DataFrame) -> int:
    if "account_id" not in frame.columns:
        return 0
    return int(frame["account_id"].dropna().nunique())


def validate_monthly_data(frame: pd.DataFrame, config: Config) -> DataQualityReport:
    """Validate a raw account-month dataframe without mutating it."""

    report = DataQualityReport(n_rows=len(frame), n_accounts=_account_count(frame))

    missing_columns = [column for column in RAW_REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        report.add(
            "missing_required_columns",
            "error",
            f"Missing required columns: {', '.join(missing_columns)}",
            count=len(missing_columns),
        )
        return report

    forbidden = [column for column in FORBIDDEN_IDENTIFIER_COLUMNS if column in frame.columns]
    if forbidden:
        report.add(
            "forbidden_personal_columns",
            "error",
            "Dataset contains personal/identifier columns that are not allowed: "
            + ", ".join(forbidden),
            count=len(forbidden),
        )

    unexpected = [column for column in frame.columns if column not in RAW_REQUIRED_COLUMNS and column != "is_synthetic"]
    if unexpected:
        report.add(
            "unexpected_columns",
            "warning",
            "Unexpected extra columns will be ignored: " + ", ".join(unexpected),
            count=len(unexpected),
        )

    # Identifiers
    account_id = frame["account_id"]
    blank_accounts = account_id.isna() | (account_id.astype(str).str.strip() == "")
    if blank_accounts.any():
        report.add(
            "missing_account_id",
            "error",
            "account_id must be present and non-blank",
            count=int(blank_accounts.sum()),
        )

    # Dates
    parsed_month = pd.to_datetime(frame["month"], errors="coerce", format="mixed")
    invalid_dates = parsed_month.isna()
    if invalid_dates.any():
        report.add(
            "invalid_month",
            "error",
            "month contains values that cannot be parsed as dates",
            count=int(invalid_dates.sum()),
            examples=_examples(frame.loc[invalid_dates, "month"]),
        )
    else:
        not_month_start = parsed_month.dt.day != 1
        if not_month_start.any():
            report.add(
                "month_not_first_day",
                "warning",
                "month values are not first-of-month; they will be normalised to month start",
                count=int(not_month_start.sum()),
                examples=_examples(frame.loc[not_month_start, "month"]),
            )

    # Segment categories
    segments = set(config.segments)
    segment_values = frame["segment"].astype("string").str.strip()
    if segments:
        invalid_segments = ~segment_values.isin(segments)
        if invalid_segments.any():
            report.add(
                "unexpected_segment",
                "error",
                "segment contains values outside the configured categories",
                count=int(invalid_segments.sum()),
                examples=_examples(frame.loc[invalid_segments, "segment"]),
            )
    if frame["segment"].isna().any():
        report.add(
            "missing_segment",
            "error",
            "segment must not be missing",
            count=int(frame["segment"].isna().sum()),
        )

    # Numeric parsing and ranges
    numeric_frame = pd.DataFrame(index=frame.index)
    for column in NUMERIC_COLUMNS:
        original_missing = frame[column].isna()
        converted = pd.to_numeric(frame[column], errors="coerce")
        numeric_frame[column] = converted
        newly_invalid = converted.isna() & ~original_missing
        if newly_invalid.any():
            report.add(
                f"non_numeric_{column}",
                "error",
                f"{column} contains non-numeric values",
                count=int(newly_invalid.sum()),
                examples=_examples(frame.loc[newly_invalid, column]),
            )

    for column in NON_NEGATIVE_COLUMNS:
        if column not in numeric_frame:
            continue
        negative = numeric_frame[column] < 0
        if negative.any():
            report.add(
                f"negative_{column}",
                "error",
                f"{column} must be non-negative",
                count=int(negative.sum()),
                examples=_examples(frame.loc[negative, column]),
            )

    if "engagement_score" in numeric_frame:
        outside = (numeric_frame["engagement_score"] < 0) | (
            numeric_frame["engagement_score"] > 100
        )
        if outside.any():
            report.add(
                "engagement_out_of_range",
                "warning",
                "engagement_score is expected to be between 0 and 100",
                count=int(outside.sum()),
                examples=_examples(frame.loc[outside, "engagement_score"]),
            )

    # Missing values
    for column in RAW_REQUIRED_COLUMNS:
        if column == "month":
            continue
        missing = frame[column].isna()
        if not missing.any():
            continue
        if column in MISSING_VALUE_TOLERATED_COLUMNS:
            report.add(
                f"missing_optional_{column}",
                "warning",
                f"{column} has missing values; the snapshot builder will expose a missingness flag",
                count=int(missing.sum()),
            )
        else:
            report.add(
                f"missing_required_{column}",
                "error",
                f"{column} must not be missing",
                count=int(missing.sum()),
            )

    # Duplicates at the account-month grain
    duplicate_mask = frame.duplicated(subset=["account_id", "month"], keep=False)
    if duplicate_mask.any():
        duplicate_pairs = frame.loc[duplicate_mask, ["account_id", "month"]].astype(str)
        report.add(
            "duplicate_account_month",
            "error",
            "Each account must have at most one row per month",
            count=int(duplicate_mask.sum()),
            examples=tuple(
                f"{row.account_id}@{row.month}" for row in duplicate_pairs.drop_duplicates().head(5).itertuples()
            ),
        )

    # History length and gaps
    valid_dates = parsed_month.notna()
    if valid_dates.any() and report.n_accounts > 0:
        working = frame.loc[valid_dates, ["account_id"]].copy()
        working["month"] = parsed_month.loc[valid_dates]
        counts = working.groupby("account_id", dropna=True)["month"].nunique()
        short = counts[counts < config.minimum_history_months]
        if len(short):
            report.add(
                "insufficient_history",
                "warning",
                "Some accounts have fewer months than the configured minimum history; "
                "they will be excluded from snapshots",
                count=int(len(short)),
                examples=tuple(f"{account} ({months}m)" for account, months in short.head(5).items()),
            )

        gap_accounts: list[str] = []
        for account, group in working.groupby("account_id", dropna=True):
            months = pd.Series(sorted(group["month"].unique()))
            if len(months) < 2:
                continue
            expected = pd.date_range(months.iloc[0], months.iloc[-1], freq="MS")
            if len(expected) != len(months):
                gap_accounts.append(str(account))
        if gap_accounts:
            report.add(
                "calendar_gaps",
                "warning",
                "Some accounts have missing calendar months; review before snapshotting",
                count=len(gap_accounts),
                examples=gap_accounts[:5],
            )

    # Account-level consistency
    if "segment" in frame.columns and report.n_accounts > 0:
        segment_counts = frame.groupby("account_id")["segment"].nunique(dropna=True)
        inconsistent = segment_counts[segment_counts > 1]
        if len(inconsistent):
            report.add(
                "inconsistent_segment",
                "warning",
                "Some accounts change segment across months; the latest segment will be used",
                count=int(len(inconsistent)),
                examples=tuple(str(item) for item in inconsistent.head(5).index),
            )

    # Optional synthetic marker
    if "is_synthetic" in frame.columns:
        values = frame["is_synthetic"].dropna().astype(str).str.lower()
        if not values.empty and not values.isin({"true", "1", "yes"}).all():
            report.add(
                "non_synthetic_rows",
                "warning",
                "Some rows are not marked as synthetic. Phase 1 expects synthetic-only data.",
                count=int((~values.isin({"true", "1", "yes"})).sum()),
            )

    zero_revenue = numeric_frame.get("monthly_revenue", pd.Series(dtype=float)) == 0
    if bool(zero_revenue.any()):
        report.add(
            "zero_revenue_months",
            "warning",
            "Some account-months have zero revenue; confirm this is real rather than missing data",
            count=int(zero_revenue.sum()),
        )

    return report


def normalise_monthly_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted copy with first-of-month dates and numeric dtypes."""

    normalised = frame.copy()
    normalised["month"] = (
        pd.to_datetime(normalised["month"], errors="coerce", format="mixed")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    normalised["account_id"] = normalised["account_id"].astype("string").str.strip()
    normalised["segment"] = normalised["segment"].astype("string").str.strip()

    for column in NUMERIC_COLUMNS:
        normalised[column] = pd.to_numeric(normalised[column], errors="coerce")

    if "is_synthetic" in normalised.columns:
        normalised["is_synthetic"] = (
            normalised["is_synthetic"]
            .astype("string")
            .str.lower()
            .map({"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False})
        )

    return normalised.sort_values(["account_id", "month"], kind="mergesort").reset_index(drop=True)


def load_monthly_data(
    path: str | Path,
    config: Config,
    strict: bool = True,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Load a CSV, validate it, and return a normalised dataframe plus report."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Data file not found: {source}")

    frame = pd.read_csv(source)
    report = validate_monthly_data(frame, config)
    if strict and report.has_errors:
        raise DataValidationError(source=str(source), report=report)

    return normalise_monthly_data(frame), report


def format_report(report: DataQualityReport) -> str:
    """Convenience alias used by command-line tools."""

    return report.summary()
