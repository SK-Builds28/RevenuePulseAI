"""Tests for the reproducible synthetic data generator."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.generate_synthetic import generate_monthly_accounts
from src.data.schema import FORBIDDEN_IDENTIFIER_COLUMNS, RAW_COLUMNS


def test_schema_is_expected(full_monthly):
    assert tuple(full_monthly.columns) == RAW_COLUMNS
    assert "account_name" not in full_monthly.columns
    assert "email" not in full_monthly.columns
    assert not set(FORBIDDEN_IDENTIFIER_COLUMNS).intersection(full_monthly.columns)


def test_no_personal_identifiers(full_monthly):
    ids = full_monthly["account_id"].astype(str)
    assert ids.str.match(r"^SYN-\d{4}$").all()
    assert not ids.str.contains("@").any()
    assert not full_monthly["segment"].astype(str).str.contains("@").any()


def test_monthly_grid_is_complete(full_monthly, config):
    per_account = full_monthly.groupby("account_id")["month"].nunique()
    assert per_account.nunique() == 1
    assert per_account.iloc[0] == config.n_months
    assert not full_monthly.duplicated(subset=["account_id", "month"]).any()
    assert full_monthly["month"].min() == config.start_month


def test_generator_is_reproducible(config):
    first = generate_monthly_accounts(config, n_accounts=12, n_months=18, seed=7)
    second = generate_monthly_accounts(config, n_accounts=12, n_months=18, seed=7)
    different = generate_monthly_accounts(config, n_accounts=12, n_months=18, seed=8)
    pd.testing.assert_frame_equal(first, second)
    assert not first["monthly_revenue"].equals(different["monthly_revenue"])


def test_values_are_realistic_and_non_negative(full_monthly, config):
    assert (full_monthly["monthly_revenue"] >= 0).all()
    assert (full_monthly["transaction_count"] >= 0).all()
    assert (full_monthly["product_count"] >= 1).all()
    assert (full_monthly["support_interactions"] >= 0).all()
    assert (full_monthly["contract_age_months"] >= 0).all()
    assert full_monthly["segment"].isin(config.segments).all()

    engagement = full_monthly["engagement_score"].dropna()
    assert engagement.between(0, 100).all()
    activity = full_monthly["days_since_last_activity"].dropna()
    assert (activity >= 0).all()

    # Missingness should exist but remain a small minority of rows.
    engagement_missing = full_monthly["engagement_score"].isna().mean()
    assert 0 < engagement_missing < 0.15


def test_growth_archetypes_create_variation(full_monthly):
    per_account = (
        full_monthly.groupby("account_id")["monthly_revenue"]
        .agg(["first", "last", "mean", "std"])
        .reset_index()
    )
    growth = (per_account["last"] - per_account["first"]) / per_account["first"].clip(lower=1)
    assert growth.min() < -0.25
    assert growth.max() > 0.25
    assert per_account["std"].gt(0).mean() > 0.95


def test_committed_demo_dataset_is_reproducible(config, project_root: Path, sample_monthly):
    generated = generate_monthly_accounts(
        config,
        n_accounts=config.demo_n_accounts,
        n_months=config.demo_n_months,
        seed=config.demo_seed,
    )
    committed_path = project_root / "data" / "sample" / "account_month_sample.csv"
    committed = pd.read_csv(committed_path, parse_dates=["month"])
    committed = committed.sort_values(["account_id", "month"], kind="mergesort").reset_index(drop=True)
    generated = generated.sort_values(["account_id", "month"], kind="mergesort").reset_index(drop=True)
    pd.testing.assert_frame_equal(committed, generated, check_dtype=False)


def test_account_ids_are_sequential_synthetic_codes(full_monthly):
    # The generator uses synthetic codes only; there are no company names.
    expected = {f"SYN-{index:04d}" for index in range(1, full_monthly["account_id"].nunique() + 1)}
    assert set(full_monthly["account_id"].astype(str)) == expected
    assert len(expected) == len(full_monthly) // full_monthly["month"].nunique()
