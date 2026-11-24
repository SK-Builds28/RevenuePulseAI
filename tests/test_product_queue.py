"""Tests for the review-queue product layer used by the Streamlit prototype."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.snapshots import build_snapshots
from src.models.preprocessing import predict_risk_probability
from src.models.train import run_training
from src.product.feedback import ReviewFeedbackStore
from src.product.recommendations import ALLOWED_ACTIONS, recommend_action
from src.product.review import (
    CORE_LEVEL_FEATURES,
    DataUnavailableError,
    ModelUnavailableError,
    build_queue,
    list_snapshot_dates,
    load_model_bundle,
    risk_band,
    risk_summary,
    score_accounts,
)


@pytest.fixture(scope="module")
def product_env(config, small_monthly, tmp_path_factory):
    """A tiny trained artifact plus snapshot CSV, independent of git-ignored files."""

    workspace = tmp_path_factory.mktemp("product")
    snapshots = build_snapshots(small_monthly, config)
    labelled = snapshots.dropna(subset=[config.target_column])
    snapshot_path = workspace / "snapshots.csv"
    labelled.to_csv(snapshot_path, index=False, date_format="%Y-%m-%d")

    model_dir = workspace / "model"
    run_training(config, snapshot_path, model_dir)
    return {
        "snapshot_path": snapshot_path,
        "model_dir": model_dir,
        "workspace": workspace,
        "latest": pd.to_datetime(labelled["snapshot_date"]).max(),
    }


@pytest.fixture(scope="module")
def queue(config, product_env):
    return build_queue(
        config,
        snapshot_date=product_env["latest"],
        snapshot_path=product_env["snapshot_path"],
        model_dir=product_env["model_dir"],
        store=ReviewFeedbackStore(product_env["workspace"] / "feedback.db"),
    )


def test_model_bundle_loads(config, product_env):
    bundle = load_model_bundle(config, product_env["model_dir"])
    assert bundle["selected_model"] in bundle["pipelines"]
    assert "account_id" not in bundle["feature_columns"]


def test_probabilities_are_bounded(config, queue):
    assert not queue.empty
    assert queue["risk_probability"].between(0.0, 1.0).all()
    assert queue["risk_probability"].notna().all()


def test_queue_is_ranked_by_risk(queue):
    assert queue["risk_probability"].is_monotonic_decreasing
    assert queue["risk_rank"].tolist() == list(range(1, len(queue) + 1))
    assert queue["risk_band"].isin({"high", "medium", "low"}).all()


def test_risk_band_uses_configured_thresholds(config):
    assert risk_band(0.99, config) == "high"
    assert risk_band(config.risk_high, config) == "high"
    assert risk_band(config.risk_medium, config) == "medium"
    assert risk_band(config.risk_medium - 0.01, config) == "low"


def test_queue_filters_apply(config, product_env, queue):
    segment = str(queue["segment"].iloc[0])
    filtered = build_queue(
        config,
        snapshot_date=product_env["latest"],
        snapshot_path=product_env["snapshot_path"],
        model_dir=product_env["model_dir"],
        store=ReviewFeedbackStore(product_env["workspace"] / "feedback.db"),
        segment=[segment],
        top_n=5,
    )
    assert len(filtered) <= 5
    assert set(filtered["segment"].astype(str)) == {segment}
    assert filtered["risk_probability"].is_monotonic_decreasing


def test_queue_starts_pending_without_feedback(queue):
    assert (queue["review_status"] == "pending").all()


def test_account_id_never_enters_the_model(config, product_env, queue):
    bundle = load_model_bundle(config, product_env["model_dir"])
    pipeline = bundle["pipelines"][bundle["selected_model"]]

    frame = pd.read_csv(product_env["snapshot_path"])
    frame = frame[pd.to_datetime(frame["snapshot_date"]) == product_env["latest"]].head(20)
    renamed = frame.copy()
    renamed["account_id"] = [f"OTHER-{index}" for index in range(len(renamed))]

    first = predict_risk_probability(pipeline, frame)
    second = predict_risk_probability(pipeline, renamed)
    assert np.allclose(first, second)


def test_missing_required_fields_raise_instead_of_scoring(config, product_env):
    frame = pd.read_csv(product_env["snapshot_path"])
    broken = frame.drop(columns=["engagement_mean_3m"])
    with pytest.raises(DataUnavailableError, match="missing required fields"):
        score_accounts(
            config, broken, load_model_bundle(config, product_env["model_dir"])
        )


def test_unscorable_rows_are_excluded_not_guessed(config, product_env):
    frame = pd.read_csv(product_env["snapshot_path"])
    frame = frame[pd.to_datetime(frame["snapshot_date"]) == product_env["latest"]].copy()
    frame.loc[frame.index[:2], list(CORE_LEVEL_FEATURES)] = np.nan

    scored = score_accounts(
        config, frame, load_model_bundle(config, product_env["model_dir"])
    )
    assert scored.attrs["skipped_rows"] == 2
    assert scored["risk_probability"].notna().all()


def test_missing_snapshot_file_raises_controlled_error(config, tmp_path):
    with pytest.raises(DataUnavailableError, match="Snapshot data not found"):
        build_queue(
            config,
            snapshot_date="2025-09-30",
            snapshot_path=tmp_path / "does_not_exist.csv",
        )


def test_missing_model_raises_controlled_error(config, product_env, tmp_path):
    with pytest.raises(ModelUnavailableError, match="Model artifact not found"):
        load_model_bundle(config, tmp_path / "no_model")


def test_dates_are_listed_newest_first(config, product_env):
    dates = list_snapshot_dates(config, snapshot_path=product_env["snapshot_path"])
    assert dates
    assert dates == sorted(dates, reverse=True)


def test_recommendations_are_deterministic_and_evidenced(queue, config):
    row = queue.iloc[0]
    first = recommend_action(row, config)
    second = recommend_action(row, config)
    assert first == second
    assert first["action"] in ALLOWED_ACTIONS
    assert first["evidence"], "A suggestion must cite account evidence"
    # Evidence strings carry real values, not placeholders.
    assert any(any(character.isdigit() for character in item) for item in first["evidence"])


def test_recommendation_rules_cover_decline_and_monitor(config):
    base = {
        "risk_band": "high",
        "risk_probability": 0.8,
        "revenue_trend_3m_pct": -0.25,
        "revenue_trend_6m_pct": 0.10,
        "transaction_trend_3m_pct": -0.10,
        "engagement_mean_3m": 65.0,
        "engagement_trend_3m_pct": -0.02,
        "days_since_activity_last": 5.0,
        "product_count_last": 3.0,
    }
    decline = recommend_action(base, config)
    assert decline["action"] == "Investigate recent revenue decline"

    quiet = dict(base)
    quiet.update(
        {
            "risk_band": "low",
            "risk_probability": 0.05,
            "revenue_trend_3m_pct": 0.02,
            "revenue_trend_6m_pct": 0.01,
            "transaction_trend_3m_pct": 0.01,
            "engagement_mean_3m": 70.0,
            "engagement_trend_3m_pct": 0.02,
            "days_since_activity_last": 3.0,
            "product_count_last": 3.0,
        }
    )
    assert recommend_action(quiet, config)["action"] == "Monitor account - no immediate action"


def test_product_layer_has_no_automated_customer_action():
    """Guardrail: the product package must not contain outbound execution code."""

    forbidden = (
        "smtplib",
        "sendgrid",
        "requests.post",
        "mailto:",
        "twilio",
        "hubspot",
        "salesforce",
        "webhook",
    )
    sources = [
        *(Path("src/product").glob("*.py")),
        Path("prototype/app.py"),
    ]
    for path in sources:
        text = path.read_text(encoding="utf-8").lower()
        for token in forbidden:
            assert token not in text, f"{path} contains outbound-action token {token!r}"


def test_risk_summary_matches_queue(queue):
    summary = risk_summary(queue)
    assert summary["reviewed"] == len(queue)
    assert summary["high"] + summary["medium"] + summary["low"] == len(queue)
    assert 0.0 <= summary["average_probability"] <= 1.0
