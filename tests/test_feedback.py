"""Tests for the local human-review feedback log."""

from __future__ import annotations

import pandas as pd
import pytest

from src.product.feedback import ReviewFeedbackStore


@pytest.fixture()
def store(tmp_path) -> ReviewFeedbackStore:
    return ReviewFeedbackStore(tmp_path / "runtime" / "review_feedback.db")


def test_feedback_record_is_written_and_read(store):
    row_id = store.record(
        account_id="SYN-0001",
        snapshot_date="2025-09-30",
        risk_score=0.72,
        risk_band="high",
        model_name="hist_gradient_boosting",
        ai_recommendation="Conduct account health review",
        decision="accepted",
        reviewer_note="Reviewed with AE; renewal conversation booked.",
        reviewer="demo-reviewer",
        recorded_at="2026-01-01T10:00:00+00:00",
    )
    assert row_id == 1
    log = store.fetch_all()
    assert len(log) == 1
    row = log.iloc[0]
    assert row["account_id"] == "SYN-0001"
    assert row["snapshot_date"] == "2025-09-30"
    assert row["risk_score"] == pytest.approx(0.72)
    assert row["decision"] == "accepted"
    assert row["ai_recommendation"] == "Conduct account health review"
    assert row["recorded_at"] == "2026-01-01T10:00:00+00:00"
    assert store.database_path.exists()


def test_edit_and_dismiss_are_recorded(store):
    store.record(
        account_id="SYN-0002",
        snapshot_date="2025-09-30",
        risk_score=0.41,
        ai_recommendation="Review engagement drop",
        decision="edited",
        edited_recommendation="Schedule a technical adoption review",
        reviewer_note="Engagement dipped after a product change.",
    )
    store.record(
        account_id="SYN-0003",
        snapshot_date="2025-09-30",
        risk_score=0.36,
        ai_recommendation="Conduct account health review",
        decision="dismissed",
        reviewer_note="Known seasonal billing pattern.",
    )
    log = store.fetch_all()
    assert log["decision"].tolist() == ["edited", "dismissed"]
    assert log.loc[0, "edited_recommendation"] == "Schedule a technical adoption review"
    assert log.loc[1, "edited_recommendation"] is None or pd.isna(
        log.loc[1, "edited_recommendation"]
    )


def test_invalid_decisions_are_rejected(store):
    with pytest.raises(ValueError, match="decision must be one of"):
        store.record(
            account_id="SYN-0004",
            snapshot_date="2025-09-30",
            risk_score=0.5,
            ai_recommendation="Monitor account - no immediate action",
            decision="sent_email",
        )
    with pytest.raises(ValueError, match="risk_score"):
        store.record(
            account_id="SYN-0004",
            snapshot_date="2025-09-30",
            risk_score=1.5,
            ai_recommendation="Monitor account - no immediate action",
            decision="accepted",
        )


def test_latest_decision_wins_for_review_status(store):
    for decision, note in (("accepted", "first"), ("dismissed", "second")):
        store.record(
            account_id="SYN-0005",
            snapshot_date="2025-09-30",
            risk_score=0.55,
            ai_recommendation="Conduct account health review",
            decision=decision,
            reviewer_note=note,
        )
    store.record(
        account_id="SYN-0006",
        snapshot_date="2025-06-30",
        risk_score=0.60,
        ai_recommendation="Schedule retention discussion",
        decision="edited",
    )
    decisions = store.latest_decisions("2025-09-30")
    assert decisions == {"SYN-0005": "dismissed"}
    assert store.latest_decisions("2025-06-30") == {"SYN-0006": "edited"}


def test_export_csv_contains_the_log(store, tmp_path):
    store.record(
        account_id="SYN-0007",
        snapshot_date="2025-09-30",
        risk_score=0.66,
        ai_recommendation="Investigate recent revenue decline",
        decision="accepted",
    )
    path = store.export_csv(tmp_path / "exports" / "review_log.csv")
    assert path.exists()
    frame = pd.read_csv(path)
    assert len(frame) == 1
    assert frame.loc[0, "account_id"] == "SYN-0007"


def test_empty_store_returns_empty_frame(tmp_path):
    store = ReviewFeedbackStore(tmp_path / "missing.db")
    assert store.fetch_all().empty
    assert store.latest_decisions("2025-09-30") == {}
