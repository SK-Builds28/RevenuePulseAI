"""SQLite audit log for human review decisions.

Every accepted, edited or dismissed recommendation is stored with the timestamp,
account, snapshot date, model score and the reviewer's own text. The store is
local-only (no network, no auth) and exists to make the human-in-the-loop step
auditable in the academic prototype.

No customer contact, pricing change or CRM write is performed here.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

DECISIONS: tuple[str, ...] = ("accepted", "edited", "dismissed")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS review_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at TEXT NOT NULL,
    account_id TEXT NOT NULL,
    snapshot_date TEXT NOT NULL,
    risk_score REAL NOT NULL,
    risk_band TEXT,
    model_name TEXT,
    ai_recommendation TEXT NOT NULL,
    decision TEXT NOT NULL,
    edited_recommendation TEXT,
    reviewer_note TEXT,
    reviewer TEXT
)
"""

COLUMNS: tuple[str, ...] = (
    "id",
    "recorded_at",
    "account_id",
    "snapshot_date",
    "risk_score",
    "risk_band",
    "model_name",
    "ai_recommendation",
    "decision",
    "edited_recommendation",
    "reviewer_note",
    "reviewer",
)


class ReviewFeedbackStore:
    """A small local SQLite store for reviewer decisions."""

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)

    # -- connection handling ------------------------------------------------
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=10.0)
        try:
            connection.execute(_SCHEMA)
            yield connection
            connection.commit()
        finally:
            connection.close()

    # -- writes -------------------------------------------------------------
    def record(
        self,
        *,
        account_id: str,
        snapshot_date: Any,
        risk_score: float,
        ai_recommendation: str,
        decision: str,
        risk_band: str | None = None,
        model_name: str | None = None,
        edited_recommendation: str | None = None,
        reviewer_note: str | None = None,
        reviewer: str | None = None,
        recorded_at: str | None = None,
    ) -> int:
        """Insert one review decision and return its row id."""

        decision_value = str(decision).strip().lower()
        if decision_value not in DECISIONS:
            raise ValueError(
                f"decision must be one of {DECISIONS}, got {decision!r}"
            )
        if not str(account_id).strip():
            raise ValueError("account_id is required")
        score = float(risk_score)
        if not 0.0 <= score <= 1.0:
            raise ValueError("risk_score must be between 0 and 1")

        timestamp = recorded_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
        snapshot = pd.Timestamp(snapshot_date).normalize().date().isoformat()

        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO review_feedback (
                    recorded_at, account_id, snapshot_date, risk_score, risk_band,
                    model_name, ai_recommendation, decision, edited_recommendation,
                    reviewer_note, reviewer
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    str(account_id),
                    snapshot,
                    score,
                    risk_band,
                    model_name,
                    str(ai_recommendation),
                    decision_value,
                    edited_recommendation,
                    reviewer_note,
                    reviewer,
                ),
            )
            return int(cursor.lastrowid)

    # -- reads --------------------------------------------------------------
    def fetch_all(self) -> pd.DataFrame:
        """Return the full audit log as a DataFrame."""

        if not self.database_path.exists():
            return pd.DataFrame(columns=list(COLUMNS))
        with self._connect() as connection:
            frame = pd.read_sql_query(
                "SELECT * FROM review_feedback ORDER BY id", connection
            )
        return frame

    def latest_decisions(self, snapshot_date: Any) -> dict[str, str]:
        """Map account id to the most recent decision for one snapshot date."""

        if not self.database_path.exists():
            return {}
        snapshot = pd.Timestamp(snapshot_date).normalize().date().isoformat()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT account_id, decision FROM review_feedback
                WHERE snapshot_date = ? ORDER BY id
                """,
                (snapshot,),
            ).fetchall()
        # Later rows overwrite earlier ones, so the last decision wins.
        return {str(account): str(decision) for account, decision in rows}

    def export_csv(self, output_path: str | Path) -> Path:
        """Write the audit log to CSV and return the path."""

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fetch_all().to_csv(path, index=False)
        return path
