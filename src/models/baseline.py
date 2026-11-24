"""Transparent, deterministic scorecard baseline for revenue-decline risk.

The baseline is deliberately small enough for a sales manager to understand:
four observable signals, fixed thresholds and additive weights. It is not
optimised on the test set and it does not use account identifiers or any
target-period information.

Signals (all available at the snapshot date):

1. ``revenue_trend_3m_pct`` at or below the configured decline threshold.
2. ``transaction_trend_3m_pct`` at or below the configured decline threshold.
3. ``days_since_activity_last`` at or above the configured inactivity threshold.
4. ``engagement_trend_3m_pct`` at or below the configured decline threshold.

Missing values never trigger a signal. This is conservative: a missing signal
adds no points rather than treating missingness as risk. The separate
``engagement_missing_ratio_3m`` feature is available to the ML models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.features.snapshots import FEATURE_COLUMNS
from src.utils.config import Config

SIGNAL_NAMES: tuple[str, ...] = (
    "revenue_decline",
    "transaction_decline",
    "inactivity",
    "engagement_decline",
)

THRESHOLD_KEYS: tuple[str, ...] = (
    "revenue_decline_pct",
    "transaction_decline_pct",
    "inactivity_days",
    "engagement_decline_pct",
)

SIGNAL_LABELS: dict[str, str] = {
    "revenue_decline": "recent revenue decline",
    "transaction_decline": "reduced transaction frequency",
    "inactivity": "increased inactivity",
    "engagement_decline": "weakening engagement",
}

# Mapping from the business signal to the actual point-in-time feature.
SIGNAL_FEATURES: dict[str, str] = {
    "revenue_decline": "revenue_trend_3m_pct",
    "transaction_decline": "transaction_trend_3m_pct",
    "inactivity": "days_since_activity_last",
    "engagement_decline": "engagement_trend_3m_pct",
}


@dataclass(frozen=True)
class BaselineScorecard:
    """A small additive scorecard with configurable weights and thresholds."""

    weights: Mapping[str, float]
    thresholds: Mapping[str, float]
    high_risk_score: float

    @classmethod
    def from_config(cls, config: Config) -> "BaselineScorecard":
        return cls(
            weights=dict(config.baseline_weights),
            thresholds=dict(config.baseline_thresholds),
            high_risk_score=float(config.baseline_high_risk_score),
        )

    @property
    def max_score(self) -> float:
        return float(sum(self.weights.values()))

    def _validate_signal_config(self) -> None:
        if set(self.weights) != set(SIGNAL_NAMES):
            raise ValueError(
                "Baseline weights must define exactly: "
                + ", ".join(SIGNAL_NAMES)
            )
        if set(self.thresholds) != set(THRESHOLD_KEYS):
            raise ValueError(
                "Baseline thresholds must define exactly: "
                + ", ".join(THRESHOLD_KEYS)
            )
        if self.max_score <= 0:
            raise ValueError("Baseline weights must sum to a positive value")

    def signal_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return boolean signal flags and additive point contributions."""

        self._validate_signal_config()
        missing = [column for column in SIGNAL_FEATURES.values() if column not in frame.columns]
        if missing:
            raise ValueError(f"Baseline input is missing feature columns: {missing}")

        signals = pd.DataFrame(index=frame.index)
        revenue = pd.to_numeric(frame["revenue_trend_3m_pct"], errors="coerce")
        transactions = pd.to_numeric(frame["transaction_trend_3m_pct"], errors="coerce")
        inactivity = pd.to_numeric(frame["days_since_activity_last"], errors="coerce")
        engagement = pd.to_numeric(frame["engagement_trend_3m_pct"], errors="coerce")

        signals["revenue_decline"] = (
            revenue <= float(self.thresholds["revenue_decline_pct"])
        ).fillna(False)
        signals["transaction_decline"] = (
            transactions <= float(self.thresholds["transaction_decline_pct"])
        ).fillna(False)
        signals["inactivity"] = (
            inactivity >= float(self.thresholds["inactivity_days"])
        ).fillna(False)
        signals["engagement_decline"] = (
            engagement <= float(self.thresholds["engagement_decline_pct"])
        ).fillna(False)

        for signal in SIGNAL_NAMES:
            signals[f"points_{signal}"] = (
                signals[signal].astype(float) * float(self.weights[signal])
            )
        return signals

    def score(self, frame: pd.DataFrame) -> pd.Series:
        """Return the additive risk score for each row."""

        signals = self.signal_frame(frame)
        points = [f"points_{signal}" for signal in SIGNAL_NAMES]
        score = signals[points].sum(axis=1)
        return pd.Series(score.to_numpy(dtype=float), index=frame.index, name="baseline_score")

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Return a probability-shaped [no-risk, risk] array.

        The score is only normalised by its maximum; it is **not** a calibrated
        probability. Ranking metrics are the primary evaluation for this model.
        """

        probability = np.clip(self.score(frame).to_numpy(dtype=float) / self.max_score, 0.0, 1.0)
        return np.column_stack([1.0 - probability, probability])

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return (self.score(frame) >= self.high_risk_score).to_numpy(dtype=int)

    def metadata(self) -> dict[str, Any]:
        return {
            "model_type": "transparent_scorecard",
            "signals": list(SIGNAL_NAMES),
            "signal_features": dict(SIGNAL_FEATURES),
            "weights": {key: float(value) for key, value in self.weights.items()},
            "thresholds": {key: float(value) for key, value in self.thresholds.items()},
            "high_risk_score": float(self.high_risk_score),
            "max_score": float(self.max_score),
            "probability_note": "score divided by maximum score; not calibrated",
        }


def score_baseline(
    frame: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """Return a copy with baseline score, probability and binary flag."""

    scorecard = BaselineScorecard.from_config(config)
    scored = frame.copy()
    scored["baseline_score"] = scorecard.score(frame)
    scored["baseline_probability"] = scored["baseline_score"] / scorecard.max_score
    scored["baseline_high_risk"] = (
        scored["baseline_score"] >= scorecard.high_risk_score
    ).astype(int)
    return scored


def rank_baseline(
    frame: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """Rank rows by score with deterministic, business-readable tie-breaks.

    Ties are broken by worse recent revenue trend, worse transaction trend,
    worse engagement trend, then ``snapshot_id``. No future information is used.
    """

    scored = score_baseline(frame, config)
    sort_columns = [
        "baseline_score",
        "revenue_trend_3m_pct",
        "transaction_trend_3m_pct",
        "engagement_trend_3m_pct",
        "snapshot_id",
    ]
    ascending = [False, True, True, True, True]
    available = [column for column in sort_columns if column in scored.columns]
    ascending = ascending[: len(available)]
    scored = scored.sort_values(
        available,
        ascending=ascending,
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    scored["baseline_rank"] = np.arange(1, len(scored) + 1)
    return scored


def baseline_feature_columns() -> tuple[str, ...]:
    """Expose the baseline inputs for tests and audit."""

    return tuple(SIGNAL_FEATURES.values())


def assert_baseline_features_allowed() -> None:
    """Fail if any baseline signal is outside the Phase 1 feature allow-list."""

    allowed = set(FEATURE_COLUMNS)
    unexpected = [column for column in baseline_feature_columns() if column not in allowed]
    if unexpected:
        raise AssertionError(f"Baseline uses non-allow-listed features: {unexpected}")
