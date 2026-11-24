"""Transparent, rule-based next-best-action suggestions.

No LLM is used. Every suggestion is produced by an ordered rule that reads the
account's own point-in-time features, and every suggestion carries the evidence
that triggered it. Wording is descriptive: a suggestion is an option for a human
reviewer, never an instruction and never an automated action.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from src.utils.config import Config

ACTION_REVENUE = "Investigate recent revenue decline"
ACTION_ENGAGEMENT = "Review engagement drop"
ACTION_HEALTH = "Conduct account health review"
ACTION_RETENTION = "Schedule retention discussion"
ACTION_ADOPTION = "Explore product adoption / cross-sell opportunity"
ACTION_MONITOR = "Monitor account - no immediate action"

# Actions a reviewer may take. Used by the UI and tests; no action leaves the app.
ALLOWED_ACTIONS: tuple[str, ...] = (
    ACTION_REVENUE,
    ACTION_ENGAGEMENT,
    ACTION_HEALTH,
    ACTION_RETENTION,
    ACTION_ADOPTION,
    ACTION_MONITOR,
)

BAND_GUIDANCE: dict[str, str] = {
    "high": "High-priority review recommended",
    "medium": "Review recommended",
    "low": "Monitor - no immediate action",
}


def _number(row: Mapping[str, Any], column: str) -> float | None:
    value = row.get(column)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def _pct(value: float | None) -> str:
    if value is None:
        return "not available"
    return f"{value * 100:+.1f}%"


def _days(value: float | None) -> str:
    if value is None:
        return "not available"
    return f"{value:.1f} days"


def recommend_action(row: Mapping[str, Any], config: Config) -> dict[str, Any]:
    """Return a deterministic suggestion plus the evidence behind it.

    Rules are evaluated in order and the first match wins:

    1. material three-month revenue decline -> investigate revenue;
    2. engagement trend decline or low engagement level -> review engagement;
    3. long inactivity gap -> account health review;
    4. large six-month revenue growth -> retention discussion (give-back risk);
    5. elevated model risk (high or medium band) -> account health review;
    6. thin product footprint on a low-risk account -> explore adoption;
    7. otherwise -> monitor.
    """

    thresholds = {
        str(key): float(value) for key, value in config.reason_settings.items()
    }
    band = str(row.get("risk_band", "low"))

    revenue_trend_3m = _number(row, "revenue_trend_3m_pct")
    revenue_trend_6m = _number(row, "revenue_trend_6m_pct")
    engagement_trend = _number(row, "engagement_trend_3m_pct")
    engagement_mean = _number(row, "engagement_mean_3m")
    inactivity = _number(row, "days_since_activity_last")
    transaction_trend = _number(row, "transaction_trend_3m_pct")
    product_count = _number(row, "product_count_last")

    if (
        revenue_trend_3m is not None
        and revenue_trend_3m <= thresholds["revenue_decline_3m_pct"]
    ):
        evidence = [
            f"Revenue trend over the last 3 months versus the prior 3 months: {_pct(revenue_trend_3m)}",
        ]
        if transaction_trend is not None:
            evidence.append(
                f"Transaction frequency trend: {_pct(transaction_trend)}"
            )
        return {
            "action": ACTION_REVENUE,
            "evidence": evidence,
            "rationale": (
                "Recent revenue is already below its prior-quarter level, so a "
                "reviewer should confirm whether the drop is structural."
            ),
        }

    engagement_triggered = (
        engagement_trend is not None
        and engagement_trend <= thresholds["engagement_decline_3m_pct"]
    ) or (
        engagement_mean is not None
        and engagement_mean <= thresholds["low_engagement_level"]
    )
    if engagement_triggered:
        evidence = []
        if engagement_mean is not None:
            evidence.append(f"Engagement level (3-month mean): {engagement_mean:.1f} / 100")
        if engagement_trend is not None:
            evidence.append(f"Engagement trend: {_pct(engagement_trend)}")
        if revenue_trend_6m is not None:
            evidence.append(f"Six-month revenue change: {_pct(revenue_trend_6m)}")
        return {
            "action": ACTION_ENGAGEMENT,
            "evidence": evidence,
            "rationale": (
                "Engagement is weak or weakening, which historically accompanies "
                "revenue risk in this dataset's synthetic patterns."
            ),
        }

    if inactivity is not None and inactivity >= thresholds["inactivity_days"]:
        return {
            "action": ACTION_HEALTH,
            "evidence": [
                f"Days since last recorded account activity: {_days(inactivity)}",
                f"Engagement level (3-month mean): "
                + (
                    f"{engagement_mean:.1f} / 100"
                    if engagement_mean is not None
                    else "not available"
                ),
            ],
            "rationale": (
                "The account has gone quiet for longer than the configured "
                "inactivity threshold; a health check is a low-cost next step."
            ),
        }

    if (
        revenue_trend_6m is not None
        and revenue_trend_6m >= thresholds["revenue_growth_6m_pct"]
    ):
        return {
            "action": ACTION_RETENTION,
            "evidence": [
                f"Six-month revenue change: {_pct(revenue_trend_6m)}",
                f"Three-month revenue change: {_pct(revenue_trend_3m)}",
            ],
            "rationale": (
                "In this synthetic data a sharp six-month run-up precedes "
                "give-back, so the model ranks the account highly even though "
                "recent revenue is rising. Confirm the account's forward pipeline."
            ),
        }

    if band in {"high", "medium"}:
        return {
            "action": ACTION_HEALTH,
            "evidence": [
                f"Model risk band: {band} ({_pct(row.get('risk_probability'))} probability)",
                f"Engagement level (3-month mean): "
                + (
                    f"{engagement_mean:.1f} / 100"
                    if engagement_mean is not None
                    else "not available"
                ),
                f"Revenue trend (3 months): {_pct(revenue_trend_3m)}",
            ],
            "rationale": (
                "The model places this account in an elevated risk band without a "
                "single dominant signal; a short health review is the appropriate "
                "first step."
            ),
        }

    if product_count is not None and product_count <= 1 and band == "low":
        return {
            "action": ACTION_ADOPTION,
            "evidence": [
                f"Active products: {product_count:.0f}",
                "Model risk band: low",
            ],
            "rationale": (
                "Risk is low and the product footprint is thin, so expansion is a "
                "more relevant conversation than retention."
            ),
        }

    return {
        "action": ACTION_MONITOR,
        "evidence": ["No configured risk or activity rule fired for this account."],
        "rationale": "Keep the account in the normal review cycle.",
    }


def band_guidance(band: str) -> str:
    """Return human-readable guidance for a risk band."""

    return BAND_GUIDANCE.get(str(band), "Review recommended")
