"""Account-level reason codes and lightweight model explainability.

Reason codes are **descriptive rules over the account's own input features**, not
causal claims and not an explanation of an individual decision. Every phrase is
derived from a feature threshold evaluated on the snapshot row, so a code can
never appear for an account whose features do not support it. Missing values
never fire a code.

No LLM and no heavy explainability dependency is used:

* logistic regression -> coefficients and odds ratios (standardised inputs);
* histogram gradient boosting -> permutation importance on the validation split.

Wording rules for the product: "factors **associated with** the risk score
include ...". Never "because of", "caused by" or "will churn".
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.features.snapshots import FEATURE_COLUMNS
from src.models.preprocessing import model_feature_frame
from src.utils.config import Config

# Default number of reason codes surfaced for one account. Kept small because a
# manager should see a short, checkable list rather than a full attribution dump.
DEFAULT_REASON_LIMIT = 4

CAUSALITY_DISCLAIMER = (
    "Reason codes list factors associated with the model's risk score. They do "
    "not show that a factor caused revenue decline and they do not show that an "
    "intervention would change the outcome."
)


class ReasonRule(NamedTuple):
    """One business-readable rule over one allow-listed feature."""

    name: str
    label: str
    feature: str
    direction: str  # "below" or "above"
    threshold_key: str


# Rule order is the display order and the truncation order for ``limit``.
REASON_RULES: tuple[ReasonRule, ...] = (
    ReasonRule(
        "revenue_decline",
        "recent revenue decline",
        "revenue_trend_3m_pct",
        "below",
        "revenue_decline_3m_pct",
    ),
    ReasonRule(
        "revenue_decline_6m",
        "revenue decline over the last six months",
        "revenue_trend_6m_pct",
        "below",
        "revenue_decline_6m_pct",
    ),
    ReasonRule(
        "revenue_growth_6m",
        "revenue well above its prior six-month average",
        "revenue_trend_6m_pct",
        "above",
        "revenue_growth_6m_pct",
    ),
    ReasonRule(
        "transaction_decline",
        "reduced transaction frequency",
        "transaction_trend_3m_pct",
        "below",
        "transaction_decline_3m_pct",
    ),
    ReasonRule(
        "inactivity",
        "increased inactivity",
        "days_since_activity_last",
        "above",
        "inactivity_days",
    ),
    ReasonRule(
        "engagement_decline",
        "weakening engagement",
        "engagement_trend_3m_pct",
        "below",
        "engagement_decline_3m_pct",
    ),
    ReasonRule(
        "low_engagement",
        "low engagement level",
        "engagement_mean_3m",
        "below",
        "low_engagement_level",
    ),
)

RULE_NAMES: tuple[str, ...] = tuple(rule.name for rule in REASON_RULES)
RULE_LABELS: dict[str, str] = {rule.name: rule.label for rule in REASON_RULES}


def reason_thresholds(config: Config) -> dict[str, float]:
    """Return the configured reason-code thresholds, validated against the rules."""

    thresholds = {
        str(key): float(value) for key, value in config.reason_settings.items()
    }
    thresholds.pop("limit", None)
    expected = {rule.threshold_key for rule in REASON_RULES}
    if set(thresholds) != expected:
        raise ValueError(
            "Reason thresholds must define exactly: " + ", ".join(sorted(expected))
        )
    return thresholds


def reason_limit(config: Config) -> int:
    return int(config.reason_settings.get("limit", DEFAULT_REASON_LIMIT))


def reason_code_frame(
    frame: pd.DataFrame,
    config: Config,
    *,
    limit: int | None = None,
) -> pd.DataFrame:
    """Return per-snapshot reason codes derived from actual feature values."""

    thresholds = reason_thresholds(config)
    maximum = reason_limit(config) if limit is None else int(limit)
    if maximum < 1:
        raise ValueError("limit must be positive")

    missing = [
        rule.feature for rule in REASON_RULES if rule.feature not in frame.columns
    ]
    if missing:
        raise ValueError(f"Reason code input is missing features: {sorted(set(missing))}")

    flags = pd.DataFrame(index=frame.index)
    for rule in REASON_RULES:
        values = pd.to_numeric(frame[rule.feature], errors="coerce")
        threshold = thresholds[rule.threshold_key]
        if rule.direction == "below":
            fired = values <= threshold
        else:
            fired = values >= threshold
        flags[rule.name] = fired.fillna(False).astype(bool)

    codes: list[list[str]] = []
    for _, row in flags.iterrows():
        row_codes = [rule.label for rule in REASON_RULES if bool(row[rule.name])]
        codes.append(row_codes[:maximum])

    output = pd.DataFrame(index=frame.index)
    output["reason_count"] = [len(value) for value in codes]
    output["reason_codes"] = ["; ".join(value) for value in codes]
    for rule in REASON_RULES:
        output[f"signal_{rule.name}"] = flags[rule.name].to_numpy(dtype=bool)
    output["disclaimer"] = CAUSALITY_DISCLAIMER
    return output


def signal_prevalence(reason_frame: pd.DataFrame) -> pd.DataFrame:
    """Summarise how often each reason code fires in a scored population."""

    rows: list[dict[str, Any]] = []
    for rule in REASON_RULES:
        column = f"signal_{rule.name}"
        if column not in reason_frame.columns:
            continue
        rows.append(
            {
                "signal": rule.name,
                "label": rule.label,
                "share_of_rows": float(reason_frame[column].mean()),
                "rows": int(reason_frame[column].sum()),
            }
        )
    return pd.DataFrame(rows)


def permutation_importance_table(
    pipeline: Any,
    frame: pd.DataFrame,
    *,
    target_column: str,
    n_repeats: int = 5,
    random_state: int = 42,
    scoring: str = "average_precision",
) -> pd.DataFrame:
    """Return permutation importance for a fitted pipeline's raw features.

    Permutation importance is model-agnostic, ships with scikit-learn and needs no
    additional dependency. Call it on the **validation** split (never the test
    split) so the explainability view stays inside the model-selection boundary.
    """

    if target_column not in frame.columns:
        raise ValueError(f"Missing target column: {target_column}")
    features = model_feature_frame(frame)
    target = frame[target_column].to_numpy().astype(int)
    if len(np.unique(target)) < 2:
        raise ValueError("Permutation importance needs both classes present")

    result = permutation_importance(
        pipeline,
        features,
        target,
        n_repeats=int(n_repeats),
        random_state=int(random_state),
        scoring=scoring,
    )
    table = pd.DataFrame(
        {
            "feature": list(FEATURE_COLUMNS),
            "importance_mean": np.asarray(result.importances_mean, dtype=float),
            "importance_std": np.asarray(result.importances_std, dtype=float),
        }
    )
    return table.sort_values(
        "importance_mean", ascending=False, kind="mergesort"
    ).reset_index(drop=True)


def reason_driver_alignment(
    importance: pd.DataFrame,
    reason_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Describe how each reason code relates to measured model importance.

    The table is descriptive: it reports the permutation importance (on
    validation) of every feature a reason code reads, so a reader can see which
    codes correspond to drivers the model actually uses.
    """

    if "feature" not in importance.columns:
        raise ValueError("Importance table needs a 'feature' column")
    values: Mapping[str, float] = dict(
        zip(importance["feature"], importance["importance_mean"])
    )
    rows: list[dict[str, Any]] = []
    for rule in REASON_RULES:
        rows.append(
            {
                "reason": rule.name,
                "label": rule.label,
                "feature": rule.feature,
                "feature_importance": float(values.get(rule.feature, float("nan"))),
                "share_of_rows": float(reason_frame[f"signal_{rule.name}"].mean()),
            }
        )
    return pd.DataFrame(rows)
