"""Review-queue services: load snapshots, score accounts and rank them.

This module is the contract between the frozen model and the Streamlit UI:

* snapshots and the model are loaded from the configured paths;
* every score comes from the saved model pipeline, never from a cached or typed
  number;
* risk bands use the configured probability thresholds;
* reason codes reuse the Phase 2 explainability logic;
* rows that cannot be scored honestly are excluded and reported.

No account is contacted, no price is changed and no CRM record is written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd

from src.features.snapshots import FEATURE_COLUMNS
from src.models.preprocessing import predict_risk_probability
from src.models.reasons import reason_code_frame
from src.product.feedback import ReviewFeedbackStore
from src.product.recommendations import recommend_action
from src.utils.config import Config

RISK_BANDS: tuple[str, ...] = ("high", "medium", "low")

CORE_LEVEL_FEATURES: tuple[str, ...] = (
    "revenue_mean_3m",
    "transaction_mean_3m",
    "engagement_mean_3m",
)

QUEUE_COLUMNS: tuple[str, ...] = (
    "account_id",
    "segment",
    "risk_probability",
    "risk_band",
    "revenue_mean_3m",
    "revenue_trend_3m_pct",
    "engagement_mean_3m",
    "days_since_activity_last",
    "reason_codes",
    "review_status",
    "risk_rank",
)


class DataUnavailableError(RuntimeError):
    """Raised when the snapshot or monthly history data cannot be read."""


class ModelUnavailableError(RuntimeError):
    """Raised when the trained model artifact cannot be loaded."""


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_model_bundle(config: Config, model_dir: Path | None = None) -> dict[str, Any]:
    """Load the frozen model bundle, raising a controlled product error."""

    directory = Path(model_dir) if model_dir is not None else config.artifacts_model_dir
    artifact = directory / "revenuepulse_model.joblib"
    if not artifact.exists():
        raise ModelUnavailableError(
            f"Model artifact not found: {artifact}. Train it first with "
            "`python -m src.models.train`."
        )
    try:
        bundle = joblib.load(artifact)
    except Exception as exc:  # pragma: no cover - defensive
        raise ModelUnavailableError(
            f"Model artifact could not be loaded: {artifact} ({exc})"
        ) from exc

    if not isinstance(bundle, Mapping) or "pipelines" not in bundle:
        raise ModelUnavailableError(
            "Model artifact has an unexpected structure; retrain with "
            "`python -m src.models.train`."
        )
    if bundle.get("selected_model") not in bundle["pipelines"]:
        raise ModelUnavailableError(
            "Model artifact does not contain the selected pipeline; retrain it."
        )
    return dict(bundle)


def load_snapshot_frame(
    config: Config,
    snapshot_date: Any | None = None,
    snapshot_path: Path | None = None,
) -> pd.DataFrame:
    """Load labelled snapshots, optionally for a single review date."""

    path = Path(snapshot_path) if snapshot_path is not None else config.full_snapshots_path
    if not path.exists():
        raise DataUnavailableError(
            f"Snapshot data not found: {path}. Build it with "
            "`python -m src.features.build_snapshots --input "
            "data/processed/account_month_full.csv "
            "--output data/processed/snapshots_full.csv`."
        )

    frame = pd.read_csv(path)
    if "snapshot_date" not in frame.columns:
        raise DataUnavailableError(
            f"Snapshot file {path} has no snapshot_date column."
        )
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce")
    if frame["snapshot_date"].isna().all():
        raise DataUnavailableError(f"Snapshot file {path} has no readable dates.")

    required = ["account_id", "segment", *FEATURE_COLUMNS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DataUnavailableError(
            "Snapshot file is missing required account fields: "
            + ", ".join(missing)
        )

    if snapshot_date is not None:
        target = pd.Timestamp(snapshot_date).normalize()
        frame = frame[frame["snapshot_date"].dt.normalize() == target]

    return frame.reset_index(drop=True)


def load_monthly_history(
    config: Config,
    account_id: str,
    snapshot_date: Any | None = None,
) -> pd.DataFrame:
    """Return monthly history for one account up to the snapshot date."""

    candidates = [config.full_data_path, config.sample_data_path]
    path = next((candidate for candidate in candidates if Path(candidate).exists()), None)
    if path is None:
        raise DataUnavailableError(
            "Monthly account history is not available. Regenerate it with "
            "`python -m src.data.generate_synthetic`."
        )

    frame = pd.read_csv(path)
    if "account_id" not in frame.columns or "month" not in frame.columns:
        raise DataUnavailableError(f"Monthly history {path} is missing required columns.")

    history = frame[frame["account_id"].astype(str) == str(account_id)].copy()
    history["month"] = pd.to_datetime(history["month"], errors="coerce")
    if snapshot_date is not None:
        history = history[history["month"] <= pd.Timestamp(snapshot_date)]
    return history.sort_values("month", kind="mergesort").reset_index(drop=True)


def list_snapshot_dates(config: Config, snapshot_path: Path | None = None) -> list[pd.Timestamp]:
    """Return available review dates, newest first."""

    frame = load_snapshot_frame(config, snapshot_path=snapshot_path)
    dates = sorted(pd.to_datetime(frame["snapshot_date"]).dt.normalize().unique())
    return [pd.Timestamp(value) for value in reversed(dates)]


# --------------------------------------------------------------------------
# Scoring and queue construction
# --------------------------------------------------------------------------
def risk_band(probability: float, config: Config) -> str:
    """Map a probability to the configured high / medium / low band."""

    value = float(probability)
    if not np.isfinite(value):
        raise ValueError("Risk probability must be finite")
    if value >= config.risk_high:
        return "high"
    if value >= config.risk_medium:
        return "medium"
    return "low"


def score_accounts(
    config: Config,
    frame: pd.DataFrame,
    bundle: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Add model probability, risk band and reason codes to snapshot rows.

    Rows whose core level features are all missing are excluded: the model would
    have to invent every input, and a score without evidence would be misleading.
    """

    if frame.empty:
        return frame.copy()

    loaded = dict(bundle) if bundle is not None else load_model_bundle(config)
    pipeline = loaded["pipelines"][loaded["selected_model"]]

    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise DataUnavailableError(
            "Cannot score accounts; missing required fields: " + ", ".join(missing)
        )

    core = frame.loc[:, list(CORE_LEVEL_FEATURES)].apply(
        pd.to_numeric, errors="coerce"
    )
    scorable = ~core.isna().all(axis=1)

    scored = frame.loc[scorable].copy().reset_index(drop=True)
    skipped = int((~scorable).sum())

    if scored.empty:
        result = scored
        result.attrs["skipped_rows"] = skipped
        return result

    probabilities = predict_risk_probability(pipeline, scored)
    scored["risk_probability"] = np.clip(probabilities, 0.0, 1.0)
    scored["risk_band"] = [
        risk_band(value, config) for value in scored["risk_probability"]
    ]

    reasons = reason_code_frame(scored, config)
    scored["reason_codes"] = reasons["reason_codes"].to_numpy()
    for column in reasons.columns:
        if column.startswith("signal_"):
            scored[column] = reasons[column].to_numpy()

    scored["baseline_score"] = _baseline_score(scored, config)
    scored["review_priority"] = scored["risk_band"].map(
        {"high": "High-priority review", "medium": "Review recommended", "low": "Monitor"}
    )

    scored = scored.sort_values(
        ["risk_probability", "account_id"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    scored["risk_rank"] = np.arange(1, len(scored) + 1)
    scored.attrs["skipped_rows"] = skipped
    return scored


def _baseline_score(frame: pd.DataFrame, config: Config) -> pd.Series:
    """Return the transparent scorecard score as a comparison column."""

    from src.models.baseline import BaselineScorecard

    return BaselineScorecard.from_config(config).score(frame)


def attach_review_status(
    frame: pd.DataFrame,
    config: Config,
    store: ReviewFeedbackStore | None = None,
) -> pd.DataFrame:
    """Attach the latest human decision for each account, or ``pending``."""

    result = frame.copy()
    if result.empty:
        result["review_status"] = pd.Series(dtype="object")
        return result

    feedback = store or ReviewFeedbackStore(config.feedback_db_path)
    snapshot = pd.Timestamp(result["snapshot_date"].iloc[0]).normalize()
    decisions = feedback.latest_decisions(snapshot)
    result["review_status"] = (
        result["account_id"].astype(str).map(decisions).fillna("pending")
    )
    return result


def build_queue(
    config: Config,
    *,
    snapshot_date: Any | None = None,
    segment: Iterable[str] | None = None,
    bands: Iterable[str] | None = None,
    top_n: int | None = None,
    model_dir: Path | None = None,
    snapshot_path: Path | None = None,
    store: ReviewFeedbackStore | None = None,
) -> pd.DataFrame:
    """Build the ranked, explained review queue for one snapshot date."""

    frame = load_snapshot_frame(config, snapshot_date, snapshot_path=snapshot_path)
    if frame.empty:
        raise DataUnavailableError(
            "No snapshot rows exist for the selected review date."
        )

    bundle = load_model_bundle(config, model_dir)
    scored = score_accounts(config, frame, bundle)

    selected_segments = [str(value) for value in segment or []]
    if selected_segments:
        scored = scored[scored["segment"].astype(str).isin(selected_segments)]

    selected_bands = [str(value) for value in bands or []]
    if selected_bands:
        scored = scored[scored["risk_band"].isin(selected_bands)]

    if top_n is not None:
        scored = scored.head(int(top_n))

    result = attach_review_status(scored, config, store=store).reset_index(drop=True)
    result["primary_reason"] = result["reason_codes"].replace("", "no reason code fired")
    result["primary_reason"] = result["primary_reason"].fillna("no reason code fired")
    result["primary_reason"] = result["primary_reason"].str.split(";").str[0]
    return result


def risk_summary(queue: pd.DataFrame) -> dict[str, Any]:
    """Return dashboard counts for a scored queue."""

    if queue.empty:
        return {
            "reviewed": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "average_probability": float("nan"),
            "skipped": int(queue.attrs.get("skipped_rows", 0)),
        }
    bands = queue["risk_band"].value_counts().to_dict()
    return {
        "reviewed": int(len(queue)),
        "high": int(bands.get("high", 0)),
        "medium": int(bands.get("medium", 0)),
        "low": int(bands.get("low", 0)),
        "average_probability": float(queue["risk_probability"].mean()),
        "skipped": int(queue.attrs.get("skipped_rows", 0)),
    }


def account_detail(
    config: Config,
    queue: pd.DataFrame,
    account_id: str,
) -> dict[str, Any] | None:
    """Return one queue row plus its recommendation for the detail page."""

    matches = queue[queue["account_id"].astype(str) == str(account_id)]
    if matches.empty:
        return None
    row = matches.iloc[0]
    recommendation = recommend_action(row, config)
    return {
        "row": row,
        "recommendation": recommendation,
        "reason_list": [
            code for code in str(row.get("reason_codes", "")).split(";") if code.strip()
        ],
    }
