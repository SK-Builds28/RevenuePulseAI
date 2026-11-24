"""Product-layer services for the RevenuePulse AI review prototype.

These modules hold the logic Streamlit pages render: loading snapshots, scoring
accounts with the frozen model, building the ranked review queue, generating
rule-based next-best-action suggestions and storing human review decisions.

They are deliberately UI-free so they can be unit tested without Streamlit.
Nothing here sends messages, changes prices or writes to a CRM.
"""

from src.product.feedback import ReviewFeedbackStore
from src.product.recommendations import recommend_action
from src.product.review import (
    DataUnavailableError,
    ModelUnavailableError,
    build_queue,
    list_snapshot_dates,
    load_model_bundle,
    risk_band,
    risk_summary,
    score_accounts,
)

__all__ = [
    "DataUnavailableError",
    "ModelUnavailableError",
    "ReviewFeedbackStore",
    "build_queue",
    "list_snapshot_dates",
    "load_model_bundle",
    "recommend_action",
    "risk_band",
    "risk_summary",
    "score_accounts",
]
