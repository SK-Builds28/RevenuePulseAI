"""Schema constants for the synthetic account-month dataset.

The raw dataset is deliberately narrow and contains no personal data. The
snapshot builder derives model features and labels from these columns.
"""

from __future__ import annotations

RAW_REQUIRED_COLUMNS: tuple[str, ...] = (
    "account_id",
    "month",
    "segment",
    "monthly_revenue",
    "transaction_count",
    "engagement_score",
    "days_since_last_activity",
    "product_count",
    "support_interactions",
    "contract_age_months",
)

RAW_OPTIONAL_COLUMNS: tuple[str, ...] = ("is_synthetic",)

RAW_COLUMNS: tuple[str, ...] = RAW_REQUIRED_COLUMNS + RAW_OPTIONAL_COLUMNS

NUMERIC_COLUMNS: tuple[str, ...] = (
    "monthly_revenue",
    "transaction_count",
    "engagement_score",
    "days_since_last_activity",
    "product_count",
    "support_interactions",
    "contract_age_months",
)

INTEGER_COLUMNS: tuple[str, ...] = (
    "transaction_count",
    "product_count",
    "support_interactions",
    "contract_age_months",
)

NON_NEGATIVE_COLUMNS: tuple[str, ...] = (
    "monthly_revenue",
    "transaction_count",
    "days_since_last_activity",
    "product_count",
    "support_interactions",
    "contract_age_months",
)

MISSING_VALUE_TOLERATED_COLUMNS: tuple[str, ...] = (
    "engagement_score",
    "days_since_last_activity",
)

FORBIDDEN_IDENTIFIER_COLUMNS: tuple[str, ...] = (
    "account_name",
    "company_name",
    "customer_name",
    "email",
    "phone",
    "contact_name",
    "address",
)
