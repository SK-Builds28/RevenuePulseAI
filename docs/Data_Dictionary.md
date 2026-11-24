# Data Dictionary - RevenuePulse AI (Phase 1)

This dictionary covers the synthetic account-month dataset, the point-in-time
snapshot fields, the model-feature allow-list, and the proxy target. It is the
reference for Phase 2 model work.

## Dataset metadata

| Property | Value |
|---|---|
| Grain | One row per `account_id` per calendar `month` |
| Committed demo | `data/sample/account_month_sample.csv` - 60 accounts x 36 months = 2,160 rows (~133 KB) |
| Full synthetic | `data/processed/account_month_full.csv` - 400 accounts x 36 months = 14,400 rows (regenerated, not committed) |
| Period | 2023-01-01 to 2025-12-01 |
| Currency | Synthetic currency units; not real INR, USD or company revenue |
| Personal data | None. No names, emails, phone numbers, addresses or real company identifiers |
| Generator | `src/data/generate_synthetic.py`, seed-controlled via `config.yaml` |
| Label prevalence | 12.8% of 3,921 point-in-time snapshots (synthetic; not a real-world claim) |

## Raw monthly fields

`Known at prediction time?` means the field is available at the snapshot date
when only rows with `month <= snapshot_date` are used. Using future rows for any
of these fields is leakage.

| Field | Type | Definition | Example | Known at prediction time? | Leakage risk | Synthetic generation |
|---|---|---|---|---|---|---|
| `account_id` | string | Synthetic account identifier. Never a model feature. | `SYN-0042` | Yes | Identifier only; must be excluded from features | Sequential `SYN-0001` ... `SYN-0400` |
| `month` | date | First day of the calendar month represented by the row. | `2024-03-01` | Yes for months up to the snapshot | High if future rows enter the feature window | Monthly grid from `start_month` for `n_months` |
| `segment` | category | Synthetic customer segment. One of `SMB`, `Mid-Market`, `Enterprise`. | `Mid-Market` | Yes | Low; static account attribute | Sampled from `synthetic.segment_mix` |
| `monthly_revenue` | float | Revenue recognised from the account in that month, in synthetic currency units. | `48250.00` | Yes for the snapshot month and earlier | **Critical** if future months are included | Base revenue x trend x seasonality x persistent shock x lognormal noise |
| `transaction_count` | integer | Number of commercial transactions/orders in the month. | `27` | Yes for the snapshot month and earlier | High if future months are included | Poisson with mean = revenue / segment average transaction value |
| `engagement_score` | float (0-100) or null | Composite product/engagement signal. Higher is better. | `64.5` | Yes for the snapshot month and earlier | High if future months are included; missingness itself is informative | Latent engagement + trend + seasonality + noise - shock penalty; 3% missing |
| `days_since_last_activity` | float (0-120) or null | Days since the last recorded account activity at month end. | `18.0` | Yes for the snapshot month and earlier | High if future months are included | Exponential draw whose scale increases as engagement falls; 2% missing |
| `product_count` | integer | Number of products/services active for the account. | `3` | Yes for the snapshot month and earlier | High if future months are included | Initial 1-5; small probability of change; declining accounts more likely to lose a product |
| `support_interactions` | integer | Support contacts or tickets in the month. | `2` | Yes for the snapshot month and earlier | High if future months are included | Poisson driven by support base, low engagement and shock state |
| `contract_age_months` | integer | Months since contract start at the row month. | `19` | Yes | Low; deterministic time feature | Random starting age 1-48 plus month index |
| `is_synthetic` | boolean | Safety marker confirming the row is generated, not real. | `True` | Yes | None; must be excluded from features | Always `True` in Phase 1 |

## Snapshot metadata fields

These fields identify a prediction point and its evaluation split. They are
**not** model inputs.

| Field | Type | Definition | Example | Known at prediction time? | Leakage risk |
|---|---|---|---|---|---|
| `snapshot_id` | string | `account_id` + snapshot date. | `SYN-0042_2024-03-31` | Yes | Identifier; exclude |
| `account_id` | string | Synthetic account identifier. | `SYN-0042` | Yes | Identifier; exclude |
| `snapshot_date` | date | Quarter-end date through which history is observed. | `2024-03-31` | Yes | Metadata; exclude |
| `split` | category | `train`, `validation` or `test`, assigned by snapshot date. | `train` | Yes | Evaluation metadata; exclude |
| `outcome_start_month` | date | First month of the future label window. | `2024-04-01` | No | **Target-period metadata**; exclude |
| `outcome_end_month` | date | Last month of the future label window. | `2024-06-01` | No | **Target-period metadata**; exclude |
| `label_eligible` | boolean | Whether the row has enough complete future history and meets the revenue floor. | `True` | No (audit only) | Exclude |

## Model-feature allow-list

The explicit allow-list lives in `src/features/snapshots.py`. Any column not on
this list is excluded from model training. All features are computed from rows
with `month <= snapshot_date`.

| Feature | Type | Definition / formula | Known at prediction time? | Leakage risk | Rationale |
|---|---|---|---|---|---|
| `segment` | category | Account segment at the snapshot. | Yes | Low | Captures structural differences in revenue and behaviour |
| `snapshot_month` | integer | Calendar month number (1-12) of the snapshot. | Yes | Low | Lets the model account for recurring seasonality |
| `history_months` | integer | Number of monthly rows available up to the snapshot. | Yes | Low | Data-availability control and short-history signal |
| `revenue_mean_3m` | float | Mean `monthly_revenue` over the latest 3 months. | Yes | Low if as-of | Current revenue scale |
| `revenue_mean_6m` | float | Mean monthly revenue over the latest 6 months. | Yes | Low if as-of | Medium-term baseline |
| `revenue_mean_12m` | float | Mean monthly revenue over the latest 12 available months. | Yes | Low if as-of | Long-term baseline and seasonality smoothing |
| `revenue_trend_3m_pct` | float | `(mean last 3m - mean prior 3m) / mean prior 3m`. | Yes | Low if as-of | Short-term deterioration signal |
| `revenue_trend_6m_pct` | float | `(mean last 6m - mean prior 6m) / mean prior 6m`; null when less than 12 months exist. | Yes | Low if as-of | Medium-term trend |
| `revenue_volatility_6m` | float | Standard deviation / mean of the latest 6 months of revenue. | Yes | Low if as-of | Distinguishes stable accounts from erratic ones |
| `transaction_mean_3m` | float | Mean transaction count over the latest 3 months. | Yes | Low if as-of | Commercial activity intensity |
| `transaction_trend_3m_pct` | float | 3-month transaction mean vs prior 3-month mean. | Yes | Low if as-of | Order-frequency deterioration |
| `revenue_per_transaction_3m` | float | Latest 3-month revenue sum / latest 3-month transaction sum. | Yes | Low if as-of | Mix shift and average deal size |
| `engagement_mean_3m` | float | Mean `engagement_score` over the latest 3 months. | Yes | Low if as-of | Product adoption and relationship strength |
| `engagement_trend_3m_pct` | float | 3-month engagement mean vs prior 3-month mean. | Yes | Low if as-of | Early disengagement |
| `engagement_missing_ratio_3m` | float | Share of the latest 3 months with missing engagement data. | Yes | Low if as-of | Data-quality control; missingness can itself be informative |
| `days_since_activity_last` | float | Most recent non-null `days_since_last_activity` at the snapshot. | Yes | Low if as-of | Recency of contact |
| `days_since_activity_mean_3m` | float | Mean activity recency over the latest 3 months. | Yes | Low if as-of | Sustained inactivity |
| `product_count_last` | float | Product count in the snapshot month. | Yes | Low if as-of | Breadth of the commercial relationship |
| `product_count_change_6m` | float | Current product count minus product count six months earlier; null when insufficient history. | Yes | Low if as-of | Product contraction/expansion |
| `support_interactions_sum_3m` | float | Sum of support interactions over the latest 3 months. | Yes | Low if as-of | Friction signal |
| `support_trend_3m_pct` | float | 3-month support sum vs prior 3-month sum; null when prior sum is zero. | Yes | Low if as-of | Rising friction |
| `contract_age_months` | float | Contract age at the snapshot. | Yes | Low | Lifecycle stage and renewal proximity proxy |

## Target and audit fields

These fields are **never** model features. They are computed from rows after the
snapshot date and are retained for evaluation and audit only.

| Field | Type | Definition | Known at prediction time? | Leakage risk |
|---|---|---|---|---|
| `reference_revenue_mean` | float | Mean `monthly_revenue` over the latest 3 months up to the snapshot. | Yes, but it is a label anchor and is excluded from features to avoid target leakage | Critical if used as a feature |
| `future_revenue_mean` | float | Mean `monthly_revenue` over the 3 months after the snapshot. | No | **Critical**; target-period variable |
| `future_revenue_change_pct` | float | `(future_revenue_mean - reference_revenue_mean) / reference_revenue_mean * 100`. | No | **Critical**; target-period variable |
| `future_revenue_decline` | integer (0/1) or null | 1 when the future mean falls by at least the configured percentage versus the reference mean, subject to the revenue floor. | No | **Critical**; this is the target |

## Label definition

For a snapshot at quarter-end date `t`:

```text
reference_revenue_mean = mean monthly_revenue over the latest 3 months ending at t
future_revenue_mean    = mean monthly_revenue over the next 3 months after t
future_revenue_decline = 1 if future_revenue_mean <= reference_revenue_mean * (1 - 20/100)
                         else 0
```

Eligibility rules:

- At least 6 months of history.
- A complete 3-month future window.
- `reference_revenue_mean >= 5000` synthetic currency units, to avoid noise from
  very small accounts.

All values are configurable in `config.yaml` under `label`.

### Why this is a proxy, not confirmed churn

The target measures a **material reduction in revenue**, not a cancellation,
contract termination or logo churn event. An account can trigger the label
because of seasonality, a one-off billing change, a delayed purchase, a product
mix shift or a genuine retention problem. The dataset does not contain a CRM
cancellation field. The report and UI must therefore say "revenue-decline risk"
rather than "confirmed churn". In Phase 2 the model output will be described as a
risk estimate and a prioritisation aid, not a churn fact.

## Data-quality rules

| Code | Severity | Meaning |
|---|---|---|
| `missing_required_columns` | error | A required raw column is absent |
| `forbidden_personal_columns` | error | A name, email, phone or address-like column is present |
| `missing_account_id` | error | Identifier is blank |
| `invalid_month` | error | Month cannot be parsed as a date |
| `duplicate_account_month` | error | More than one row for an account-month |
| `unexpected_segment` | error | Segment outside the configured categories |
| `negative_*` | error | A non-negative numeric field is negative |
| `non_numeric_*` | error | A numeric column contains text or invalid values |
| `missing_required_*` | error | A mandatory field has missing values |
| `missing_optional_engagement_score` | warning | Engagement is missing; a missingness feature is exposed |
| `missing_optional_days_since_last_activity` | warning | Activity recency is missing |
| `insufficient_history` | warning | Account has fewer months than the minimum; it will be excluded |
| `calendar_gaps` | warning | An account is missing calendar months |
| `inconsistent_segment` | warning | Segment changes within an account; latest value is used |
| `engagement_out_of_range` | warning | Engagement outside 0-100 |
| `zero_revenue_months` | warning | Revenue is exactly zero; confirm it is not missing data |
| `non_synthetic_rows` | warning | Rows not marked `is_synthetic=True` in a Phase 1 dataset |
| `unexpected_columns` | warning | Extra columns are ignored |

## Leakage controls

1. **As-of filtering.** Features use only rows with `month <= snapshot_date`.
2. **Separate outcome window.** The label uses only rows after the snapshot date.
3. **Explicit allow-list.** `FEATURE_COLUMNS` in `src/features/snapshots.py` is an
   allow-list; identifiers, split metadata, target-period fields and
   `is_synthetic` are excluded.
4. **Time-ordered splits.** `train` snapshots end before `validation`, which ends
   before `test`; the snapshot cadence is quarterly and the horizon is three
   months, so label windows do not overlap across splits.
5. **Duplicate protection.** Duplicate account-month rows fail validation;
   duplicate account/snapshot rows fail the snapshot build.
6. **Automated leakage tests.** `tests/test_snapshots.py` changes future revenue
   and asserts that earlier features are unchanged; changing past revenue does
   change them.
7. **No target encoding or global aggregates.** Features are computed per account
   from that account's history only.

## Synthetic-data limitations

- These are generated behaviours, not observed customer behaviour.
- The generator embeds assumptions about seasonality, decline events, engagement
  and support. Real data will differ.
- The 12.8% label prevalence is a property of this generator, not a market
  statistic.
- The label is a revenue-decline proxy, not confirmed churn.
- Monthly averages smooth short spikes and troughs.
- Some trailing features are intentionally null for short histories; Phase 2
  must impute them inside the model pipeline.
- No causal claim is possible: a flagged account is not proven to have been saved
  by an intervention.
- `data/processed/account_month_full.csv` is regenerated and git-ignored; only
  the small sample is committed.
