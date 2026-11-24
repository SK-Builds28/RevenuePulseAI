# Phase 1 Build Notes - RevenuePulse AI

**Status:** Phase 1 complete (repository foundation + data foundation). No model,
no Streamlit UI, no recommendation engine and no LLM integration are included.

## 1. Objective

Build a reproducible foundation that can support Phase 2 model work without
leaking future information:

- configurable assumptions in `config.yaml`;
- synthetic B2B account-month data with no personal information;
- a documented data dictionary;
- point-in-time snapshots;
- a transparent revenue-decline label;
- data-quality checks with readable errors;
- automated leakage and reproducibility tests.

## 2. What was built

| Area | Files | Purpose |
|---|---|---|
| Configuration | `config.yaml`, `src/utils/config.py` | Typed access to seeds, dataset size, label rules, splits, thresholds, model names, flags and paths; cross-field validation |
| Synthetic data | `src/data/generate_synthetic.py` | Seeded generator for stable, growing, declining, seasonal and noisy accounts |
| Schema | `src/data/schema.py` | Required columns, numeric fields, non-negative rules, forbidden personal fields |
| Quality | `src/data/quality.py`, `src/data/validate_data.py` | Blocking errors and warnings, normalisation, CLI report |
| Snapshots | `src/features/snapshots.py`, `src/features/build_snapshots.py` | Quarter-end snapshots, feature allow-list, label, split assignment, time-ordering checks |
| Tests | `tests/` | 47 tests covering config, generator, quality, labels, splits, duplicate handling, leakage and documentation coverage |
| Documentation | `README.md`, `AGENTS.md`, `docs/` | Setup, status, data dictionary, decisions and limitations |

## 3. Key decisions

| Decision | Choice | Reason |
|---|---|---|
| Data grain | Account-month | Matches CRM/billing exports and supports point-in-time snapshots |
| Snapshot cadence | Quarter-end | Matches the review workflow and avoids overlapping 90-day labels |
| Prediction horizon | 3 months | Long enough for intervention, short enough to evaluate |
| Reference window | 3 months | Mirrors the future window and limits seasonal mismatch |
| Decline rule | Future 3-month mean <= 80% of trailing 3-month mean | Transparent, configurable, and not a fabricated churn flag |
| Minimum history | 6 months | Enough for a 3-vs-3 trend and a stable baseline |
| Revenue floor | 5,000 synthetic units | Avoids unstable percentage changes on tiny accounts |
| Split strategy | Time-ordered train/validation/test | Mimics deployment and prevents future-to-past leakage |
| Feature policy | Explicit allow-list | Identifiers and target-period columns cannot enter the model by accident |
| Data format | CSV | No extra parquet dependency; inspectable by faculty |
| Config format | YAML + dataclass wrapper | Readable for a capstone, no heavyweight dependency |
| Model code | Not started | Phase 2 scope |
| UI | Not started | Phase 3 scope |
| LLM | Disabled and rejected by config validation | Core capability must remain local and free |
| Opportunity scoring | Disabled | Clearly secondary and out of Phase 1 |

## 4. Assumptions

- The capstone is an individual submission, subject to faculty confirmation. The
  assignment packet says teams of 3-5; the Topic Approval Form says individual.
- All data is synthetic. No real company, customer or personal data is used.
- The unit of analysis is one B2B account over calendar months.
- The primary risk question is material **revenue decline**, not confirmed logo
  churn. There is no cancellation field in the dataset.
- A three-month forward window is a useful proxy for a quarterly account review.
- Each account has at most one row per month; duplicates are treated as errors.
- Accounts with less than six months of history are excluded from snapshots.
- A minimum trailing revenue of 5,000 synthetic units avoids percentage noise.
- Splits are defined by snapshot date, not randomly.
- Model features must be available at the snapshot date; target-period fields are
  retained for evaluation only.
- Synthetic currency units are used deliberately so no real financial result is
  implied.
- Model names in config are declarations only; Phase 2 will implement them.

## 5. Synthetic data design

Five behavioural archetypes are sampled per account:

- **stable** - flat trend, low shock hazard;
- **growing** - positive monthly trend and improving engagement;
- **declining** - negative trend, lower engagement, more support friction, higher
  shock hazard;
- **seasonal** - strong annual seasonality with a stable trend;
- **noisy** - volatile revenue and engagement with moderate decline risk.

Revenue is generated from base revenue by segment, trend, seasonality, persistent
shock events and lognormal noise. Engagement, activity recency, transactions and
support interactions are correlated with the latent state but contain noise, so
the prediction task is not trivial. Decline events are persistent (a shock
multiplier), with a 30% chance of gradual recovery after 6-13 months.

The generator does **not** emit the archetype or a pre-computed label. Labels are
derived by the snapshot builder from observed future revenue, keeping the label
rule transparent and avoiding a generator-side shortcut.

## 6. Label definition and proxy rationale

For each quarter-end snapshot `t`:

```text
reference = mean(monthly_revenue over the 3 months ending at t)
future    = mean(monthly_revenue over the 3 months after t)
label     = 1 if future <= reference * (1 - 20/100), else 0
```

Eligibility: at least 6 months of history, a complete 3-month future window, and
`reference >= 5000` synthetic units.

This is a **proxy for account risk**, not confirmed churn. A decline can come from
seasonality, billing timing, a product mix change or a genuine retention problem.
The label deliberately does not claim to identify cancellation. Phase 2 model
output must be described as a risk estimate and a prioritisation aid.

## 7. Data quality and leakage controls

### Blocking errors

- missing required columns;
- forbidden personal/identifier columns;
- blank account IDs;
- invalid dates;
- duplicate account-month rows;
- unexpected segments;
- non-numeric values;
- negative revenue or counts;
- missing mandatory fields.

### Warnings

- missing engagement or activity recency;
- insufficient history;
- calendar gaps;
- inconsistent segment over time;
- engagement outside 0-100;
- zero-revenue months;
- non-synthetic rows;
- unexpected extra columns.

### Leakage controls

1. History is filtered to `month <= snapshot_date`.
2. The label window is strictly after the snapshot.
3. The feature allow-list excludes IDs, split metadata, target fields and
   `is_synthetic`.
4. Splits are time-ordered; quarterly snapshots plus a three-month horizon avoid
   overlapping label windows.
5. Duplicate account-month and account-snapshot rows are rejected.
6. A dedicated test modifies future revenue and asserts that earlier features do
   not change; modifying past revenue does change them.
7. No global statistics or target encoding are used in feature construction.

## 8. Dataset facts (current run)

| Fact | Value |
|---|---|
| Committed demo | 60 accounts x 36 months = 2,160 rows, ~133 KB |
| Full synthetic | 400 accounts x 36 months = 14,400 rows |
| Snapshot rows | 3,921 |
| Snapshot dates | 2023-06-30 to 2025-09-30 |
| Snapshot count per account | Up to 10 quarter-ends |
| Label positives | 503 / 3,921 = **12.8%** |
| Train | 2,395 rows, 329 positives, 13.7% |
| Validation | 778 rows, 80 positives, 10.3% |
| Test | 748 rows, 94 positives, 12.6% |
| Demo snapshot rows | 594, 13.0% positive |
| Missing engagement | 411 / 14,400 (2.9%) |
| Missing activity recency | 320 / 14,400 (2.2%) |

The 12.8% prevalence is a property of this generator, not a market statistic. It
is lower than the 20-25% planning target because the 20% decline threshold is
conservative and the three-month averages smooth short shocks. It was not tuned
to make a model look good; Phase 2 will report the actual baseline and model
results.

## 9. Tests executed

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Result at Phase 1 completion:

```text
47 passed in 20.75s
```

The suite covers:

- configuration defaults, safety flags and invalid configurations;
- generator schema, reproducibility, value ranges and archetype variation;
- committed sample reproducibility;
- required columns, duplicates, invalid dates, negative revenue, unexpected
  segments, insufficient history, missing optional fields and forbidden personal
  columns;
- label correctness, configurable threshold, split ordering, target separation,
  duplicate handling and the future-revenue leakage test;
- documentation coverage for every raw field, feature and target.

## 10. Hardcoded assumptions that remain

These are intentionally visible here rather than hidden in prose:

- Archetype parameters (growth, engagement trend, support base, shock hazard,
  noise sigma, seasonal amplitude) are in `_ARCHETYPE_PARAMS`.
- Shock multiplier range is 0.58-0.82; recovery occurs after 6-13 months with 30%
  probability.
- Base revenue uses a lognormal multiplier with sigma 0.25.
- Per-account jitter for growth, engagement, support, hazard and noise is
  hardcoded.
- Synthetic currency units are used; no FX or real pricing is implied.
- Snapshot frequency is restricted to `quarter_end` by config validation.
- Feature formulas and window lengths are code-level definitions; the configured
  `feature_windows_months` is documented but the explicit formulas use 3/6/12
  months.
- The validation range for engagement is 0-100; activity recency is clipped to
  0-120 in the generator but only checked for non-negativity by the validator.
- The split dates are tied to the 36-month generated period.
- No precision@K, false-alarm or business thresholds are evaluated in Phase 1;
  the config values are hypotheses/placeholders.
- No model, baseline, calibration or performance result exists yet.

## 11. Alternatives considered

| Alternative | Why not chosen |
|---|---|
| Weekly snapshots | Overlapping 90-day labels create leakage; quarterly cadence is cleaner |
| Random train/test split | Leaks future information into the past; unacceptable for this product |
| Generator emits the label directly | Hides the label rule and makes leakage harder to audit |
| Hardcoded "churn" flag | Not supported by the data and would overstate what is measured |
| Parquet output | Adds a dependency; CSV is sufficient and inspectable |
| Pydantic configuration | Extra dependency for a small capstone; dataclass wrapper is enough |
| Streamlit UI in Phase 1 | Would distract from data correctness and violate the phase boundary |
| LLM-generated labels or summaries | Off by default and not needed for the core prediction |
| Real company data | Explicitly prohibited for this assignment |

## 12. Known limitations and ambiguities

1. **Individual vs team:** the packet specifies 3-5 students; the Topic Approval
   Form says individual. This remains subject to faculty confirmation.
2. **Environment:** the machine initially had only the Microsoft Store Python
   stub. Python 3.12.10 was installed and a local `.venv` created. Git was not
   installed at the time of this build, so no literal `git commit` exists. The
   small demo CSV is present in the deliverable and `.gitignore` is configured
   to track it while ignoring regenerated full data. Install Git and run
   `git init`, `git add .`, `git commit -m "Phase 1 data foundation"` to create
   the repository commit.
3. **Label prevalence:** 12.8%, not the planning target of 20-25%. The threshold
   was not weakened to inflate positives.
4. **Short-history nulls:** `revenue_trend_6m_pct` is null for ~20% of snapshots
   and `product_count_change_6m` for ~10%, because the first snapshots have only
   6-11 months of history. Phase 2 must impute inside a pipeline fitted on
   training data only.
5. **Zero prior support:** `support_trend_3m_pct` is null when the prior three
   months have zero support interactions; the same imputation rule applies.
6. **Monthly averages:** the label and features smooth short spikes. A one-month
   dip does not necessarily trigger the label, by design.
7. **No cancellation data:** the target is a revenue-decline proxy, not confirmed
   churn.
8. **No real-world validation:** synthetic results cannot establish real
   predictive performance, willingness to pay or retention impact.
9. **No UI/model yet:** there is nothing to demo beyond data generation,
   validation, snapshots and tests. This is intentional for Phase 1.
10. **Date parsing:** mixed-format dates are parsed with `format="mixed"` and
    invalid values are reported as errors. Production CSVs should still use
    ISO dates for clarity.

## 13. Phase 2 handoff

Phase 2 should add, without changing the data contract:

- a transparent weighted-scorecard baseline;
- logistic regression and histogram gradient boosting;
- preprocessing inside a scikit-learn pipeline, fitted on train only;
- calibration on validation;
- precision@K, recall@K, lift@K, PR-AUC, Brier score and bootstrap confidence
  intervals;
- per-segment error analysis;
- a model card with actual results, including limitations.

Phase 2 must not tune on the test set, must not hardcode metrics, and must report
the synthetic-data caveat alongside every result.
