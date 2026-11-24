# AGENTS.md - RevenuePulse AI

Operating rules for any AI coding agent working in this repository.

## Project context

- Individual MBA ZG583 capstone (subject to faculty confirmation).
- Product: a sales opportunity and revenue-decline risk copilot for B2B account
  teams.
- Primary decision: which accounts are most likely to show material revenue
  decline in the next 90 days, and why.
- Current phase: **Phase 2 - baseline, models, evaluation and model documentation**
  (complete). Phase 3 (UI, review queue, HITL logging) has not started.
- No real company, customer or personal data is allowed.
- All data in this repository is synthetic.

## Non-negotiable guardrails

1. **Synthetic only.** Do not add real customer, company, CRM, billing or
   personal data. Do not generate names, emails, phone numbers or addresses.
2. **No fabricated evidence.** Do not invent model scores, precision, revenue,
   customer quotes, interview results or commercial traction. Every reported
   number must come from a reproducible script or a cited source.
3. **No hardcoded results.** Model metrics and labels must be computed, not typed
   into code or documentation as if they were outputs.
4. **Point-in-time discipline.** Features for a snapshot may use only data on or
   before the snapshot date. The label uses only the future outcome window.
5. **Human review is mandatory.** The eventual product is decision support. No
   automated customer contact, pricing change or offer execution.
6. **LLM off by default.** The core prediction must work locally with
   scikit-learn. Do not add an LLM dependency unless a later phase explicitly
   asks for an optional, disabled-by-default adapter.
7. **Respect the phase boundary.** Do not build the Streamlit UI, recommendation
   engine, opportunity scoring or model training in Phase 1.
8. **Configuration over constants.** Material assumptions belong in
   `config.yaml`; document any remaining hardcoded assumption in the build notes.
9. **No secrets.** Never commit `.env`, API keys, tokens or credentials.
10. **Explain limitations.** Synthetic data cannot establish real-world
    performance. State this wherever results appear.

## Current phase boundaries

Phase 1 includes:

- `config.yaml` and its validated loader;
- synthetic account-month generation;
- data dictionary and data-quality checks;
- point-in-time snapshots, features and the revenue-decline label;
- leakage and reproducibility tests;
- documentation.

Phase 1 excluded:

- model training or evaluation;
- Streamlit pages or any UI;
- recommendation/action catalogue;
- LLM text generation;
- opportunity/expansion scoring;
- real integrations, auth or multi-tenancy.

Phase 2 (current) includes:

- the transparent weighted-scorecard baseline;
- train-fitted preprocessing, logistic regression and histogram gradient boosting;
- validation-only model selection and a single locked test evaluation;
- Precision@K, recall@K, lift@K, PR-AUC, ROC-AUC, Brier, calibration, segment and
  error analysis, and account-level bootstrap intervals;
- reason codes, a saved model artifact, model metadata and the model card.

Phase 2 excludes and Phase 3 has not started:

- Streamlit UI, review queue, account detail page;
- accept/edit/dismiss and SQLite HITL logging;
- action recommendation engine;
- CRM integration, LLM functionality, email generation, pricing automation.

Phase 3 begins only when the user explicitly asks for it.

## Environment

- Windows 10/11, PowerShell.
- Python 3.12 in `.venv`.
- Use `.\.venv\Scripts\python.exe` directly; the global `python` command may be
  the Microsoft Store stub.
- Git may not be installed; do not assume it exists.

## Commands

Run from the `RevenuePulse_AI` directory.

```powershell
# Install
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Generate the committed demo dataset
.\.venv\Scripts\python.exe -m src.data.generate_synthetic --demo

# Generate the full synthetic dataset
.\.venv\Scripts\python.exe -m src.data.generate_synthetic

# Validate raw data
.\.venv\Scripts\python.exe -m src.data.validate_data --input data/sample/account_month_sample.csv

# Build snapshots
.\.venv\Scripts\python.exe -m src.features.build_snapshots --input data/sample/account_month_sample.csv --output data/processed/snapshots_sample.csv

# Train, select on validation, freeze the model
.\.venv\Scripts\python.exe -m src.models.train

# Single locked test evaluation
.\.venv\Scripts\python.exe -m src.evaluation.evaluate

# Tests
.\.venv\Scripts\python.exe -m pytest
```

## Architecture conventions

- `src/data/` - schema, ingestion, quality, synthetic generation.
- `src/features/` - as-of snapshots, feature allow-list, label, split assignment.
- `src/models/` - baseline, preprocessing, training and reason codes.
- `src/evaluation/` - metrics, bootstrap and the locked evaluation CLI.
- `artifacts/` - generated model and evaluation outputs; git-ignored, reproducible.
- `src/utils/` - configuration and shared helpers.
- `tests/` - pytest; tests must not require network access.
- `docs/` - Markdown, GitHub-readable.
- `prototype/` - Phase 3 only.
- `runtime/` - local SQLite and logs; git-ignored.

Prefer small pure functions with docstrings. Keep Streamlit out of domain logic.
Use CSV for inspectability; do not add parquet/pyarrow without a clear reason.

## Data contract

Raw grain: one row per `account_id` per `month`.

Required columns:

`account_id`, `month`, `segment`, `monthly_revenue`, `transaction_count`,
`engagement_score`, `days_since_last_activity`, `product_count`,
`support_interactions`, `contract_age_months`.

Optional: `is_synthetic`.

Do not add account names or any personal/company identifiers. The data
dictionary in `docs/Data_Dictionary.md` is the source of truth.

## Modelling rules for Phase 2

- Baseline first: transparent weighted scorecard.
- Models: logistic regression and histogram gradient boosting.
- Preprocessing inside a scikit-learn pipeline, fitted on train only.
- Time-based train/validation/test; never a random split.
- Calibrate on validation; evaluate on test once.
- Report PR-AUC, precision@K, recall@K, lift@K, Brier score and bootstrap
  confidence intervals, plus per-segment results.
- Do not claim real-world performance from synthetic data.
- Do not tune the test set or change gates after seeing results.
- Keep the feature allow-list in `src/features/snapshots.py` authoritative.

## Definition of done for a change

1. The code runs from the documented command.
2. Relevant tests pass; add tests for new invariants.
3. No future data enters features.
4. No fabricated metrics or claims.
5. `config.yaml` remains the place for assumptions.
6. Documentation reflects the actual implementation.
7. Phase boundaries are respected.

If a requirement conflicts with these rules, stop and ask the user rather than
silently weakening a guardrail.
