# RevenuePulse AI - Streamlit Prototype

Decision-support prototype for account-level revenue-decline prioritisation.
**Synthetic data only. Human review mandatory. No automated customer action.**

## Run

From the `RevenuePulse_AI` directory:

```powershell
.\.venv\Scripts\python.exe -m streamlit run prototype/app.py
```

Requirements: `pip install -r requirements.txt`. The app needs:

- `data/processed/snapshots_full.csv` (build with `python -m src.features.build_snapshots`)
- `artifacts/model/revenuepulse_model.joblib` (build with `python -m src.models.train`)

If either is missing the page shows a controlled error with the command to rebuild it.
The review log is created on first use at `runtime/review_feedback.db` (git-ignored).

## Pages

| Page | Purpose |
|---|---|
| Executive Dashboard | Review counts by risk band, average probability, model in use, synthetic disclaimer |
| Review Queue | Ranked accounts with risk probability/band, recent revenue, trend, engagement, activity recency, primary reason, review status; filters and CSV export |
| Account Detail | Probability and band, monthly revenue chart, commercial signals, reason codes, rule-based next-best-action with evidence |
| Human Review | Accept / edit / dismiss with note, saved to SQLite with full audit context; log table and CSV export |

## Logic lives outside the UI

`prototype/app.py` only renders. The reusable logic is in `src/product/`:

- `review.py` - load snapshots and model, score, band, explain, rank, summarise
- `recommendations.py` - deterministic, evidence-linked next-best-action rules (no LLM)
- `feedback.py` - SQLite review log and CSV export

This keeps the queue, recommendation and audit rules unit-testable without Streamlit.

## Guardrails

- Scores come from the frozen model artifact; nothing is hardcoded.
- Rows whose core level fields are all missing are excluded, never assigned a fake score.
- The app cannot email, call, price, discount or write to a CRM; a test scans the
  product layer for outbound integrations.
- Medium risk is presented as "Review recommended"; the disclaimer states that the
  score is a prioritisation signal, not a churn prediction.
- No authentication, no roles, no multi-user separation: academic MVP limitation.

## Screenshots

Captures of all four pages, the reason codes and the audit log live in
[`../screenshots/`](../screenshots/).

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_product_queue.py tests/test_feedback.py tests/test_prototype_app.py
```
