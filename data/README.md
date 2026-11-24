# Data Folder

## What is here

| Path | Committed? | Purpose |
|---|---|---|
| `sample/account_month_sample.csv` | Yes | 60-account x 36-month demonstration dataset for demos, tests and code review |
| `sample/` future files | Yes, if small | Additional tiny fixtures only |
| `processed/account_month_full.csv` | No (git-ignored) | 400-account x 36-month dataset regenerated from the configured seed |
| `processed/snapshots*.csv` | No (git-ignored) | Point-in-time snapshots built from a raw file |

The full dataset is not committed because it is fully reproducible from
`config.yaml` and the generator.

## Synthetic-data disclaimer

Every row is generated. There are no real companies, customers, people, emails,
phone numbers or financial results in this folder. Revenue values are in
synthetic currency units. Do not present them as market data.

## Grain and schema

One row per `account_id` per `month`.

Required columns:

- `account_id` - synthetic code such as `SYN-0042`
- `month` - first day of the month
- `segment` - `SMB`, `Mid-Market` or `Enterprise`
- `monthly_revenue` - synthetic revenue for the month
- `transaction_count` - transactions/orders in the month
- `engagement_score` - 0-100 or null
- `days_since_last_activity` - 0-120 or null
- `product_count` - active products
- `support_interactions` - support contacts in the month
- `contract_age_months` - months since contract start

Optional:

- `is_synthetic` - safety marker, always true in Phase 1

See `docs/Data_Dictionary.md` for definitions, generation logic and leakage
notes.

## Regenerate

From the project root:

```powershell
# Small committed demo (60 accounts x 36 months)
.\.venv\Scripts\python.exe -m src.data.generate_synthetic --demo

# Full synthetic dataset (400 accounts x 36 months)
.\.venv\Scripts\python.exe -m src.data.generate_synthetic
```

The demo uses seed `20260101`; the full dataset uses the global seed `42`. Both
are configured in `config.yaml`.

## Validate

```powershell
.\.venv\Scripts\python.exe -m src.data.validate_data --input data/sample/account_month_sample.csv
```

Blocking errors include duplicate account-months, invalid dates, negative
revenue, forbidden personal columns and unexpected segments. Missing engagement
or activity recency is a warning because the snapshot builder exposes a
missingness feature.

## Build snapshots

```powershell
.\.venv\Scripts\python.exe -m src.features.build_snapshots `
  --input data/sample/account_month_sample.csv `
  --output data/processed/snapshots_sample.csv
```

Snapshots are quarter-end and use only history up to the snapshot date. The label
uses the next three months and is never a feature.
