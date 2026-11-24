"""Generate reproducible synthetic B2B account-month data.

The generator creates five behavioural archetypes (stable, growing, declining,
seasonal, noisy), adds persistent revenue shocks and measurement noise, and
never emits names, emails, phone numbers or real company identifiers.

Run from the project root:

    python -m src.data.generate_synthetic --demo
    python -m src.data.generate_synthetic

The second command regenerates the full configured dataset with the configured
random seed. No real customer data is used anywhere in this project.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.schema import RAW_COLUMNS
from src.utils.config import Config, load_config

_ARCHETYPE_PARAMS: dict[str, dict[str, float]] = {
    "stable": {
        "growth": 0.0008,
        "engagement_base": 72.0,
        "engagement_trend": 0.0,
        "support_base": 0.55,
        "hazard": 0.008,
        "noise_sigma": 0.08,
        "seasonal_amplitude": 0.07,
    },
    "growing": {
        "growth": 0.012,
        "engagement_base": 74.0,
        "engagement_trend": 0.45,
        "support_base": 0.40,
        "hazard": 0.002,
        "noise_sigma": 0.08,
        "seasonal_amplitude": 0.06,
    },
    "declining": {
        "growth": -0.030,
        "engagement_base": 62.0,
        "engagement_trend": -0.85,
        "support_base": 1.20,
        "hazard": 0.100,
        "noise_sigma": 0.10,
        "seasonal_amplitude": 0.08,
    },
    "seasonal": {
        "growth": 0.0,
        "engagement_base": 70.0,
        "engagement_trend": 0.0,
        "support_base": 0.60,
        "hazard": 0.020,
        "noise_sigma": 0.08,
        "seasonal_amplitude": 0.24,
    },
    "noisy": {
        "growth": -0.002,
        "engagement_base": 66.0,
        "engagement_trend": -0.20,
        "support_base": 0.90,
        "hazard": 0.050,
        "noise_sigma": 0.18,
        "seasonal_amplitude": 0.10,
    },
}


def _normalise_probabilities(values: dict[str, float]) -> tuple[list[str], np.ndarray]:
    keys = list(values)
    probabilities = np.array([values[key] for key in keys], dtype=float)
    if probabilities.sum() <= 0:
        raise ValueError("Probabilities must sum to a positive value")
    return keys, probabilities / probabilities.sum()


def generate_monthly_accounts(
    config: Config,
    n_accounts: int | None = None,
    n_months: int | None = None,
    seed: int | None = None,
    start_month: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return a tidy account-month dataframe with no personal data.

    Args:
        config: Validated project configuration.
        n_accounts: Optional override for the number of synthetic accounts.
        n_months: Optional override for months per account.
        seed: Optional override for the NumPy random seed.
        start_month: Optional first month (``YYYY-MM`` or timestamp).
    """

    account_count = int(n_accounts if n_accounts is not None else config.n_accounts)
    month_count = int(n_months if n_months is not None else config.n_months)
    random_seed = int(seed if seed is not None else config.seed)
    start = pd.Timestamp(start_month if start_month is not None else config.start_month)
    start = start.to_period("M").to_timestamp()

    if account_count <= 0:
        raise ValueError("n_accounts must be positive")
    if month_count <= 0:
        raise ValueError("n_months must be positive")

    rng = np.random.default_rng(random_seed)
    months = pd.date_range(start=start, periods=month_count, freq="MS")

    segment_names, segment_probabilities = _normalise_probabilities(config.segment_mix)
    archetype_names, archetype_probabilities = _normalise_probabilities(config.archetype_mix)

    missing_engagement = float(config.missing_rates.get("engagement_score", 0.0))
    missing_activity = float(config.missing_rates.get("days_since_last_activity", 0.0))

    rows: list[dict[str, object]] = []

    for account_index in range(account_count):
        account_id = f"SYN-{account_index + 1:04d}"
        segment = str(rng.choice(segment_names, p=segment_probabilities))
        archetype = str(rng.choice(archetype_names, p=archetype_probabilities))
        params = _ARCHETYPE_PARAMS[archetype]

        base_revenue = float(config.segment_base_revenue[segment]) * float(
            rng.lognormal(mean=0.0, sigma=0.25)
        )
        growth = float(params["growth"]) + float(rng.normal(0.0, 0.0035))
        engagement_base = float(params["engagement_base"]) + float(rng.normal(0.0, 7.0))
        engagement_trend = float(params["engagement_trend"]) + float(rng.normal(0.0, 0.10))
        support_base = float(params["support_base"]) * float(rng.lognormal(0.0, 0.25))
        hazard = float(params["hazard"]) * float(rng.lognormal(0.0, 0.45))
        noise_sigma = float(params["noise_sigma"]) * float(rng.lognormal(0.0, 0.18))
        seasonal_amplitude = float(params["seasonal_amplitude"]) + float(
            rng.uniform(-0.02, 0.02)
        )
        seasonal_phase = float(rng.uniform(0.0, 12.0))
        avg_transaction_value = float(config.segment_avg_transaction_value[segment])
        contract_age_start = int(rng.integers(1, 49))
        product_count = int(rng.integers(1, 6))

        shock_multiplier = 1.0
        recovery_month: int | None = None

        for month_index, month in enumerate(months):
            month_number = month.month
            seasonal_factor = 1.0 + seasonal_amplitude * float(
                np.sin(2.0 * np.pi * (month_number - seasonal_phase) / 12.0)
            )
            trend_factor = (1.0 + growth) ** month_index

            # Persistent decline events create genuine future revenue drops.
            if recovery_month is not None and month_index >= recovery_month:
                shock_multiplier = min(1.0, shock_multiplier * 1.18)
                if shock_multiplier >= 0.98:
                    shock_multiplier = 1.0
                    recovery_month = None

            if shock_multiplier == 1.0 and rng.random() < hazard:
                shock_multiplier = float(rng.uniform(0.58, 0.82))
                if rng.random() < 0.30:
                    recovery_month = month_index + int(rng.integers(6, 13))

            revenue_noise = float(rng.lognormal(mean=0.0, sigma=noise_sigma))
            revenue = max(
                0.0,
                base_revenue
                * trend_factor
                * seasonal_factor
                * shock_multiplier
                * revenue_noise,
            )

            engagement = (
                engagement_base
                + engagement_trend * month_index
                + 5.0 * float(np.sin(2.0 * np.pi * month_number / 12.0))
                + float(rng.normal(0.0, 6.0))
                - 32.0 * (1.0 - shock_multiplier)
            )
            engagement = float(np.clip(engagement, 0.0, 100.0))

            activity_scale = 2.0 + (100.0 - engagement) / 5.0
            days_since_activity = float(rng.exponential(activity_scale))
            days_since_activity = float(np.clip(days_since_activity, 0.0, 120.0))

            expected_transactions = revenue / max(avg_transaction_value, 1.0)
            transaction_count = int(rng.poisson(max(expected_transactions, 0.0)))

            if archetype == "declining" and rng.random() < 0.02:
                product_count = max(1, product_count - 1)
            elif archetype == "growing" and rng.random() < 0.015:
                product_count = min(8, product_count + 1)
            elif rng.random() < 0.005:
                product_count = int(np.clip(product_count + rng.choice([-1, 1]), 1, 8))

            support_lambda = (
                support_base
                * (1.0 + max(0.0, 65.0 - engagement) / 35.0)
                * (1.0 + 0.35 * (1.0 - shock_multiplier))
            )
            support_interactions = int(rng.poisson(max(support_lambda, 0.0)))

            if rng.random() < missing_engagement:
                engagement_value: float | None = None
            else:
                engagement_value = round(engagement, 2)

            if rng.random() < missing_activity:
                activity_value: float | None = None
            else:
                activity_value = round(days_since_activity, 1)

            rows.append(
                {
                    "account_id": account_id,
                    "month": month,
                    "segment": segment,
                    "monthly_revenue": round(revenue, 2),
                    "transaction_count": transaction_count,
                    "engagement_score": engagement_value,
                    "days_since_last_activity": activity_value,
                    "product_count": product_count,
                    "support_interactions": support_interactions,
                    "contract_age_months": contract_age_start + month_index,
                    "is_synthetic": True,
                }
            )

    frame = pd.DataFrame(rows, columns=list(RAW_COLUMNS))
    frame["month"] = pd.to_datetime(frame["month"]).dt.to_period("M").dt.to_timestamp()
    frame["segment"] = frame["segment"].astype("string")
    frame["account_id"] = frame["account_id"].astype("string")
    frame = frame.sort_values(["account_id", "month"], kind="mergesort").reset_index(drop=True)
    return frame


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate the small committed demonstration dataset instead of the full dataset.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional output CSV path.")
    parser.add_argument("--accounts", type=int, default=None, help="Override account count.")
    parser.add_argument("--months", type=int, default=None, help="Override month count.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)

    if args.demo:
        n_accounts = args.accounts if args.accounts is not None else config.demo_n_accounts
        n_months = args.months if args.months is not None else config.demo_n_months
        seed = args.seed if args.seed is not None else config.demo_seed
        output = args.output if args.output is not None else config.demo_output_path
    else:
        n_accounts = args.accounts if args.accounts is not None else config.n_accounts
        n_months = args.months if args.months is not None else config.n_months
        seed = args.seed if args.seed is not None else config.seed
        output = args.output if args.output is not None else config.full_data_path

    frame = generate_monthly_accounts(
        config, n_accounts=n_accounts, n_months=n_months, seed=seed
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, date_format="%Y-%m-%d")

    print(f"Wrote {len(frame):,} rows for {frame['account_id'].nunique():,} accounts")
    print(
        f"Period: {frame['month'].min().date()} to {frame['month'].max().date()} "
        f"({frame['month'].nunique()} months)"
    )
    print(f"Seed: {seed}")
    print(f"Output: {output}")
    print("Synthetic data only - not real company or customer data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
