"""Build and save point-in-time snapshots from a monthly account CSV.

Usage from the project root:

    python -m src.features.build_snapshots \
        --input data/sample/account_month_sample.csv \
        --output data/processed/snapshots_sample.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.quality import load_monthly_data
from src.features.snapshots import (
    assert_time_ordering,
    build_snapshots,
    label_prevalence,
)
from src.utils.config import load_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--input", type=Path, default=None, help="Raw account-month CSV")
    parser.add_argument("--output", type=Path, default=None, help="Snapshot CSV output path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)

    input_path = args.input or config.sample_data_path
    output_path = args.output or config.snapshots_path

    monthly, quality = load_monthly_data(input_path, config, strict=True)
    print(quality.summary())
    if quality.warnings:
        print("Warnings do not block snapshot construction.")

    snapshots = build_snapshots(monthly, config, validate=False)
    assert_time_ordering(snapshots)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshots.to_csv(output_path, index=False, date_format="%Y-%m-%d")

    print("")
    print(f"Snapshots: {len(snapshots):,} rows")
    print(f"Accounts: {snapshots['account_id'].nunique() if not snapshots.empty else 0:,}")
    print(
        "Snapshot dates: "
        f"{snapshots['snapshot_date'].min().date() if not snapshots.empty else 'n/a'} to "
        f"{snapshots['snapshot_date'].max().date() if not snapshots.empty else 'n/a'}"
    )
    if not snapshots.empty:
        split_counts = snapshots["split"].value_counts().to_dict()
        print(f"Split rows: {split_counts}")
    prevalence = label_prevalence(snapshots, config.target_column)
    print(f"Label prevalence ({config.target_column}): {prevalence:.3f}" if prevalence == prevalence else "Label prevalence: n/a")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
