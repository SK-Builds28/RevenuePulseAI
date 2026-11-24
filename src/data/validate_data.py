"""Validate a raw account-month CSV without building snapshots.

Usage:

    python -m src.data.validate_data --input data/sample/account_month_sample.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.quality import validate_monthly_data
from src.utils.config import load_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Raw account-month CSV (defaults to configured sample data).",
    )
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)
    path = args.input or config.sample_data_path

    if not path.exists():
        print(f"File not found: {path}")
        return 2

    frame = pd.read_csv(path)
    report = validate_monthly_data(frame, config)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.summary())
    return 1 if report.has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
