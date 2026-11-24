"""Shared pytest fixtures for Phase 1."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.generate_synthetic import generate_monthly_accounts
from src.data.quality import load_monthly_data
from src.utils.config import Config, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def config(project_root: Path) -> Config:
    return load_config(project_root / "config.yaml")


@pytest.fixture(scope="session")
def sample_monthly(config: Config) -> pd.DataFrame:
    path = config.sample_data_path
    if not path.exists():
        pytest.skip(f"Committed sample dataset not found: {path}")
    frame, _ = load_monthly_data(path, config, strict=True)
    return frame


@pytest.fixture(scope="session")
def small_monthly(config: Config) -> pd.DataFrame:
    """A small deterministic cohort used by snapshot and quality tests."""

    return generate_monthly_accounts(
        config, n_accounts=60, n_months=config.n_months, seed=123
    )


@pytest.fixture(scope="session")
def full_monthly(config: Config) -> pd.DataFrame:
    return generate_monthly_accounts(
        config, n_accounts=config.n_accounts, n_months=config.n_months, seed=config.seed
    )
