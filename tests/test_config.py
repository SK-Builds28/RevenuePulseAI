"""Tests for configuration loading and guardrails."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from src.utils.config import ConfigError, load_config


def _write_variant(config, tmp_path: Path, mutate) -> Path:
    raw = copy.deepcopy(dict(config.raw))
    mutate(raw)
    path = tmp_path / "config_variant.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_config_loads_core_assumptions(config):
    assert config.seed == 42
    assert config.n_accounts == 400
    assert config.n_months == 36
    assert config.horizon_months == 3
    assert config.reference_window_months == 3
    assert config.decline_threshold_pct == 20.0
    assert config.minimum_history_months == 6
    assert config.train_end < config.validation_end < config.test_end
    assert config.risk_medium < config.risk_high
    assert config.primary_model == "logistic_regression"
    assert config.challenger_model == "hist_gradient_boosting"


def test_phase_one_safety_flags_are_off(config):
    assert config.llm_enabled is False
    assert config.llm_recommendations_enabled is False
    assert config.opportunity_scoring_enabled is False
    assert config.synthetic_banner_enabled is True


def test_paths_resolve_relative_to_config(config):
    assert config.sample_data_path.is_absolute()
    assert config.sample_data_path.parent.name == "sample"
    # Phase 3 prototype audit log; the path is configuration, not code.
    assert config.feedback_db_path.name == "review_feedback.db"
    assert config.feedback_db_path.parent.name == "runtime"


def test_llm_cannot_be_enabled_in_phase_one(config, tmp_path):
    path = _write_variant(
        config,
        tmp_path,
        lambda raw: raw["llm"].__setitem__("enabled", True),
    )
    with pytest.raises(ConfigError, match="LLM usage is disabled"):
        load_config(path)


def test_split_dates_must_be_ordered(config, tmp_path):
    def mutate(raw):
        raw["splits"]["train_end"] = "2025-06-30"
        raw["splits"]["validation_end"] = "2024-12-31"

    path = _write_variant(config, tmp_path, mutate)
    with pytest.raises(ConfigError, match="strictly increasing"):
        load_config(path)


def test_split_end_must_fit_generated_data(config, tmp_path):
    path = _write_variant(
        config,
        tmp_path,
        lambda raw: raw["splits"].__setitem__("test_end", "2027-12-31"),
    )
    with pytest.raises(ConfigError, match="after the generated data ends"):
        load_config(path)


def test_segment_mix_must_sum_to_one(config, tmp_path):
    def mutate(raw):
        raw["synthetic"]["segment_mix"]["SMB"] = 0.10

    path = _write_variant(config, tmp_path, mutate)
    with pytest.raises(ConfigError, match="segment_mix must sum to 1.0"):
        load_config(path)


def test_risk_thresholds_must_be_sane(config, tmp_path):
    path = _write_variant(
        config,
        tmp_path,
        lambda raw: raw["risk"].__setitem__("medium_threshold", 0.80),
    )
    with pytest.raises(ConfigError, match="risk thresholds"):
        load_config(path)


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "does_not_exist.yaml")
