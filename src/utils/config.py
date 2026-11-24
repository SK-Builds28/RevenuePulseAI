"""Configuration loading and validation.

Phase 1 keeps every material assumption in ``config.yaml``. Code reads values
through this module so experiments can be reproduced by changing one file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml


class ConfigError(ValueError):
    """Raised when configuration is missing or internally inconsistent."""


_MISSING = object()

_REQUIRED_SECTIONS = (
    "project",
    "random",
    "synthetic",
    "label",
    "snapshots",
    "splits",
    "risk",
    "evaluation",
    "reasons",
    "baseline",
    "models",
    "features",
    "llm",
    "feedback",
    "artifacts",
    "paths",
)

_SUPPORTED_ARCHETYPES = {"stable", "growing", "declining", "seasonal", "noisy"}


def _as_timestamp(value: Any, key: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(value).normalize()
    except Exception as exc:  # pragma: no cover - defensive
        raise ConfigError(f"{key} must be a valid date, got {value!r}") from exc


def _as_month_start(value: Any, key: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(value).to_period("M").to_timestamp()
    except Exception as exc:  # pragma: no cover - defensive
        raise ConfigError(f"{key} must be a valid month, got {value!r}") from exc


@dataclass(frozen=True)
class Config:
    """Read-only view over the YAML configuration."""

    raw: Mapping[str, Any]
    path: Path

    @property
    def project_root(self) -> Path:
        return self.path.parent

    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted_key.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted_key: str) -> Any:
        value = self.get(dotted_key, _MISSING)
        if value is _MISSING:
            raise ConfigError(f"Missing required configuration key: {dotted_key}")
        return value

    def resolve_path(self, path_value: str) -> Path:
        path = Path(path_value)
        return path if path.is_absolute() else self.project_root / path

    # -- frequently used, typed accessors ---------------------------------
    @property
    def project_version(self) -> str:
        return str(self.require("project.version"))

    @property
    def seed(self) -> int:
        return int(self.require("random.seed"))

    @property
    def n_accounts(self) -> int:
        return int(self.require("synthetic.n_accounts"))

    @property
    def n_months(self) -> int:
        return int(self.require("synthetic.n_months"))

    @property
    def start_month(self) -> pd.Timestamp:
        return _as_month_start(self.require("synthetic.start_month"), "synthetic.start_month")

    @property
    def segments(self) -> tuple[str, ...]:
        return tuple(str(s) for s in self.require("synthetic.segment_mix").keys())

    @property
    def segment_mix(self) -> dict[str, float]:
        return {str(k): float(v) for k, v in self.require("synthetic.segment_mix").items()}

    @property
    def segment_base_revenue(self) -> dict[str, float]:
        return {
            str(k): float(v)
            for k, v in self.require("synthetic.segment_base_revenue").items()
        }

    @property
    def segment_avg_transaction_value(self) -> dict[str, float]:
        return {
            str(k): float(v)
            for k, v in self.require("synthetic.segment_avg_transaction_value").items()
        }

    @property
    def archetype_mix(self) -> dict[str, float]:
        return {str(k): float(v) for k, v in self.require("synthetic.archetype_mix").items()}

    @property
    def missing_rates(self) -> dict[str, float]:
        return {
            str(k): float(v)
            for k, v in self.require("synthetic.missing_rates").items()
        }

    @property
    def demo_n_accounts(self) -> int:
        return int(self.require("synthetic.demonstration.n_accounts"))

    @property
    def demo_n_months(self) -> int:
        return int(self.require("synthetic.demonstration.n_months"))

    @property
    def demo_seed(self) -> int:
        return int(self.require("synthetic.demonstration.seed"))

    @property
    def demo_output_path(self) -> Path:
        return self.resolve_path(str(self.require("synthetic.demonstration.output_path")))

    @property
    def horizon_months(self) -> int:
        return int(self.require("label.prediction_horizon_months"))

    @property
    def reference_window_months(self) -> int:
        return int(self.require("label.reference_window_months"))

    @property
    def decline_threshold_pct(self) -> float:
        return float(self.require("label.decline_threshold_pct"))

    @property
    def minimum_reference_revenue(self) -> float:
        return float(self.require("label.minimum_reference_revenue"))

    @property
    def minimum_history_months(self) -> int:
        return int(self.require("label.minimum_history_months"))

    @property
    def target_column(self) -> str:
        return str(self.require("label.target_column"))

    @property
    def drop_rows_below_minimum_revenue(self) -> bool:
        return bool(self.require("label.drop_rows_below_minimum_revenue"))

    @property
    def snapshot_frequency(self) -> str:
        return str(self.require("snapshots.frequency"))

    @property
    def feature_windows(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self.require("snapshots.feature_windows_months"))

    @property
    def drop_incomplete_horizon(self) -> bool:
        return bool(self.require("snapshots.drop_incomplete_horizon"))

    @property
    def train_end(self) -> pd.Timestamp:
        return _as_timestamp(self.require("splits.train_end"), "splits.train_end")

    @property
    def validation_end(self) -> pd.Timestamp:
        return _as_timestamp(self.require("splits.validation_end"), "splits.validation_end")

    @property
    def test_end(self) -> pd.Timestamp:
        return _as_timestamp(self.require("splits.test_end"), "splits.test_end")

    @property
    def risk_high(self) -> float:
        return float(self.require("risk.high_threshold"))

    @property
    def risk_medium(self) -> float:
        return float(self.require("risk.medium_threshold"))

    @property
    def precision_at_k(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self.require("evaluation.precision_at_k"))

    @property
    def k_fractions(self) -> tuple[float, ...]:
        return tuple(float(v) for v in self.require("evaluation.k_fractions"))

    @property
    def false_alarm_tolerance(self) -> float:
        return float(self.require("evaluation.false_alarm_tolerance"))

    @property
    def false_alarm_tolerance_status(self) -> str:
        return str(self.require("evaluation.false_alarm_tolerance_status"))

    @property
    def bootstrap_iterations(self) -> int:
        return int(self.require("evaluation.bootstrap_iterations"))

    @property
    def bootstrap_seed(self) -> int:
        return int(self.require("evaluation.bootstrap_seed"))

    @property
    def calibration_bins(self) -> int:
        return int(self.require("evaluation.calibration_bins"))

    @property
    def primary_k(self) -> int:
        return int(self.require("evaluation.primary_k"))

    @property
    def model_selection_metric(self) -> str:
        return str(self.require("evaluation.model_selection_metric"))

    @property
    def selection_tiebreak_metric(self) -> str:
        return str(self.require("evaluation.selection_tiebreak_metric"))

    @property
    def interpretability_margin(self) -> float:
        return float(self.require("evaluation.interpretability_margin"))

    @property
    def reason_settings(self) -> dict[str, Any]:
        return {
            str(key): value for key, value in self.require("reasons").items()
        }

    @property
    def baseline_weights(self) -> dict[str, float]:
        return {
            str(key): float(value)
            for key, value in self.require("baseline.weights").items()
        }

    @property
    def baseline_thresholds(self) -> dict[str, float]:
        return {
            str(key): float(value)
            for key, value in self.require("baseline.thresholds").items()
        }

    @property
    def baseline_high_risk_score(self) -> float:
        return float(self.require("baseline.high_risk_score"))

    @property
    def binary_threshold(self) -> float:
        return float(self.require("models.binary_threshold"))

    @property
    def logistic_regression_params(self) -> dict[str, Any]:
        return {
            str(key): value
            for key, value in self.require("models.logistic_regression").items()
        }

    @property
    def hist_gradient_boosting_params(self) -> dict[str, Any]:
        return {
            str(key): value
            for key, value in self.require("models.hist_gradient_boosting").items()
        }

    @property
    def primary_model(self) -> str:
        return str(self.require("models.primary"))

    @property
    def challenger_model(self) -> str:
        return str(self.require("models.challenger"))

    @property
    def model_random_state(self) -> int:
        return int(self.require("models.random_state"))

    @property
    def opportunity_scoring_enabled(self) -> bool:
        return bool(self.require("features.opportunity_scoring"))

    @property
    def llm_recommendations_enabled(self) -> bool:
        return bool(self.require("features.llm_recommendations"))

    @property
    def synthetic_banner_enabled(self) -> bool:
        return bool(self.require("features.synthetic_banner"))

    @property
    def llm_enabled(self) -> bool:
        return bool(self.require("llm.enabled"))

    @property
    def llm_provider(self) -> str:
        return str(self.require("llm.provider"))

    @property
    def feedback_db_path(self) -> Path:
        return self.resolve_path(str(self.require("feedback.database_path")))

    @property
    def artifacts_model_dir(self) -> Path:
        return self.resolve_path(str(self.require("artifacts.model_dir")))

    @property
    def artifacts_evaluation_dir(self) -> Path:
        return self.resolve_path(str(self.require("artifacts.evaluation_dir")))

    @property
    def sample_data_path(self) -> Path:
        return self.resolve_path(str(self.require("paths.sample_data")))

    @property
    def full_data_path(self) -> Path:
        return self.resolve_path(str(self.require("paths.full_data")))

    @property
    def snapshots_path(self) -> Path:
        return self.resolve_path(str(self.require("paths.snapshots")))

    @property
    def full_snapshots_path(self) -> Path:
        return self.resolve_path(str(self.require("paths.full_snapshots")))

    @property
    def sample_snapshots_path(self) -> Path:
        return self.resolve_path(str(self.require("paths.sample_snapshots")))


def load_config(path: str | Path = "config.yaml") -> Config:
    """Load and validate a YAML configuration file."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, Mapping):
        raise ConfigError("Configuration root must be a mapping")

    config = Config(raw=raw, path=config_path)
    validate_config(config)
    return config


def _check_probability(value: float, key: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ConfigError(f"{key} must be between 0 and 1, got {value}")


def validate_config(config: Config) -> None:
    """Validate cross-field assumptions and fail with actionable messages."""

    for section in _REQUIRED_SECTIONS:
        if not isinstance(config.get(section), Mapping):
            raise ConfigError(f"Missing or invalid configuration section: {section}")

    if config.seed < 0:
        raise ConfigError("random.seed must be non-negative")

    if config.n_accounts <= 0:
        raise ConfigError("synthetic.n_accounts must be positive")
    if config.n_months <= 0:
        raise ConfigError("synthetic.n_months must be positive")

    if config.minimum_history_months < 3:
        raise ConfigError("label.minimum_history_months must be at least 3")
    if config.horizon_months < 1:
        raise ConfigError("label.prediction_horizon_months must be at least 1")
    if config.reference_window_months < 1:
        raise ConfigError("label.reference_window_months must be at least 1")
    if not 0 < config.decline_threshold_pct < 100:
        raise ConfigError("label.decline_threshold_pct must be between 0 and 100")
    if config.minimum_reference_revenue < 0:
        raise ConfigError("label.minimum_reference_revenue must be non-negative")

    required_months = (
        config.minimum_history_months + config.horizon_months + 1
    )
    if config.n_months < required_months:
        raise ConfigError(
            "synthetic.n_months must cover minimum history plus prediction horizon; "
            f"need at least {required_months}, got {config.n_months}"
        )

    segment_mix = config.segment_mix
    if not segment_mix:
        raise ConfigError("synthetic.segment_mix must not be empty")
    if abs(sum(segment_mix.values()) - 1.0) > 1e-6:
        raise ConfigError("synthetic.segment_mix must sum to 1.0")
    for segment in segment_mix:
        if segment not in config.segment_base_revenue:
            raise ConfigError(f"Missing segment_base_revenue for {segment}")
        if segment not in config.segment_avg_transaction_value:
            raise ConfigError(f"Missing segment_avg_transaction_value for {segment}")

    archetype_mix = config.archetype_mix
    if abs(sum(archetype_mix.values()) - 1.0) > 1e-6:
        raise ConfigError("synthetic.archetype_mix must sum to 1.0")
    unexpected = set(archetype_mix) - _SUPPORTED_ARCHETYPES
    if unexpected:
        raise ConfigError(f"Unsupported synthetic archetypes: {sorted(unexpected)}")

    for field, rate in config.missing_rates.items():
        _check_probability(rate, f"synthetic.missing_rates.{field}")

    if config.snapshot_frequency != "quarter_end":
        raise ConfigError(
            "Phase 1 supports snapshots.frequency='quarter_end' only; "
            f"got {config.snapshot_frequency!r}"
        )
    if not config.feature_windows:
        raise ConfigError("snapshots.feature_windows_months must not be empty")
    if any(window <= 0 for window in config.feature_windows):
        raise ConfigError("snapshots.feature_windows_months must be positive")

    train_end, validation_end, test_end = (
        config.train_end,
        config.validation_end,
        config.test_end,
    )
    if not train_end < validation_end < test_end:
        raise ConfigError(
            "Split dates must be strictly increasing: "
            "splits.train_end < splits.validation_end < splits.test_end"
        )

    data_end = config.start_month + pd.DateOffset(months=config.n_months - 1)
    data_end = data_end + pd.offsets.MonthEnd(0)
    if test_end > data_end:
        raise ConfigError(
            f"splits.test_end ({test_end.date()}) is after the generated data ends "
            f"({data_end.date()}); increase n_months or move the split dates"
        )

    if not 0 < config.risk_medium < config.risk_high < 1:
        raise ConfigError(
            "risk thresholds must satisfy 0 < medium_threshold < high_threshold < 1"
        )

    if not config.precision_at_k or any(k <= 0 for k in config.precision_at_k):
        raise ConfigError("evaluation.precision_at_k must contain positive integers")
    for fraction in config.k_fractions:
        _check_probability(fraction, "evaluation.k_fractions")
    _check_probability(config.false_alarm_tolerance, "evaluation.false_alarm_tolerance")
    if config.bootstrap_iterations <= 0:
        raise ConfigError("evaluation.bootstrap_iterations must be positive")
    if config.bootstrap_seed < 0:
        raise ConfigError("evaluation.bootstrap_seed must be non-negative")
    if config.calibration_bins < 2:
        raise ConfigError("evaluation.calibration_bins must be at least 2")
    if config.primary_k <= 0:
        raise ConfigError("evaluation.primary_k must be positive")
    if config.primary_k not in config.precision_at_k:
        raise ConfigError(
            "evaluation.primary_k must be one of evaluation.precision_at_k"
        )
    allowed_selection_metrics = {"precision_at_k", "pr_auc"}
    if config.model_selection_metric not in allowed_selection_metrics:
        raise ConfigError(
            "evaluation.model_selection_metric must be one of "
            f"{sorted(allowed_selection_metrics)}"
        )
    if config.selection_tiebreak_metric not in allowed_selection_metrics:
        raise ConfigError(
            "evaluation.selection_tiebreak_metric must be one of "
            f"{sorted(allowed_selection_metrics)}"
        )
    if config.selection_tiebreak_metric == config.model_selection_metric:
        raise ConfigError(
            "evaluation.selection_tiebreak_metric must differ from the primary metric"
        )
    if config.interpretability_margin < 0:
        raise ConfigError("evaluation.interpretability_margin must be non-negative")

    baseline_signals = {
        "revenue_decline",
        "transaction_decline",
        "inactivity",
        "engagement_decline",
    }
    weights = config.baseline_weights
    thresholds = config.baseline_thresholds
    if set(weights) != baseline_signals:
        raise ConfigError(
            "baseline.weights must define exactly: "
            + ", ".join(sorted(baseline_signals))
        )
    baseline_threshold_keys = {
        "revenue_decline_pct",
        "transaction_decline_pct",
        "inactivity_days",
        "engagement_decline_pct",
    }
    if set(thresholds) != baseline_threshold_keys:
        raise ConfigError(
            "baseline.thresholds must define exactly: "
            + ", ".join(sorted(baseline_threshold_keys))
        )
    if any(value < 0 for value in weights.values()):
        raise ConfigError("baseline.weights must be non-negative")
    if not 0 <= config.baseline_high_risk_score <= sum(weights.values()):
        raise ConfigError(
            "baseline.high_risk_score must be between 0 and the maximum score"
        )

    reasons = config.reason_settings
    required_reason_keys = {
        "revenue_decline_3m_pct",
        "revenue_decline_6m_pct",
        "revenue_growth_6m_pct",
        "transaction_decline_3m_pct",
        "inactivity_days",
        "engagement_decline_3m_pct",
        "low_engagement_level",
        "limit",
    }
    if set(reasons) != required_reason_keys:
        raise ConfigError(
            "reasons must define exactly: "
            + ", ".join(sorted(required_reason_keys))
        )
    if int(reasons["limit"]) < 1:
        raise ConfigError("reasons.limit must be positive")
    if float(reasons["inactivity_days"]) < 0:
        raise ConfigError("reasons.inactivity_days must be non-negative")

    _check_probability(config.binary_threshold, "models.binary_threshold")

    lr_params = config.logistic_regression_params
    if float(lr_params.get("C", 0)) <= 0:
        raise ConfigError("models.logistic_regression.C must be positive")
    if int(lr_params.get("max_iter", 0)) <= 0:
        raise ConfigError("models.logistic_regression.max_iter must be positive")
    if lr_params.get("class_weight") not in (None, "balanced"):
        raise ConfigError(
            "models.logistic_regression.class_weight must be null or 'balanced'"
        )

    hgb_params = config.hist_gradient_boosting_params
    if float(hgb_params.get("learning_rate", 0)) <= 0:
        raise ConfigError(
            "models.hist_gradient_boosting.learning_rate must be positive"
        )
    if int(hgb_params.get("max_iter", 0)) <= 0:
        raise ConfigError("models.hist_gradient_boosting.max_iter must be positive")
    if int(hgb_params.get("max_leaf_nodes", 0)) < 2:
        raise ConfigError(
            "models.hist_gradient_boosting.max_leaf_nodes must be at least 2"
        )
    if int(hgb_params.get("min_samples_leaf", 0)) < 1:
        raise ConfigError(
            "models.hist_gradient_boosting.min_samples_leaf must be positive"
        )
    if float(hgb_params.get("l2_regularization", 0)) < 0:
        raise ConfigError(
            "models.hist_gradient_boosting.l2_regularization must be non-negative"
        )
    if hgb_params.get("class_weight") not in (None, "balanced"):
        raise ConfigError(
            "models.hist_gradient_boosting.class_weight must be null or 'balanced'"
        )
    if not isinstance(hgb_params.get("early_stopping"), bool):
        raise ConfigError(
            "models.hist_gradient_boosting.early_stopping must be true or false"
        )

    if not config.project_version.strip():
        raise ConfigError("project.version must be non-empty")

    if not config.primary_model or not config.challenger_model:
        raise ConfigError("models.primary and models.challenger must be non-empty")
    if config.primary_model == config.challenger_model:
        raise ConfigError("models.primary and models.challenger must differ")

    if config.llm_enabled:
        raise ConfigError(
            "LLM usage is disabled in Phase 1. Set llm.enabled: false; "
            "the optional adapter is a later-phase item."
        )
    if config.llm_recommendations_enabled:
        raise ConfigError(
            "features.llm_recommendations must remain false in Phase 1."
        )
    if config.opportunity_scoring_enabled:
        raise ConfigError(
            "features.opportunity_scoring is a later-phase, experimental item. "
            "Set it to false for Phase 1."
        )

    for key in (
        "feedback.database_path",
        "artifacts.model_dir",
        "artifacts.evaluation_dir",
        "paths.sample_data",
        "paths.full_data",
        "paths.snapshots",
        "paths.full_snapshots",
        "paths.sample_snapshots",
    ):
        if not str(config.require(key)).strip():
            raise ConfigError(f"{key} must be a non-empty path")
