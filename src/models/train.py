"""Train, compare and freeze the RevenuePulse AI revenue-decline model.

Usage from the project root::

    python -m src.models.train

The command:

1. loads the point-in-time snapshots (no feature engineering is redone here);
2. fits both candidate models on the TRAIN split only;
3. compares the transparent baseline, logistic regression and histogram gradient
   boosting on the VALIDATION split;
4. selects one primary model with the rule in ``config.yaml``;
5. saves the fitted pipelines, metadata and validation metrics.

The test split is never read by this module. ``src.evaluation.evaluate`` performs
the single locked test evaluation.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.evaluation.metrics import evaluate_predictions
from src.features.snapshots import (
    FEATURE_COLUMNS,
    FORBIDDEN_FEATURE_COLUMNS,
    assert_time_ordering,
    label_prevalence,
)
from src.models.baseline import (
    SIGNAL_FEATURES,
    BaselineScorecard,
    assert_baseline_features_allowed,
    score_baseline,
)
from src.models.preprocessing import (
    CATEGORICAL_FEATURES,
    HIST_GRADIENT_BOOSTING,
    LOGISTIC_REGRESSION,
    NUMERIC_FEATURES,
    build_model_pipeline,
    model_feature_frame,
    model_metadata_parameters,
    predict_risk_probability,
)
from src.utils.config import Config, load_config

BASELINE_NAME = "baseline"
ML_MODELS: tuple[str, ...] = (LOGISTIC_REGRESSION, HIST_GRADIENT_BOOSTING)
MODEL_LABELS: dict[str, str] = {
    BASELINE_NAME: "Transparent scorecard",
    LOGISTIC_REGRESSION: "Logistic regression",
    HIST_GRADIENT_BOOSTING: "Histogram gradient boosting",
}
ARTIFACT_FILENAME = "revenuepulse_model.joblib"
METADATA_FILENAME = "model_metadata.json"
VALIDATION_FILENAME = "validation_metrics.json"


# --------------------------------------------------------------------------
# Data loading and split handling
# --------------------------------------------------------------------------
def load_snapshots(path: Path, config: Config) -> pd.DataFrame:
    """Load labelled snapshots and validate the time ordering of the splits."""

    if not path.exists():
        raise FileNotFoundError(
            f"Snapshot file not found: {path}. Rebuild it with "
            "`python -m src.features.build_snapshots --input "
            "data/processed/account_month_full.csv "
            "--output data/processed/snapshots_full.csv`."
        )

    frame = pd.read_csv(path)
    for column in ("snapshot_date", "outcome_start_month", "outcome_end_month"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")

    if config.target_column not in frame.columns:
        raise ValueError(f"Snapshot file has no target column: {config.target_column}")

    frame = frame.dropna(subset=[config.target_column]).copy()
    frame[config.target_column] = frame[config.target_column].astype(int)
    frame = frame.sort_values(["snapshot_date", "account_id"], kind="mergesort")
    frame = frame.reset_index(drop=True)

    expected = {"train", "validation", "test"}
    unexpected = set(frame["split"].unique()) - expected
    if unexpected:
        raise ValueError(f"Unexpected split labels: {sorted(unexpected)}")
    assert_time_ordering(frame)

    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Snapshot file is missing feature columns: {missing}")

    return frame


def split_frames(snapshots: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return train / validation / test frames, each reset to a clean index."""

    frames: dict[str, pd.DataFrame] = {}
    for split in ("train", "validation", "test"):
        subset = snapshots[snapshots["split"] == split].reset_index(drop=True)
        if subset.empty:
            raise ValueError(f"Split {split!r} has no labelled snapshots")
        frames[split] = subset
    return frames


def assert_no_forbidden_columns(features: pd.DataFrame) -> None:
    """Fail if identifiers, split metadata or target fields reach the model matrix."""

    unexpected = [
        column for column in FORBIDDEN_FEATURE_COLUMNS if column in features.columns
    ]
    if unexpected:
        raise AssertionError(f"Forbidden columns in model matrix: {unexpected}")
    if "account_id" in features.columns:
        raise AssertionError("account_id must never be a model feature")


# --------------------------------------------------------------------------
# Fitting and scoring
# --------------------------------------------------------------------------
def fit_models(train: pd.DataFrame, config: Config) -> dict[str, Any]:
    """Fit every candidate ML pipeline on the training split only."""

    features = model_feature_frame(train)
    assert_no_forbidden_columns(features)
    target = train[config.target_column].to_numpy().astype(int)

    pipelines: dict[str, Any] = {}
    for model_name in ML_MODELS:
        pipeline = build_model_pipeline(model_name, config)
        pipeline.fit(features, target)
        pipelines[model_name] = pipeline
    return pipelines


def baseline_probabilities(frame: pd.DataFrame, config: Config) -> np.ndarray:
    """Return the scorecard score normalised by its maximum (not calibrated)."""

    scorecard = BaselineScorecard.from_config(config)
    assert_baseline_features_allowed()
    scored = score_baseline(frame, config)
    return np.clip(
        scored["baseline_score"].to_numpy(dtype=float) / scorecard.max_score, 0.0, 1.0
    )


def score_all_models(
    frame: pd.DataFrame,
    pipelines: Mapping[str, Any],
    config: Config,
) -> dict[str, np.ndarray]:
    """Return positive-class risk scores for the baseline and each fitted model."""

    scores: dict[str, np.ndarray] = {
        BASELINE_NAME: baseline_probabilities(frame, config)
    }
    for model_name, pipeline in pipelines.items():
        scores[model_name] = predict_risk_probability(pipeline, frame)
    for name, values in scores.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite scores for {name}")
        if values.min() < 0.0 or values.max() > 1.0:
            raise ValueError(f"Scores for {name} fall outside [0, 1]")
    return scores


def evaluate_frame(
    frame: pd.DataFrame,
    scores: Mapping[str, np.ndarray],
    config: Config,
) -> dict[str, Any]:
    """Evaluate every model on one labelled split with the configured metric set."""

    results: dict[str, Any] = {}
    for name, values in scores.items():
        results[name] = evaluate_predictions(
            frame,
            target_column=config.target_column,
            scores=values,
            k_values=config.precision_at_k,
            fractions=config.k_fractions,
            primary_k=config.primary_k,
            binary_threshold=config.binary_threshold,
            probability_like=True,
        )
        results[name]["label"] = MODEL_LABELS.get(name, name)
    return results


def compact_metrics(results: Mapping[str, Any], primary_k: int) -> dict[str, dict[str, float]]:
    """Reduce a metric bundle to the two model-selection numbers."""

    compact: dict[str, dict[str, float]] = {}
    for name, metrics in results.items():
        compact[name] = {
            "precision_at_k": float(metrics["ranking"][str(primary_k)]["precision"]),
            "pr_auc": float(metrics["pr_auc"]),
            "recall_at_k": float(metrics["ranking"][str(primary_k)]["recall"]),
            "lift_at_k": float(metrics["ranking"][str(primary_k)]["lift"]),
        }
    return compact


# --------------------------------------------------------------------------
# Model selection (validation only)
# --------------------------------------------------------------------------
def _safe(value: Any) -> float:
    number = float(value)
    return number if np.isfinite(number) else float("-inf")


def select_primary_model(
    validation: Mapping[str, Mapping[str, float]],
    config: Config,
) -> dict[str, Any]:
    """Choose the primary model from validation scores using the configured rule.

    Rule (declared in ``config.yaml`` before any test evaluation):

    1. Rank the ML candidates by ``evaluation.model_selection_metric``, then by
       ``evaluation.selection_tiebreak_metric``.
    2. If the winner is not logistic regression and its lead over logistic
       regression is within ``evaluation.interpretability_margin``, prefer
       logistic regression for interpretability.
    3. The transparent baseline is a reference, never a fitted candidate; a
       boolean records whether the chosen model actually beat it.
    """

    metric = str(config.model_selection_metric)
    tiebreak = str(config.selection_tiebreak_metric)
    if BASELINE_NAME not in validation:
        raise ValueError("Validation comparison must include the baseline")

    candidates = {
        name: values for name, values in validation.items() if name in ML_MODELS
    }
    if not candidates:
        raise ValueError("No ML candidates supplied for model selection")

    ordered = sorted(
        candidates.items(),
        key=lambda item: (
            -_safe(item[1].get(metric, float("nan"))),
            -_safe(item[1].get(tiebreak, float("nan"))),
            0 if item[0] == LOGISTIC_REGRESSION else 1,
        ),
    )
    best_name, best_values = ordered[0]
    best_value = _safe(best_values.get(metric, float("nan")))
    margin = float(config.interpretability_margin)

    selected = best_name
    margin_applied = False
    reason = (
        f"{MODEL_LABELS[best_name]} had the best validation {metric} "
        f"({best_value:.4f} at k={config.primary_k})."
    )

    if best_name != LOGISTIC_REGRESSION and LOGISTIC_REGRESSION in candidates:
        lr_value = _safe(candidates[LOGISTIC_REGRESSION].get(metric, float("nan")))
        if best_value - lr_value <= margin:
            selected = LOGISTIC_REGRESSION
            margin_applied = True
            reason = (
                f"{MODEL_LABELS[best_name]} led validation {metric} by only "
                f"{best_value - lr_value:.4f} (margin {margin:.2f}), so logistic "
                "regression was preferred for interpretability."
            )

    selected_values = candidates[selected]
    baseline_values = validation[BASELINE_NAME]
    beats_baseline = _safe(selected_values.get(metric, float("nan"))) > _safe(
        baseline_values.get(metric, float("nan"))
    )

    return {
        "selection_metric": metric,
        "tiebreak_metric": tiebreak,
        "primary_k": int(config.primary_k),
        "interpretability_margin": margin,
        "best_ml_model": best_name,
        "selected_model": selected,
        "margin_applied": margin_applied,
        "beats_baseline": bool(beats_baseline),
        "validation_scores": {
            name: {key: float(value) for key, value in values.items()}
            for name, values in validation.items()
        },
        "reason": reason,
        "note": (
            "Selection used the validation split only. The test split was not "
            "read before the model was frozen."
        ),
    }


# --------------------------------------------------------------------------
# Artifact writing
# --------------------------------------------------------------------------
def _period(frame: pd.DataFrame, target_column: str) -> dict[str, Any]:
    return {
        "start": str(pd.to_datetime(frame["snapshot_date"]).min().date()),
        "end": str(pd.to_datetime(frame["snapshot_date"]).max().date()),
        "rows": int(len(frame)),
        "accounts": int(frame["account_id"].nunique()),
        "positives": int(frame[target_column].sum()),
        "prevalence": float(frame[target_column].mean()),
    }


def build_metadata(
    *,
    selection: Mapping[str, Any],
    config: Config,
    splits: Mapping[str, pd.DataFrame],
    metrics: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
    pipelines: Mapping[str, Any],
    trained_at: str,
    snapshot_path: Path,
) -> dict[str, Any]:
    """Build the JSON metadata stored beside the model artifact."""

    selected = str(selection["selected_model"])
    return {
        "model_name": selected,
        "model_label": MODEL_LABELS[selected],
        "model_type": type(pipelines[selected].named_steps["classifier"]).__name__,
        "version": config.project_version,
        "trained_at_utc": trained_at,
        "synthetic_data": True,
        "synthetic_data_note": (
            "Trained on generated account histories. Performance is a property of "
            "the synthetic generator and is not real-world churn evidence."
        ),
        "target_column": config.target_column,
        "label_definition": {
            "prediction_horizon_months": config.horizon_months,
            "reference_window_months": config.reference_window_months,
            "decline_threshold_pct": config.decline_threshold_pct,
            "minimum_reference_revenue": config.minimum_reference_revenue,
            "minimum_history_months": config.minimum_history_months,
        },
        "feature_columns": list(FEATURE_COLUMNS),
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "excluded_columns": [column for column in FORBIDDEN_FEATURE_COLUMNS],
        "training_period": _period(splits["train"], config.target_column),
        "validation_period": _period(splits["validation"], config.target_column),
        "test_period": _period(splits["test"], config.target_column),
        "random_seed": int(config.model_random_state),
        "bootstrap_seed": int(config.bootstrap_seed),
        "bootstrap_iterations": int(config.bootstrap_iterations),
        "model_parameters": model_metadata_parameters(pipelines[selected]),
        "candidate_parameters": {
            name: model_metadata_parameters(pipeline)
            for name, pipeline in pipelines.items()
        },
        "class_imbalance_handling": {
            "logistic_regression_class_weight": config.logistic_regression_params.get(
                "class_weight"
            ),
            "hist_gradient_boosting_class_weight": config.hist_gradient_boosting_params.get(
                "class_weight"
            ),
            "note": (
                "Class weighting, where enabled, changes the probability scale and "
                "therefore the Brier score; ranking metrics are the primary "
                "evaluation and were unaffected in the validation comparison."
            ),
        },
        "preprocessing": {
            "numeric": "median imputation (train statistics)"
            + (", then standardisation" if selected == LOGISTIC_REGRESSION else ""),
            "categorical": "most-frequent imputation, one-hot with handle_unknown='ignore'",
            "fit_scope": "training split only",
        },
        "primary_precision_at_k": int(config.primary_k),
        "binary_threshold": float(config.binary_threshold),
        "selection": {
            key: value
            for key, value in selection.items()
            if key in {
                "selection_metric",
                "tiebreak_metric",
                "selected_model",
                "best_ml_model",
                "margin_applied",
                "beats_baseline",
                "reason",
                "note",
            }
        },
        "selected_metrics": {
            "train": metrics["train"][selected],
            "validation": validation_metrics[selected],
        },
        "calibration_applied": False,
        "calibration_note": (
            "No post-hoc calibration was fitted. Calibration is assessed and "
            "reported as a limitation instead."
        ),
        "snapshot_source": str(snapshot_path),
        "label_prevalence_train": label_prevalence(splits["train"], config.target_column),
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "intended_use": "Decision support for human account review; no automated action.",
    }


def save_training_outputs(
    *,
    output_dir: Path,
    pipelines: Mapping[str, Any],
    metadata: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Persist the model bundle, metadata and validation metrics."""

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / ARTIFACT_FILENAME
    metadata_path = output_dir / METADATA_FILENAME

    bundle = {
        "selected_model": metadata["model_name"],
        "pipelines": {name: pipeline for name, pipeline in pipelines.items()},
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": metadata["target_column"],
        "binary_threshold": metadata["binary_threshold"],
        "version": metadata["version"],
        "trained_at_utc": metadata["trained_at_utc"],
        "random_seed": metadata["random_seed"],
        "synthetic_data": True,
    }
    joblib.dump(bundle, artifact_path)

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=False, default=str)
        handle.write("\n")

    evaluation_dir = output_dir.parent / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    with (evaluation_dir / VALIDATION_FILENAME).open("w", encoding="utf-8") as handle:
        json.dump(validation_metrics, handle, indent=2, default=str)
        handle.write("\n")

    return artifact_path, metadata_path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--snapshots",
        type=Path,
        default=None,
        help="Snapshot CSV (default: paths.full_snapshots from config.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Model artifact directory (default: artifacts.model_dir)",
    )
    return parser


def run_training(
    config: Config,
    snapshot_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run the whole training and selection flow and return a summary dict."""

    snapshots = load_snapshots(snapshot_path, config)
    splits = split_frames(snapshots)

    print(f"Snapshots: {len(snapshots):,} rows from {snapshot_path}")
    for split, frame in splits.items():
        print(
            f"  {split:<10} rows={len(frame):>5} accounts={frame['account_id'].nunique():>4} "
            f"positives={int(frame[config.target_column].sum()):>3} "
            f"prevalence={frame[config.target_column].mean():.3f} "
            f"({frame['snapshot_date'].min().date()} to {frame['snapshot_date'].max().date()})"
        )

    pipelines = fit_models(splits["train"], config)

    train_scores = score_all_models(splits["train"], pipelines, config)
    validation_scores = score_all_models(splits["validation"], pipelines, config)
    train_metrics = evaluate_frame(splits["train"], train_scores, config)
    validation_metrics = evaluate_frame(splits["validation"], validation_scores, config)

    compact = compact_metrics(validation_metrics, config.primary_k)
    selection = select_primary_model(compact, config)

    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metadata = build_metadata(
        selection=selection,
        config=config,
        splits=splits,
        metrics={"train": train_metrics},
        validation_metrics=validation_metrics,
        pipelines=pipelines,
        trained_at=trained_at,
        snapshot_path=snapshot_path,
    )

    artifact_path, metadata_path = save_training_outputs(
        output_dir=output_dir,
        pipelines=pipelines,
        metadata=metadata,
        validation_metrics=validation_metrics,
    )

    print("")
    print(f"Validation comparison (k={config.primary_k}):")
    header = f"  {'model':<26}{'P@K':>8}{'R@K':>8}{'Lift@K':>9}{'PR-AUC':>9}"
    print(header)
    for name in (BASELINE_NAME, *ML_MODELS):
        values = compact[name]
        print(
            f"  {MODEL_LABELS[name]:<26}{values['precision_at_k']:>8.3f}"
            f"{values['recall_at_k']:>8.3f}{values['lift_at_k']:>9.2f}"
            f"{values['pr_auc']:>9.3f}"
        )
    print("")
    print(f"Selected model: {selection['selected_model']}")
    print(f"Reason: {selection['reason']}")
    print(f"Model artifact: {artifact_path}")
    print(f"Metadata: {metadata_path}")

    return {
        "selection": selection,
        "artifact_path": str(artifact_path),
        "metadata_path": str(metadata_path),
        "validation_metrics": validation_metrics,
        "train_metrics": train_metrics,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)
    snapshot_path = Path(args.snapshots or config.full_snapshots_path)
    output_dir = Path(args.output_dir or config.artifacts_model_dir)
    run_training(config, snapshot_path, output_dir)

    print("")
    print(
        "Next: python -m src.evaluation.evaluate  "
        "(single locked test evaluation; do not tune on the test split)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
