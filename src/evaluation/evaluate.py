"""Single locked test evaluation of the frozen RevenuePulse AI model.

Usage from the project root::

    python -m src.evaluation.evaluate

The model is frozen by ``src.models.train`` using train and validation data only,
so this module is the one place where the test split is scored. It writes
machine-readable results to ``artifacts/evaluation`` and prints a summary.

Nothing here changes the model, its hyperparameters or the selection rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from src.evaluation.bootstrap import account_cluster_bootstrap
from src.evaluation.metrics import (
    calibration_table,
    evaluate_predictions,
    ranking_comparison_table,
    segment_analysis,
)
from src.features.snapshots import FEATURE_COLUMNS
from src.models.baseline import BaselineScorecard
from src.models.preprocessing import (
    HIST_GRADIENT_BOOSTING,
    LOGISTIC_REGRESSION,
    logistic_regression_coefficients,
    model_feature_frame,
    predict_risk_probability,
)
from src.models.reasons import (
    permutation_importance_table,
    reason_code_frame,
    reason_driver_alignment,
    signal_prevalence,
)
from src.models.train import (
    ARTIFACT_FILENAME,
    BASELINE_NAME,
    METADATA_FILENAME,
    ML_MODELS,
    MODEL_LABELS,
    load_snapshots,
    split_frames,
)
from src.utils.config import Config, load_config

ERROR_COLUMNS: tuple[str, ...] = (
    "snapshot_id",
    "account_id",
    "snapshot_date",
    "segment",
    "selected_score",
    "baseline_score",
    "selected_rank",
    "baseline_rank",
    "future_revenue_decline",
    "future_revenue_change_pct",
    "revenue_trend_3m_pct",
    "revenue_trend_6m_pct",
    "revenue_mean_3m",
    "transaction_trend_3m_pct",
    "engagement_trend_3m_pct",
    "engagement_mean_3m",
    "days_since_activity_last",
    "days_since_activity_mean_3m",
    "history_months",
    "revenue_volatility_6m",
    "support_trend_3m_pct",
    "product_count_change_6m",
)


def _jsonable(value: Any) -> Any:
    """Convert numpy / pandas scalars into plain JSON types."""

    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.date().isoformat()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if pd.isna(value):
        return None
    return str(value)


def load_artifact(model_dir: Path) -> dict[str, Any]:
    """Load the frozen model bundle and its metadata."""

    artifact_path = model_dir / ARTIFACT_FILENAME
    metadata_path = model_dir / METADATA_FILENAME
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {artifact_path}. Train first with "
            "`python -m src.models.train`."
        )
    bundle = joblib.load(artifact_path)
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {"bundle": bundle, "metadata": metadata, "artifact_path": artifact_path}


def score_models(
    frame: pd.DataFrame,
    bundle: Mapping[str, Any],
    config: Config,
) -> dict[str, np.ndarray]:
    """Score a frame with the baseline and every frozen pipeline."""

    scorecard = BaselineScorecard.from_config(config)
    scored = frame.copy()
    scored["baseline_score"] = scorecard.score(frame)
    baseline_probability = np.clip(
        scored["baseline_score"].to_numpy(dtype=float) / scorecard.max_score, 0.0, 1.0
    )

    scores: dict[str, np.ndarray] = {BASELINE_NAME: baseline_probability}
    for name, pipeline in bundle["pipelines"].items():
        scores[name] = predict_risk_probability(pipeline, frame)
    return scores


def rank_frame(
    frame: pd.DataFrame,
    scores: Mapping[str, np.ndarray],
    *,
    selected_model: str,
) -> pd.DataFrame:
    """Attach scores and deterministic ranks for error analysis."""

    ranked = frame.copy()
    ranked["selected_score"] = scores[selected_model]
    ranked["baseline_score"] = scores[BASELINE_NAME]
    # Stable sort on score, then snapshot_id, keeps ties reproducible.
    ranked = ranked.sort_values(
        ["selected_score", "snapshot_id"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    ranked["selected_rank"] = np.arange(1, len(ranked) + 1)

    baseline_order = frame.assign(
        baseline_score=scores[BASELINE_NAME]
    ).sort_values(
        ["baseline_score", "snapshot_id"], ascending=[False, True], kind="mergesort"
    )
    baseline_rank = pd.Series(
        np.arange(1, len(baseline_order) + 1), index=baseline_order["snapshot_id"].to_numpy()
    )
    ranked["baseline_rank"] = ranked["snapshot_id"].map(baseline_rank).astype(int)
    return ranked


def _error_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in ERROR_COLUMNS if column in frame.columns]
    table = frame.loc[:, columns].copy()
    if "snapshot_date" in table.columns:
        table["snapshot_date"] = pd.to_datetime(table["snapshot_date"]).dt.date
    return table.reset_index(drop=True)


def error_tables(
    ranked: pd.DataFrame,
    *,
    target_column: str,
    primary_k: int,
    positive_score_floor: float = 0.5,
    negative_score_ceiling: float = 0.5,
    limit: int = 15,
) -> dict[str, pd.DataFrame]:
    """Build the four error-analysis tables requested in Phase 2."""

    in_selected = ranked["selected_rank"] <= primary_k
    in_baseline = ranked["baseline_rank"] <= primary_k
    positive = ranked[target_column] == 1
    negative = ranked[target_column] == 0

    high_false_positives = ranked[negative & in_selected & (ranked["selected_score"] >= positive_score_floor)]
    low_false_negatives = ranked[positive & ~in_selected & (ranked["selected_score"] <= negative_score_ceiling)]

    baseline_only = ranked[positive & in_baseline & ~in_selected]
    model_only = ranked[positive & in_selected & ~in_baseline]

    return {
        "high_scoring_false_positives": _error_table(
            high_false_positives.sort_values(
                ["selected_score", "snapshot_id"], ascending=[False, True], kind="mergesort"
            ).head(limit)
        ),
        "low_scoring_false_negatives": _error_table(
            low_false_negatives.sort_values(
                ["selected_score", "snapshot_id"], ascending=[True, True], kind="mergesort"
            ).head(limit)
        ),
        "baseline_only_hits": _error_table(
            baseline_only.sort_values(
                ["selected_score", "snapshot_id"], ascending=[False, True], kind="mergesort"
            ).head(limit)
        ),
        "model_only_hits": _error_table(
            model_only.sort_values(
                ["selected_score", "snapshot_id"], ascending=[False, True], kind="mergesort"
            ).head(limit)
        ),
        "top_k_selection": _error_table(ranked[in_selected].head(primary_k)),
    }


def validation_explainability(
    *,
    bundle: Mapping[str, Any],
    splits: Mapping[str, pd.DataFrame],
    config: Config,
    evaluation_dir: Path,
) -> dict[str, Any]:
    """Persist coefficient and permutation-importance tables (validation only)."""

    summary: dict[str, Any] = {}
    validation = splits["validation"]

    if LOGISTIC_REGRESSION in bundle["pipelines"]:
        coefficients = logistic_regression_coefficients(
            bundle["pipelines"][LOGISTIC_REGRESSION]
        )
        coefficients.to_csv(evaluation_dir / "logistic_coefficients.csv", index=False)
        summary["logistic_top_drivers"] = coefficients.head(8).to_dict(orient="records")

    if HIST_GRADIENT_BOOSTING in bundle["pipelines"]:
        importance = permutation_importance_table(
            bundle["pipelines"][HIST_GRADIENT_BOOSTING],
            validation,
            target_column=config.target_column,
            random_state=config.model_random_state,
        )
        importance.to_csv(
            evaluation_dir / "hgb_permutation_importance.csv", index=False
        )
        summary["hgb_top_features"] = importance.head(10).to_dict(orient="records")

    return summary


def run_evaluation(
    config: Config,
    *,
    snapshot_path: Path,
    model_dir: Path,
    output_dir: Path,
    split: str = "test",
) -> dict[str, Any]:
    """Score the frozen model once on the requested split and persist results."""

    loaded = load_artifact(model_dir)
    bundle = loaded["bundle"]
    metadata = loaded["metadata"]
    selected_model = str(bundle["selected_model"])

    snapshots = load_snapshots(snapshot_path, config)
    splits = split_frames(snapshots)
    if split not in splits:
        raise ValueError(f"Unknown split {split!r}; expected train, validation or test")
    frame = splits[split]

    output_dir.mkdir(parents=True, exist_ok=True)

    scores = score_models(frame, bundle, config)
    metrics = {
        name: evaluate_predictions(
            frame,
            target_column=config.target_column,
            scores=values,
            k_values=config.precision_at_k,
            fractions=config.k_fractions,
            primary_k=config.primary_k,
            binary_threshold=config.binary_threshold,
            probability_like=True,
        )
        for name, values in scores.items()
    }
    for name, entry in metrics.items():
        entry["label"] = MODEL_LABELS.get(name, name)

    ranking_table = ranking_comparison_table(metrics, config.precision_at_k)
    ranking_table.to_csv(output_dir / "precision_at_k.csv", index=False)

    calibration: dict[str, pd.DataFrame] = {}
    for name in scores:
        table = calibration_table(
            frame[config.target_column], scores[name], n_bins=config.calibration_bins
        )
        calibration[name] = table
        table.to_csv(output_dir / f"calibration_{name}.csv", index=False)

    bootstrap = account_cluster_bootstrap(
        frame,
        target_column=config.target_column,
        model_scores=scores,
        primary_k=config.primary_k,
        iterations=config.bootstrap_iterations,
        seed=config.bootstrap_seed,
        baseline_name=BASELINE_NAME,
    )

    segments: dict[str, pd.DataFrame] = {}
    for name in (BASELINE_NAME, selected_model):
        table = segment_analysis(
            frame,
            target_column=config.target_column,
            scores=scores[name],
            binary_predictions=(scores[name] >= config.binary_threshold).astype(int),
            fraction=max(config.k_fractions) if config.k_fractions else 0.2,
        )
        segments[name] = table
        table.to_csv(output_dir / f"segments_{name}.csv", index=False)

    ranked = rank_frame(frame, scores, selected_model=selected_model)
    errors = error_tables(
        ranked,
        target_column=config.target_column,
        primary_k=config.primary_k,
    )
    for name, table in errors.items():
        table.to_csv(output_dir / f"error_{name}.csv", index=False)

    explainability = validation_explainability(
        bundle=bundle,
        splits=splits,
        config=config,
        evaluation_dir=output_dir,
    )

    reasons = reason_code_frame(ranked, config)
    ranked_reasons = pd.concat(
        [
            ranked.loc[:, ["snapshot_id", "account_id", "selected_score", "selected_rank", config.target_column]],
            reasons,
        ],
        axis=1,
    )
    ranked_reasons.head(50).to_csv(output_dir / "selected_top_50_reasons.csv", index=False)
    reason_prevalence = signal_prevalence(reasons)
    reason_prevalence.to_csv(output_dir / "reason_code_prevalence.csv", index=False)

    importance_path = output_dir / "hgb_permutation_importance.csv"
    if importance_path.exists():
        alignment = reason_driver_alignment(
            pd.read_csv(importance_path), reasons
        )
        alignment.to_csv(output_dir / "reason_driver_alignment.csv", index=False)
        explainability["reason_driver_alignment"] = alignment.to_dict(orient="records")

    results = {
        "split": split,
        "selected_model": selected_model,
        "synthetic_data": True,
        "synthetic_data_note": (
            "All account histories are generated. These are synthetic-data results "
            "and not evidence of real-world churn prediction performance."
        ),
        "metadata_snapshot": {
            key: metadata.get(key)
            for key in (
                "model_name",
                "model_type",
                "version",
                "trained_at_utc",
                "training_period",
                "validation_period",
                "test_period",
                "selection",
            )
        },
        "frame": {
            "rows": int(len(frame)),
            "accounts": int(frame["account_id"].nunique()),
            "positives": int(frame[config.target_column].sum()),
            "prevalence": float(frame[config.target_column].mean()),
            "start": str(pd.to_datetime(frame["snapshot_date"]).min().date()),
            "end": str(pd.to_datetime(frame["snapshot_date"]).max().date()),
        },
        "metrics": metrics,
        "bootstrap": bootstrap,
        "calibration": {
            name: table.to_dict(orient="records") for name, table in calibration.items()
        },
        "segments": {
            name: table.to_dict(orient="records") for name, table in segments.items()
        },
        "error_counts": {
            name: int(len(table)) for name, table in errors.items()
        },
        "explainability": explainability,
        "reason_code_prevalence": reason_prevalence.to_dict(orient="records"),
        "scores": {
            name: {
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
            }
            for name, values in scores.items()
        },
    }

    with (output_dir / f"{split}_results.json").open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(results), handle, indent=2)
        handle.write("\n")

    _print_summary(results, config)
    return results


def _print_summary(results: Mapping[str, Any], config: Config) -> None:
    frame = results["frame"]
    print("")
    print("=" * 78)
    print(f"SYNTHETIC-DATA EVALUATION ({results['split']} split, locked model)")
    print("=" * 78)
    print(
        f"Rows={frame['rows']:,} accounts={frame['accounts']:,} "
        f"positives={frame['positives']} prevalence={frame['prevalence']:.3f} "
        f"({frame['start']} to {frame['end']})"
    )
    print(f"Selected model: {results['selected_model']}  (selection locked on validation)")

    print("")
    print(f"Ranking metrics (primary k={config.primary_k}):")
    print(f"  {'model':<26}{'P@K':>8}{'R@K':>8}{'Lift':>8}{'FA rate':>9}{'PR-AUC':>9}{'ROC-AUC':>9}{'Brier':>8}")
    for name, metrics in results["metrics"].items():
        primary = metrics["primary_ranking"]
        print(
            f"  {metrics['label']:<26}{primary['precision']:>8.3f}"
            f"{primary['recall']:>8.3f}{primary['lift']:>8.2f}"
            f"{primary['false_alarm_rate']:>9.3f}{metrics['pr_auc']:>9.3f}"
            f"{metrics['roc_auc']:>9.3f}{metrics['brier']:>8.3f}"
        )

    print("")
    print("Precision@K across configured capacities:")
    print(f"  {'model':<26}" + "".join(f"{'K=' + str(k):>10}" for k in config.precision_at_k))
    for name, metrics in results["metrics"].items():
        row = "".join(
            f"{metrics['ranking'][str(k)]['precision']:>10.3f}"
            for k in config.precision_at_k
        )
        print(f"  {metrics['label']:<26}{row}")

    bootstrap = results["bootstrap"]
    print("")
    print(
        f"Account-level bootstrap 95% intervals "
        f"({bootstrap['iterations']} iterations, seed {bootstrap['seed']}, "
        f"{bootstrap['n_accounts']} accounts):"
    )
    for name, entry in bootstrap["models"].items():
        precision = entry["precision_at_k"]
        line = (
            f"  {MODEL_LABELS.get(name, name):<26} P@K {precision['point']:.3f} "
            f"[{precision['lower']:.3f}, {precision['upper']:.3f}]  "
            f"PR-AUC {entry['pr_auc']['point']:.3f} "
            f"[{entry['pr_auc']['lower']:.3f}, {entry['pr_auc']['upper']:.3f}]"
        )
        difference = entry.get("precision_difference_vs_baseline_at_k")
        if difference:
            line += (
                f"  delta P@K {difference['point']:+.3f} "
                f"[{difference['lower']:+.3f}, {difference['upper']:+.3f}]"
            )
        print(line)

    print("")
    print("Calibration (equal-count bins, observed vs predicted):")
    for name in (BASELINE_NAME, results["selected_model"]):
        table = results["calibration"][name]
        gaps = [
            abs(row["mean_predicted"] - row["observed_positive_rate"])
            for row in table
        ]
        print(
            f"  {MODEL_LABELS.get(name, name):<26} mean |gap| {np.mean(gaps):.3f} "
            f"max |gap| {np.max(gaps):.3f}"
        )

    print("")
    print("Segments (selected model, top fraction ranking):")
    for row in results["segments"][results["selected_model"]]:
        print(
            f"  {row['segment']:<12} n={row['n_rows']:>4} prevalence={row['prevalence']:.3f} "
            f"P@frac={row['precision_at_fraction']:.3f} lift={row['lift_at_fraction']:.2f} "
            f"FA={row['false_alarm_rate']:.3f} [{row['status']}]"
        )

    print("")
    print("Reason-code prevalence in the scored population:")
    for row in results["reason_code_prevalence"]:
        print(f"  {row['label']:<32} {row['share_of_rows']:.3f}")
    print("")
    print("Reminder: synthetic data only. No real-world performance is implied.")


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
        "--model-dir",
        type=Path,
        default=None,
        help="Trained artifact directory (default: artifacts.model_dir)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Result directory (default: artifacts.evaluation_dir)",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "validation", "test"],
        help="Split to score; the documented run uses 'test' once",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)
    run_evaluation(
        config,
        snapshot_path=Path(args.snapshots or config.full_snapshots_path),
        model_dir=Path(args.model_dir or config.artifacts_model_dir),
        output_dir=Path(args.output_dir or config.artifacts_evaluation_dir),
        split=args.split,
    )
    print("")
    print(f"Artifacts: {Path(args.output_dir or config.artifacts_evaluation_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
