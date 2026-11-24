"""Account-level bootstrap confidence intervals.

Snapshot rows for the same account are correlated. Resampling rows independently
would understate uncertainty, so the bootstrap resamples unique accounts with
replacement and keeps all of each sampled account's snapshot rows together.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.evaluation.metrics import pr_auc, ranking_metrics


def _percentile_interval(values: Sequence[float]) -> tuple[float | None, float | None, int]:
    clean = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if len(clean) == 0:
        return None, None, 0
    lower, upper = np.percentile(clean, [2.5, 97.5])
    return float(lower), float(upper), int(len(clean))


def account_cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    target_column: str,
    model_scores: Mapping[str, Sequence[float]],
    primary_k: int,
    iterations: int,
    seed: int,
    baseline_name: str = "baseline",
) -> dict[str, Any]:
    """Bootstrap Precision@K and PR-AUC by resampling accounts.

    Returns point estimates, percentile 95% intervals, valid replicate counts and
    the paired Precision@K difference/ratio versus the baseline.
    """

    if "account_id" not in frame.columns:
        raise ValueError("Account-level bootstrap requires an 'account_id' column")
    if target_column not in frame.columns:
        raise ValueError(f"Missing target column: {target_column}")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if primary_k <= 0:
        raise ValueError("primary_k must be positive")

    y = frame[target_column].to_numpy().astype(int)
    groups = {
        str(account): np.asarray(indices, dtype=int)
        for account, indices in frame.groupby("account_id", sort=True).indices.items()
    }
    account_ids = sorted(groups)
    if baseline_name not in model_scores:
        raise ValueError(f"Baseline model {baseline_name!r} is missing from model_scores")

    scores = {
        str(name): np.asarray(values, dtype=float)
        for name, values in model_scores.items()
    }
    for name, values in scores.items():
        if len(values) != len(frame):
            raise ValueError(f"Score length mismatch for model {name!r}")
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite scores for model {name!r}")

    point_precision = {
        name: ranking_metrics(y, values, primary_k)["precision"]
        for name, values in scores.items()
    }
    point_pr_auc = {
        name: pr_auc(y, values)
        for name, values in scores.items()
    }

    precision_samples: dict[str, list[float]] = {name: [] for name in scores}
    pr_auc_samples: dict[str, list[float]] = {name: [] for name in scores}
    precision_diff_samples: dict[str, list[float]] = {
        name: [] for name in scores if name != baseline_name
    }
    precision_ratio_samples: dict[str, list[float]] = {
        name: [] for name in scores if name != baseline_name
    }

    rng = np.random.default_rng(int(seed))
    n_accounts = len(account_ids)

    for _ in range(int(iterations)):
        sampled_accounts = rng.choice(account_ids, size=n_accounts, replace=True)
        indices = np.concatenate([groups[str(account)] for account in sampled_accounts])
        y_rep = y[indices]
        if len(np.unique(y_rep)) < 2:
            # Precision@K is still defined with one class; PR-AUC is not.
            pr_auc_valid = False
        else:
            pr_auc_valid = True

        replicate_precision: dict[str, float] = {}
        for name, values in scores.items():
            precision = ranking_metrics(y_rep, values[indices], primary_k)["precision"]
            replicate_precision[name] = precision
            precision_samples[name].append(precision)
            if pr_auc_valid:
                pr_auc_samples[name].append(pr_auc(y_rep, values[indices]))

        baseline_precision = replicate_precision[baseline_name]
        for name in precision_diff_samples:
            difference = replicate_precision[name] - baseline_precision
            precision_diff_samples[name].append(difference)
            if baseline_precision > 0:
                precision_ratio_samples[name].append(
                    replicate_precision[name] / baseline_precision
                )

    results: dict[str, Any] = {
        "method": "account_cluster_percentile_bootstrap",
        "iterations": int(iterations),
        "seed": int(seed),
        "primary_k": int(primary_k),
        "n_rows": int(len(frame)),
        "n_accounts": int(n_accounts),
        "baseline_name": baseline_name,
        "interval": "95% percentile interval (2.5th, 97.5th)",
        "resampling_unit": "account_id (all snapshot rows kept together)",
        "models": {},
    }

    for name in scores:
        lower, upper, valid = _percentile_interval(precision_samples[name])
        pr_lower, pr_upper, pr_valid = _percentile_interval(pr_auc_samples[name])
        model_result: dict[str, Any] = {
            "precision_at_k": {
                "point": float(point_precision[name]),
                "lower": lower,
                "upper": upper,
                "valid_replicates": valid,
            },
            "pr_auc": {
                "point": float(point_pr_auc[name]) if np.isfinite(point_pr_auc[name]) else None,
                "lower": pr_lower,
                "upper": pr_upper,
                "valid_replicates": pr_valid,
            },
        }
        if name != baseline_name:
            diff_lower, diff_upper, diff_valid = _percentile_interval(
                precision_diff_samples[name]
            )
            ratio_lower, ratio_upper, ratio_valid = _percentile_interval(
                precision_ratio_samples[name]
            )
            model_result["precision_difference_vs_baseline_at_k"] = {
                "point": float(point_precision[name] - point_precision[baseline_name]),
                "lower": diff_lower,
                "upper": diff_upper,
                "valid_replicates": diff_valid,
            }
            model_result["precision_ratio_vs_baseline_at_k"] = {
                "point": (
                    float(point_precision[name] / point_precision[baseline_name])
                    if point_precision[baseline_name] > 0
                    else None
                ),
                "lower": ratio_lower,
                "upper": ratio_upper,
                "valid_replicates": ratio_valid,
            }
        results["models"][name] = model_result

    results["limitations"] = (
        "Synthetic test set; wide intervals are expected. The percentile "
        "bootstrap does not correct model-selection uncertainty, because model "
        "selection used the validation split rather than the test split."
    )
    return results
