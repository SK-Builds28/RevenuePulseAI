"""Train-fitted preprocessing and model pipelines.

All imputation statistics, scaling parameters and one-hot categories are learned
from the training split only. Validation and test frames are transformed through
``Pipeline.transform`` / ``Pipeline.predict_proba`` and never used to fit the
preprocessor.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.snapshots import FEATURE_COLUMNS
from src.utils.config import Config

CATEGORICAL_FEATURES: tuple[str, ...] = ("segment",)
NUMERIC_FEATURES: tuple[str, ...] = tuple(
    column for column in FEATURE_COLUMNS if column not in CATEGORICAL_FEATURES
)

LOGISTIC_REGRESSION = "logistic_regression"
HIST_GRADIENT_BOOSTING = "hist_gradient_boosting"
SUPPORTED_MODELS: tuple[str, ...] = (LOGISTIC_REGRESSION, HIST_GRADIENT_BOOSTING)


def build_preprocessor(*, scale_numeric: bool) -> ColumnTransformer:
    """Build a preprocessing transformer without fitting it.

    Parameters
    ----------
    scale_numeric:
        Standardise numeric features for linear models. Tree models do not need
        scaling, so the HGB pipeline keeps the imputed values on their original
        scale.
    """

    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(NUMERIC_FEATURES)),
            ("categorical", categorical_pipeline, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model_pipeline(model_name: str, config: Config) -> Pipeline:
    """Return an unfitted sklearn pipeline for a supported model name."""

    if model_name == LOGISTIC_REGRESSION:
        classifier = LogisticRegression(**config.logistic_regression_params)
        preprocessor = build_preprocessor(scale_numeric=True)
    elif model_name == HIST_GRADIENT_BOOSTING:
        classifier = HistGradientBoostingClassifier(
            random_state=config.model_random_state,
            **config.hist_gradient_boosting_params,
        )
        preprocessor = build_preprocessor(scale_numeric=False)
    else:
        raise ValueError(
            f"Unsupported model {model_name!r}; expected one of {SUPPORTED_MODELS}"
        )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def model_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Select exactly the allow-listed feature columns for fitting/prediction.

    Raises if a feature is missing. ``account_id``, target columns and other
    metadata can never enter the model matrix through this function.
    """

    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")
    return frame.loc[:, list(FEATURE_COLUMNS)].copy()


def transformed_feature_names(pipeline: Pipeline) -> list[str]:
    """Return the post-preprocessing feature names for a fitted pipeline."""

    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is None:
        raise ValueError("Pipeline has no 'preprocessor' step")
    return [str(name) for name in preprocessor.get_feature_names_out()]


def predict_risk_probability(pipeline: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities from a fitted pipeline."""

    features = model_feature_frame(frame)
    probabilities = pipeline.predict_proba(features)
    positive = np.asarray(probabilities)[:, 1]
    if not np.isfinite(positive).all():
        raise ValueError("Model produced non-finite risk probabilities")
    return np.clip(positive, 0.0, 1.0)


def logistic_regression_coefficients(pipeline: Pipeline) -> pd.DataFrame:
    """Return coefficients and odds ratios with transformed feature names."""

    classifier = pipeline.named_steps.get("classifier")
    if not isinstance(classifier, LogisticRegression):
        raise ValueError("Pipeline classifier is not LogisticRegression")

    names = transformed_feature_names(pipeline)
    coefficients = np.asarray(classifier.coef_).reshape(-1)
    if len(names) != len(coefficients):
        raise ValueError(
            "Transformed feature names and coefficients have different lengths"
        )

    frame = pd.DataFrame(
        {
            "transformed_feature": names,
            "coefficient": coefficients.astype(float),
            "odds_ratio": np.exp(coefficients.astype(float)),
        }
    )
    frame["absolute_coefficient"] = frame["coefficient"].abs()
    return frame.sort_values(
        "absolute_coefficient", ascending=False, kind="mergesort"
    ).reset_index(drop=True)


def model_metadata_parameters(pipeline: Pipeline) -> dict[str, Any]:
    """Return JSON-ish classifier parameters for the model metadata file."""

    classifier = pipeline.named_steps.get("classifier")
    if classifier is None:
        raise ValueError("Pipeline has no 'classifier' step")
    return {
        str(key): _json_safe(value)
        for key, value in classifier.get_params(deep=False).items()
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
