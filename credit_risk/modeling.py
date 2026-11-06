"""Reproducible model development, calibration and artifact persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import PROJECT_ROOT, model_config, policy_config
from .data import MODEL_INPUT_COLUMNS, TARGET, validate_credit_data
from .features import RiskFeatureTransformer


def make_estimators(config: dict[str, Any] | None = None) -> dict[str, Pipeline]:
    """Create the transparent baseline and nonlinear challenger."""
    config = config or model_config()
    seed = int(config["random_state"])
    challenger = config["challenger"]
    baseline = Pipeline(
        [
            ("features", RiskFeatureTransformer()),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=2000, random_state=seed, solver="lbfgs"),
            ),
        ]
    )
    gradient_boosting = Pipeline(
        [
            ("features", RiskFeatureTransformer()),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=float(challenger["learning_rate"]),
                    max_iter=int(challenger["max_iter"]),
                    max_leaf_nodes=int(challenger["max_leaf_nodes"]),
                    min_samples_leaf=int(challenger["min_samples_leaf"]),
                    l2_regularization=float(challenger["l2_regularization"]),
                    random_state=seed,
                ),
            ),
        ]
    )
    return {"logistic_regression": baseline, "hist_gradient_boosting": gradient_boosting}


def split_development_data(
    frame: pd.DataFrame, config: dict[str, Any] | None = None
) -> dict[str, pd.DataFrame | pd.Series]:
    """Create stratified 60/20/20 train/validation/test partitions."""
    config = config or model_config()
    X = frame[MODEL_INPUT_COLUMNS].copy()
    y = frame[TARGET].astype(int).copy()
    indices = frame.index.to_numpy()
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=float(config["test_size"]),
        stratify=y,
        random_state=int(config["random_state"]),
    )
    validation_fraction_of_train_val = float(config["validation_size"]) / (
        1.0 - float(config["test_size"])
    )
    train_idx, validation_idx = train_test_split(
        train_val_idx,
        test_size=validation_fraction_of_train_val,
        stratify=y.loc[train_val_idx],
        random_state=int(config["random_state"]),
    )
    return {
        "X_train": X.loc[train_idx],
        "y_train": y.loc[train_idx],
        "X_validation": X.loc[validation_idx],
        "y_validation": y.loc[validation_idx],
        "X_test": X.loc[test_idx],
        "y_test": y.loc[test_idx],
        "test_index": pd.Series(test_idx, name="source_index"),
    }


def calibration_intercept_slope(y_true: pd.Series | np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    """Estimate calibration intercept and slope from log-odds."""
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(probability / (1 - probability)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(logits, np.asarray(y_true, dtype=int))
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def evaluate_probabilities(
    y_true: pd.Series | np.ndarray,
    probability: np.ndarray,
    threshold: float = 0.50,
) -> dict[str, Any]:
    """Evaluate ranking, calibration and threshold behavior."""
    y_array = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-8, 1 - 1e-8)
    prediction = (probability >= threshold).astype(int)
    auc = roc_auc_score(y_array, probability)
    fpr, tpr, _ = roc_curve(y_array, probability)
    intercept, slope = calibration_intercept_slope(y_array, probability)
    matrix = confusion_matrix(y_array, prediction, labels=[0, 1])
    return {
        "roc_auc": float(auc),
        "gini": float(2 * auc - 1),
        "pr_auc": float(average_precision_score(y_array, probability)),
        "brier_score": float(brier_score_loss(y_array, probability)),
        "log_loss": float(log_loss(y_array, probability, labels=[0, 1])),
        "ks": float(np.max(tpr - fpr)),
        "accuracy": float(accuracy_score(y_array, prediction)),
        "precision": float(precision_score(y_array, prediction, zero_division=0)),
        "recall": float(recall_score(y_array, prediction, zero_division=0)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "threshold": float(threshold),
        "confusion_matrix": matrix.tolist(),
    }


def calibration_table(
    y_true: pd.Series | np.ndarray, probability: np.ndarray, bins: int = 10
) -> pd.DataFrame:
    """Return observed and predicted default rates by equal-count PD band."""
    table = pd.DataFrame({"observed": np.asarray(y_true), "predicted": probability})
    table["pd_band"] = pd.qcut(table["predicted"], q=bins, duplicates="drop")
    return (
        table.groupby("pd_band", observed=False)
        .agg(
            accounts=("observed", "size"),
            mean_predicted_pd=("predicted", "mean"),
            observed_default_rate=("observed", "mean"),
        )
        .reset_index()
        .assign(pd_band=lambda value: value["pd_band"].astype(str))
    )


def _fit_calibration_candidates(
    estimator: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    cv: int,
    threshold: float,
) -> tuple[str, dict[str, dict[str, Any]]]:
    results: dict[str, dict[str, Any]] = {}
    uncalibrated = clone(estimator).fit(X_train, y_train)
    results["uncalibrated"] = evaluate_probabilities(
        y_validation, uncalibrated.predict_proba(X_validation)[:, 1], threshold
    )
    for method in ("sigmoid", "isotonic"):
        calibrated = CalibratedClassifierCV(clone(estimator), method=method, cv=cv, n_jobs=-1)
        calibrated.fit(X_train, y_train)
        results[method] = evaluate_probabilities(
            y_validation, calibrated.predict_proba(X_validation)[:, 1], threshold
        )
    selected = min(
        ("sigmoid", "isotonic"),
        key=lambda method: (results[method]["brier_score"], results[method]["log_loss"]),
    )
    return selected, results


def train_and_evaluate(frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """Train, select calibration on validation data, refit, and test once."""
    validate_credit_data(frame, strict=True)
    config = model_config()
    policy = policy_config()
    split = split_development_data(frame, config)
    estimators = make_estimators(config)
    threshold = float(config["classification_threshold"])
    selection: dict[str, Any] = {}
    selected_methods: dict[str, str] = {}

    for name, estimator in estimators.items():
        method, candidate_results = _fit_calibration_candidates(
            estimator,
            split["X_train"],
            split["y_train"],
            split["X_validation"],
            split["y_validation"],
            cv=int(config["calibration_cv"]),
            threshold=threshold,
        )
        selected_methods[name] = method
        selection[name] = {"selected_method": method, "candidates": candidate_results}

    def validation_utility(model_name: str) -> float:
        metrics = selection[model_name]["candidates"][selected_methods[model_name]]
        return float(metrics["roc_auc"] - metrics["brier_score"])

    baseline_utility = validation_utility("logistic_regression")
    challenger_utility = validation_utility("hist_gradient_boosting")
    champion_name = (
        "hist_gradient_boosting"
        if challenger_utility >= baseline_utility - 0.005
        else "logistic_regression"
    )

    X_development = pd.concat([split["X_train"], split["X_validation"]]).sort_index()
    y_development = pd.concat([split["y_train"], split["y_validation"]]).sort_index()
    fitted_models: dict[str, Any] = {}
    test_metrics: dict[str, Any] = {}
    test_probabilities: dict[str, np.ndarray] = {}

    for name, estimator in estimators.items():
        uncalibrated = clone(estimator).fit(X_development, y_development)
        uncalibrated_probability = uncalibrated.predict_proba(split["X_test"])[:, 1]
        calibrated = CalibratedClassifierCV(
            clone(estimator),
            method=selected_methods[name],
            cv=int(config["final_calibration_cv"]),
            n_jobs=-1,
        ).fit(X_development, y_development)
        calibrated_probability = calibrated.predict_proba(split["X_test"])[:, 1]
        fitted_models[name] = calibrated
        test_probabilities[name] = calibrated_probability
        test_metrics[name] = {
            "uncalibrated": evaluate_probabilities(
                split["y_test"], uncalibrated_probability, threshold
            ),
            "calibrated": evaluate_probabilities(
                split["y_test"], calibrated_probability, threshold
            ),
            "calibration_method": selected_methods[name],
            "calibration_table": calibration_table(
                split["y_test"], calibrated_probability
            ).to_dict(orient="records"),
        }

    champion_probability = test_probabilities[champion_name]
    test_indices = split["X_test"].index
    predictions = frame.loc[test_indices, ["ID", TARGET, "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]].copy()
    predictions["PD_LOGISTIC"] = test_probabilities["logistic_regression"]
    predictions["PD_CHALLENGER"] = test_probabilities["hist_gradient_boosting"]
    predictions["PD_CHAMPION"] = champion_probability

    reference_columns = ["ID", *MODEL_INPUT_COLUMNS, "SEX", "EDUCATION", "MARRIAGE", "AGE", TARGET]
    reference_data = frame.loc[X_development.index, reference_columns].sample(
        n=min(5000, len(X_development)), random_state=int(config["random_state"])
    )
    bundle = {
        "model_version": "1.0.0",
        "champion_name": champion_name,
        "models": fitted_models,
        "selected_calibration_methods": selected_methods,
        "selection_results": selection,
        "test_metrics": test_metrics,
        "model_config": config,
        "policy_config": policy,
        "model_input_columns": MODEL_INPUT_COLUMNS,
        "reference_data": reference_data,
        "training_summary": {
            "rows": len(frame),
            "development_rows": len(X_development),
            "test_rows": len(split["X_test"]),
            "default_rate": float(frame[TARGET].mean()),
            "test_default_rate": float(split["y_test"].mean()),
        },
    }
    return bundle, predictions


def save_training_artifacts(
    bundle: dict[str, Any],
    predictions: pd.DataFrame,
    artifact_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Persist the model bundle and human-readable evaluation outputs."""
    directory = Path(artifact_dir or PROJECT_ROOT / "artifacts")
    directory.mkdir(parents=True, exist_ok=True)
    bundle_path = directory / "model_bundle.joblib"
    metrics_path = directory / "metrics.json"
    predictions_path = directory / "test_predictions.csv"
    joblib.dump(bundle, bundle_path)
    serializable = {
        key: bundle[key]
        for key in (
            "model_version",
            "champion_name",
            "selected_calibration_methods",
            "selection_results",
            "test_metrics",
            "model_config",
            "policy_config",
            "training_summary",
        )
    }
    metrics_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    predictions.to_csv(predictions_path, index=False)
    return {"bundle": bundle_path, "metrics": metrics_path, "predictions": predictions_path}


def load_model_bundle(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path or PROJECT_ROOT / "artifacts/model_bundle.joblib")
    return joblib.load(path)
