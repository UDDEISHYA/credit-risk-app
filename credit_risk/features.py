"""Leakage-aware behavioral features used by both models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .data import BILL_COLUMNS, MODEL_INPUT_COLUMNS, PAYMENT_COLUMNS, PAY_STATUS_COLUMNS


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.abs().clip(lower=1.0)
    return (numerator / denominator).replace([np.inf, -np.inf], 0.0).clip(-5.0, 10.0)


def build_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create stable, interpretable features from application-time fields.

    Demographics and the target are deliberately absent. Bill and payment
    values are normalized where possible so the models can compare customers
    with different credit limits.
    """
    missing = sorted(set(MODEL_INPUT_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Cannot build model features; missing: {missing}")

    source = frame[MODEL_INPUT_COLUMNS].astype(float).copy()
    limit = source["LIMIT_BAL"].clip(lower=1.0)
    features = pd.DataFrame(index=frame.index)
    features["log_limit_bal"] = np.log1p(limit)

    for column in PAY_STATUS_COLUMNS:
        features[column.lower()] = source[column]

    for index, column in enumerate(BILL_COLUMNS, start=1):
        features[f"utilization_{index}"] = (source[column] / limit).clip(-2.0, 5.0)

    for index, (payment, bill) in enumerate(zip(PAYMENT_COLUMNS, BILL_COLUMNS), start=1):
        features[f"payment_to_bill_{index}"] = _safe_ratio(source[payment], source[bill])

    pay_status = source[PAY_STATUS_COLUMNS]
    utilizations = features[[f"utilization_{i}" for i in range(1, 7)]]
    payment_ratios = features[[f"payment_to_bill_{i}" for i in range(1, 7)]]
    features["max_delinquency"] = pay_status.max(axis=1)
    features["months_delinquent"] = (pay_status > 0).sum(axis=1)
    features["months_severely_delinquent"] = (pay_status >= 2).sum(axis=1)
    features["avg_utilization"] = utilizations.mean(axis=1)
    features["max_utilization"] = utilizations.max(axis=1)
    features["recent_utilization"] = utilizations["utilization_1"]
    features["avg_payment_to_bill"] = payment_ratios.mean(axis=1)
    features["zero_payment_months"] = (source[PAYMENT_COLUMNS] == 0).sum(axis=1)
    features["log_total_payments"] = np.log1p(source[PAYMENT_COLUMNS].sum(axis=1).clip(lower=0))
    features["bill_trend"] = ((source["BILL_AMT1"] - source["BILL_AMT6"]) / limit).clip(-5, 5)
    features["delinquency_trend"] = source["PAY_0"] - source["PAY_6"]
    return features.astype(float)


class RiskFeatureTransformer(TransformerMixin, BaseEstimator):
    """Scikit-learn compatible wrapper around :func:`build_model_features`."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "RiskFeatureTransformer":
        self.feature_names_in_ = np.asarray(MODEL_INPUT_COLUMNS, dtype=object)
        self.feature_names_out_ = np.asarray(build_model_features(X).columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return build_model_features(X).to_numpy(dtype=float)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        return self.feature_names_out_
