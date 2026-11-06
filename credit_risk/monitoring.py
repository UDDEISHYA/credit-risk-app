"""Small, auditable monitoring utilities for drift and model health."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .data import MODEL_INPUT_COLUMNS, TARGET, validate_credit_data
from .decisioning import predict_pd
from .modeling import evaluate_probabilities


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Calculate PSI using reference quantile bins and guarded proportions."""
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()
    if reference.empty or current.empty:
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0 if np.isclose(reference.mean(), current.mean()) else 1.0
    edges[0], edges[-1] = -np.inf, np.inf
    reference_counts, _ = np.histogram(reference, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)
    reference_share = np.clip(reference_counts / max(reference_counts.sum(), 1), 1e-5, None)
    current_share = np.clip(current_counts / max(current_counts.sum(), 1), 1e-5, None)
    return float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))


def psi_status(value: float) -> str:
    if np.isnan(value):
        return "UNKNOWN"
    if value < 0.10:
        return "GREEN"
    if value < 0.25:
        return "AMBER"
    return "RED"


def feature_drift_table(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    columns = columns or MODEL_INPUT_COLUMNS
    rows = []
    for column in columns:
        value = population_stability_index(reference[column], current[column])
        rows.append({"feature": column, "psi": value, "status": psi_status(value)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False, na_position="last").reset_index(drop=True)


def monitor_batch(bundle: dict[str, Any], current: pd.DataFrame) -> dict[str, Any]:
    """Monitor data quality, feature/score drift and performance when labels exist."""
    reference = bundle["reference_data"]
    quality = validate_credit_data(current, strict=False).to_dict()
    feature_drift = feature_drift_table(reference, current)
    current_pd = predict_pd(bundle, current)
    reference_pd = predict_pd(bundle, reference)
    score_psi = population_stability_index(pd.Series(reference_pd), pd.Series(current_pd))
    performance = None
    if TARGET in current and current[TARGET].notna().all():
        performance = evaluate_probabilities(current[TARGET], current_pd)
    worst_status = "GREEN"
    if (feature_drift["status"] == "RED").any() or psi_status(score_psi) == "RED":
        worst_status = "RED"
    elif (feature_drift["status"] == "AMBER").any() or psi_status(score_psi) == "AMBER":
        worst_status = "AMBER"
    action = {
        "GREEN": "Continue routine monitoring.",
        "AMBER": "Investigate shifted inputs and increase monitoring frequency.",
        "RED": "Investigate incoming data before relying on automated decisions.",
    }[worst_status]
    return {
        "overall_status": worst_status,
        "recommended_action": action,
        "data_quality": quality,
        "feature_drift": feature_drift,
        "score_psi": score_psi,
        "score_status": psi_status(score_psi),
        "performance": performance,
        "average_pd": float(np.mean(current_pd)),
    }


def simulate_drift(frame: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Create a transparent adverse scenario for dashboard demonstration."""
    rng = np.random.default_rng(random_state)
    shifted = frame.copy()
    selected = rng.random(len(shifted)) < 0.55
    bill_columns = [column for column in shifted if column.startswith("BILL_AMT")]
    shifted[[*bill_columns, "LIMIT_BAL"]] = shifted[
        [*bill_columns, "LIMIT_BAL"]
    ].astype(float)
    shifted.loc[selected, bill_columns] = shifted.loc[selected, bill_columns] * 1.40
    shifted.loc[selected, "LIMIT_BAL"] = shifted.loc[selected, "LIMIT_BAL"] * 0.80
    delayed = selected & (rng.random(len(shifted)) < 0.70)
    shifted.loc[delayed, "PAY_0"] = np.maximum(shifted.loc[delayed, "PAY_0"], 2)
    return shifted
