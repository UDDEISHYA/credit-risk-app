"""Translate calibrated probabilities into controlled lending decisions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import policy_config
from .data import BILL_COLUMNS, MODEL_INPUT_COLUMNS, PAYMENT_COLUMNS, PAY_STATUS_COLUMNS


def score_from_pd(
    pd_value: float,
    base_score: float = 600,
    base_good_to_bad_odds: float = 20,
    points_to_double_odds: float = 50,
) -> int:
    """Convert PD to a traditional odds-based score (higher is safer)."""
    probability = float(np.clip(pd_value, 1e-6, 1 - 1e-6))
    factor = points_to_double_odds / np.log(2.0)
    good_to_bad_odds = (1.0 - probability) / probability
    score = base_score + factor * np.log(good_to_bad_odds / base_good_to_bad_odds)
    return int(np.clip(np.rint(score), 300, 850))


def risk_grade(pd_value: float) -> str:
    if pd_value < 0.02:
        return "A"
    if pd_value < 0.05:
        return "B"
    if pd_value < 0.10:
        return "C"
    if pd_value < 0.20:
        return "D"
    return "E"


def predict_pd(bundle: dict[str, Any], applicants: pd.DataFrame) -> np.ndarray:
    """Score applicants with the persisted champion model."""
    missing = sorted(set(MODEL_INPUT_COLUMNS) - set(applicants.columns))
    if missing:
        raise ValueError(f"Missing scoring inputs: {missing}")
    model = bundle["models"][bundle["champion_name"]]
    return model.predict_proba(applicants[MODEL_INPUT_COLUMNS])[:, 1]


def _reason_scenarios(applicant: pd.DataFrame, reference: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    row = applicant.copy()
    limit = float(row.iloc[0]["LIMIT_BAL"])
    scenarios: list[tuple[str, pd.DataFrame]] = []

    improved = row.copy()
    improved.loc[:, PAY_STATUS_COLUMNS] = improved[PAY_STATUS_COLUMNS].clip(upper=0)
    scenarios.append(("Recent or repeated repayment delays", improved))

    improved = row.copy()
    for column in BILL_COLUMNS:
        improved.loc[:, column] = np.minimum(improved[column], 0.30 * limit)
    scenarios.append(("High balance relative to the available credit limit", improved))

    improved = row.copy()
    for payment, bill in zip(PAYMENT_COLUMNS, BILL_COLUMNS):
        target_payment = np.maximum(0.10 * improved[bill].clip(lower=0), improved[payment])
        improved.loc[:, payment] = target_payment
    scenarios.append(("Low recent payments relative to billed balances", improved))

    improved = row.copy()
    recent_floor = np.minimum(improved["BILL_AMT6"], improved["BILL_AMT1"])
    improved.loc[:, "BILL_AMT1"] = recent_floor
    improved.loc[:, "BILL_AMT2"] = np.minimum(improved["BILL_AMT2"], recent_floor)
    scenarios.append(("Recent billed balances are increasing", improved))

    improved = row.copy()
    reference_limit = float(reference["LIMIT_BAL"].quantile(0.75))
    improved.loc[:, "LIMIT_BAL"] = max(limit, reference_limit)
    scenarios.append(("Low credit limit relative to the observed balance pattern", improved))
    return scenarios


def reason_codes(
    bundle: dict[str, Any], applicant: pd.DataFrame, max_reasons: int = 4
) -> list[dict[str, float | str]]:
    """Create model-agnostic, behavioral reason codes using controlled perturbations.

    Each scenario improves one behavioral risk theme. The PD reduction is used
    to rank reasons. It is a local sensitivity explanation, not a causal claim.
    """
    applicant = applicant.iloc[[0]][MODEL_INPUT_COLUMNS].copy()
    base_probability = float(predict_pd(bundle, applicant)[0])
    impacts: list[dict[str, float | str]] = []
    for label, scenario in _reason_scenarios(applicant, bundle["reference_data"]):
        improved_probability = float(predict_pd(bundle, scenario)[0])
        reduction = max(0.0, base_probability - improved_probability)
        if reduction > 1e-4:
            impacts.append(
                {
                    "reason": label,
                    "pd_reduction": reduction,
                    "improved_pd": improved_probability,
                }
            )
    impacts.sort(key=lambda item: float(item["pd_reduction"]), reverse=True)
    if not impacts:
        impacts.append(
            {
                "reason": "No single tested behavioral factor materially dominates this score",
                "pd_reduction": 0.0,
                "improved_pd": base_probability,
            }
        )
    return impacts[:max_reasons]


def make_decision(
    bundle: dict[str, Any],
    applicant: pd.DataFrame,
    requested_amount: float | None = None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return PD, score, grade, financial metrics, decision and explanations."""
    policy = policy or bundle.get("policy_config") or policy_config()
    applicant = applicant.iloc[[0]].copy()
    probability = float(predict_pd(bundle, applicant)[0])
    limit = float(applicant.iloc[0]["LIMIT_BAL"])
    exposure = float(requested_amount if requested_amount is not None else limit)
    exposure = max(0.0, exposure)
    grade = risk_grade(probability)
    score_settings = policy["score"]
    score = score_from_pd(
        probability,
        base_score=float(score_settings["base_score"]),
        base_good_to_bad_odds=float(score_settings["base_good_to_bad_odds"]),
        points_to_double_odds=float(score_settings["points_to_double_odds"]),
    )
    expected_loss = probability * float(policy["lgd"]) * exposure
    expected_interest = (1.0 - probability) * float(policy["annual_interest_rate"]) * exposure
    operating_cost = float(policy["operating_cost_rate"]) * exposure
    expected_profit = expected_interest - expected_loss - operating_cost

    if probability < float(policy["approve_pd_max"]):
        decision = "APPROVE"
    elif probability < float(policy["reject_pd_min"]):
        decision = "MANUAL_REVIEW"
    else:
        decision = "REJECT"

    max_delay = int(applicant[PAY_STATUS_COLUMNS].max(axis=1).iloc[0])
    policy_reasons: list[str] = []
    if max_delay >= int(policy["severe_delinquency_months"]):
        decision = "REJECT"
        policy_reasons.append("Severe repayment delay rule triggered")
    elif max_delay >= int(policy["review_delinquency_months"]) and decision == "APPROVE":
        decision = "MANUAL_REVIEW"
        policy_reasons.append("Repayment-delay review rule triggered")
    if expected_profit < 0 and decision == "APPROVE":
        decision = "MANUAL_REVIEW"
        policy_reasons.append("Scenario expected profit is negative")

    multiplier = float(policy["recommended_limit_multiplier"][grade])
    recommended_amount = 0.0 if decision == "REJECT" else min(exposure, limit * multiplier)
    recommended_amount = float(np.floor(recommended_amount / 1000.0) * 1000.0)
    return {
        "pd": probability,
        "credit_score": score,
        "risk_grade": grade,
        "decision": decision,
        "expected_loss": expected_loss,
        "expected_profit": expected_profit,
        "requested_amount": exposure,
        "recommended_amount": recommended_amount,
        "currency": policy["currency"],
        "policy_reasons": policy_reasons,
        "reason_codes": reason_codes(bundle, applicant),
    }


def portfolio_policy_frontier(
    probabilities: np.ndarray,
    observed_default: pd.Series | np.ndarray,
    exposure: pd.Series | np.ndarray,
    approve_cutoffs: np.ndarray | None = None,
    reject_cutoff: float = 0.20,
    lgd: float = 0.60,
    interest_rate: float = 0.15,
    operating_cost_rate: float = 0.02,
) -> pd.DataFrame:
    """Compare three-way decision policies without retraining the model."""
    probabilities = np.asarray(probabilities, dtype=float)
    observed = np.asarray(observed_default, dtype=int)
    exposure = np.asarray(exposure, dtype=float)
    cutoffs = approve_cutoffs if approve_cutoffs is not None else np.linspace(0.02, 0.16, 15)
    rows: list[dict[str, float]] = []
    for cutoff in cutoffs:
        approved = probabilities < cutoff
        review = (probabilities >= cutoff) & (probabilities < reject_cutoff)
        rejected = probabilities >= reject_cutoff
        approved_exposure = exposure[approved]
        approved_pd = probabilities[approved]
        profit = (
            (1 - approved_pd) * interest_rate * approved_exposure
            - approved_pd * lgd * approved_exposure
            - operating_cost_rate * approved_exposure
        )
        rows.append(
            {
                "approve_pd_cutoff": float(cutoff),
                "approval_rate": float(approved.mean()),
                "manual_review_rate": float(review.mean()),
                "rejection_rate": float(rejected.mean()),
                "observed_bad_rate_approved": float(observed[approved].mean()) if approved.any() else np.nan,
                "expected_loss_approved": float(np.sum(approved_pd * lgd * approved_exposure)),
                "expected_profit_approved": float(np.sum(profit)),
                "average_approved_amount": float(approved_exposure.mean()) if approved.any() else 0.0,
            }
        )
    return pd.DataFrame(rows)

