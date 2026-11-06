from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime_guard import require_compatible_runtime

require_compatible_runtime()

import pandas as pd
import streamlit as st

from common import get_bundle, money, page_setup
from credit_risk.data import BILL_COLUMNS, MODEL_INPUT_COLUMNS, PAYMENT_COLUMNS, PAY_STATUS_COLUMNS
from credit_risk.decisioning import make_decision


page_setup("Applicant decision", "🧾")
bundle = get_bundle()
reference = bundle["reference_data"]
defaults = reference[MODEL_INPUT_COLUMNS].median(numeric_only=True)

with st.form("application"):
    limit = st.number_input("Existing credit limit (TWD)", 10_000, 1_000_000, int(defaults["LIMIT_BAL"]), step=10_000)
    requested = st.number_input("Scenario exposure / requested amount (TWD)", 1_000, 1_000_000, int(limit), step=1_000)
    st.markdown("**Repayment status** — negative/zero values are current; positive values indicate months delayed.")
    status_columns = st.columns(6)
    statuses = {
        name: status_columns[index].number_input(name, -2, 8, int(defaults[name]))
        for index, name in enumerate(PAY_STATUS_COLUMNS)
    }
    with st.expander("Monthly bill and payment history"):
        values = {}
        for bill, payment in zip(BILL_COLUMNS, PAYMENT_COLUMNS):
            first, second = st.columns(2)
            values[bill] = first.number_input(bill, value=int(defaults[bill]), step=1_000)
            values[payment] = second.number_input(payment, min_value=0, value=max(0, int(defaults[payment])), step=500)
    submitted = st.form_submit_button("Generate controlled decision", type="primary")

if submitted:
    row = {"LIMIT_BAL": limit, **statuses, **values}
    applicant = pd.DataFrame([row], columns=MODEL_INPUT_COLUMNS)
    result = make_decision(bundle, applicant, requested_amount=requested)
    st.subheader(result["decision"].replace("_", " "))
    metrics = st.columns(5)
    metrics[0].metric("Calibrated PD", f"{result['pd']:.2%}")
    metrics[1].metric("Credit score", result["credit_score"])
    metrics[2].metric("Risk grade", result["risk_grade"])
    metrics[3].metric("Expected loss", money(result["expected_loss"], result["currency"]))
    metrics[4].metric("Recommended amount", money(result["recommended_amount"], result["currency"]))
    if result["policy_reasons"]:
        st.info("; ".join(result["policy_reasons"]))
    st.markdown("**Principal behavioral factors**")
    for index, reason in enumerate(result["reason_codes"], start=1):
        st.write(f"{index}. {reason['reason']} (local PD sensitivity: {reason['pd_reduction']:.2%})")
    st.caption("Sensitivities are associations under controlled input changes, not causal promises or legal adverse-action reasons.")

