from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime_guard import require_compatible_runtime

require_compatible_runtime()

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import get_bundle, money, page_setup
from credit_risk.data import MODEL_INPUT_COLUMNS
from credit_risk.decisioning import make_decision


page_setup("What-if simulator", "🎛️")
bundle = get_bundle()
reference = bundle["reference_data"].reset_index(drop=True)
customer_id = st.selectbox("Illustrative customer ID", reference["ID"].head(250).tolist())
base = reference.loc[reference["ID"] == customer_id, MODEL_INPUT_COLUMNS].iloc[[0]].copy()

first, second = st.columns(2)
new_limit = first.slider("Credit limit", 10_000, 1_000_000, int(base.iloc[0]["LIMIT_BAL"]), 10_000)
new_pay_0 = second.slider("Most recent repayment status", -2, 8, int(base.iloc[0]["PAY_0"]))
third, fourth = st.columns(2)
new_bill = third.number_input("Most recent bill", value=int(base.iloc[0]["BILL_AMT1"]), step=1_000)
new_payment = fourth.number_input("Most recent payment", min_value=0, value=int(base.iloc[0]["PAY_AMT1"]), step=500)

scenario = base.copy()
scenario.loc[:, "LIMIT_BAL"] = new_limit
scenario.loc[:, "PAY_0"] = new_pay_0
scenario.loc[:, "BILL_AMT1"] = new_bill
scenario.loc[:, "PAY_AMT1"] = new_payment
base_result = make_decision(bundle, base)
scenario_result = make_decision(bundle, scenario)

comparison = pd.DataFrame(
    {
        "metric": ["PD", "Credit score", "Expected loss"],
        "Current": [base_result["pd"], base_result["credit_score"] / 1000, base_result["expected_loss"] / max(base_result["requested_amount"], 1)],
        "Scenario": [scenario_result["pd"], scenario_result["credit_score"] / 1000, scenario_result["expected_loss"] / max(scenario_result["requested_amount"], 1)],
    }
)
figure = go.Figure()
for label in ("Current", "Scenario"):
    figure.add_bar(name=label, x=comparison["metric"], y=comparison[label])
figure.update_layout(barmode="group", title="Normalized metric comparison")
st.plotly_chart(figure, use_container_width=True)
current, changed = st.columns(2)
current.markdown(f"**Current:** {base_result['decision']} · PD {base_result['pd']:.2%} · score {base_result['credit_score']}")
changed.markdown(f"**Scenario:** {scenario_result['decision']} · PD {scenario_result['pd']:.2%} · score {scenario_result['credit_score']}")
changed.caption(f"Scenario expected loss: {money(scenario_result['expected_loss'], scenario_result['currency'])}")
st.warning("Changing one field demonstrates model sensitivity; it does not prove the change would cause a real person's risk to improve.")

