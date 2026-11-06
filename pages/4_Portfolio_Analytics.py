from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime_guard import require_compatible_runtime

require_compatible_runtime()

import numpy as np
import plotly.express as px
import streamlit as st

from common import get_bundle, get_predictions, page_setup
from credit_risk.decisioning import portfolio_policy_frontier


page_setup("Portfolio analytics", "📊")
bundle = get_bundle()
predictions = get_predictions()
policy = bundle["policy_config"]
predictions["grade"] = predictions["PD_CHAMPION"].map(
    lambda value: "A" if value < 0.02 else "B" if value < 0.05 else "C" if value < 0.10 else "D" if value < 0.20 else "E"
)
predictions["expected_loss"] = predictions["PD_CHAMPION"] * policy["lgd"] * predictions["LIMIT_BAL"]
grade_summary = (
    predictions.groupby("grade", observed=False)
    .agg(
        applications=("ID", "size"),
        observed_default_rate=("DEFAULT_NEXT_MONTH", "mean"),
        expected_loss=("expected_loss", "sum"),
        average_limit=("LIMIT_BAL", "mean"),
    )
    .reindex(list("ABCDE"))
    .reset_index()
)

st.subheader("Risk-grade portfolio view")
grade_columns = st.columns(3)
grade_columns[0].plotly_chart(px.bar(grade_summary, x="grade", y="applications", title="Applications by grade"), use_container_width=True)
grade_columns[1].plotly_chart(px.bar(grade_summary, x="grade", y="observed_default_rate", title="Observed default by grade"), use_container_width=True)
grade_columns[2].plotly_chart(px.bar(grade_summary, x="grade", y="expected_loss", title="Expected loss by grade"), use_container_width=True)
st.dataframe(
    grade_summary.style.format({"observed_default_rate": "{:.1%}", "expected_loss": "{:,.0f}", "average_limit": "{:,.0f}"}),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Policy frontier")
frontier = portfolio_policy_frontier(
    predictions["PD_CHAMPION"],
    predictions["DEFAULT_NEXT_MONTH"],
    predictions["LIMIT_BAL"],
    approve_cutoffs=np.linspace(0.02, 0.16, 15),
    reject_cutoff=policy["reject_pd_min"],
    lgd=policy["lgd"],
    interest_rate=policy["annual_interest_rate"],
    operating_cost_rate=policy["operating_cost_rate"],
)
st.dataframe(frontier.style.format({"approval_rate": "{:.1%}", "manual_review_rate": "{:.1%}", "rejection_rate": "{:.1%}", "observed_bad_rate_approved": "{:.1%}", "expected_loss_approved": "{:,.0f}", "expected_profit_approved": "{:,.0f}"}), use_container_width=True, hide_index=True)

left, right = st.columns(2)
with left:
    melted = frontier.melt(id_vars="approve_pd_cutoff", value_vars=["approval_rate", "manual_review_rate", "rejection_rate"], var_name="rate", value_name="value")
    st.plotly_chart(px.line(melted, x="approve_pd_cutoff", y="value", color="rate", markers=True, title="Decision-rate frontier"), use_container_width=True)
with right:
    st.plotly_chart(px.line(frontier, x="approval_rate", y="observed_bad_rate_approved", markers=True, title="Approval versus observed bad rate"), use_container_width=True)
st.plotly_chart(px.line(frontier, x="approval_rate", y="expected_profit_approved", markers=True, title="Scenario expected profit frontier"), use_container_width=True)
st.caption("Profit uses configured scenario assumptions; the dataset contains no contractual pricing or recovery values.")

