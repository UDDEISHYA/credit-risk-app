from __future__ import annotations

from runtime_guard import require_compatible_runtime

require_compatible_runtime()

import plotly.express as px
import streamlit as st

from common import get_bundle, get_predictions, money, page_setup


page_setup("Explainable Credit Underwriting")
bundle = get_bundle()
predictions = get_predictions()
policy = bundle["policy_config"]
predictions["decision"] = "MANUAL_REVIEW"
predictions.loc[predictions["PD_CHAMPION"] < policy["approve_pd_max"], "decision"] = "APPROVE"
predictions.loc[predictions["PD_CHAMPION"] >= policy["reject_pd_min"], "decision"] = "REJECT"
predictions["grade"] = predictions["PD_CHAMPION"].map(
    lambda value: "A" if value < 0.02 else "B" if value < 0.05 else "C" if value < 0.10 else "D" if value < 0.20 else "E"
)

st.caption(
    "A portfolio demonstration that turns calibrated next-month default probabilities into "
    "scores, financial impact, decisions, explanations and monitoring."
)
columns = st.columns(5)
columns[0].metric("Test applications", f"{len(predictions):,}")
columns[1].metric("Approval rate", f"{(predictions['decision'] == 'APPROVE').mean():.1%}")
columns[2].metric("Average calibrated PD", f"{predictions['PD_CHAMPION'].mean():.1%}")
expected_loss = (predictions["PD_CHAMPION"] * policy["lgd"] * predictions["LIMIT_BAL"]).sum()
columns[3].metric("Expected portfolio loss", money(expected_loss, policy["currency"]))
columns[4].metric("Latest demo monitoring", "GREEN", help="Reference batch compared with itself; use the Monitoring page for stress scenarios.")

left, right = st.columns(2)
with left:
    st.plotly_chart(
        px.histogram(
            predictions,
            x="grade",
            category_orders={"grade": list("ABCDE")},
            title="Risk-grade distribution",
            color="grade",
        ),
        use_container_width=True,
    )
with right:
    funnel = predictions["decision"].value_counts().rename_axis("decision").reset_index(name="applications")
    st.plotly_chart(px.funnel(funnel, x="applications", y="decision", title="Decision funnel"), use_container_width=True)

champion_metrics = bundle["test_metrics"][bundle["champion_name"]]["calibrated"]
st.subheader("Current model")
st.write(
    f"Champion: **{bundle['champion_name'].replace('_', ' ').title()}** · "
    f"ROC-AUC **{champion_metrics['roc_auc']:.3f}** · "
    f"Brier score **{champion_metrics['brier_score']:.3f}**"
)
st.warning("Educational model only. It must not be used for real lending or adverse action.")

